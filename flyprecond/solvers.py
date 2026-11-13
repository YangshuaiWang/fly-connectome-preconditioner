"""Trained circuits as preconditioners: loading, transfer to other grids, FGMRES and the multilevel cycle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

from .connectome import compile_flyvis_connectome
from .controls import cut_boundary_input
from .field_operator import RetinotopicFlyOperator
from .hex_pde import HexPoissonGrid, make_poisson_dataset
from .krylov import flexible_gmres, model_direction

ROOT = Path(__file__).resolve().parents[1]
RTOL = 1e-8
COARSEST = 4
TEST_SEED = 20261003
N_TEST = {8: 64, 16: 64, 32: 32, 64: 16}


# --------------------------------------------------------------------------- circuits
def build_circuit(config: dict, extent: int, seed: int = 0, root: Path = ROOT):
    """The fly circuit on the lattice of the given extent, with untrained parameters."""
    grid = HexPoissonGrid.build(extent)
    graph = compile_flyvis_connectome(Path(root) / config["connectome"], extent=extent)
    if config.get("boundary_mode", "cut") == "cut":
        graph, _ = cut_boundary_input(graph, grid.axial, grid.interior)
    model = RetinotopicFlyOperator(graph, grid.axial, grid.interior, steps=config["steps"], seed=seed,
                                   input_encoding=config["input_encoding"], input_current_scale=config["input_current_scale"],
                                   release=config.get("release", "relu"))
    return model, grid


def transfer(model: RetinotopicFlyOperator, new: RetinotopicFlyOperator) -> RetinotopicFlyOperator:
    """Copy all type-shared parameters; pair gains are copied as effective gains, since the initial pair gains
    depend on the graph."""
    if model.pair_keys != new.pair_keys or model.cell_types != new.cell_types or model.output_types != new.output_types:
        raise ValueError("type sets differ between the two graphs")
    with torch.no_grad():
        new.log_gain_multiplier.copy_(model.log_gain_multiplier + torch.log(model.initial_pair_gain) - torch.log(new.initial_pair_gain))
        for name in ("bias", "raw_tau", "log_input_gain", "decoder"):
            getattr(new, name).copy_(getattr(model, name))
    return new.eval()


def load_circuit(path, extent: int | None = None, root: Path = ROOT):
    """Rebuild a trained circuit from its parameter file, on the training grid or transferred to another extent."""
    data = json.loads(Path(path).read_text())
    config, seed = data["config"], data["seed"]
    model, grid = build_circuit(config, config["extent"], seed, root)
    with torch.no_grad():
        for name, values in data["parameters"].items():
            target = getattr(model, name)
            target.copy_(torch.tensor(values, dtype=target.dtype))
    model.eval()
    if extent is None or extent == config["extent"]:
        return model, grid
    new, grid = build_circuit(config, extent, seed, root)
    return transfer(model, new), grid


def save_parameters(model: RetinotopicFlyOperator, config: dict, seed: int, path) -> None:
    keys = ("connectome", "extent", "steps", "input_encoding", "input_current_scale", "boundary_mode", "release")
    params = {n: p.detach().double().tolist() for n, p in model.named_parameters()}
    Path(path).write_text(json.dumps({"seed": seed, "config": {k: config[k] for k in keys}, "parameters": params}))


# --------------------------------------------------------------------------- forcings and FGMRES
def balanced_forcings(extent: int, n: int | None = None, seed: int = TEST_SEED) -> torch.Tensor:
    """Normalized test forcings of the balanced family on the lattice of the given extent."""
    grid = HexPoissonGrid.build(extent)
    data = make_poisson_dataset(grid, n_train=128, n_validation=32, n_test=64, n_ood=64, seed=seed, family="balanced",
                                n_ood_high_band=64)
    n = n or N_TEST.get(extent, 8)
    return torch.tensor(data.splits["test"].forcing[:n] / data.normalization.force_scale, dtype=torch.float64)


def interior_direction(apply_interior, interior):
    """Wrap a map on interior arrays (batch, n_int) as an FGMRES direction on lattice fields."""
    mask = torch.as_tensor(interior, dtype=torch.bool)

    def direction(values):
        z = torch.zeros_like(values)
        z[:, mask] = torch.as_tensor(apply_interior(values[:, mask].numpy()), dtype=values.dtype)
        return z
    return direction


def circuit_direction(model, grid):
    """The circuit as a right preconditioner; Arnoldi vectors are rescaled to unit RMS before each call."""
    return model_direction(model, torch.float32, grid.interior, residual_rms=1.0)


def fgmres_iterations(direction, forcing, grid, cap: int = 500) -> np.ndarray:
    """FGMRES iteration counts per forcing (zero initial guess, no restart, relative tolerance 1e-8)."""
    res = flexible_gmres(direction, forcing, grid.laplacian, grid.interior, rtol=RTOL, restart=cap, max_iterations=cap,
                         keep_sparse=grid.n_sites > 5000)
    converged = np.array([s == "converged" for s in res.status])
    return np.where(converged, res.iterations.numpy().astype(int), cap)


# --------------------------------------------------------------------------- multilevel cycle
def prolongation(fine: HexPoissonGrid, coarse: HexPoissonGrid) -> sparse.csr_matrix:
    """Interior-to-interior linear prolongation (n_int_fine x n_int_coarse)."""
    fi = np.flatnonzero(fine.interior)
    ci = np.flatnonzero(coarse.interior)
    cpos = {int(j): k for k, j in enumerate(ci)}
    rows, cols, vals = [], [], []
    for r, i in enumerate(fi):
        u, v = (int(x) for x in fine.axial[i])
        if u % 2 == 0 and v % 2 == 0:
            pairs, w = [(u, v)], 1.0
        elif u % 2 != 0 and v % 2 == 0:
            pairs, w = [(u - 1, v), (u + 1, v)], 0.5
        elif u % 2 == 0:
            pairs, w = [(u, v - 1), (u, v + 1)], 0.5
        else:
            pairs, w = [(u - 1, v + 1), (u + 1, v - 1)], 0.5
        for a, b in pairs:
            j = coarse._lookup.get((a // 2, b // 2))
            if j is not None and j in cpos:
                rows.append(r)
                cols.append(cpos[j])
                vals.append(w)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(fi), len(ci)))


class CircuitLevel:
    """The trained circuit on one level, as a map on interior arrays, scaled by the mesh factor (h/h_tr)^2."""

    def __init__(self, path, extent, train_extent=8, root: Path = ROOT):
        self.model, self.grid = load_circuit(path, extent, root)
        self.direction = circuit_direction(self.model, self.grid)
        self.mask = np.asarray(self.grid.interior)
        self.scale = (train_extent / extent) ** 2

    def __call__(self, r: np.ndarray) -> np.ndarray:
        full = np.zeros((r.shape[0], len(self.mask)))
        full[:, self.mask] = r
        out = self.direction(torch.tensor(full, dtype=torch.float64)).numpy()
        return self.scale * out[:, self.mask]


class MultilevelCircuit:
    """Galerkin V(1,0) ('vpre') or V(1,1) ('vcycle') cycle with the circuit as level correction on every level except
    the coarsest (extent 4, solved by LU). Each level correction takes the minimal-residual step length."""

    def __init__(self, path, extent, mode="vcycle", root: Path = ROOT):
        if mode not in ("vcycle", "vpre"):
            raise ValueError(mode)
        self.mode = mode
        extents, e = [], extent
        while e > COARSEST:
            extents.append(e)
            e //= 2
        self.extents = extents
        self.grids = [HexPoissonGrid.build(x) for x in extents] + [HexPoissonGrid.build(COARSEST)]
        self.A = [g.interior_matrix.tocsr() for g in self.grids]
        self.P = [prolongation(self.grids[k], self.grids[k + 1]) for k in range(len(extents))]
        self.R = [p.T.tocsr() / 4.0 for p in self.P]
        self.circuits = [CircuitLevel(path, x, root=root) for x in extents]
        self.coarse = splu(self.A[-1].tocsc())

    def _coarse_solve(self, r):
        return np.stack([self.coarse.solve(x) for x in r])

    def _corr(self, k, r):
        c = self.circuits[k](r)
        ac = (self.A[k] @ c.T).T
        num, den = np.sum(r * ac, axis=1), np.sum(ac * ac, axis=1)
        w = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        return c * w[:, None], ac * w[:, None]

    def _cycle(self, r, k=0):
        if k == len(self.circuits):
            return self._coarse_solve(r)
        x, ax = self._corr(k, r)
        x = x + (self.P[k] @ self._cycle((self.R[k] @ (r - ax).T).T, k + 1).T).T
        if self.mode == "vpre":
            return x
        y, _ = self._corr(k, r - (self.A[k] @ x.T).T)
        return x + y

    def __call__(self, r):
        return self._cycle(r)
