#!/usr/bin/env python3
"""Train the cell-type parameters of the fly circuit by minimizing the normalized Dirichlet energy.

The defaults are the settings of the paper: lattice extent 8 (169 DOFs), K = 16 Euler steps, 1000 Adam updates,
128 balanced training forcings and the boundary cut. One run takes about an hour on one CPU core.

Example:
    python scripts/train_circuit.py --seeds 0 --output runs/measured
    python scripts/train_circuit.py --model shell_rewired --seeds 0 --output runs/shell
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyprecond.connectome import compile_flyvis_connectome  # noqa: E402
from flyprecond.controls import cut_boundary_input, rewire_cell_type_degrees  # noqa: E402
from flyprecond.field_operator import RetinotopicFlyOperator  # noqa: E402
from flyprecond.hex_pde import HexPoissonGrid, make_poisson_dataset  # noqa: E402
from flyprecond.solvers import save_parameters  # noqa: E402


def arguments(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connectome", type=Path, default=Path("data/fib25-fib19_v2.2.json"))
    p.add_argument("--output", type=Path, default=Path("runs/measured"))
    p.add_argument("--model", choices=["fly", "type_rewired", "shell_rewired"], default="fly",
                   help="measured wiring, or the wiring rewired within cell-type pairs (type) or within distance shells (shell)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--extent", type=int, default=8)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.015)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--data-seed", type=int, default=20261003)
    p.add_argument("--threads", type=int, default=1)
    return p.parse_args(argv)


def energy(prediction, forcing, operator, interior):
    """Dirichlet energy 1/2 u^T A u - u^T f divided by ||f||^2, averaged over the batch."""
    au = prediction @ operator.T
    per_sample = (0.5 * prediction * au - prediction * forcing)[:, interior].mean(1)
    return (per_sample / forcing[:, interior].square().mean(1).clamp_min(1e-8)).mean()


def predict(model, forcing, batch=16):
    return torch.cat([model(x) for x in forcing.split(batch)])


@torch.no_grad()
def calibrate_decoder(model, forcing, operator, interior, ridge=1e-6) -> None:
    """Refit the readout weights from the normal equations of the training energy."""
    features = torch.cat([model.readout_features(x) for x in forcing.split(16)]).double()
    applied = torch.einsum("ij,njk->nik", operator.double(), features)
    weights = forcing[:, interior].double().square().mean(1).clamp_min(1e-8).reciprocal()
    hessian = torch.einsum("npi,npj,n->ij", features, applied, weights) / len(forcing)
    rhs = torch.einsum("npi,np,n->i", features, forcing.double(), weights) / len(forcing)
    scale = hessian.diag().clamp_min(1e-20).sqrt()
    scaled = hessian / (scale[:, None] * scale[None, :])
    normalized = torch.linalg.solve(scaled + ridge * torch.eye(len(scale)), rhs / scale)
    model.decoder.copy_((normalized / scale).to(model.decoder.dtype))


def train_one(model, dataset, grid, operator, args, seed, destination) -> dict:
    scale = dataset.normalization.force_scale
    forcing = torch.tensor(dataset.splits["train"].forcing / scale, dtype=torch.float32)
    validation = torch.tensor(dataset.splits["validation"].forcing / scale, dtype=torch.float32)
    generator = torch.Generator().manual_seed(50_000 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.updates, eta_min=args.lr * 0.1)
    history, best, best_state, best_update = [], float("inf"), None, 0
    start = perf_counter()
    for update in range(args.updates + 1):
        if update % args.validate_every == 0 or update == args.updates:
            with torch.no_grad():
                val = energy(predict(model, validation), validation, operator, grid.interior).item()
            history.append({"update": update, "validation_energy": val, "seconds": perf_counter() - start})
            if np.isfinite(val) and val < best:
                best, best_state, best_update = val, copy.deepcopy(model.state_dict()), update
            print(f"{destination.name} update={update:4d} validation energy={val:.6g} ({history[-1]['seconds']:.0f} s)", flush=True)
        if update == args.updates:
            break
        batch = forcing[torch.randint(len(forcing), (args.batch_size,), generator=generator)]
        optimizer.zero_grad(set_to_none=True)
        loss = energy(model(batch), batch, operator, grid.interior)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            model.log_gain_multiplier.clamp_(-6, 4)
            model.log_input_gain.clamp_(-4, 3)
            model.raw_tau.clamp_(-9, 0)
    model.load_state_dict(best_state)
    calibrate_decoder(model, forcing, operator, grid.interior)
    with torch.no_grad():
        calibrated = energy(predict(model, validation), validation, operator, grid.interior).item()
    if not (np.isfinite(calibrated) and calibrated < best):
        model.load_state_dict(best_state)
    else:
        best = calibrated
    return {"best_update": best_update, "validation_energy": best, "history": history}


def main(argv=None) -> int:
    args = arguments(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    grid = HexPoissonGrid.build(args.extent)
    dataset = make_poisson_dataset(grid, n_train=128, n_validation=32, n_test=64, n_ood=64, seed=args.data_seed,
                                   family="balanced")
    operator = torch.tensor(grid.laplacian.toarray(), dtype=torch.float32)
    graph = compile_flyvis_connectome(args.connectome, extent=args.extent)
    config = {"connectome": str(args.connectome), "extent": args.extent, "steps": args.steps, "input_encoding": "contrast",
              "input_current_scale": 0.1, "boundary_mode": "cut", "release": "relu"}
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        circuit = graph
        if args.model != "fly":
            circuit, _ = rewire_cell_type_degrees(graph, seed=10_000 + seed, swaps_per_edge=20 if args.model == "shell_rewired" else 5,
                                                  preserve_distance=args.model != "type_rewired",
                                                  distance_bin_width=2.0 if args.model == "shell_rewired" else None)
        circuit, _ = cut_boundary_input(circuit, grid.axial, grid.interior)
        model = RetinotopicFlyOperator(circuit, grid.axial, grid.interior, steps=args.steps, seed=seed,
                                       input_encoding=config["input_encoding"], input_current_scale=config["input_current_scale"],
                                       release=config["release"])
        destination = args.output / f"{args.model}_seed{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        record = train_one(model, dataset, grid, operator, args, seed, destination)
        (destination / "history.json").write_text(json.dumps(record, indent=1))
        if args.model == "fly":
            save_parameters(model, config, seed, destination / "parameters.json")
        else:
            torch.save(model.state_dict(), destination / "state_dict.pt")
        print(f"finished {args.model} seed {seed}: validation energy {record['validation_energy']:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
