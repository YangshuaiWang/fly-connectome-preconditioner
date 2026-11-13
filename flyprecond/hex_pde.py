"""Poisson fields on the same axial hexagonal lattice as the FlyVis retina.

Targets are *discrete* Dirichlet solutions of a six-neighbour finite-difference
operator. Agreement with these targets measures operator-learning error, not
error relative to the continuum PDE. A mesh-refinement study is required for
the latter. Forcing is generated before the solve and is never fitted from a
chosen solution. Boundary forcing values are retained as input observations;
the PDE and all normalization statistics use interior sites only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import splu


_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
_SPLIT_IDS = {
    "train": 11,
    "validation": 23,
    "test": 37,
    "ood_localized": 53,
    "ood_high_frequency": 71,
    "ood_high_band": 89,
}


@dataclass(frozen=True)
class HexPoissonGrid:
    """Unit-circumradius hexagon with homogeneous Dirichlet boundary values.

    ``laplacian`` represents positive ``-Delta`` on interior rows, including
    their boundary-neighbour columns, and is zero on boundary rows. Its interior
    principal submatrix ``interior_matrix`` is symmetric positive definite.
    Matrices can be converted with ``.toarray()`` for use as torch buffers.
    """

    extent: int
    axial: NDArray[np.int64]
    xy: NDArray[np.float64]
    h: float
    interior: NDArray[np.bool_]
    boundary: NDArray[np.bool_]
    interior_indices: NDArray[np.int64]
    laplacian: sparse.csr_matrix
    interior_matrix: sparse.csr_matrix
    _factor: Any = field(repr=False, compare=False)
    _lookup: dict[tuple[int, int], int] = field(repr=False, compare=False)

    @classmethod
    def build(cls, extent: int) -> HexPoissonGrid:
        """Build a grid in the coordinate order used by the connectome compiler."""
        if isinstance(extent, bool) or int(extent) != extent or extent < 1:
            raise ValueError("extent must be a positive integer")
        extent = int(extent)
        axial = np.asarray(
            [
                (u, v)
                for u in range(-extent, extent + 1)
                for v in range(max(-extent, -extent - u), min(extent, extent - u) + 1)
            ],
            dtype=np.int64,
        )
        h = 1.0 / extent
        xy = h * np.column_stack(
            (axial[:, 0] + 0.5 * axial[:, 1], np.sqrt(3) / 2 * axial[:, 1])
        )
        radius = np.max(np.column_stack((np.abs(axial), np.abs(axial.sum(1)))), 1)
        boundary = radius == extent
        interior = ~boundary
        interior_indices = np.flatnonzero(interior)
        lookup = {tuple(coordinate): i for i, coordinate in enumerate(axial)}
        rows, columns, entries = [], [], []
        weight = 2.0 / (3.0 * h * h)
        for i in interior_indices:
            u, v = axial[i]
            rows.append(i)
            columns.append(i)
            entries.append(6 * weight)
            for du, dv in _NEIGHBOURS:
                rows.append(i)
                columns.append(lookup[(u + du, v + dv)])
                entries.append(-weight)
        laplacian = sparse.csr_matrix(
            (entries, (rows, columns)), shape=(len(axial), len(axial))
        )
        interior_matrix = laplacian[interior_indices][:, interior_indices].tocsr()
        return cls(
            extent=extent,
            axial=axial,
            xy=xy,
            h=h,
            interior=interior,
            boundary=boundary,
            interior_indices=interior_indices,
            laplacian=laplacian,
            interior_matrix=interior_matrix,
            _factor=splu(interior_matrix.tocsc()),
            _lookup=lookup,
        )

    @property
    def n_sites(self) -> int:
        return len(self.axial)

    def coordinate_indices(self, u: ArrayLike, v: ArrayLike) -> NDArray[np.int64]:
        """Map connectome node coordinates to field sites; repeated sites are OK."""
        u_array, v_array = np.broadcast_arrays(u, v)
        if np.any(u_array != np.round(u_array)) or np.any(v_array != np.round(v_array)):
            raise ValueError("axial coordinates must be integers")
        try:
            indices = [
                self._lookup[(int(a), int(b))]
                for a, b in zip(u_array.flat, v_array.flat)
            ]
        except KeyError as error:
            raise ValueError(f"coordinate outside extent {self.extent}: {error}") from error
        return np.asarray(indices, dtype=np.int64).reshape(u_array.shape)

    def _fields(self, values: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != self.n_sites:
            raise ValueError(f"expected a final field dimension of {self.n_sites}")
        if not np.isfinite(array).all():
            raise ValueError("fields must contain only finite values")
        return array

    def apply_laplacian(self, values: ArrayLike) -> NDArray[np.float64]:
        """Apply ``-Delta_h`` to NumPy fields shaped ``(..., n_sites)``.

        Boundary values are used by adjacent interior stencils; boundary rows
        return zero. This also permits consistency checks with nonzero boundary
        test functions. ``solve`` itself imposes a zero boundary.
        """
        array = self._fields(values)
        flat = array.reshape(-1, self.n_sites)
        return np.asarray((self.laplacian @ flat.T).T).reshape(array.shape)

    def solve(self, forcing: ArrayLike) -> NDArray[np.float64]:
        """Solve interior equations by sparse LU; return full zero-boundary fields."""
        array = self._fields(forcing)
        flat = array.reshape(-1, self.n_sites)
        solution = np.zeros_like(flat)
        if len(flat):
            solution[:, self.interior] = self._factor.solve(flat[:, self.interior].T).T
        return solution.reshape(array.shape)


@dataclass(frozen=True)
class FieldSplit:
    forcing: NDArray[np.float64]
    solution: NDArray[np.float64]
    seed: int
    family: str


@dataclass(frozen=True)
class FieldNormalization:
    """Scalar RMS scales fitted only on physical training interior fields.

    No mean subtraction or per-sample scaling is used. By default both fields
    use ``force_scale`` so the normalized PDE is unchanged and a physics-only
    model need not use solution labels to choose its output scale.

    Optionally, if ``f_n=f/force_scale`` and ``u_n=u/solution_scale``, the equation
    becomes ``residual_scale * (-Delta_h u_n) = f_n``. The solution RMS is an
    optional diagnostic or supervised-training scale, not required by the PDE.
    """

    force_scale: float
    solution_scale: float

    @classmethod
    def fit(cls, training: FieldSplit, grid: HexPoissonGrid) -> FieldNormalization:
        if training.forcing.shape[0] == 0:
            raise ValueError("normalization requires nonempty training data")
        force_scale = float(np.sqrt(np.mean(training.forcing[:, grid.interior] ** 2)))
        solution_scale = float(np.sqrt(np.mean(training.solution[:, grid.interior] ** 2)))
        if min(force_scale, solution_scale) <= np.finfo(np.float64).tiny:
            raise ValueError("normalization requires nonzero training field scales")
        return cls(force_scale=force_scale, solution_scale=solution_scale)

    @property
    def residual_scale(self) -> float:
        return self.solution_scale / self.force_scale

    def normalize(
        self, split: FieldSplit, *, use_solution_scale: bool = False
    ) -> FieldSplit:
        return FieldSplit(
            forcing=split.forcing / self.force_scale,
            solution=split.solution / (
                self.solution_scale if use_solution_scale else self.force_scale
            ),
            seed=split.seed,
            family=split.family,
        )


@dataclass(frozen=True)
class PoissonDataset:
    grid: HexPoissonGrid
    splits: dict[str, FieldSplit]
    normalization: FieldNormalization

    def normalized(
        self, split_name: str, *, use_solution_scale: bool = False
    ) -> FieldSplit:
        return self.normalization.normalize(
            self.splits[split_name], use_solution_scale=use_solution_scale
        )


def _flat_spectrum_field(
    grid: HexPoissonGrid, rng: np.random.Generator, eigenvectors: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Zero-boundary field with i.i.d. unit-variance coefficients on all interior modes."""
    values = np.zeros(grid.n_sites)
    values[grid.interior] = eigenvectors @ rng.normal(size=eigenvectors.shape[1])
    return values


def _forcing_fields(
    grid: HexPoissonGrid, n_samples: int, *, seed: int, family: str
) -> NDArray[np.float64]:
    """Draw Gaussian mixtures plus smooth random fields or OOD forcing fields.

    ``balanced`` alternates: even samples are ordinary ``smooth`` draws, odd
    samples replace the smooth draw by a flat-spectrum random combination of
    all interior eigenmodes (separate random stream), rescaled to the interior
    RMS of the replaced smooth draw so both halves carry comparable energy. The
    ``smooth`` path itself is unchanged, so earlier data seeds regenerate
    bit-for-bit, and even ``balanced`` samples equal the ``smooth`` samples.
    ``white`` draws i.i.d. standard normal values at the interior
    sites, a flat expected spectrum in every orthonormal basis that needs no
    eigenvectors; it is a separate branch and leaves the other families unchanged.
    The families built from eigenvectors (flat-spectrum ``balanced`` samples and
    ``high_band``) depend on the eigenvector basis inside degenerate eigenspaces,
    which LAPACK may return differently for different BLAS thread counts; the
    recorded runs used single-threaded BLAS.
    """
    rng = np.random.default_rng(seed)
    xy = grid.xy
    result = np.zeros((n_samples, grid.n_sites))
    eigenvectors, flat_rng = None, None
    if family == "balanced":
        eigenvectors = np.linalg.eigh(grid.interior_matrix.toarray())[1]
        # A separate stream keeps the main stream identical to the smooth
        # family, so even samples equal the smooth draws bit-for-bit.
        flat_rng = np.random.default_rng(np.random.SeedSequence([seed, 97]))
    for sample in range(n_samples):
        draw = family
        if family == "balanced":
            draw = "smooth"
        if draw in ("smooth", "localized"):
            n_blobs = int(rng.integers(2, 7)) if draw == "smooth" else int(rng.integers(1, 4))
            values = np.zeros(grid.n_sites)
            for _ in range(n_blobs):
                # Sub-cell jitter avoids a fixed basis; width floors prevent
                # single-pixel sources on the coarse prototype meshes.
                center = xy[rng.choice(grid.interior_indices)].copy()
                center += rng.uniform(-0.2, 0.2, 2) * grid.h
                if draw == "smooth":
                    width = rng.uniform(0.25, 0.6)
                else:
                    width = rng.uniform(max(0.055, 0.65 * grid.h), max(0.11, 0.95 * grid.h))
                aspect = rng.uniform(0.7, 1.4)
                second_width = width * aspect
                if draw == "localized":
                    second_width = max(second_width, 0.65 * grid.h)
                angle = rng.uniform(0, 2 * np.pi)
                rotation = np.asarray(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                displacement = (xy - center) @ rotation
                squared_radius = (displacement[:, 0] / width) ** 2
                squared_radius += (displacement[:, 1] / second_width) ** 2
                values += rng.normal() * np.exp(-0.5 * squared_radius)
            if draw == "smooth":
                # A separate broad RBF random field plus affine forcing increases
                # diversity without selecting analytically solvable eigenmodes.
                anchors = rng.uniform(-0.85, 0.85, (8, 2))
                kernel = np.exp(-np.sum((xy[:, None, :] - anchors) ** 2, axis=2) / 0.6**2)
                values += 0.3 * kernel @ rng.normal(size=8) / np.sqrt(8)
                values += 0.2 * (rng.normal() + xy @ rng.normal(size=2))
            if family == "balanced" and sample % 2 == 1:
                assert eigenvectors is not None and flat_rng is not None
                flat = _flat_spectrum_field(grid, flat_rng, eigenvectors)
                smooth_rms = np.sqrt(np.mean(values[grid.interior] ** 2))
                flat_rms = np.sqrt(np.mean(flat[grid.interior] ** 2))
                values = flat * (smooth_rms / max(flat_rms, np.finfo(np.float64).tiny))
        elif family == "white":
            values = np.zeros(grid.n_sites)
            values[grid.interior] = rng.normal(size=int(grid.interior.sum()))
        elif family == "high_band":
            # Random combination of the top-tertile interior eigenmodes of this
            # grid's discrete operator: a grid-specific high-band probe, unlike
            # the continuous-frequency ``high_frequency`` family, whose energy on
            # coarse grids falls mostly in the low and middle discrete bands.
            if eigenvectors is None:
                values_all, vectors_all = np.linalg.eigh(grid.interior_matrix.toarray())
                high_mask = values_all > np.quantile(values_all, 2 / 3) + 1e-10
                eigenvectors = vectors_all[:, high_mask]
            values = np.zeros(grid.n_sites)
            coefficients = rng.normal(size=eigenvectors.shape[1])
            values[grid.interior] = eigenvectors @ coefficients / np.sqrt(len(coefficients))
        elif family == "high_frequency":
            values = np.zeros(grid.n_sites)
            for _ in range(5):
                angle = rng.uniform(0, 2 * np.pi)
                direction = np.asarray([np.cos(angle), np.sin(angle)])
                frequency = rng.uniform(min(2.5, 0.28 / grid.h), min(4.5, 0.42 / grid.h))
                phase = rng.uniform(0, 2 * np.pi)
                values += rng.normal() * np.cos(2 * np.pi * frequency * (xy @ direction) + phase)
            values /= np.sqrt(5)
        else:
            raise ValueError(f"unknown forcing family: {family}")
        result[sample] = values
    return result


IN_DISTRIBUTION_FAMILIES = ("smooth", "balanced", "white")


def make_poisson_dataset(
    grid: HexPoissonGrid,
    *,
    n_train: int = 128,
    n_validation: int = 32,
    n_test: int = 64,
    n_ood: int = 32,
    seed: int = 1729,
    family: str = "smooth",
    n_ood_high_band: int = 0,
) -> PoissonDataset:
    """Create independent reproducible splits and fit train-only normalization.

    Fixed split identifiers ensure changing a split's sample count does not
    change any other split. Localized OOD Gaussian widths range from
    ``max(0.055, 0.65*h)`` to ``max(0.11, 0.95*h)``; the second anisotropic axis
    is also floored at ``0.65*h``. High-frequency forcing ranges from
    ``min(2.5, 0.28/h)`` to ``min(4.5, 0.42/h)`` cycles per unit, below the
    nearest-neighbour Nyquist limit. These grid-dependent bounds prevent extreme
    aliasing on prototype meshes, but OOD results still concern discrete
    operators; they are not a continuum convergence study. No OOD data enter
    normalization fitting.

    ``family`` selects the in-distribution forcing family used by the train,
    validation and test splits (``smooth`` or ``balanced``); the two OOD families
    are unchanged. The default reproduces earlier datasets exactly.
    ``n_ood_high_band > 0`` adds an ``ood_high_band`` split of grid-specific
    top-tertile eigenmode combinations; it uses its own split identifier, so
    the other splits are unaffected.
    """
    if family not in IN_DISTRIBUTION_FAMILIES:
        raise ValueError(f"unknown in-distribution family: {family}")
    counts = {
        "train": n_train,
        "validation": n_validation,
        "test": n_test,
        "ood_localized": n_ood,
        "ood_high_frequency": n_ood,
    }
    if n_ood_high_band:
        counts["ood_high_band"] = n_ood_high_band
    splits = {}
    for name, count in counts.items():
        if isinstance(count, bool) or int(count) != count or count < 0:
            raise ValueError("sample counts must be nonnegative integers")
        if name == "train" and count == 0:
            raise ValueError("training data must be nonempty")
        split_seed = int(np.random.SeedSequence([seed, _SPLIT_IDS[name]]).generate_state(1)[0])
        split_family = name.removeprefix("ood_") if name.startswith("ood_") else family
        forcing = _forcing_fields(grid, int(count), seed=split_seed, family=split_family)
        splits[name] = FieldSplit(
            forcing=forcing,
            solution=grid.solve(forcing),
            seed=split_seed,
            family=split_family,
        )
    return PoissonDataset(
        grid=grid,
        splits=splits,
        normalization=FieldNormalization.fit(splits["train"], grid),
    )
