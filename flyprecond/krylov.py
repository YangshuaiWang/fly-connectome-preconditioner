"""Classical flexible GMRES with optional learned right preconditioning.

Arnoldi orthogonalization, residual evaluation, and the least-squares solve are
external numerical algorithms, not biological mechanisms. A model supplies
directions only. Accuracy of this hybrid solver must not be described as the
accuracy of the fruit-fly circuit alone. There is no classical fallback and no
reference-solution input.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class KrylovResult:
    """Batched solutions with per-field stopping and honest cost accounting."""

    solution: torch.Tensor
    relative_residual: torch.Tensor
    iterations: torch.Tensor
    preconditioner_calls: torch.Tensor
    operator_applications: torch.Tensor
    restart_cycles: torch.Tensor
    status: tuple[str, ...]
    residual_history: torch.Tensor
    batch_direction_calls: int
    timings: dict[str, float]

    def summary(self) -> dict:
        relative = self.relative_residual.detach().cpu().numpy()
        iterations = self.iterations.detach().cpu().numpy()
        calls = self.preconditioner_calls.detach().cpu().numpy()
        return {
            "algorithm": "classical restarted flexible GMRES with supplied right preconditioner",
            "reference_solutions_used": False,
            "classical_fallback_used": False,
            "outer_dtype": "float64",
            "true_residual_confirmed": True,
            "n_fields": len(self.status),
            "status_counts": dict(Counter(self.status)),
            "converged_fraction": self.status.count("converged") / len(self.status),
            "relative_residual_mean": float(relative.mean()),
            "relative_residual_median": float(np.median(relative)),
            "relative_residual_max": float(relative.max()),
            "iterations_mean": float(iterations.mean()),
            "iterations_max": int(iterations.max()),
            "preconditioner_calls_mean": float(calls.mean()),
            "preconditioner_calls_total": int(calls.sum()),
            "batch_direction_calls": self.batch_direction_calls,
            "operator_applications_total": int(self.operator_applications.sum().item()),
            "restart_cycles_max": int(self.restart_cycles.max().item()),
            "timings": self.timings,
        }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _prepare_problem(forcing, operator, interior, initial, keep_sparse=False):
    f = torch.as_tensor(forcing)
    if f.ndim != 2 or min(f.shape) == 0 or not f.is_floating_point():
        raise ValueError("forcing must be a nonempty floating-point (batch, sites) tensor")
    if not torch.isfinite(f).all():
        raise ValueError("forcing must be finite")
    f = f.to(dtype=torch.float64)
    mask = torch.as_tensor(interior, dtype=torch.bool, device=f.device)
    if mask.shape != (f.shape[1],) or not mask.any():
        raise ValueError("interior must be a nonempty mask of field sites")
    if callable(operator) and not hasattr(operator, "shape"):
        # a callable operator(values, rows) maps a (batch, sites) float64 tensor to A_k v_k field by
        # field, rows giving the indices of the fields (None: all); the default matrix paths are unchanged.
        matrix = operator
    elif keep_sparse and hasattr(operator, "tocsr"):
        # large grids keep the operator sparse (CSR); the default dense path is unchanged.
        csr = operator.tocsr()
        matrix = torch.sparse_csr_tensor(
            torch.as_tensor(csr.indptr, dtype=torch.int64), torch.as_tensor(csr.indices, dtype=torch.int64),
            torch.as_tensor(csr.data, dtype=f.dtype), size=csr.shape, device=f.device)
        if csr.shape != (f.shape[1], f.shape[1]) or not np.isfinite(csr.data).all():
            raise ValueError("operator must be a finite square matrix matching the sites")
    else:
        if hasattr(operator, "toarray"):
            operator = operator.toarray()
        matrix = torch.as_tensor(operator, dtype=f.dtype, device=f.device)
        if matrix.layout != torch.strided:
            matrix = matrix.to_dense()
        if matrix.shape != (f.shape[1], f.shape[1]) or not torch.isfinite(matrix).all():
            raise ValueError("operator must be a finite square matrix matching the sites")
    if initial is None:
        x = torch.zeros_like(f)
    else:
        x = torch.as_tensor(initial, dtype=f.dtype, device=f.device).clone()
        if x.shape != f.shape or not torch.isfinite(x).all():
            raise ValueError("initial must be a finite field batch matching forcing")
    return f * mask, matrix, mask, x * mask


@torch.no_grad()
def flexible_gmres(
    direction: Callable[[torch.Tensor], torch.Tensor],
    forcing: torch.Tensor,
    operator,
    interior,
    *,
    initial: torch.Tensor | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 20,
    max_iterations: int = 80,
    breakdown_tolerance: float = 32 * np.finfo(np.float64).eps,
    keep_sparse: bool = False,
) -> KrylovResult:
    """Solve independent fields by batched restarted flexible right GMRES.

    The outer algorithm stores ``z_j=direction(v_j)`` separately from each
    Arnoldi vector ``v_j`` and constructs ``x=x_start+sum_j z_j*y_j``. It thus
    supports nonlinear or iteration-dependent preconditioners. Orthogonalization
    is two-pass modified Gram--Schmidt, followed by independent Givens rotations
    and triangular solves. All outer arithmetic and returned fields are float64.

    The default initial guess is zero for a fair identity-direction baseline
    (``lambda residual: residual``). To start at a learned prediction, pass it
    explicitly as ``initial`` and account for that additional model call outside
    this function. ``preconditioner_calls`` counts calls made here per field;
    ``batch_direction_calls`` counts actual calls with a batch of active fields.

    A true ``f-Ax`` is recomputed after every valid Arnoldi step and is the only
    convergence criterion. Each field stops independently, and unsuccessful
    directions never trigger an identity/Jacobi fallback. Boundary forcing and
    solution values are clamped to zero. For a zero RHS, the reported relative
    residual is its absolute norm. Residual history has one row per global inner
    step, with already stopped fields retaining their final residual.

    Dense NumPy/Torch or SciPy sparse operators are accepted; by default this
    prototype converts the small lattice operator to dense. With
    ``keep_sparse=True`` a SciPy sparse operator is kept in CSR form (
    large grids). The supplied interior operator must be nonsingular.
    """
    for value, name, minimum in ((restart, "restart", 1),
                                 (max_iterations, "max_iterations", 0)):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for value, name in ((rtol, "rtol"), (atol, "atol"),
                        (breakdown_tolerance, "breakdown_tolerance")):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    setup_start = perf_counter()
    f, matrix, mask, x = _prepare_problem(forcing, operator, interior, initial, keep_sparse=keep_sparse)
    _sync(f.device)
    setup_seconds = perf_counter() - setup_start
    solve_start = perf_counter()
    batch, sites = f.shape
    width = min(int(restart), int(mask.sum().item()))
    iterations = torch.zeros(batch, dtype=torch.long, device=f.device)
    calls = torch.zeros_like(iterations)
    applications = torch.zeros_like(iterations)
    cycles = torch.zeros_like(iterations)
    batch_calls = 0
    direction_seconds = 0.0

    def apply(values, rows=None):
        if callable(matrix) and not isinstance(matrix, torch.Tensor):
            # rows: the indices of the fields in values (None: all fields in order)
            return matrix(values, rows) * mask
        if matrix.layout == torch.sparse_csr:
            return (matrix @ values.T).T * mask
        return (values @ matrix.T) * mask

    residual = f - apply(x)
    applications += 1
    f_norm = f.norm(dim=1)
    denominator = torch.where(f_norm > 0, f_norm, torch.ones_like(f_norm))
    tolerance = atol + rtol * f_norm
    norm = residual.norm(dim=1)
    if not torch.isfinite(norm).all():
        raise ValueError("initial residual is nonfinite")
    active = norm > tolerance
    statuses = ["max_iterations" if value else "converged" for value in active.tolist()]
    history = [(norm / denominator).clone()]

    def stop(indices, status):
        active[indices] = False
        for index in indices.tolist():
            statuses[index] = status

    for start in range(0, int(max_iterations), width):
        indices = torch.nonzero(active, as_tuple=True)[0]
        if not len(indices):
            break
        inner_width = min(width, int(max_iterations) - start)
        cycles[indices] += 1
        base = x.clone()
        v = f.new_zeros((batch, inner_width + 1, sites))
        z = f.new_zeros((batch, inner_width, sites))
        triangular = f.new_zeros((batch, inner_width, inner_width))
        cosine = f.new_zeros((batch, inner_width))
        sine = f.new_zeros((batch, inner_width))
        rhs = f.new_zeros((batch, inner_width + 1))
        rhs[indices, 0] = norm[indices]
        v[indices, 0] = residual[indices] / norm[indices, None]

        for j in range(inner_width):
            indices = torch.nonzero(active, as_tuple=True)[0]
            if not len(indices):
                break
            calls[indices] += 1
            _sync(f.device)
            direction_start = perf_counter()
            proposed = direction(v[indices, j].clone())
            _sync(f.device)
            direction_seconds += perf_counter() - direction_start
            batch_calls += 1
            proposed = torch.as_tensor(proposed, dtype=f.dtype, device=f.device)
            if proposed.shape != (len(indices), sites):
                raise ValueError("direction must return a matching (batch, sites) field")
            proposed = torch.where(mask, proposed, torch.zeros_like(proposed))
            finite = torch.isfinite(proposed).all(1)
            stop(indices[~finite], "nonfinite_direction")
            indices, proposed = indices[finite], proposed[finite]
            nonzero = proposed.norm(dim=1) > 0
            stop(indices[~nonzero], "zero_direction")
            indices, proposed = indices[nonzero], proposed[nonzero]
            if len(indices):
                z[indices, j] = proposed
                w = apply(proposed, indices)
                applications[indices] += 1
                image_norm = w.norm(dim=1)
                finite = torch.isfinite(w).all(1) & torch.isfinite(image_norm)
                stop(indices[~finite], "nonfinite_operator_image")
                indices, w, image_norm = indices[finite], w[finite], image_norm[finite]
                nonzero = image_norm > 0
                stop(indices[~nonzero], "zero_operator_image")
                indices, w, image_norm = indices[nonzero], w[nonzero], image_norm[nonzero]
            if len(indices):
                column = f.new_zeros((len(indices), j + 2))
                for _ in range(2):
                    for k in range(j + 1):
                        coefficient = (v[indices, k] * w).sum(1)
                        column[:, k] += coefficient
                        w -= coefficient[:, None] * v[indices, k]
                next_norm = w.norm(dim=1)
                column[:, j + 1] = next_norm
                happy_breakdown = next_norm <= breakdown_tolerance * image_norm
                # Apply the previous Givens rotations to this new column.
                for k in range(j):
                    upper, lower = column[:, k].clone(), column[:, k + 1].clone()
                    column[:, k] = cosine[indices, k] * upper + sine[indices, k] * lower
                    column[:, k + 1] = -sine[indices, k] * upper + cosine[indices, k] * lower
                pivot = torch.hypot(column[:, j], column[:, j + 1])
                independent = torch.isfinite(pivot) & (pivot > breakdown_tolerance * image_norm)
                stop(indices[~independent], "dependent_direction_breakdown")
                indices = indices[independent]
                column, pivot = column[independent], pivot[independent]
                w, next_norm = w[independent], next_norm[independent]
                happy_breakdown = happy_breakdown[independent]
            if len(indices):
                c, s = column[:, j] / pivot, column[:, j + 1] / pivot
                cosine[indices, j], sine[indices, j] = c, s
                triangular[indices, :j, j] = column[:, :j]
                triangular[indices, j, j] = pivot
                current_rhs, next_rhs = rhs[indices, j].clone(), rhs[indices, j + 1].clone()
                rhs[indices, j] = c * current_rhs + s * next_rhs
                rhs[indices, j + 1] = -s * current_rhs + c * next_rhs
                coefficients = torch.linalg.solve_triangular(
                    triangular[indices, :j + 1, :j + 1],
                    rhs[indices, :j + 1, None], upper=True,
                ).squeeze(-1)
                candidate = base[indices] + (z[indices, :j + 1]
                                              * coefficients[:, :, None]).sum(1)
                candidate *= mask
                true_residual = f[indices] - apply(candidate, indices)
                applications[indices] += 1
                candidate_norm = true_residual.norm(dim=1)
                finite = torch.isfinite(candidate).all(1) & torch.isfinite(candidate_norm)
                stop(indices[~finite], "nonfinite_iterate")
                accepted = indices[finite]
                x[accepted] = candidate[finite]
                residual[accepted] = true_residual[finite]
                norm[accepted] = candidate_norm[finite]
                iterations[accepted] += 1
                converged = candidate_norm[finite] <= tolerance[accepted]
                stop(accepted[converged], "converged")
                stalled = happy_breakdown[finite] & ~converged
                stop(accepted[stalled], "arnoldi_breakdown_true_residual_not_converged")
                continuing = finite.clone()
                continuing[finite] = ~converged & ~stalled
                keep = indices[continuing]
                v[keep, j + 1] = w[continuing] / next_norm[continuing, None]
            history.append((norm / denominator).clone())

    # Independently recompute the final true residual even for early breakdowns.
    residual = f - apply(x)
    applications += 1
    norm = residual.norm(dim=1)
    for index, status in enumerate(statuses):
        if status == "converged" and norm[index] > tolerance[index]:
            statuses[index] = "final_true_residual_check_failed"
    history[-1] = (norm / denominator).clone()
    _sync(f.device)
    solve_seconds = perf_counter() - solve_start
    return KrylovResult(
        solution=x, relative_residual=norm / denominator, iterations=iterations,
        preconditioner_calls=calls, operator_applications=applications,
        restart_cycles=cycles, status=tuple(statuses),
        residual_history=torch.stack(history), batch_direction_calls=batch_calls,
        timings={"setup_seconds": setup_seconds, "solve_seconds": solve_seconds,
                 "preconditioner_seconds": direction_seconds,
                 "outer_seconds": max(0.0, solve_seconds - direction_seconds)},
    )


def model_direction(
    model: Callable[[torch.Tensor], torch.Tensor],
    dtype: torch.dtype,
    interior,
    residual_rms: float | None = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Adapt a field model into a nonlinear right preconditioner.

    Arnoldi input vectors are not generally physical forcing samples. Optionally
    rescale each vector to ``residual_rms`` on the interior before calling the
    model, then undo that amplitude scaling on its output. This keeps calls in
    the training amplitude range but is an engineered adapter, not a biological
    claim or an assumption that the model is exactly homogeneous. ``dtype`` is
    the model input dtype, typically float32; the output matches the float64
    outer vector dtype. Use an evaluation-mode model on the caller's device.
    """
    if residual_rms is not None and (not np.isfinite(residual_rms) or residual_rms <= 0):
        raise ValueError("residual_rms must be finite and positive")
    if not dtype.is_floating_point:
        raise ValueError("model dtype must be floating point")

    @torch.no_grad()
    def direction(values):
        mask = torch.as_tensor(interior, dtype=torch.bool, device=values.device)
        if mask.shape != (values.shape[1],) or not mask.any():
            raise ValueError("interior must be a nonempty mask of field sites")
        values = values * mask
        if residual_rms is None:
            multiplier = values.new_ones((len(values),))
        else:
            rms = values[:, mask].square().mean(1).sqrt()
            multiplier = torch.where(rms > 0, residual_rms / rms.clamp_min(
                torch.finfo(values.dtype).tiny), torch.ones_like(rms))
        response = model((values * multiplier[:, None]).to(dtype=dtype))
        response = torch.as_tensor(response, dtype=values.dtype, device=values.device)
        if response.shape != values.shape:
            raise ValueError("model must return a field matching its input")
        return torch.where(mask, response / multiplier[:, None], torch.zeros_like(response))

    return direction
