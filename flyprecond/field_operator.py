"""Retinotopic field solvers with FlyVis passive graded-release dynamics.

This is an independently implemented PDE adaptation of the equations in
TuragaLab/flyvis/network/dynamics.py (MIT), not pretrained FlyVis or a measured
whole-brain simulation. The input encoding, task, and decoder are our choices.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .connectome import ConnectomeGraph


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RetinotopicFlyOperator(nn.Module):
    """Field -> photoreceptors -> signed circuit -> shared local decoder.

    One positive gain per source/target cell-type pair and one bias/time constant
    per cell type. There are no learned dense spatial projections. Voltages are
    evolved by explicit Euler with ReLU applied to presynaptic voltage (or, with
    release='linear', the voltage itself). The
    decoder reads only the official output populations, at the same column.
    Subtracting the zero-stimulus response makes u(0)=0 exactly.
    """

    def __init__(
        self,
        graph: ConnectomeGraph,
        axial: np.ndarray,
        interior: np.ndarray,
        *,
        steps: int = 16,
        dt: float = 0.02,
        initial_gain: float = 0.01,
        input_encoding: str = "contrast",
        input_current_scale: float = 0.1,
        seed: int = 0,
        coefficient_encoding: str | None = None,
        release: str = "relu",
    ) -> None:
        super().__init__()
        if release not in ("relu", "linear"):
            raise ValueError("release must be 'relu' (rectified presynaptic voltage) or 'linear'")
        self.release = release
        if steps < 1 or not 0 < dt < 0.05:
            raise ValueError("Need positive steps and 0 < dt < initial tau=.05")
        self.steps, self.dt = steps, dt
        if input_encoding not in ("contrast", "onoff") or input_current_scale <= 0:
            raise ValueError("Invalid stimulus encoding or scale")
        if coefficient_encoding not in (None, "sum", "split", "modulate"):
            raise ValueError(
                "coefficient_encoding must be None, 'sum', 'split' or 'modulate'"
            )
        self.coefficient_encoding = coefficient_encoding
        self.input_encoding, self.input_current_scale = (
            input_encoding,
            input_current_scale,
        )
        self.n_nodes, self.n_points = graph.n_nodes, len(axial)
        self.register_buffer("interior", torch.tensor(interior, dtype=torch.float32))
        column = {tuple(uv): i for i, uv in enumerate(axial)}
        types = sorted(set(graph.node_type))
        type_lookup = {t: i for i, t in enumerate(types)}
        type_ids = np.array([type_lookup[t] for t in graph.node_type])
        self.register_buffer("type_ids", torch.tensor(type_ids, dtype=torch.long))
        node_column = np.array(
            [column[(int(u), int(v))] for u, v in zip(graph.node_u, graph.node_v)]
        )
        input_nodes = graph.input_indices
        input_types = sorted(set(graph.node_type[input_nodes]))
        input_lookup = {t: i for i, t in enumerate(input_types)}
        input_type_ids = np.array(
            [input_lookup[t] for t in graph.node_type[input_nodes]]
        )
        self.register_buffer("input_nodes", torch.tensor(input_nodes, dtype=torch.long))
        self.register_buffer("input_columns", torch.tensor(node_column[input_nodes]))
        self.register_buffer("input_type_ids", torch.tensor(input_type_ids))
        # An explicit task adapter: complementary light increments/decrements.
        polarities = np.array(
            [1 if i < len(input_types) // 2 else -1 for i in input_type_ids]
        )
        self.register_buffer("input_polarities", torch.tensor(polarities))
        # Variable-coefficient task adapter: the (zero-mean) log coefficient is a
        # second local contrast signal. "sum": every receptor type receives both
        # signals with separate type-shared gains; "split": the first half of the
        # receptor types (sorted) receive only the forcing, the second half only
        # the coefficient. Both are numerical adapters, not retinal physiology.
        if coefficient_encoding is not None:
            # Registered only for coefficient models so that earlier Poisson
            # checkpoints still load with strict=True.
            half = len(input_types) // 2
            forcing_mask = np.ones(len(input_type_ids))
            coefficient_mask = np.ones(len(input_type_ids))
            if coefficient_encoding == "split":
                forcing_mask = (input_type_ids < half).astype(np.float64)
                coefficient_mask = (input_type_ids >= half).astype(np.float64)
            self.register_buffer(
                "forcing_receptor_mask",
                torch.tensor(forcing_mask, dtype=torch.float32),
            )
            self.register_buffer(
                "coefficient_receptor_mask",
                torch.tensor(coefficient_mask, dtype=torch.float32),
            )
        output_types = sorted(set(graph.node_type[graph.node_role == "output"]))
        node_lookup = {
            (t, int(u), int(v)): i
            for i, (t, u, v) in enumerate(
                zip(graph.node_type, graph.node_u, graph.node_v)
            )
        }
        output_nodes = [
            [node_lookup[(t, int(u), int(v))] for t in output_types] for u, v in axial
        ]
        self.register_buffer("output_nodes", torch.tensor(output_nodes))

        pair_keys = sorted(
            set(zip(graph.node_type[graph.source], graph.node_type[graph.target]))
        )
        pair_lookup = {key: i for i, key in enumerate(pair_keys)}
        pair_ids = np.array(
            [
                pair_lookup[(s, t)]
                for s, t in zip(
                    graph.node_type[graph.source], graph.node_type[graph.target]
                )
            ]
        )
        means = np.bincount(pair_ids, weights=graph.synapse_count) / np.bincount(
            pair_ids
        )
        self.register_buffer("edge_pair", torch.tensor(pair_ids))
        self.register_buffer(
            "signed_counts",
            torch.tensor(graph.sign * graph.synapse_count, dtype=torch.float32),
        )
        self.register_buffer(
            "initial_pair_gain", torch.tensor(initial_gain / means, dtype=torch.float32)
        )
        # Sorted indices permit is_coalesced=True and avoid a coalesce per step.
        order = np.lexsort((graph.source, graph.target))
        pairs = np.stack([graph.target[order], graph.source[order]])
        if len(np.unique(pairs, axis=1).T) != graph.n_edges:
            raise ValueError("Field model requires a simple directed graph")
        self.register_buffer("edge_indices", torch.tensor(pairs))
        self.register_buffer("edge_order", torch.tensor(order))

        generator = torch.Generator().manual_seed(seed)
        self.log_gain_multiplier = nn.Parameter(torch.zeros(len(pair_keys)))
        self.bias = nn.Parameter(
            0.5 + 0.05 * torch.randn(len(types), generator=generator)
        )
        # tau = dt + softplus(raw_tau); initial tau exactly .05 seconds.
        self.raw_tau = nn.Parameter(
            torch.full((len(types),), float(np.log(np.expm1(0.05 - dt))))
        )
        self.log_input_gain = nn.Parameter(torch.zeros(len(input_types)))
        if coefficient_encoding in ("sum", "split"):
            self.log_coefficient_gain = nn.Parameter(torch.zeros(len(input_types)))
        elif coefficient_encoding == "modulate":
            # Multiplicative adapter: the forcing stimulus at a receptor is scaled
            # by a^(-gamma_type), gamma initialized to 1 (first-order f/a
            # correction for slowly varying a). log a = 0 recovers the Poisson
            # model exactly.
            self.coefficient_exponent = nn.Parameter(torch.ones(len(input_types)))
        self.decoder = nn.Parameter(
            0.05 * torch.randn(len(output_types), generator=generator)
        )
        self.cell_types = [str(t) for t in types]
        self.pair_keys = [(str(s), str(t)) for s, t in pair_keys]
        self.output_types = [str(t) for t in output_types]

    def readout_features(
        self, forcing: torch.Tensor, coefficient: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch = forcing.shape[0]
        forcing = forcing * self.interior
        if self.coefficient_encoding is None:
            if coefficient is not None:
                raise ValueError("this model has no coefficient channel")
            # Include one zero stimulus; a shared local subtraction removes
            # baseline boundary artifacts without extra decoder parameters.
            forcing = torch.cat([forcing, forcing.new_zeros((1, self.n_points))])
            n_rows = batch + 1
        else:
            if coefficient is None or coefficient.shape != forcing.shape:
                raise ValueError("coefficient field required, shaped like forcing")
            # Baseline per sample: zero forcing with the same coefficient, so
            # u(f=0; a) = 0 exactly for every coefficient field.
            coefficient = coefficient * self.interior
            forcing = torch.cat([forcing, torch.zeros_like(forcing)])
            coefficient = torch.cat([coefficient, coefficient])
            n_rows = 2 * batch
        stimulus = forcing[:, self.input_columns]
        if self.input_encoding == "onoff":
            stimulus = F.relu(stimulus * self.input_polarities)
        # Contrast uses small signed current deviations around the bias-set
        # operating point. This task adapter is not measured retinal physiology.
        stimulus = stimulus * self.log_input_gain.exp()[self.input_type_ids]
        if self.coefficient_encoding in ("sum", "split"):
            stimulus = stimulus * self.forcing_receptor_mask + (
                coefficient[:, self.input_columns]
                * self.log_coefficient_gain.exp()[self.input_type_ids]
                * self.coefficient_receptor_mask
            )
        elif self.coefficient_encoding == "modulate":
            stimulus = stimulus * torch.exp(
                -self.coefficient_exponent[self.input_type_ids]
                * coefficient[:, self.input_columns]
            )
        stimulus = self.input_current_scale * stimulus
        external = forcing.new_zeros((n_rows, self.n_nodes)).index_copy(
            1, self.input_nodes, stimulus
        )
        states = self._integrate(external, n_rows, keep_steps=(self.steps,))
        outputs = states[self.steps][:, self.output_nodes]
        return (
            (outputs[:batch] - outputs[batch:])
            * self.interior[None, :, None]
            / self.input_current_scale
        )

    def _integrate(
        self, external: torch.Tensor, n_rows: int, keep_steps: tuple[int, ...]
    ) -> dict[int, torch.Tensor]:
        """Euler rollout; returns the full node state after each requested step."""
        bias = self.bias[self.type_ids]
        alpha = self.dt / (self.dt + F.softplus(self.raw_tau[self.type_ids]))
        state = bias.unsqueeze(0).expand(n_rows, -1)
        gains = self.initial_pair_gain * self.log_gain_multiplier.exp()
        values = (self.signed_counts * gains[self.edge_pair])[self.edge_order]
        matrix = torch.sparse_coo_tensor(
            self.edge_indices,
            values,
            (self.n_nodes, self.n_nodes),
            is_coalesced=True,
            check_invariants=False,
        )
        kept: dict[int, torch.Tensor] = {}
        for step in range(1, self.steps + 1):
            presynaptic = F.relu(state) if self.release == "relu" else state
            recurrent = torch.sparse.mm(matrix, presynaptic.T).T
            state = state + alpha * (-state + bias + recurrent + external)
            if step in keep_steps:
                kept[step] = state
        return kept

    @torch.no_grad()
    def node_responses(
        self,
        forcing: torch.Tensor,
        steps: tuple[int, ...],
        coefficient: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        """Baseline-subtracted responses of ALL nodes after each requested step.

        Diagnostic helper (read-only): same stimulus construction and baseline
        subtraction as ``readout_features`` but without restricting to the output
        populations or the final step. Returns ``{step: (batch, n_nodes)}`` in
        the same units as ``readout_features`` (divided by the current scale).
        """
        if any(s < 1 or s > self.steps for s in steps):
            raise ValueError("requested steps must lie in [1, steps]")
        batch = forcing.shape[0]
        forcing = forcing * self.interior
        if self.coefficient_encoding is None:
            if coefficient is not None:
                raise ValueError("this model has no coefficient channel")
            forcing = torch.cat([forcing, forcing.new_zeros((1, self.n_points))])
            n_rows = batch + 1
        else:
            if coefficient is None or coefficient.shape[0] != batch:
                raise ValueError("coefficient field required, shaped like forcing")
            coefficient = coefficient * self.interior
            forcing = torch.cat([forcing, torch.zeros_like(forcing)])
            coefficient = torch.cat([coefficient, coefficient])
            n_rows = 2 * batch
        stimulus = forcing[:, self.input_columns]
        if self.input_encoding == "onoff":
            stimulus = F.relu(stimulus * self.input_polarities)
        stimulus = stimulus * self.log_input_gain.exp()[self.input_type_ids]
        if self.coefficient_encoding in ("sum", "split"):
            stimulus = stimulus * self.forcing_receptor_mask + (
                coefficient[:, self.input_columns]
                * self.log_coefficient_gain.exp()[self.input_type_ids]
                * self.coefficient_receptor_mask
            )
        elif self.coefficient_encoding == "modulate":
            stimulus = stimulus * torch.exp(
                -self.coefficient_exponent[self.input_type_ids]
                * coefficient[:, self.input_columns]
            )
        stimulus = self.input_current_scale * stimulus
        external = forcing.new_zeros((n_rows, self.n_nodes)).index_copy(
            1, self.input_nodes, stimulus
        )
        states = self._integrate(external, n_rows, keep_steps=tuple(steps))
        return {
            step: (state[:batch] - state[batch:]) / self.input_current_scale
            for step, state in states.items()
        }

    def forward(
        self, forcing: torch.Tensor, coefficient: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.readout_features(forcing, coefficient) @ self.decoder

    def diagnostics(self) -> dict:
        tau = self.dt + F.softplus(self.raw_tau.detach())
        gain = self.log_gain_multiplier.detach().exp()
        return {
            "tau_min": tau.min().item(),
            "tau_max": tau.max().item(),
            "gain_multiplier_min": gain.min().item(),
            "gain_multiplier_max": gain.max().item(),
            "parameters": parameter_count(self),
            "steps": self.steps,
            "dt": self.dt,
            "input_encoding": self.input_encoding,
            "input_current_scale": self.input_current_scale,
            "coefficient_encoding": self.coefficient_encoding,
        }
