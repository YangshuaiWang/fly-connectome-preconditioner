"""Constrained graph null models for field-to-field connectome experiments.

The controls change which retinotopic cells are connected while keeping the
cell-type circuit, directed degrees, receptor/readout mapping, and edge attributes
fixed. A target's weighted incoming strength is *not* preserved: synapse counts
stay attached to source-edge slots, rather than to target cells.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .connectome import ConnectomeGraph


def edge_squared_distances(graph: ConnectomeGraph) -> NDArray[np.int64]:
    """Squared distance in axial hex coordinates, in lattice-spacing units."""

    du = graph.node_u[graph.target] - graph.node_u[graph.source]
    dv = graph.node_v[graph.target] - graph.node_v[graph.source]
    return du * du + du * dv + dv * dv


def edge_distance_shells(
    graph: ConnectomeGraph, *, width: float
) -> NDArray[np.int64]:
    """Return zero for same-column edges, else ``ceil(length / width)``."""

    if not math.isfinite(width) or width <= 0:
        raise ValueError("distance_bin_width must be finite and positive")
    return np.ceil(np.sqrt(edge_squared_distances(graph)) / width).astype(np.int64)


def _histogram(values: NDArray[np.int64]) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(unique, counts, strict=True)
    }


def graph_summary(graph: ConnectomeGraph) -> dict[str, Any]:
    """Return JSON-safe counts and locality diagnostics, without dense matrices."""

    distances = edge_squared_distances(graph)
    unique_distances, counts = np.unique(distances, return_counts=True)
    edge_ids = graph.source * graph.n_nodes + graph.target
    incoming = np.bincount(graph.target, minlength=graph.n_nodes)
    outgoing = np.bincount(graph.source, minlength=graph.n_nodes)
    return {
        "n_nodes": graph.n_nodes,
        "n_edges": graph.n_edges,
        "n_cell_types": int(np.unique(graph.node_type).size),
        "n_input_nodes": int(np.count_nonzero(graph.node_role == "input")),
        "n_output_nodes": int(np.count_nonzero(graph.node_role == "output")),
        "n_self_loops": int(np.count_nonzero(graph.source == graph.target)),
        "n_parallel_edges": int(graph.n_edges - np.unique(edge_ids).size),
        "min_in_degree": int(incoming.min()) if graph.n_nodes else 0,
        "max_in_degree": int(incoming.max()) if graph.n_nodes else 0,
        "min_out_degree": int(outgoing.min()) if graph.n_nodes else 0,
        "max_out_degree": int(outgoing.max()) if graph.n_nodes else 0,
        "mean_squared_distance": float(distances.mean()) if distances.size else 0.0,
        "mean_distance": float(np.sqrt(distances).mean()) if distances.size else 0.0,
        "same_column_fraction": float(np.mean(distances == 0))
        if distances.size
        else 0.0,
        "squared_distance_histogram": {
            str(int(distance)): int(count)
            for distance, count in zip(unique_distances, counts, strict=True)
        },
    }


def _edge_groups(graph: ConnectomeGraph) -> NDArray[np.int64]:
    _, node_types = np.unique(graph.node_type, return_inverse=True)
    _, sign_ids = np.unique(graph.sign, return_inverse=True)
    n_types = max(1, int(node_types.max()) + 1) if node_types.size else 1
    n_signs = max(1, int(sign_ids.max()) + 1) if sign_ids.size else 1
    keys = (
        node_types[graph.source] * n_types + node_types[graph.target]
    ) * n_signs + sign_ids
    return np.unique(keys, return_inverse=True)[1]


def assert_rewiring_invariants(
    original: ConnectomeGraph,
    rewired: ConnectomeGraph,
    *,
    preserve_distance: bool = False,
    distance_bin_width: float | None = None,
) -> None:
    """Check degree/type/sign/mapping invariants and simple-graph constraints.

    Incoming degrees are checked separately for each source-type/target-type/sign
    group, which is stronger than checking total per-node in-degrees alone.
    Existing self-loops may be removed, but no new self-loop may be introduced.
    """

    for name in (
        "node_type",
        "node_u",
        "node_v",
        "node_role",
        "source",
        "sign",
        "synapse_count",
    ):
        np.testing.assert_array_equal(getattr(original, name), getattr(rewired, name))
    assert original.extent == rewired.extent
    assert original.source_spec == rewired.source_spec
    assert original.target.shape == rewired.target.shape
    np.testing.assert_array_equal(
        original.node_type[original.target], rewired.node_type[rewired.target]
    )
    groups = _edge_groups(original)
    np.testing.assert_array_equal(
        np.sort(groups * original.n_nodes + original.target),
        np.sort(groups * rewired.n_nodes + rewired.target),
    )
    edge_ids = rewired.source * rewired.n_nodes + rewired.target
    assert np.unique(edge_ids).size == rewired.n_edges, "Parallel edges introduced"
    old_loops = set(original.source[original.source == original.target].tolist())
    new_loops = set(rewired.source[rewired.source == rewired.target].tolist())
    assert new_loops <= old_loops, "New self-loops introduced"
    if distance_bin_width is not None:
        if not preserve_distance:
            raise ValueError("distance_bin_width requires preserve_distance=True")
        np.testing.assert_array_equal(
            edge_distance_shells(original, width=distance_bin_width),
            edge_distance_shells(rewired, width=distance_bin_width),
        )
    elif preserve_distance:
        np.testing.assert_array_equal(
            edge_squared_distances(original), edge_squared_distances(rewired)
        )


def rewire_cell_type_degrees(
    graph: ConnectomeGraph,
    *,
    seed: int,
    swaps_per_edge: float = 5.0,
    preserve_distance: bool = False,
    distance_bin_width: float | None = None,
) -> tuple[ConnectomeGraph, dict[str, Any]]:
    """Swap targets within cell-type/sign groups under a fixed proposal budget.

    ``swaps_per_edge`` is the number of *attempts* per edge, not a guarantee of
    accepted swaps. This bounds work even when exact distance constraints make
    mixing impossible. The strict control preserves each edge's squared lattice
    distance, not just its population histogram. It should never be described as
    a fully mixed or uniform null ensemble: the metadata records actual mixing.

    With ``preserve_distance=True`` and a positive ``distance_bin_width``, preserve
    each edge's coarse length shell instead of its exact length. Same-column edges
    remain in shell zero; other edges use ``ceil(length / distance_bin_width)``.
    This deliberately weaker locality constraint can mix where exact distance
    preservation is nearly frozen. The exact distance histogram can then change.

    All graph node arrays and source/sign/count arrays remain unchanged. Per-node
    in/out degrees, including degrees within each cell-type/sign group, are
    preserved. Multi-edge inputs are rejected so that duplicate prevention is
    unambiguous. Arrays in the input graph are never mutated.
    """

    if not math.isfinite(swaps_per_edge) or swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must be finite and nonnegative")
    if distance_bin_width is not None:
        if not preserve_distance:
            raise ValueError("distance_bin_width requires preserve_distance=True")
        distance_labels = edge_distance_shells(graph, width=distance_bin_width)
    else:
        distance_labels = edge_squared_distances(graph)
    edge_ids = graph.source * graph.n_nodes + graph.target
    edge_set = set(edge_ids.tolist())
    if len(edge_set) != graph.n_edges:
        raise ValueError("Null models require a graph without parallel edges")
    original_edge_set = edge_set.copy()
    target = graph.target.copy()
    groups = _edge_groups(graph)
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(np.diff(groups[order])) + 1
    group_edges = np.split(order, boundaries) if graph.n_edges else []
    distances = edge_squared_distances(graph)
    rng = np.random.default_rng(seed)
    attempted = 0
    successful = 0
    rejected = {"trivial": 0, "self_loop": 0, "duplicate": 0, "distance": 0}
    unproposable_edges = 0
    u, v = graph.node_u, graph.node_v

    for indices in group_edges:
        if indices.size < 2:
            unproposable_edges += int(indices.size)
            continue
        n_attempts = math.ceil(swaps_per_edge * indices.size)
        # Vectorized RNG keeps the loop lightweight; topology updates are serial.
        proposals = rng.integers(indices.size, size=(n_attempts, 2))
        for local_i, local_j in proposals:
            attempted += 1
            i, j = int(indices[local_i]), int(indices[local_j])
            a, b = int(graph.source[i]), int(graph.source[j])
            x, y = int(target[i]), int(target[j])
            if a == b or x == y:
                rejected["trivial"] += 1
                continue
            if a == y or b == x:
                rejected["self_loop"] += 1
                continue
            new_i, new_j = a * graph.n_nodes + y, b * graph.n_nodes + x
            if new_i in edge_set or new_j in edge_set:
                rejected["duplicate"] += 1
                continue
            if preserve_distance:
                du_i, dv_i = int(u[y] - u[a]), int(v[y] - v[a])
                du_j, dv_j = int(u[x] - u[b]), int(v[x] - v[b])
                label_i = du_i * du_i + du_i * dv_i + dv_i * dv_i
                label_j = du_j * du_j + du_j * dv_j + dv_j * dv_j
                if distance_bin_width is not None:
                    label_i = math.ceil(math.sqrt(label_i) / distance_bin_width)
                    label_j = math.ceil(math.sqrt(label_j) / distance_bin_width)
                if label_i != distance_labels[i] or label_j != distance_labels[j]:
                    rejected["distance"] += 1
                    continue
            edge_set.remove(a * graph.n_nodes + x)
            edge_set.remove(b * graph.n_nodes + y)
            edge_set.add(new_i)
            edge_set.add(new_j)
            target[i], target[j] = y, x
            successful += 1

    rewired = replace(graph, target=target)
    assert_rewiring_invariants(
        graph,
        rewired,
        preserve_distance=preserve_distance,
        distance_bin_width=distance_bin_width,
    )
    changed = int(np.count_nonzero(target != graph.target))
    changed_topology = len(original_edge_set - edge_set)
    denominator = max(1, graph.n_edges)
    rewired_distances = edge_squared_distances(rewired)
    original_histogram, rewired_histogram = (
        _histogram(distances),
        _histogram(rewired_distances),
    )
    histogram_keys = original_histogram.keys() | rewired_histogram.keys()
    control = "type_degree_distance" if preserve_distance else "type_degree"
    if distance_bin_width is not None:
        control = "type_degree_distance_shell"
    metadata: dict[str, Any] = {
        "control": control,
        "seed": int(seed),
        "attempts_per_edge": float(swaps_per_edge),
        "attempted_swaps": attempted,
        "successful_swaps": successful,
        "acceptance_fraction": successful / max(1, attempted),
        "rejected_swaps": rejected,
        "unproposable_edges": unproposable_edges,
        "changed_edge_count": changed,
        "changed_edge_fraction": changed / denominator,
        "changed_topology_edge_count": changed_topology,
        "changed_topology_edge_fraction": changed_topology / denominator,
        "preserves_edge_squared_distance": bool(
            preserve_distance and distance_bin_width is None
        ),
        "preserves_edge_distance_shell": distance_bin_width is not None,
        "distance_bin_width": distance_bin_width,
        "squared_distance_changed_edge_fraction": float(
            np.count_nonzero(distances != rewired_distances) / denominator
        ),
        "squared_distance_histogram_preserved": (
            original_histogram == rewired_histogram
        ),
        "squared_distance_histogram_l1_change": sum(
            abs(original_histogram.get(key, 0) - rewired_histogram.get(key, 0))
            for key in histogram_keys
        ),
        "preserves_target_weighted_strength": False,
        "invariants_verified": True,
        "mixing_warning": (
            "Fewer than 10% of original connections changed; this is a weakly "
            "mixed constrained control, not evidence from an independent topology."
            if changed_topology / denominator < 0.1
            else "Finite target-swap chain; uniform sampling is not established."
        ),
        "original": graph_summary(graph),
        "rewired": graph_summary(rewired),
    }
    if distance_bin_width is not None:
        original_bins = _histogram(distance_labels)
        rewired_bins = _histogram(
            edge_distance_shells(rewired, width=distance_bin_width)
        )
        metadata["distance_bin_histograms"] = {
            "original": original_bins,
            "rewired": rewired_bins,
        }
        metadata["distance_bin_histogram_preserved"] = original_bins == rewired_bins
    return rewired, metadata


def cut_boundary_input(
    graph: ConnectomeGraph, axial: np.ndarray, interior: np.ndarray, *, reach: str = "all"
) -> tuple[ConnectomeGraph, dict[str, Any]]:
    """Remove every synapse from a neuron in a boundary column to a neuron in an
    interior column .

    Boundary columns keep their neurons and dynamics but can no longer influence
    the interior, an absorbing boundary analogous to the zero exterior values of
    the discrete Dirichlet operator. Signs, counts and all other synapses are kept.

    ``reach``  restricts the cut to synapses that
    connect adjacent columns ("near", hexagonal distance 1) or that span two or
    more columns ("far"); the default "all" is the round-10 cut.
    """
    column = {tuple(int(x) for x in uv): i for i, uv in enumerate(axial)}
    node_column = np.array([column[(int(u), int(v))] for u, v in zip(graph.node_u, graph.node_v)])
    node_interior = np.asarray(interior, dtype=bool)[node_column]
    remove = ~node_interior[graph.source] & node_interior[graph.target]
    info: dict[str, Any] = {"control": "cut_boundary_input"}
    if reach != "all":
        du = graph.node_u[graph.target] - graph.node_u[graph.source]
        dv = graph.node_v[graph.target] - graph.node_v[graph.source]
        distance = (np.abs(du) + np.abs(dv) + np.abs(du + dv)) // 2
        if reach == "near":
            remove &= distance == 1
        elif reach == "far":
            remove &= distance >= 2
        else:
            raise ValueError(f"unknown reach {reach!r}")
        info["reach"] = reach
        info["synapses_removed"] = float(graph.synapse_count[remove].sum())
    keep = ~remove
    pruned = replace(graph, source=graph.source[keep], target=graph.target[keep], sign=graph.sign[keep], synapse_count=graph.synapse_count[keep])
    return pruned, {**info, "edges_removed": int(remove.sum()), "edges_kept": int(keep.sum())}
