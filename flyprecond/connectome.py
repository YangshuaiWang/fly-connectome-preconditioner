"""Compile the public FlyVis average-filter specification into a sparse graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


@dataclass(frozen=True)
class ConnectomeGraph:
    """A compact directed graph derived from the FlyVis connectome specification."""

    node_type: NDArray[np.str_]
    node_u: NDArray[np.int64]
    node_v: NDArray[np.int64]
    node_role: NDArray[np.str_]
    source: NDArray[np.int64]
    target: NDArray[np.int64]
    sign: NDArray[np.float64]
    synapse_count: NDArray[np.float64]
    source_spec: str
    extent: int

    @property
    def n_nodes(self) -> int:
        return int(self.node_type.size)

    @property
    def n_edges(self) -> int:
        return int(self.source.size)

    @property
    def input_indices(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.node_role == "input")

    @property
    def non_input_indices(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.node_role != "input")

    def recurrent_matrix(self, spectral_scale: float = 0.92) -> sparse.csr_matrix:
        """Return a stable target-by-source sparse propagation matrix.

        Signed synapse counts are normalized by the absolute incoming strength of
        each target. ``spectral_scale`` controls recurrent gain without changing
        topology.
        """

        raw = self.sign * np.log1p(self.synapse_count)
        incoming = np.bincount(
            self.target, weights=np.abs(raw), minlength=self.n_nodes
        )
        values = spectral_scale * raw / np.maximum(incoming[self.target], 1.0)
        return sparse.csr_matrix(
            (values, (self.target, self.source)),
            shape=(self.n_nodes, self.n_nodes),
        )


def _hex_coordinates(extent: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for u in range(-extent, extent + 1):
        lower = max(-extent, -extent - u)
        upper = min(extent, extent - u)
        coords.extend((u, v) for v in range(lower, upper + 1))
    return coords


def compile_flyvis_connectome(
    spec_path: str | Path,
    *,
    extent: int = 2,
) -> ConnectomeGraph:
    """Compile the official FlyVis JSON without interpolating unmeasured offsets.

    FlyVis optionally fills gaps inside the convex hull of measured filters. The
    pilot deliberately disables that interpolation so every compiled edge comes
    from an offset explicitly present in the public specification.
    """

    path = Path(spec_path)
    spec = json.loads(path.read_text())
    input_types = set(spec["input_units"])
    output_types = set(spec["output_units"])

    node_types: list[str] = []
    node_u: list[int] = []
    node_v: list[int] = []
    node_roles: list[str] = []
    lookup: dict[tuple[str, int, int], int] = {}

    full_hex = _hex_coordinates(extent)
    for node in spec["nodes"]:
        cell_type = node["name"]
        pattern, args = node["pattern"]
        if pattern == "single":
            coords = [(0, 0)]
        else:
            if pattern == "tile":
                u_stride = v_stride = int(args)
            elif pattern == "stride":
                u_stride, v_stride = map(int, args)
            else:
                raise ValueError(f"Unsupported FlyVis node pattern: {pattern}")
            coords = [
                (u, v)
                for u, v in full_hex
                if u % u_stride == 0 and v % v_stride == 0
            ]

        if cell_type in input_types:
            role = "input"
        elif cell_type in output_types:
            role = "output"
        else:
            role = "intermediate"
        for u, v in coords:
            index = len(node_types)
            node_types.append(cell_type)
            node_u.append(u)
            node_v.append(v)
            node_roles.append(role)
            lookup[(cell_type, u, v)] = index

    sources: list[int] = []
    targets: list[int] = []
    signs: list[float] = []
    counts: list[float] = []
    indices_by_type: dict[str, list[int]] = {}
    for index, cell_type in enumerate(node_types):
        indices_by_type.setdefault(cell_type, []).append(index)

    for edge in spec["edges"]:
        src_type = edge["src"]
        target_type = edge["tar"]
        sign = float(edge["alpha"])
        for (du, dv), count in edge["offsets"]:
            for source in indices_by_type.get(src_type, []):
                target = lookup.get(
                    (target_type, node_u[source] + int(du), node_v[source] + int(dv))
                )
                if target is None:
                    continue
                sources.append(source)
                targets.append(target)
                signs.append(sign)
                counts.append(float(count))

    return ConnectomeGraph(
        node_type=np.asarray(node_types, dtype=str),
        node_u=np.asarray(node_u, dtype=np.int64),
        node_v=np.asarray(node_v, dtype=np.int64),
        node_role=np.asarray(node_roles, dtype=str),
        source=np.asarray(sources, dtype=np.int64),
        target=np.asarray(targets, dtype=np.int64),
        sign=np.asarray(signs, dtype=np.float64),
        synapse_count=np.asarray(counts, dtype=np.float64),
        source_spec=str(path),
        extent=extent,
    )


def rewire_targets(graph: ConnectomeGraph, *, seed: int) -> ConnectomeGraph:
    """Create a degree-preserving directed-multigraph null model.

    A permutation of edge targets preserves every source out-degree, every target
    in-degree, the complete edge-weight/sign multiset, and the total edge count.
    It intentionally destroys cell-type and retinotopic motifs. Parallel edges can
    occur, matching the multigraph interpretation of synapse bundles.
    """

    rng = np.random.default_rng(seed)
    return replace(graph, target=rng.permutation(graph.target))

