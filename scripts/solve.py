#!/usr/bin/env python3
"""FGMRES on the hexagonal Poisson problem with a trained circuit as preconditioner.

Example:
    python scripts/solve.py --parameters parameters/circuit_seed0.json --extent 32 --methods none single vcycle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyprecond.hex_pde import HexPoissonGrid  # noqa: E402
from flyprecond.solvers import (MultilevelCircuit, balanced_forcings, circuit_direction, fgmres_iterations,  # noqa: E402
                                interior_direction, load_circuit)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parameters", type=Path, default=Path("parameters/circuit_seed0.json"))
    parser.add_argument("--extent", type=int, default=16, help="lattice extent: 8 (169 DOFs), 16 (721), 32 (2,977), 64 (12,097)")
    parser.add_argument("--methods", nargs="+", default=["none", "single", "vpre", "vcycle"],
                        choices=["none", "single", "vpre", "vcycle"],
                        help="none: no preconditioner; single: single-level circuit; vpre/vcycle: circuit V(1,0)/V(1,1)")
    parser.add_argument("--n-forcings", type=int, default=0, help="use only the first n test forcings")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    grid = HexPoissonGrid.build(args.extent)
    f = balanced_forcings(args.extent)
    if args.n_forcings:
        f = f[: args.n_forcings]
    print(f"extent {args.extent}: {int(grid.interior.sum())} DOFs, {len(f)} balanced test forcings")
    for method in args.methods:
        t0 = time.perf_counter()
        if method == "none":
            direction = interior_direction(lambda r: r, grid.interior)
        elif method == "single":
            model, g = load_circuit(args.parameters, args.extent)
            direction = circuit_direction(model, g)
        else:
            direction = interior_direction(MultilevelCircuit(args.parameters, args.extent, method), grid.interior)
        it = fgmres_iterations(direction, f, grid)
        print(f"  {method:7s} median {np.median(it):6.1f}  range {it.min()}-{it.max()}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
