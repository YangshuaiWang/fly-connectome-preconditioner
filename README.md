# Fly connectome preconditioner

Python implementation of a neuroscience-inspired preconditioner built from the fly visual connectome for elliptic
partial differential equations.

The preconditioner is the optic-lobe circuit of the FlyVis connectome template (65 cell types, 34 output types) placed
on a hexagonal lattice. Only its 776 cell-type parameters are trained, by minimizing the Dirichlet energy on one
coarse grid. The trained circuit then serves as a nonlinear right preconditioner for flexible GMRES, on its own or as
the level correction of a Galerkin multigrid cycle.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_connectome.py          # downloads data/fib25-fib19_v2.2.json from the FlyVis repository
```

The code was run with Python 3.12, NumPy 1.26, SciPy 1.14 and PyTorch 2.5 on CPU.

## Solving with a trained circuit

`parameters/` contains ten trained circuits from independent initializations
(`circuit_seed0.json`, ..., `circuit_seed9.json`).

```bash
python scripts/solve.py --parameters parameters/circuit_seed0.json --extent 16
```

This solves the discrete Poisson problem on the lattice of extent 16 (721 DOFs) for the balanced test forcings and
prints the FGMRES iteration counts without preconditioning (`none`), with the single-level circuit (`single`) and with
the circuit V(1,0) and V(1,1) cycles (`vpre`, `vcycle`). Extents 8, 16, 32 and 64 correspond to 169, 721, 2,977 and
12,097 DOFs.

## Training a circuit

```bash
python scripts/train_circuit.py --seeds 0 --output runs/measured
python scripts/solve.py --parameters runs/measured/fly_seed0/parameters.json --extent 32
```

`--model type_rewired` and `--model shell_rewired` train the circuit on rewired control templates.
A training run takes about one hour on one CPU core. A new run follows the same training protocol as the supplied
circuits in `parameters/`.

## Contents

| Path | Content |
|---|---|
| `flyprecond/connectome.py` | Compilation of the FlyVis template on a hexagonal lattice |
| `flyprecond/controls.py` | Boundary cut and degree-preserving rewiring |
| `flyprecond/field_operator.py` | Neural dynamics, input and readout of the circuit |
| `flyprecond/hex_pde.py` | Hexagonal lattice, discrete Laplacian and forcing families |
| `flyprecond/krylov.py` | Flexible GMRES with a nonlinear right preconditioner |
| `flyprecond/solvers.py` | Loading and transfer of trained circuits, multilevel cycle |
| `scripts/` | Connectome download, training and solving |
| `parameters/` | Trained parameters of the ten circuits |

## License

MIT License (see `LICENSE`). The neural dynamics follow the FlyVis model of Lappalainen et al. (Nature, 2024), whose
code and connectome template are available at https://github.com/TuragaLab/flyvis under the MIT License.
