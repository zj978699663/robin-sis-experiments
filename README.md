# Robin–SIS experiments

## Scope

This package reproduces Tables 1–3 and Figure 1 of *Enriched Fourier-Galerkin finite elements preserving the first mean corrector for a logistic SIS reaction-diffusion system with fast periodic Robin outflow*.

## Run

Use Python 3.13.9 and put `pdflatex` on `PATH`. The figure uses Computer Modern and the LaTeX packages `fontenc`, `amsmath`, `amssymb`, `bm`, and `upgreek`. Run from this directory.

```text
python -m pip install numpy==2.3.5 scipy==1.16.3 numba==0.62.1 matplotlib==3.10.6 threadpoolctl==3.5.0
python ex1.py
python ex2.py
```

## Tables, figures and checks

Each command can run on its own. The solvers use one linear algebra thread. The disk and direct time marches take longer than the other solves.

| Paper | Command | Calculation | Output in `build/` |
|---|---|---|---|
| §6.1 Interval | `python ex1.py` | Solve the interval problems with `h=1/10` and `Q=160`, and the recovery test with `N=3`, `Q=80`. | `tables/table1.csv`, `figures/fig1a.pdf`–`figures/fig1d.pdf` |
| §6.1 Unit disk | `python ex1.py` | Generate the curved mesh and solve all three periods with `N=3`, `Q=64` and quadrature orders 64, 4, and 8. | `tables/table2.csv` |
| §6.2 Method comparison | `python ex2.py` | Solve all 18 mesh–period pairs with each of the three methods. Compute the field errors against a Richardson reference from 5120 and 10240 elements. Use the medians from `data/timings.csv` for the paper's time columns. | `tables/table3.csv` |

`common.py` contains the interval solver and table functions. The table checks use the values in `reference/`. The timings are the three original measurements per method and case.
