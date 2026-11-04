# Replication package

Code for "Nonlinear Calvo Pricing, the Reset-Price Floor, and the Paradox of Flexibility at the Zero Lower Bound" (Minseong Kim).

## Contents

| File | Purpose |
|---|---|
| `replication_calvo_zlb.py` | Solves the exact nonlinear models and reproduces every table and the computational validation in Appendix C. |
| `results.json` | Output of a complete run of the script. |
| `requirements.txt` | Package versions used for that run. |

## Requirements

Python 3 with NumPy and SciPy. The included `results.json` was produced with Python 3.12.3, NumPy 2.4.4 and SciPy 1.17.1:

```
pip install -r requirements.txt
```

## Usage

```
python replication_calvo_zlb.py              # all parts
python replication_calvo_zlb.py t1 t3        # selected parts only
```

Each part prints its results and writes them to `results.json` (an existing file is updated part by part).

| Part | Output |
|---|---|
| `t1` | Table 1: existence thresholds |
| `t2` | Table 2: first trap period away from the existence boundary |
| `t3` | Table 3: approach to the existence boundary |
| `t4` | Table 4: time aggregation |
| `t5` | Table 5: first-order formulas against exact thresholds |
| `t6` | Table 6: sensitivity to the calibration |
| `bound` | Numerical illustration of Proposition 3 (not tabulated in the paper) |
| `checks` | Computational validation reported in Appendix C |

A complete run takes about one to two minutes on a laptop.

## Methods

All solvers use the full nonlinear model; no equation is linearized except in the log-linear benchmark (`loglin`), which is reported for comparison. The experiment is a perfect-foresight trap: the natural rate equals `rbar` for periods 0 to T-1 with the nominal rate at zero, and from period T on inflation is zero, which is the exact terminal regime of Lemma 1.

- **Calvo, firm-specific labor markets** (`calvo_industry`): exact backward recursion. Post-exit sums are in closed form.
- **Calvo, economy-wide labor market** (`resid_econ`, `calvo_economy_path`, `refine_boundary`): price dispersion is a backward-looking state, so the trap periods are solved jointly by a Powell hybrid method with continuation in theta, imposing the reset-price floor of Lemma 2 by construction. Post-exit sums over the infinite horizon are computed exactly (`post_exit_v2`): in closed form when sigma = 1, and otherwise term by term until dispersion equals one to machine precision, with the remaining geometric tail in closed form. Continuation in theta stalls slightly before the existence boundary because Pi_0 falls very fast there, so the boundary is located by fixing log Pi_0 along a sequence down to -10 and solving for theta; stopping the sequence at log Pi_0 = -3 instead of -10 changes the boundary by less than 1e-7.
- **Rotemberg** (`rotemberg`): backward recursion with the unknown L = log k, where k is the share of output left after price-adjustment costs. This keeps solutions accurate when adjustment costs absorb almost all output. At each date the solver keeps the largest solution in L, which is the solution with the mildest deflation because Pi is increasing in L. To find all solutions (`all_roots`), it evaluates the residual of the Rotemberg condition on a grid of values of L. Wherever the residual changes sign between two neighboring grid points, a solution lies between them, and the solver locates it by repeatedly halving that interval (bisection). The grid has extra points near L = 0, where deflation is mild, and near the lower end of the domain, where adjustment costs absorb almost all output. A grid can miss two solutions that lie close together between the same two grid points, because the residual then does not change sign there. To catch such pairs, the solver takes each point where the residual comes closest to zero without crossing it and refines it with a numerical optimizer, which tests whether the residual crosses zero nearby. Non-existence is recorded when no solution exists or the solution would require L < -700, the limit of double precision. At the baseline calibration the equation has at most one solution at every date for all trap lengths and natural rates of Table 1, on a theta grid with step 0.005. Part `checks` reports this count, using only roots with residual below 1e-8. Taking the largest solution guards against spurious sign changes that rounding error creates where Pi is numerically zero.
- **Thresholds** for the firm-specific Calvo and Rotemberg models are found by bisection in theta, which assumes existence on an interval of theta; part `checks` verifies this on a grid (see below).
- **Computational validation** (`checks`, `calvo_economy_T1_roots`): on a theta grid with step 0.005, part `checks` verifies that existence holds on a single interval for the firm-specific Calvo and Rotemberg models at T = 1 and T = 8, and for the economy-wide Calvo model at T = 1. With T = 1 the price index, the dispersion equation and the Euler equation give every other variable as a function of the reset price Z_0, so the economy-wide model reduces to one equation in Z_0, the reset condition. `calvo_economy_T1_roots` evaluates the residual of this equation on a fine grid of values of Z_0 above the reset-price floor and locates every sign change, which finds all equilibria. For T > 1 this validation is not available, and the economy-wide threshold refers to the end of the deflationary branch followed from high theta. Part `checks` also reports the Rotemberg threshold at T = 8 with grids of 1,001, 2,001 and 8,001 points, the economy-wide threshold at T = 8 with the continuation step halved or the residual tolerance tightened to 1e-12, and the economy-wide thresholds at T = 1 and T = 8 with initial dispersion S_{-1} equal to 1, 1.0005 and 1.005.

Baseline calibration (quarterly): sigma = 1, phi = 1, eps = 6, beta = 0.99.
