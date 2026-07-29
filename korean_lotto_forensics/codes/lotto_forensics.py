"""
lotto_forensics.py
===================
Statistical fairness audit for Korean Lotto 6/45.

Implements sixteen tests. Eleven are ported or reformulated from
election-forensics methodology; five are natural to lottery auditing,
including out-of-sample predictive backtests and exact combinatorial
null distributions for structural claims about draw sequences.

  L1  Ball-frequency chi-square (chi2(44))
  L2  Last-digit test, corrected non-uniform null
  L3  Draw-sum distribution test (exact theoretical)
  L4  Odd/even and high/low split tests (hypergeometric)
  L5  Pairwise co-occurrence test (990 pairs)
  L6  0s-and-5s mechanical-bias test
  L7  Last-digit mean and variance (corrected null)
  L8  Simulation-adjusted chi-square (Shikano and Mack 2011)
  L9  Temporal autocorrelation (per ball, Ljung-Box)
  L10 Summary-statistic variance test
  L11 Cross-ball correlation matrix (Fisher Z test)
  L12 Cold-number mean-reversion backtest (out-of-sample)
  L13 Hot/cold half-split correlation (simulation-based null)
  L14 Inter-appearance gap coefficient of variation (simulation-based null)
  L15 Momentum-following backtest (out-of-sample, mirror image of L12)
  L16 Within-draw adjacency clustering (exact combinatorial null)

Usage
-----
  python lotto_forensics.py lotto_data.csv                          # all draws, L1-L12
  python lotto_forensics.py lotto_data.csv --from 500 --to 999       # draw range
  python lotto_forensics.py lotto_data.csv --lag 10                  # custom L9 lag
  python lotto_forensics.py lotto_data.csv --l12-windows 18 19 20 21 22
  python lotto_forensics.py lotto_data.csv --skip-l12                # skip the L12 backtest
  python lotto_forensics.py lotto_data.csv --extra-tests             # also run L13-L16
  python lotto_forensics.py lotto_data.csv --extended-sweep          # L12/L15 window x era sweep
  python lotto_forensics.py                                          # synthetic demo

CSV format (auto-detected column names):
  draw_no, date, b1, b2, b3, b4, b5, b6[, bonus]
  or ball1..ball6 / num1..num6 (encoding utf-8 or cp949)

Dependencies
------------
  pip install pandas numpy scipy matplotlib koreanize-matplotlib
"""

import argparse
import json
import math
import sys
import warnings
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare, hypergeom, norm, pearsonr
from scipy.stats import chi2 as chi2_dist

warnings.filterwarnings("ignore")

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pass

matplotlib.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────
N_POOL = 45           # number pool
K_DRAW = 6            # balls per draw
C_N_K  = math.comb(N_POOL, K_DRAW)   # 8,145,060

_LD_POP = {d: 0 for d in range(10)}
for _b in range(1, N_POOL + 1):
    _LD_POP[_b % 10] += 1

EXPECTED_LD_PROPS = np.array([_LD_POP[d] / N_POOL for d in range(10)])
EXPECTED_LD_MEAN  = sum(d * _LD_POP[d] for d in range(10)) / N_POOL
EXPECTED_LD_VAR   = (
    sum(d**2 * _LD_POP[d] for d in range(10)) / N_POOL - EXPECTED_LD_MEAN**2
)

EXPECTED_SUM_MEAN = K_DRAW * (N_POOL + 1) / 2
POP_VAR           = (N_POOL**2 - 1) / 12
EXPECTED_SUM_VAR  = (
    K_DRAW * POP_VAR * (N_POOL - K_DRAW) / (N_POOL - 1)
)

N_ODD  = 23
N_EVEN = 22
N_HIGH = 23
N_LOW  = 22

P_COOCCUR   = math.comb(N_POOL - 2, K_DRAW - 2) / C_N_K
P_ZERO_FIVE = 9 / N_POOL


# ─────────────────────────────────────────────
# DATA LOADING & FILTERING
# ─────────────────────────────────────────────
def load_lotto_data(path: str):
    """
    Read lotto draw history from CSV.
    Returns (DataFrame, ball_col_list[6], bonus_col_or_None).
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")

    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_").str.replace("번호", "")
    )

    for pattern in [
        lambda c: len(c) == 2 and c[0] == "b" and c[1].isdigit(),
        lambda c: c.startswith("ball") and c[4:].isdigit(),
        lambda c: c.startswith("num")  and c[3:].isdigit(),
        lambda c: len(c) == 2 and c[0] == "n" and c[1].isdigit(),
        lambda c: c.startswith("당첨"),
    ]:
        cols = sorted([c for c in df.columns if pattern(c)])
        if len(cols) >= 6:
            ball_cols = cols[:6]
            break
    else:
        raise ValueError(
            f"Cannot identify ball columns in: {list(df.columns)}\n"
            "Rename them to b1..b6 or ball1..ball6."
        )

    bonus_candidates = [c for c in df.columns if "bonus" in c or "보너스" in c]
    bonus_col = bonus_candidates[0] if bonus_candidates else None

    for c in ball_cols + ([bonus_col] if bonus_col else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=ball_cols).reset_index(drop=True)

    for draw_col in ["draw_no", "회차", "round", "draw"]:
        if draw_col in df.columns:
            df = df.sort_values(draw_col).reset_index(drop=True)
            break

    print(f"  Loaded {len(df):,} draws | balls: {ball_cols} | bonus: {bonus_col}")
    return df, ball_cols, bonus_col


def filter_by_draw_range(
    df: pd.DataFrame,
    draw_from: int | None,
    draw_to:   int | None,
) -> pd.DataFrame:
    """
    Keep only rows whose draw_no is within [draw_from, draw_to].
    """
    draw_col = None
    for candidate in ["draw_no", "회차", "round", "draw"]:
        if candidate in df.columns:
            draw_col = candidate
            break

    if draw_col is None:
        if draw_from is not None or draw_to is not None:
            raise ValueError(
                "Cannot filter by draw range: no draw_no column found in CSV."
            )
        return df

    if draw_from is not None:
        df = df[df[draw_col] >= draw_from]
    if draw_to is not None:
        df = df[df[draw_col] <= draw_to]

    df = df.reset_index(drop=True)

    if len(df) == 0:
        lo = draw_from if draw_from is not None else "start"
        hi = draw_to   if draw_to   is not None else "end"
        raise ValueError(f"No draws found in range [{lo}, {hi}].")

    lo_actual = int(df[draw_col].min())
    hi_actual = int(df[draw_col].max())
    print(f"  Filtered to draws {lo_actual}-{hi_actual}  ({len(df):,} draws retained)")
    return df


def extract_balls(df: pd.DataFrame, ball_cols: list) -> np.ndarray:
    return df[ball_cols].values.astype(int)


def generate_synthetic_data(n_draws: int = 1100, seed: int = 42):
    rng  = np.random.default_rng(seed)
    pool = np.arange(1, N_POOL + 1)
    rows = []
    for i in range(n_draws):
        draw  = sorted(rng.choice(pool, K_DRAW, replace=False))
        bonus = int(rng.choice([b for b in pool if b not in draw]))
        rows.append([i + 1, f"draw_{i+1}"] + draw + [bonus])
    df = pd.DataFrame(
        rows,
        columns=["draw_no", "date", "b1", "b2", "b3", "b4", "b5", "b6", "bonus"],
    )
    return df, ["b1", "b2", "b3", "b4", "b5", "b6"], "bonus"


# ─────────────────────────────────────────────
# THEORETICAL TOOLS
# ─────────────────────────────────────────────
def compute_exact_sum_distribution() -> dict:
    max_s = sum(range(N_POOL - K_DRAW + 1, N_POOL + 1))
    dp    = np.zeros((K_DRAW + 1, max_s + 1), dtype=np.int64)
    dp[0, 0] = 1
    for ball in range(1, N_POOL + 1):
        for j in range(min(ball, K_DRAW), 0, -1):
            end = max_s + 1 - ball
            if end > 0:
                dp[j, ball : max_s + 1] += dp[j - 1, :end]
    return {
        s: int(dp[K_DRAW, s]) / C_N_K
        for s in range(max_s + 1)
        if dp[K_DRAW, s] > 0
    }


def merge_bins(obs_arr, exp_arr, min_exp=5.0):
    merged_obs, merged_exp = [], []
    acc_o, acc_e = 0.0, 0.0
    for o, e in zip(obs_arr, exp_arr):
        acc_o += o
        acc_e += e
        if acc_e >= min_exp:
            merged_obs.append(acc_o)
            merged_exp.append(acc_e)
            acc_o, acc_e = 0.0, 0.0

    if acc_o > 0 or acc_e > 0:
        if merged_obs:
            merged_obs[-1] += acc_o
            merged_exp[-1] += acc_e
        else:
            merged_obs.append(acc_o)
            merged_exp.append(acc_e)

    total_obs = sum(merged_obs)
    total_exp = sum(merged_exp)
    if total_exp > 0:
        merged_exp = [e * (total_obs / total_exp) for e in merged_exp]

    return merged_obs, merged_exp


# ─────────────────────────────────────────────
# TEST IMPLEMENTATIONS
# ─────────────────────────────────────────────

def test_L1_ball_frequency(balls: np.ndarray, logs: list) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L1] Ball-Frequency Chi-Square  χ²(44)")
    log("=" * 62)

    n_draws = len(balls)
    exp     = n_draws * K_DRAW / N_POOL

    counts = np.zeros(N_POOL, dtype=int)
    for row in balls:
        counts[row - 1] += 1

    exp_arr = np.full(N_POOL, exp)
    exp_arr = exp_arr * (np.sum(counts) / np.sum(exp_arr))

    chi2_stat, p_val = chisquare(counts, f_exp=exp_arr)

    log(f"  N draws              : {n_draws:,}")
    log(f"  Expected per ball    : {exp:.2f}")
    log(f"  χ²(44) = {chi2_stat:.4f}   p = {p_val:.4f}")
    log("  Result: " + ("PASS" if p_val > 0.05 else "FAIL *"))

    p_appear = K_DRAW / N_POOL
    se       = math.sqrt(n_draws * p_appear * (1 - p_appear))
    zscores  = (counts - exp) / se
    top3     = np.argsort(np.abs(zscores))[::-1][:3]
    log("  Top-3 outlier balls (|z|-ranked):")
    for idx in top3:
        log(f"    Ball {idx+1:2d}: count={counts[idx]:4d}  z={zscores[idx]:+.3f}")

    return {"counts": counts, "expected": exp, "chi2": chi2_stat, "p": p_val,
            "zscores": zscores}


def test_L2_last_digit(balls: np.ndarray, logs: list) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L2] Last-Digit Test (Corrected Null for 1-45)")
    log("=" * 62)

    flat = balls.flatten()
    ld   = flat % 10
    n    = len(ld)
    obs  = np.array([np.sum(ld == d) for d in range(10)])
    exp  = EXPECTED_LD_PROPS * n

    exp = exp * (np.sum(obs) / np.sum(exp))
    chi2_stat, p_val = chisquare(obs, f_exp=exp)

    log(f"  N ball appearances   : {n:,}")
    log(f"  χ²(9) = {chi2_stat:.4f}   p = {p_val:.4f}")
    log("  Result: " + ("PASS" if p_val > 0.05 else "FAIL *"))
    log("  digit  observed  expected  ratio")
    for d in range(10):
        log(f"    {d}     {obs[d]:5d}   {exp[d]:7.1f}  {obs[d]/exp[d]:.3f}")

    return {"obs": obs, "exp": exp, "chi2": chi2_stat, "p": p_val}


def test_L3_draw_sum(balls: np.ndarray, logs: list) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L3] Draw-Sum Distribution Test")
    log("=" * 62)

    sums    = balls.sum(axis=1)
    n_draws = len(sums)

    obs_mean = float(sums.mean())
    obs_std  = float(sums.std())

    se_mean  = (EXPECTED_SUM_VAR / n_draws) ** 0.5
    z_mean   = (obs_mean - EXPECTED_SUM_MEAN) / se_mean
    p_mean   = 2 * (1 - norm.cdf(abs(z_mean)))

    log(f"  N draws              : {n_draws:,}")
    log(f"  Observed mean        : {obs_mean:.3f}   (expected {EXPECTED_SUM_MEAN:.1f})")
    log(f"  (a) Z-test on mean   : z = {z_mean:+.4f}   p = {p_mean:.4f}")

    theo = compute_exact_sum_distribution()
    all_s = sorted(theo)
    obs_cnt = np.array([np.sum(sums == s) for s in all_s])
    exp_cnt = np.array([theo[s] * n_draws  for s in all_s])

    m_obs, m_exp = merge_bins(obs_cnt, exp_cnt, min_exp=5.0)
    chi2_stat, p_chi2 = chisquare(m_obs, f_exp=m_exp)
    df_chi = len(m_obs) - 1

    log(f"  (b) χ²({df_chi}) binned-sum test = {chi2_stat:.4f}   p = {p_chi2:.4f}  "
        + ("PASS" if p_chi2 > 0.05 else "FAIL *"))

    return {
        "sums": sums, "theo_dist": theo,
        "obs_mean": obs_mean, "obs_std": obs_std,
        "z_mean": z_mean, "p_mean": p_mean,
        "chi2": chi2_stat, "p": p_chi2,
    }


def test_L4_splits(balls: np.ndarray, logs: list) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L4] Odd/Even & High/Low Split Tests  (Hypergeometric Null)")
    log("=" * 62)

    n_draws = len(balls)
    out = {}

    configs = [
        ("Odd  (1,3,…,45)",   lambda r: r % 2 == 1,  N_ODD),
        ("High (23-45)",        lambda r: r >= 23,     N_HIGH),
    ]
    for label, cond, n_pop in configs:
        per_draw = np.array([np.sum(cond(row)) for row in balls])
        obs      = np.bincount(per_draw, minlength=K_DRAW + 1)[: K_DRAW + 1]
        hg       = hypergeom(N_POOL, n_pop, K_DRAW)
        exp      = np.array([hg.pmf(k) * n_draws for k in range(K_DRAW + 1)])
        m_obs, m_exp = merge_bins(obs, exp)
        chi2_stat, p_val = chisquare(m_obs, f_exp=m_exp)
        df_val = len(m_obs) - 1

        log(f"\n  [{label}]  (N_success_in_pool = {n_pop})")
        log(f"  χ²({df_val}) = {chi2_stat:.4f}   p = {p_val:.4f}  "
            + ("PASS" if p_val > 0.05 else "FAIL *"))

        out[label] = {"per_draw": per_draw, "obs": obs, "exp": exp,
                      "chi2": chi2_stat, "p": p_val}

    return out


def test_L5_pairwise(balls: np.ndarray, logs: list) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L5] Pairwise Co-Occurrence Test  (990 pairs)")
    log("=" * 62)

    n_draws = len(balls)
    exp_per_pair = n_draws * P_COOCCUR

    pair_idx = {p: i for i, p in enumerate(combinations(range(1, N_POOL + 1), 2))}
    counts   = np.zeros(len(pair_idx), dtype=int)

    for row in balls:
        for pair in combinations(sorted(row), 2):
            counts[pair_idx[pair]] += 1

    exp_arr   = np.full(len(counts), exp_per_pair)
    exp_arr = exp_arr * (np.sum(counts) / np.sum(exp_arr))

    chi2_stat, p_val = chisquare(counts, f_exp=exp_arr)
    df_val = len(counts) - 1

    se      = math.sqrt(exp_per_pair * (1 - P_COOCCUR))
    zscores = (counts - exp_per_pair) / se

    log(f"  Expected per pair: {exp_per_pair:.2f}  (se ≈ {se:.2f})")
    log(f"  χ²({df_val}) = {chi2_stat:.4f}   p = {p_val:.4f}  "
        + ("PASS" if p_val > 0.05 else "FAIL *"))

    return {
        "counts": counts, "pairs": list(pair_idx.keys()), "expected": exp_per_pair,
        "zscores": zscores, "chi2": chi2_stat, "p": p_val,
    }


def test_L6_zeros_fives(balls: np.ndarray, logs: list) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L6] 0s-and-5s Test  (Mechanical Bias Indicator, ported F8)")
    log("=" * 62)

    flat  = balls.flatten()
    n     = len(flat)
    count = int(np.sum((flat % 10 == 0) | (flat % 10 == 5)))
    p_hat = count / n
    se    = math.sqrt(P_ZERO_FIVE * (1 - P_ZERO_FIVE) / n)
    z     = (p_hat - P_ZERO_FIVE) / se
    p_val = 2 * (1 - norm.cdf(abs(z)))

    log(f"  Z-statistic          : {z:+.4f}")
    log(f"  p-value              : {p_val:.4f}  " + ("PASS" if p_val > 0.05 else "FAIL *"))

    return {"count_05": count, "p_hat": p_hat, "z": z, "p": p_val}


def test_L7_ld_mean_var(balls: np.ndarray, logs: list) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L7] Last-Digit Mean & Variance  (Corrected Null, ported F9)")
    log("=" * 62)

    flat = balls.flatten()
    ld   = flat % 10
    n    = len(ld)

    obs_mean = float(ld.mean())
    obs_var  = float(ld.var())

    se_mean  = math.sqrt(EXPECTED_LD_VAR / n)
    z_mean   = (obs_mean - EXPECTED_LD_MEAN) / se_mean
    p_mean   = 2 * (1 - norm.cdf(abs(z_mean)))

    chi2_var  = (n - 1) * obs_var / EXPECTED_LD_VAR
    p_var_low = chi2_dist.cdf(chi2_var, df=n - 1)
    p_var     = 2 * min(p_var_low, 1 - p_var_low)

    log(f"  Z (mean)             : {z_mean:+.4f}   p = {p_mean:.4f}  "
        + ("PASS" if p_mean > 0.05 else "FAIL *"))
    log(f"  χ² (variance)        : {chi2_var:.2f}   p = {p_var:.4f}  "
        + ("PASS" if p_var > 0.05 else "FAIL *"))

    return {
        "obs_mean": obs_mean, "obs_var": obs_var,
        "z_mean": z_mean, "p_mean": p_mean,
        "chi2_var": chi2_var, "p_var": p_var,
    }


def test_L8_sim_chi2(balls: np.ndarray, logs: list, n_sim: int = 500, batch: int = 100) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log(f"  [L8] Simulation-Adjusted Chi-Square  (S&M 2011, ported F7d)")
    log("=" * 62)

    n_draws    = len(balls)
    exp        = n_draws * K_DRAW / N_POOL
    counts_obs = np.zeros(N_POOL, dtype=int)
    for row in balls:
        counts_obs[row - 1] += 1
    obs_chi2 = float(np.sum((counts_obs - exp) ** 2 / exp))

    rng      = np.random.default_rng(42)
    sim_vals = []
    processed = 0
    while processed < n_sim:
        b        = min(batch, n_sim - processed)
        n_total  = b * n_draws
        rand_mat = rng.random((n_total, N_POOL))
        perm_mat = np.argsort(rand_mat, axis=1)[:, :K_DRAW]
        perm_3d  = perm_mat.reshape(b, n_draws, K_DRAW)
        for i in range(b):
            flat = perm_3d[i].flatten()
            cnts = np.bincount(flat, minlength=N_POOL)
            sim_vals.append(float(np.sum((cnts - exp) ** 2 / exp)))
        processed += b

    sim_arr     = np.array(sim_vals)
    sim_95      = float(np.percentile(sim_arr, 95))
    textbook_95 = chi2_dist.ppf(0.95, df=N_POOL - 1)
    chi2_trans  = obs_chi2 * textbook_95 / sim_95 if sim_95 > 0 else float("nan")
    p_empirical = float(np.mean(sim_arr >= obs_chi2))

    log(f"  χ²_trans             : {chi2_trans:.4f}")
    log(f"  Empirical p-value    : {p_empirical:.4f}  " + ("PASS" if chi2_trans <= textbook_95 else "FAIL *"))

    return {
        "obs_chi2": obs_chi2, "sim_arr": sim_arr,
        "sim_95": sim_95, "chi2_trans": chi2_trans,
        "textbook_95": textbook_95, "p_empirical": p_empirical,
    }


def test_L9_autocorrelation(balls: np.ndarray, logs: list, max_lag: int = 5) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log(f"  [L9] Temporal Autocorrelation  (lags 1-{max_lag}, per ball)")
    log("=" * 62)

    n_draws = len(balls)
    indicator = np.zeros((n_draws, N_POOL), dtype=np.float64)
    for i, row in enumerate(balls):
        indicator[i, row - 1] = 1.0

    ac_matrix = np.zeros((N_POOL, max_lag))
    for b in range(N_POOL):
        x  = indicator[:, b]
        xc = x - x.mean()
        denom = float(np.dot(xc, xc))
        if denom == 0: continue
        for lag in range(1, max_lag + 1):
            if n_draws - lag > 0:
                r = float(np.dot(xc[lag:], xc[:-lag])) / denom
                ac_matrix[b, lag - 1] = r
            else:
                ac_matrix[b, lag - 1] = 0.0

    lag_vec = np.arange(1, max_lag + 1)
    # Avoid division by zero if lag is somehow larger than draws
    weights = np.where(n_draws - lag_vec > 0, 1.0 / (n_draws - lag_vec), 0)
    lb_stat = n_draws * (n_draws + 2) * float(np.sum(ac_matrix ** 2 * weights[np.newaxis, :]))
    lb_df   = N_POOL * max_lag
    lb_p    = float(1 - chi2_dist.cdf(lb_stat, df=lb_df))

    log(f"  Ljung-Box Q({lb_df}) = {lb_stat:.4f}   p = {lb_p:.4f}  " + ("PASS" if lb_p > 0.05 else "FAIL *"))

    return {"ac_matrix": ac_matrix, "n_draws": n_draws, "lb_stat": lb_stat, "lb_p": lb_p, "max_lag": max_lag}


def test_L10_variance(balls: np.ndarray, logs: list) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L10] Summary-Statistic Variance Test  (ported F3)")
    log("=" * 62)

    n_draws = len(balls)
    out     = {}

    def chi2_var_test(label, obs_var, theo_var):
        chi2_v    = (n_draws - 1) * obs_var / theo_var
        p_low     = chi2_dist.cdf(chi2_v, df=n_draws - 1)
        p_twotail = 2 * min(p_low, 1 - p_low)
        log(f"  [{label}] χ²({n_draws-1}) = {chi2_v:.2f}   p = {p_twotail:.4f}  " + ("PASS" if p_twotail > 0.05 else "FAIL *"))
        return {"obs_var": obs_var, "theo_var": theo_var, "chi2": chi2_v, "p": p_twotail}

    sums = balls.sum(axis=1)
    out["sum"] = chi2_var_test("Sum of 6 balls", float(sums.var(ddof=1)), EXPECTED_SUM_VAR)

    odd_per_draw = np.array([int(np.sum(row % 2 == 1)) for row in balls])
    theo_var_odd = K_DRAW * (N_ODD / N_POOL) * (N_EVEN / N_POOL) * (N_POOL - K_DRAW) / (N_POOL - 1)
    out["odd"] = chi2_var_test("Odd-ball count per draw", float(odd_per_draw.var(ddof=1)), theo_var_odd)

    ranges         = balls.max(axis=1) - balls.min(axis=1)
    rng_mc         = np.random.default_rng(0)
    pool           = np.arange(1, N_POOL + 1)
    mc_ranges      = np.array([np.ptp(rng_mc.choice(pool, K_DRAW, replace=False)) for _ in range(10_000)])
    out["range"]   = chi2_var_test("Draw range (max−min)", float(ranges.var(ddof=1)), float(mc_ranges.var()))

    return out


def test_L11_cross_correlation(balls: np.ndarray, logs: list) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L11] Cross-Ball Correlation Matrix (Pearson)")
    theo_corr = -1.0 / (N_POOL - 1)
    log(f"  Theoretical structural pair correlation: {theo_corr:.5f} (-1/44)")
    log("=" * 62)

    n_draws = len(balls)
    indicator = np.zeros((n_draws, N_POOL), dtype=np.float64)
    for i, row in enumerate(balls):
        indicator[i, row - 1] = 1.0

    # Calculate Pearson correlation matrix
    corr_matrix = np.corrcoef(indicator.T)

    # Extract the 990 unique upper-triangle correlations
    triu_idx = np.triu_indices(N_POOL, k=1)
    obs_corrs = corr_matrix[triu_idx]

    # Fisher Z-transformation to test significance
    def fisher_z(r):
        r = np.clip(r, -0.9999, 0.9999)
        return 0.5 * np.log((1 + r) / (1 - r))

    z_obs  = fisher_z(obs_corrs)
    z_theo = fisher_z(theo_corr)
    se_z   = 1.0 / math.sqrt(max(n_draws - 3, 1))

    z_stats = (z_obs - z_theo) / se_z
    p_vals  = 2 * (1 - norm.cdf(np.abs(z_stats)))

    # Global heuristic chi-square test (assumes weak dependence)
    chi2_approx = float(np.sum(z_stats**2))
    p_chi2 = float(1 - chi2_dist.cdf(chi2_approx, df=len(obs_corrs)))

    log(f"  Total pairs          : {len(obs_corrs)}")
    log(f"  Approx global χ²({len(obs_corrs)}) = {chi2_approx:.2f}   p = {p_chi2:.4f}")
    log("  Result (Global): " + ("PASS" if p_chi2 > 0.05 else "FAIL *"))

    return {
        "corr_matrix": corr_matrix,
        "obs_corrs": obs_corrs,
        "z_stats": z_stats,
        "chi2_approx": chi2_approx,
        "p_chi2": p_chi2,
    }


def _l12_least_freq_6(window_draws: np.ndarray) -> tuple | None:
    """Return the six lowest-frequency balls in a window, or None if the
    six/seven boundary is tied (ambiguous)."""
    counts = np.zeros(N_POOL, dtype=int)
    for row in window_draws:
        counts[row - 1] += 1
    order = np.argsort(counts, kind="stable")  # ball indices 0..44, ascending count
    sixth_count = counts[order[5]]
    seventh_count = counts[order[6]]
    if sixth_count != seventh_count:
        chosen = tuple(sorted(int(b) + 1 for b in order[:6]))
        return chosen
    return None


def _l12_predict(balls: np.ndarray, t: int, base_window: int, max_extend: int = 80):
    """Predict the six coldest balls for draw index t, extending the lookback
    window on ties per the settling rule until a unique set is found."""
    L = base_window
    while True:
        start = max(t - L, 0)
        window = balls[start:t]
        result = _l12_least_freq_6(window) if len(window) > 0 else None
        if result is not None:
            return result, L - base_window, True
        L += 1
        if L - base_window > max_extend or start == 0:
            counts = np.zeros(N_POOL, dtype=int)
            for row in window:
                counts[row - 1] += 1
            order = np.lexsort((np.arange(N_POOL), counts))  # tie-break by ball number
            chosen = tuple(sorted(int(b) + 1 for b in order[:6]))
            return chosen, L - base_window, False


def test_L12_mean_reversion(balls: np.ndarray, logs: list, windows=(18, 19, 20, 21, 22)) -> dict:
    """Out-of-sample backtest of the popular cold-number mean-reversion
    heuristic: forecast the next draw with the six balls least frequent in
    the recent past, extending the lookback window on ties, and compare the
    hit rate to the theoretical rate of picking six numbers at random."""
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L12] Cold-Number Mean-Reversion Backtest (out-of-sample)")
    log("=" * 62)

    n_draws = len(balls)
    theo_mean = K_DRAW * K_DRAW / N_POOL
    # exact hypergeometric variance of the match count
    theo_var = 0.0
    for k in range(K_DRAW + 1):
        pk = hypergeom(N_POOL, K_DRAW, K_DRAW).pmf(k)
        theo_var += (k - theo_mean) ** 2 * pk

    out = {}
    for w in windows:
        if n_draws <= w:
            continue
        matches, extends = [], []
        for t in range(w, n_draws):
            pred, ext, _settled = _l12_predict(balls, t, w)
            actual = set(int(x) for x in balls[t])
            matches.append(len(actual & set(pred)))
            extends.append(ext)
        matches = np.array(matches)
        n = len(matches)
        se = math.sqrt(theo_var / n)
        z = (matches.mean() - theo_mean) / se
        pct_ext = float(np.mean(np.array(extends) > 0) * 100)
        out[w] = {
            "n_tests": n,
            "mean_matches": float(matches.mean()),
            "match_dist": np.bincount(matches, minlength=K_DRAW + 1) / n * 100,
            "avg_extra_draws": float(np.mean(extends)),
            "pct_needed_extension": pct_ext,
            "z": z,
        }
        log(f"  Window {w:>2}: n={n:5d}  mean_matches={matches.mean():.4f}  "
            f"z={z:+.3f}  avg_extra_draws={np.mean(extends):.2f}  "
            f"needed_extension={pct_ext:.1f}%")

    log(f"\n  Theoretical (random-guess) mean matches: {theo_mean:.4f}")
    any_significant = any(abs(r["z"]) >= 1.96 for r in out.values())
    log("  Result: " + ("FAIL \u2605 (at least one window significant)" if any_significant
                          else "PASS (no window distinguishable from random guessing)"))

    return {"windows": out, "theo_mean": theo_mean, "theo_var": theo_var}


def test_L13_hot_cold_split(balls: np.ndarray, logs: list, n_sim: int = 5000, seed: int = 42) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L13] Hot/Cold Half-Split Correlation")
    log("=" * 62)

    n = len(balls)
    mid = n // 2
    c1 = np.bincount(balls[:mid].flatten(), minlength=N_POOL + 1)[1:]
    c2 = np.bincount(balls[mid:].flatten(), minlength=N_POOL + 1)[1:]
    r_obs, _ = pearsonr(c1, c2)

    rng = np.random.default_rng(seed)
    sims = np.empty(n_sim)
    for i in range(n_sim):
        d1 = _gen_fair_draws(mid, rng)
        d2 = _gen_fair_draws(n - mid, rng)
        s1 = np.bincount(d1.flatten(), minlength=N_POOL + 1)[1:]
        s2 = np.bincount(d2.flatten(), minlength=N_POOL + 1)[1:]
        sims[i] = pearsonr(s1, s2)[0]
    p_empirical = float(np.mean(np.abs(sims) >= abs(r_obs)))

    log(f"  Split: {mid} / {n - mid} draws")
    log(f"  Observed r = {r_obs:+.4f}   simulated null: mean={sims.mean():+.4f} sd={sims.std():.4f}")
    log(f"  Empirical p-value (n_sim={n_sim}): {p_empirical:.4f}  "
        + ("PASS" if p_empirical > 0.05 else "FAIL *"))

    robustness = {}
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = int(n * frac)
        a = np.bincount(balls[:m].flatten(), minlength=N_POOL + 1)[1:]
        b = np.bincount(balls[m:].flatten(), minlength=N_POOL + 1)[1:]
        robustness[frac] = float(pearsonr(a, b)[0])
    log("  Robustness across split fractions (r): "
        + ", ".join(f"{f:.1f}={r:+.3f}" for f, r in robustness.items()))

    return {
        "r_obs": float(r_obs), "sim_mean": float(sims.mean()), "sim_std": float(sims.std()),
        "p_empirical": p_empirical, "n_sim": n_sim, "n1": mid, "n2": n - mid,
        "robustness_by_split_fraction": robustness,
    }


def _mean_gap_cv(balls_arr: np.ndarray) -> float:
    cvs = []
    for num in range(1, N_POOL + 1):
        appearances = np.where(np.any(balls_arr == num, axis=1))[0]
        if len(appearances) >= 3:
            gaps = np.diff(appearances)
            mg = float(np.mean(gaps))
            if mg > 0:
                cvs.append(float(np.std(gaps, ddof=1)) / mg)
    return float(np.mean(cvs)) if cvs else float("nan")


def test_L14_gap_cv(balls: np.ndarray, logs: list, n_sim: int = 2000, seed: int = 43) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L14] Inter-Appearance Gap Coefficient of Variation")
    log("=" * 62)

    n = len(balls)
    obs_cv = _mean_gap_cv(balls)
    theo_cv = math.sqrt(1 - K_DRAW / N_POOL)

    rng = np.random.default_rng(seed)
    sims = np.empty(n_sim)
    for i in range(n_sim):
        sims[i] = _mean_gap_cv(_gen_fair_draws(n, rng))
    p_empirical = min(2 * min(np.mean(sims <= obs_cv), np.mean(sims >= obs_cv)), 1.0)

    log(f"  Observed mean gap CV     : {obs_cv:.4f}")
    log(f"  Textbook geometric CV    : {theo_cv:.4f}")
    log(f"  Simulated null: mean={sims.mean():.4f}  sd={sims.std():.4f}")
    log(f"  Empirical p-value (n_sim={n_sim}): {p_empirical:.4f}  "
        + ("PASS" if p_empirical > 0.05 else "FAIL *"))

    return {
        "obs_cv": obs_cv, "theo_cv_geometric": theo_cv,
        "sim_mean": float(sims.mean()), "sim_std": float(sims.std()),
        "p_empirical": float(p_empirical), "n_sim": n_sim,
    }


def _most_freq_6(window_draws: np.ndarray) -> tuple | None:
    counts = np.zeros(N_POOL, dtype=int)
    for row in window_draws:
        counts[row - 1] += 1
    order = np.argsort(-counts, kind="stable")
    if counts[order[5]] != counts[order[6]]:
        return tuple(sorted(int(b) + 1 for b in order[:6]))
    return None


def _predict_momentum(balls: np.ndarray, t: int, base_window: int, max_extend: int = 80):
    L = base_window
    while True:
        start = max(t - L, 0)
        window = balls[start:t]
        result = _most_freq_6(window) if len(window) > 0 else None
        if result is not None:
            return result, L - base_window
        L += 1
        if L - base_window > max_extend or start == 0:
            counts = np.zeros(N_POOL, dtype=int)
            for row in window:
                counts[row - 1] += 1
            order = np.lexsort((np.arange(N_POOL), -counts))
            return tuple(sorted(int(b) + 1 for b in order[:6])), L - base_window


def test_L15_momentum(balls: np.ndarray, logs: list, windows=(18, 19, 20, 21, 22)) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L15] Momentum-Following Backtest (out-of-sample)")
    log("=" * 62)

    n_draws = len(balls)
    theo_mean = K_DRAW * K_DRAW / N_POOL
    theo_var = sum((k - theo_mean) ** 2 * hypergeom(N_POOL, K_DRAW, K_DRAW).pmf(k) for k in range(K_DRAW + 1))

    out = {}
    for w in windows:
        matches, extends = [], []
        for t in range(w, n_draws):
            pred, ext = _predict_momentum(balls, t, w)
            actual = set(int(x) for x in balls[t])
            matches.append(len(actual & set(pred)))
            extends.append(ext)
        matches = np.array(matches)
        n = len(matches)
        se = math.sqrt(theo_var / n)
        z = (matches.mean() - theo_mean) / se
        pct_ext = float(np.mean(np.array(extends) > 0) * 100)
        out[w] = {
            "n_tests": n, "mean_matches": float(matches.mean()), "z": float(z),
            "avg_extra_draws": float(np.mean(extends)), "pct_needed_extension": pct_ext,
        }
        log(f"  Window {w:>2}: n={n:5d}  mean_matches={matches.mean():.4f}  "
            f"z={z:+.3f}  avg_extra_draws={np.mean(extends):.2f}  needed_extension={pct_ext:.1f}%")

    log(f"\n  Theoretical (random-guess) mean matches: {theo_mean:.4f}")
    any_significant = any(abs(r["z"]) >= 1.96 for r in out.values())
    log("  Result: " + ("FAIL * (at least one window significant)" if any_significant
                          else "PASS (no window distinguishable from random guessing)"))

    return {"windows": out, "theo_mean": theo_mean}


def _exact_adjacency_pmf() -> dict:
    """Exact distribution of the number of adjacent-integer pairs among a
    uniformly random 6-of-45 draw, via dynamic programming over the 45
    balls in order (analogous to the exact draw-sum derivation for L3)."""
    max_k = K_DRAW - 1
    dp = np.zeros((K_DRAW + 1, 2, max_k + 1), dtype=object)
    dp[0][0][0] = 1
    for _ball in range(1, N_POOL + 1):
        ndp = np.zeros((K_DRAW + 1, 2, max_k + 1), dtype=object)
        for j in range(K_DRAW + 1):
            for prev in range(2):
                for k in range(max_k + 1):
                    c = dp[j][prev][k]
                    if c == 0:
                        continue
                    ndp[j][0][k] += c
                    if j + 1 <= K_DRAW:
                        nk = k + 1 if prev == 1 else k
                        if nk <= max_k:
                            ndp[j + 1][1][nk] += c
        dp = ndp
    pmf, total = {}, 0
    for prev in range(2):
        for k in range(max_k + 1):
            c = int(dp[K_DRAW][prev][k])
            pmf[k] = pmf.get(k, 0) + c
            total += c
    assert total == C_N_K, (total, C_N_K)
    return {k: v / C_N_K for k, v in pmf.items()}


def _count_adjacent_pairs(row: np.ndarray) -> int:
    s = sorted(int(x) for x in row)
    return sum(1 for a, b in zip(s, s[1:]) if b - a == 1)


def test_L16_adjacency(balls: np.ndarray, logs: list) -> dict:
    log = lambda m: (print(m), logs.append(m))
    log("\n" + "=" * 62)
    log("  [L16] Within-Draw Adjacency Clustering  (exact null)")
    log("=" * 62)

    pmf = _exact_adjacency_pmf()
    max_k = max(pmf)
    theo_mean = sum(k * p for k, p in pmf.items())
    theo_var = sum((k - theo_mean) ** 2 * p for k, p in pmf.items())

    n_draws = len(balls)
    obs_per_draw = np.array([_count_adjacent_pairs(row) for row in balls])
    obs_mean = float(obs_per_draw.mean())

    se = math.sqrt(theo_var / n_draws)
    z = (obs_mean - theo_mean) / se
    p_z = 2 * (1 - norm.cdf(abs(z)))

    obs_hist = np.bincount(obs_per_draw, minlength=max_k + 1)
    exp_hist = np.array([pmf.get(k, 0) * n_draws for k in range(max_k + 1)])
    m_obs, m_exp = merge_bins(obs_hist, exp_hist, min_exp=5.0)
    chi2_stat, chi2_p = chisquare(m_obs, f_exp=m_exp)

    log(f"  Exact theoretical mean adjacent pairs/draw : {theo_mean:.4f}")
    log(f"  Observed mean                              : {obs_mean:.4f}")
    log(f"  Z-test: z = {z:+.4f}   p = {p_z:.4f}  " + ("PASS" if p_z > 0.05 else "FAIL *"))
    log(f"  Binned chi-square({len(m_obs)-1}) = {chi2_stat:.4f}   p = {chi2_p:.4f}  "
        + ("PASS" if chi2_p > 0.05 else "FAIL *"))

    return {
        "pmf": pmf, "theo_mean": theo_mean, "theo_var": theo_var,
        "obs_mean": obs_mean, "z": float(z), "p_z": float(p_z),
        "chi2": float(chi2_stat), "chi2_df": len(m_obs) - 1, "chi2_p": float(chi2_p),
        "obs_hist": obs_hist.tolist(), "exp_hist": [float(x) for x in exp_hist], "n_draws": n_draws,
    }


def _gen_fair_draws(n: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized generation of n fair 6-of-45 draws via the random-permutation
    trick: rank N_POOL uniform draws per row and keep the lowest K_DRAW ranks."""
    rand_mat = rng.random((n, N_POOL))
    return np.argsort(rand_mat, axis=1)[:, :K_DRAW] + 1


def run_window_era_sweep(balls: np.ndarray, logs: list, mode: str = "cold",
                          windows=range(18, 51), out_path: str = "l12_extended_sweep.png") -> dict:
    """Extend the L12/L15 backtest to a wider range of lookback windows and
    report results separately within each of the five temporal windows used
    for the L1-L11 static audit, in addition to the full history, to check
    whether the (null) predictive result is stable across eras."""
    log = lambda m: (print(m), logs.append(m))
    label = "L12 cold-number" if mode == "cold" else "L15 momentum-following"
    log("\n" + "=" * 62)
    log(f"  [{('L12' if mode == 'cold' else 'L15')}-extended] {label} backtest, windows {windows.start}-{windows.stop - 1}, by era")
    log("=" * 62)

    n_draws = len(balls)
    theo_mean = K_DRAW * K_DRAW / N_POOL
    theo_var = sum((k - theo_mean) ** 2 * hypergeom(N_POOL, K_DRAW, K_DRAW).pmf(k) for k in range(K_DRAW + 1))

    predict_fn = _l12_predict if mode == "cold" else _predict_momentum

    eras = {
        "W1 (1-250)": (0, 250), "W2 (251-500)": (250, 500), "W3 (501-750)": (500, 750),
        "W4 (751-1000)": (750, 1000), "W5 (1001-end)": (1000, n_draws), "Full": (0, n_draws),
    }

    matches_by_w = {}
    for w in windows:
        matches = np.full(n_draws, -1, dtype=int)
        for t in range(w, n_draws):
            pred = predict_fn(balls, t, w)[0]
            actual = set(int(x) for x in balls[t])
            matches[t] = len(actual & set(pred))
        matches_by_w[w] = matches

    table = {}
    for era, (lo, hi) in eras.items():
        table[era] = {}
        for w in windows:
            m = matches_by_w[w]
            idx = np.arange(max(lo, w), hi)
            vals = m[idx]
            vals = vals[vals >= 0]
            n = len(vals)
            if n == 0:
                continue
            mean_m = float(vals.mean())
            se = math.sqrt(theo_var / n)
            z = (mean_m - theo_mean) / se
            table[era][w] = {"n": n, "mean": mean_m, "z": z}

    for era, wdict in table.items():
        zs = np.array([wdict[w]["z"] for w in wdict])
        n_sig = int(np.sum(np.abs(zs) >= 1.96))
        log(f"  {era:>16}: mean(z)={zs.mean():+.3f}  min(z)={zs.min():+.3f}  "
            f"max(z)={zs.max():+.3f}  n_sig(|z|>=1.96)={n_sig}/{len(zs)}")

    fig, ax = plt.subplots(figsize=(14, 5))
    era_names = list(eras.keys())
    window_list = list(windows)
    z_grid = np.array([[table[e][w]["z"] for w in window_list] for e in era_names])
    im = ax.imshow(z_grid, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(0, len(window_list), 2))
    ax.set_xticklabels([window_list[i] for i in range(0, len(window_list), 2)], fontsize=8)
    ax.set_yticks(range(len(era_names)))
    ax.set_yticklabels(era_names, fontsize=10)
    ax.set_xlabel("Base lookback window (draws)")
    ax.set_title(f"{label} backtest: z-score by era and window size")
    fig.colorbar(im, ax=ax, label="z-score")
    for i in range(len(era_names)):
        for j, w in enumerate(window_list):
            if abs(z_grid[i, j]) >= 1.96:
                ax.text(j, i, "*", ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    log(f"  Sweep heatmap saved to '{out_path}'")

    return {"theo_mean": theo_mean, "theo_var": theo_var, "table": table}
# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
def _pass_fail(p: float, threshold: float = 0.05) -> str:
    return "PASS" if p > threshold else "FAIL *"

def plot_dashboard(balls: np.ndarray, results: dict, out_path: str = "lotto_forensics_dashboard.png", range_label: str = "") -> None:
    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    title = "Korean Lotto 6/45 Statistical Fairness Audit Dashboard"
    if range_label: title += f"  [{range_label}]"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    r1, r2, r3, r4 = results["L1"], results["L2"], results["L3"], results["L4"]
    r5, r6, r7, r8 = results["L5"], results["L6"], results["L7"], results["L8"]
    r9, r10, r11   = results["L9"], results["L10"], results["L11"]
    n_draws = len(balls)
    ball_nums = np.arange(1, N_POOL + 1)
    W = 0.4

    # --- ROW 0 ---
    ax = axes[0, 0]
    clr = ["tomato" if abs(z) > 2 else "steelblue" for z in r1["zscores"]]
    ax.bar(ball_nums, r1["counts"], color=clr, edgecolor="none", alpha=0.85)
    ax.axhline(r1["expected"], color="black", ls="--", lw=1.5, label=f"Exp ({r1['expected']:.0f})")
    ax.set_title(f"[L1] Ball Frequency  χ²(44)={r1['chi2']:.2f}  p={r1['p']:.3f}  {_pass_fail(r1['p'])}", fontsize=9)
    ax.set_xlabel("Ball number", fontsize=8); ax.set_ylabel("Count", fontsize=8)

    ax = axes[0, 1]
    ax.hist(r3["sums"], bins=range(int(r3["sums"].min()), int(r3["sums"].max()) + 2), alpha=0.60, color="steelblue", density=True, label="Observed")
    sv = sorted(r3["theo_dist"])
    ax.plot(sv, [r3["theo_dist"][s] for s in sv], "r-", lw=1.5, label="Exact Theo")
    ax.set_title(f"[L3] Draw Sum  p={r3['p']:.3f}  {_pass_fail(r3['p'])}", fontsize=9)
    ax.set_xlabel("Sum", fontsize=8); ax.legend(fontsize=7)

    ax = axes[0, 2]
    r4k = list(r4.keys())[0]
    k_vals = np.arange(K_DRAW + 1)
    ax.bar(k_vals - W/2, r4[r4k]["obs"], W, alpha=0.85, color="mediumpurple", label="Obs")
    ax.bar(k_vals + W/2, r4[r4k]["exp"], W, alpha=0.65, color="slategray", label="Exp")
    ax.set_title(f"[L4] Odd Count  p={r4[r4k]['p']:.3f}  {_pass_fail(r4[r4k]['p'])}", fontsize=9)
    ax.set_xlabel("# odd balls", fontsize=8); ax.set_xticks(k_vals)

    ax = axes[0, 3]
    r4k2 = list(r4.keys())[1] if len(r4) > 1 else list(r4.keys())[0]
    ax.bar(k_vals - W/2, r4[r4k2]["obs"], W, alpha=0.85, color="teal", label="Obs")
    ax.bar(k_vals + W/2, r4[r4k2]["exp"], W, alpha=0.65, color="slategray", label="Exp")
    ax.set_title(f"[L4] High Count  p={r4[r4k2]['p']:.3f}  {_pass_fail(r4[r4k2]['p'])}", fontsize=9)
    ax.set_xlabel("# high balls", fontsize=8); ax.set_xticks(k_vals)

    # --- ROW 1 ---
    ax = axes[1, 0]
    digits = np.arange(10)
    ax.bar(digits - W/2, r2["obs"], W, alpha=0.85, color="salmon")
    ax.bar(digits + W/2, r2["exp"], W, alpha=0.65, color="slategray")
    ax.set_title(f"[L2] Last-Digit  p={r2['p']:.3f}  {_pass_fail(r2['p'])}", fontsize=9)
    ax.set_xticks(digits)

    ax = axes[1, 1]
    ax.hist(r8["sim_arr"], bins=35, alpha=0.75, color="mediumseagreen", edgecolor="none")
    ax.axvline(r8["obs_chi2"], color="red", lw=2.0, ls="--", label=f"Obs: {r8['obs_chi2']:.2f}")
    ax.set_title(f"[L8] Sim-Adj χ²  emp_p={r8['p_empirical']:.3f}  {_pass_fail(r8['p_empirical'])}", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    ax.hist(r5["zscores"], bins=40, alpha=0.75, color="darkcyan", density=True)
    xn = np.linspace(-4, 4, 200)
    ax.plot(xn, norm.pdf(xn), "r-", lw=1.5, label="N(0,1)")
    ax.set_title(f"[L5] Pair Co-occurrence Z  p={r5['p']:.3f}  {_pass_fail(r5['p'])}", fontsize=9)

    ax = axes[1, 3]
    ax.bar(ball_nums, r1["zscores"], color=clr, edgecolor="none", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(2, color="red", lw=1.0, ls=":", label="±2σ")
    ax.axhline(-2, color="red", lw=1.0, ls=":")
    ax.set_title("[L1] Ball Frequency Z-scores", fontsize=9)

    # --- ROW 2 ---
    ax = axes[2, 0]
    im = ax.imshow(r9["ac_matrix"].T, aspect="auto", cmap="RdBu_r", vmin=-0.15, vmax=0.15, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Autocorrelation")
    ax.set_title(f"[L9] Temporal Autocorr  p={r9['lb_p']:.3f}  {_pass_fail(r9['lb_p'])}", fontsize=9)

    # Use dynamic ticks up to max_lag, ensuring we don't crowd it if max_lag is massive
    max_lag_display = min(r9["max_lag"], 20)
    ax.set_yticks(range(max_lag_display))
    ax.set_yticklabels([f"Lag {i+1}" for i in range(max_lag_display)], fontsize=7)

    ax = axes[2, 1]
    ac1 = r9["ac_matrix"][:, 0]
    thr = 1.96 / math.sqrt(r9["n_draws"])
    clr3 = ["tomato" if abs(r) > thr else "steelblue" for r in ac1]
    ax.bar(ball_nums, ac1, color=clr3, edgecolor="none", alpha=0.85)
    ax.axhline(thr, color="red", lw=1.0, ls=":")
    ax.axhline(-thr, color="red", lw=1.0, ls=":")
    ax.set_title("[L9] Lag-1 Autocorrelation per Ball", fontsize=9)

    ax = axes[2, 2]
    labels_v = ["Sum", "Odd count", "Range"]
    ratios = [r10["sum"]["obs_var"]/r10["sum"]["theo_var"], r10["odd"]["obs_var"]/r10["odd"]["theo_var"], r10["range"]["obs_var"]/r10["range"]["theo_var"]]
    pvals_v = [r10["sum"]["p"], r10["odd"]["p"], r10["range"]["p"]]
    clr4 = ["tomato" if p < 0.05 else "steelblue" for p in pvals_v]
    bars = ax.bar(labels_v, ratios, color=clr4, edgecolor="none", alpha=0.85)
    ax.axhline(1.0, color="black", lw=1.5, ls="--")
    for bar, p in zip(bars, pvals_v): ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"p={p:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("[L10] Variance Ratio (obs / theo)", fontsize=9)

    ax = axes[2, 3]
    plot_corr = r11["corr_matrix"].copy()
    np.fill_diagonal(plot_corr, np.nan)
    im2 = ax.imshow(plot_corr, cmap="RdBu_r", vmin=-0.1, vmax=0.1, interpolation="none")
    fig.colorbar(im2, ax=ax, label="Pearson r")
    ax.set_title(f"[L11] Cross-Ball Correlation Heatmap", fontsize=9)
    ax.set_xlabel("Ball number")

    # --- ROW 3 ---
    ax = axes[3, 0]
    ax.hist(r11["z_stats"], bins=40, alpha=0.75, color="purple", density=True, edgecolor="none")
    ax.plot(xn, norm.pdf(xn), "r-", lw=1.5, label="N(0,1)")
    ax.set_title(f"[L11] Cross-Ball Z-scores  p={r11['p_chi2']:.3f}  {_pass_fail(r11['p_chi2'])}", fontsize=9)
    ax.set_xlabel("Fisher Z-score")

    r12 = results.get("L12")

    ax = axes[3, 1]
    if r12:
        w_list = sorted(r12["windows"])
        means = [r12["windows"][w]["mean_matches"] for w in w_list]
        zs = [r12["windows"][w]["z"] for w in w_list]
        clr5 = ["tomato" if abs(z) >= 1.96 else "steelblue" for z in zs]
        ax.bar([str(w) for w in w_list], means, color=clr5, edgecolor="none", alpha=0.85)
        ax.axhline(r12["theo_mean"], color="black", ls="--", lw=1.5, label=f"Random guess ({r12['theo_mean']:.3f})")
        ax.set_title("[L12] Mean Matches by Lookback Window", fontsize=9)
        ax.set_xlabel("Base window (draws)", fontsize=8)
        ax.legend(fontsize=7)
    else:
        ax.axis("off")

    ax = axes[3, 2]
    if r12:
        w_list = sorted(r12["windows"])
        k_vals = np.arange(K_DRAW + 1)
        mid_w = w_list[len(w_list) // 2]
        dist = r12["windows"][mid_w]["match_dist"]
        theo_dist = np.array([hypergeom(N_POOL, K_DRAW, K_DRAW).pmf(k) * 100 for k in k_vals])
        ax.bar(k_vals - W/2, dist, W, alpha=0.85, color="mediumpurple", label=f"Obs (W={mid_w})")
        ax.bar(k_vals + W/2, theo_dist, W, alpha=0.65, color="slategray", label="Theory")
        ax.set_title("[L12] Match-Count Distribution", fontsize=9)
        ax.set_xlabel("# balls matched", fontsize=8); ax.set_xticks(k_vals)
        ax.legend(fontsize=7)
    else:
        ax.axis("off")

    ax = axes[3, 3]
    ax.axis("off")
    def _pline(tag, label, p): return f"  {'✓' if p > 0.05 else '✗'} {tag:<4} {label:<28} p={p:.4f}"
    lines = [
        "══════ AUDIT SUMMARY ══════",
        _pline("L1",  "Ball frequency",          r1["p"]),
        _pline("L2",  "Last-digit (corr null)",  r2["p"]),
        _pline("L3",  "Draw sum distrib",         r3["p"]),
        _pline("L4",  "Odd/even split",           r4[list(r4)[0]]["p"]),
        _pline("L4",  "High/low split",           r4[list(r4)[1 if len(r4)>1 else 0]]["p"]),
        _pline("L5",  "Pair co-occurrence",       r5["p"]),
        _pline("L6",  "0s & 5s bias",             r6["p"]),
        _pline("L7",  "LD mean (corr null)",      r7["p_mean"]),
        _pline("L7",  "LD variance",              r7["p_var"]),
        _pline("L8",  "Sim-adj chi-square",       r8["p_empirical"]),
        _pline("L9",  f"Autocorr (max_lag={r9['max_lag']})", r9["lb_p"]),
        _pline("L10", "Variance (sum)",           r10["sum"]["p"]),
        _pline("L10", "Variance (odd count)",     r10["odd"]["p"]),
        _pline("L10", "Variance (range)",         r10["range"]["p"]),
        _pline("L11", "Cross-ball correlation",   r11["p_chi2"]),
    ]
    if r12:
        for w in sorted(r12["windows"]):
            z = r12["windows"][w]["z"]
            p_like = 2 * (1 - norm.cdf(abs(z)))  # two-tailed p from the z-statistic
            lines.append(_pline("L12", f"Cold-number backtest (W={w})", p_like))
    n_fail = sum(1 for ln in lines[1:] if "✗" in ln)
    lines += ["", f"  Failures: {n_fail} / {len(lines)-1}", f"  Draws analysed: {n_draws:,}", f"  Range: {range_label if range_label else 'all'}"]
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=8.5, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.75))
    ax.set_title("Test Summary", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n  Dashboard saved to '{out_path}'")

def save_report(logs: list, out_path: str = "lotto_forensics_report.txt") -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(["=" * 65, "  KOREAN LOTTO 6/45 STATISTICAL FAIRNESS AUDIT REPORT", "=" * 65, ""] + logs))
    print(f"  Text report saved to '{out_path}'")

# ─────────────────────────────────────────────
# JSON EXPORT
# ─────────────────────────────────────────────
def _json_safe(obj):
    """Recursively convert numpy scalars/arrays, tuples, and non-string dict
    keys into plain JSON-serializable structures."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Korean Lotto 6/45 Statistical Fairness Audit"
    )
    parser.add_argument("csv", nargs="?", default=None, help="Path to CSV file (omit for synthetic demo)")
    parser.add_argument("--from", dest="draw_from", type=int, default=None, help="First draw_no to include")
    parser.add_argument("--to",   dest="draw_to",   type=int, default=None, help="Last draw_no to include")
    parser.add_argument("--lag",  dest="lag",       type=int, default=5,    help="Max lag period for L9 Autocorrelation test (default: 5)")
    parser.add_argument("--l12-windows", dest="l12_windows", type=int, nargs="+", default=[18, 19, 20, 21, 22],
                         help="Lookback windows for the L12 cold-number backtest (default: 18 19 20 21 22)")
    parser.add_argument("--skip-l12", dest="skip_l12", action="store_true",
                         help="Skip the L12 out-of-sample backtest")
    parser.add_argument("--extra-tests", dest="extra_tests", action="store_true",
                         help="Also run L13-L16 (hot/cold split correlation, gap CV, momentum backtest, adjacency clustering)")
    parser.add_argument("--extended-sweep", dest="extended_sweep", action="store_true",
                         help="Run the widened window (18-50) x temporal-era sweep for L12 (and L15 if --extra-tests is set)")
    parser.add_argument("--sweep-windows", dest="sweep_windows", type=int, nargs=2, default=[18, 50],
                         metavar=("MIN", "MAX"), help="Inclusive window range for --extended-sweep (default: 18 50)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        df, ball_cols, bonus_col = load_lotto_data(args.csv)
        df = filter_by_draw_range(df, args.draw_from, args.draw_to)
        draw_col = next((c for c in ["draw_no", "회차", "round", "draw"] if c in df.columns), None)
        if draw_col:
            lo, hi = int(df[draw_col].min()), int(df[draw_col].max())
            range_label, range_tag = f"draws {lo}-{hi}", f"{lo}_{hi}"
        else:
            range_label, range_tag = "", "all"
        tag = f"{args.csv.replace('.csv', '').replace('.CSV', '')}_{range_tag}"
    else:
        print("\n  [!] No data file provided; using synthetic fair draws (demo).")
        df, ball_cols, bonus_col = generate_synthetic_data(n_draws=1100)
        range_label, tag = "synthetic demo", "synthetic_demo"

    balls = extract_balls(df, ball_cols)
    logs = []

    print(f"\n  Running forensic tests on {len(balls):,} draws...")
    results = {
        "L1":  test_L1_ball_frequency(balls, logs),
        "L2":  test_L2_last_digit(balls, logs),
        "L3":  test_L3_draw_sum(balls, logs),
        "L4":  test_L4_splits(balls, logs),
        "L5":  test_L5_pairwise(balls, logs),
        "L6":  test_L6_zeros_fives(balls, logs),
        "L7":  test_L7_ld_mean_var(balls, logs),
        "L8":  test_L8_sim_chi2(balls, logs, n_sim=500, batch=100),
        "L9":  test_L9_autocorrelation(balls, logs, max_lag=args.lag),
        "L10": test_L10_variance(balls, logs),
        "L11": test_L11_cross_correlation(balls, logs),
    }
    if not args.skip_l12:
        results["L12"] = test_L12_mean_reversion(balls, logs, windows=tuple(args.l12_windows))

    if args.extra_tests:
        results["L13"] = test_L13_hot_cold_split(balls, logs)
        results["L14"] = test_L14_gap_cv(balls, logs)
        results["L15"] = test_L15_momentum(balls, logs)
        results["L16"] = test_L16_adjacency(balls, logs)

    if args.extended_sweep:
        w_min, w_max = args.sweep_windows
        sweep_range = range(w_min, w_max + 1)
        if not args.skip_l12:
            results["L12_extended"] = run_window_era_sweep(
                balls, logs, mode="cold", windows=sweep_range,
                out_path=f"l12_extended_sweep_{tag}.png",
            )
        if args.extra_tests:
            results["L15_extended"] = run_window_era_sweep(
                balls, logs, mode="hot", windows=sweep_range,
                out_path=f"l15_extended_sweep_{tag}.png",
            )

    plot_dashboard(balls, results, out_path=f"lotto_forensics_dashboard_{tag}.png", range_label=range_label)
    save_report(logs, out_path=f"lotto_forensics_report_{tag}.txt")

    json_path = f"lotto_forensics_results_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"  Full results JSON saved to '{json_path}'")


if __name__ == "__main__":
    main()
