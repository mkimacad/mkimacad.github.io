"""Replication code for "Nonlinear Calvo Pricing, the Reset-Price Floor, and the
Paradox of Flexibility at the Zero Lower Bound".

Experiment (perfect foresight). A discount-factor shock sets the natural rate to
rbar for t = 0, ..., T-1 while the nominal rate is held at zero. From T on,
inflation is zero and real marginal cost is at its steady-state value, which is
the exact terminal regime of Lemma 1 in the paper. All solvers use the full
nonlinear model; post-exit sums over the infinite horizon are computed exactly
(closed form or exact geometric tail), with no truncation. Baseline calibration
(quarterly): sigma = 1, phi = 1, eps = 6, beta = 0.99.

Usage:
    python replication_calvo_zlb.py                # all parts
    python replication_calvo_zlb.py t1 t3 checks   # selected parts only
Parts and the tables they reproduce:
    t1      Table 1, existence thresholds
    t2      Table 2, first trap period away from the existence boundary
    t3      Table 3, approach to the existence boundary
    t4      Table 4, time aggregation
    t5      Table 5, first-order formulas against exact thresholds
    t6      Table 6, sensitivity to the calibration
    bound   numerical illustration of Proposition 3 (not tabulated)
    checks  computational validation reported in Appendix C
Results are printed and also written to results.json.
"""
import json
import sys
import time

import numpy as np
from scipy.optimize import brentq, root

S_INIT = 1.0  # S_{-1}; 1 is the zero-inflation steady state
BASE = dict(sigma=1.0, phi=1.0, eps=6.0, beta=0.99)


def yss(p):
    """Steady-state output, from Y^(sigma+phi) = (eps-1)/eps."""
    return ((p['eps'] - 1) / p['eps']) ** (1 / (p['sigma'] + p['phi']))


def zfloor(theta, e):
    """Reset-price floor of Lemma 2: Z_t > (1-theta)^(1/(eps-1))."""
    return (1 - theta) ** (1 / (e - 1))


def pi_from_z(z, theta, e):
    """Gross inflation implied by the exact Calvo price index."""
    return ((1 - (1 - theta) * z ** (1 - e)) / theta) ** (1 / (e - 1))


# ---------------------------------------------------------------------------
# Calvo, firm-specific labor markets: exact backward recursion
# ---------------------------------------------------------------------------
def calvo_industry(theta, T, rbar, p=BASE):
    """Return the equilibrium path, or None if the reset-price floor binds."""
    s, f, e, b = p['sigma'], p['phi'], p['eps'], p['beta']
    zeta = s + f
    y0, mc = yss(p), (e - 1) / e
    bs, zmin = 1 / (1 + rbar), zfloor(theta, e)
    Y, Pi = np.empty(T), np.ones(T + 1)
    v1 = y0 ** (1 - s) * mc / (1 - b * theta)
    v2 = y0 ** (1 - s) / (1 - b * theta)
    y_next = y0
    for t in range(T - 1, -1, -1):
        Y[t] = (bs * y_next ** (-s) / Pi[t + 1]) ** (-1 / s)
        v1 = Y[t] ** (1 - s) * Y[t] ** zeta + bs * theta * Pi[t + 1] ** (e * (1 + f)) * v1
        v2 = Y[t] ** (1 - s) + bs * theta * Pi[t + 1] ** (e - 1) * v2
        z = (e / (e - 1) * v1 / v2) ** (1 / (1 + e * f))
        if not np.isfinite(z) or z <= zmin:
            return None
        Pi[t] = pi_from_z(z, theta, e)
        y_next = Y[t]
    return dict(Pi=Pi[:T], x=np.log(Y / y0))


def post_exit_v2(s_T, theta, p=BASE):
    """Exact infinite sum V_{2,T} = sum_k (beta*theta)^k Y_{T+k}^(1-sigma) after exit.

    After exit Pi = 1, S_{T+k} - 1 = theta^k (S_T - 1) and Y_{T+k} = Y S_{T+k}^(-phi/zeta)
    (Lemma 1), so each term is y0^(1-sigma) (beta*theta)^k (1 + (S_T-1) theta^k)^a with
    a = -phi (1-sigma)/zeta. For sigma = 1, a = 0 and the sum is y0^(1-sigma)/(1-beta*theta)
    in closed form. Otherwise the terms are summed until theta^k |S_T-1| < 1e-17, where the
    dispersion factor equals one to machine precision, and the remaining geometric tail is
    added in closed form. No truncation of the infinite horizon is involved.
    """
    s, f, b = p['sigma'], p['phi'], p['beta']
    zeta = s + f
    y0 = yss(p)
    a = -f * (1 - s) / zeta
    bt = b * theta
    if a == 0.0 or s_T == 1.0:
        return y0 ** (1 - s) / (1 - bt)
    dev = abs(s_T - 1)
    if not np.isfinite(dev):
        return np.nan  # a Newton trial point outside the domain; the solver rejects it
    K = 1 if dev < 1e-17 else int(np.ceil(np.log(1e-17 / dev) / np.log(theta)))
    K = min(max(K, 1), 2_000_000)
    k = np.arange(K)
    head = np.sum(bt ** k * (1 + (s_T - 1) * theta ** k) ** a)
    return y0 ** (1 - s) * (head + bt ** K / (1 - bt))


# ---------------------------------------------------------------------------
# Calvo, economy-wide labor market: Newton solver with continuation in theta
# ---------------------------------------------------------------------------
def resid_econ(u, theta, T, rbar, p=BASE, s_init=None):
    """Residuals of the reset-price conditions for t = 0, ..., T-1.

    Unknowns are u_t with Z_t = floor + exp(u_t), so the floor holds by construction.
    """
    s, f, e, b = p['sigma'], p['phi'], p['eps'], p['beta']
    zeta = s + f
    y0, mc, bs = yss(p), (e - 1) / e, 1 / (1 + rbar)
    Z = zfloor(theta, e) + np.exp(u)
    Pi = pi_from_z(Z, theta, e)
    S = np.empty(T)
    s_prev = S_INIT if s_init is None else s_init  # dispersion before the trap, S_{-1}
    for t in range(T):
        s_prev = (1 - theta) * Z[t] ** (-e) + theta * Pi[t] ** e * s_prev
        S[t] = s_prev
    # exact exit regime (Lemma 1): Pi = 1, MC_t = MC and Y_t = y0 * S_t^(-phi/zeta) for t >= T,
    # with S_T = 1 + theta (S_{T-1} - 1); V_{1,T} = MC V_{2,T} with V_{2,T} an exact infinite sum
    s_T = 1 + theta * (S[-1] - 1)
    v2 = post_exit_v2(s_T, theta, p)
    v1 = mc * v2
    y_next = y0 * s_T ** (-f / zeta)
    Pin = np.append(Pi, 1.0)
    Y, R = np.empty(T), np.empty(T)
    for t in range(T - 1, -1, -1):
        Y[t] = (bs * y_next ** (-s) / Pin[t + 1]) ** (-1 / s)
        v1 = Y[t] ** (1 - s) * Y[t] ** zeta * S[t] ** f + bs * theta * Pin[t + 1] ** e * v1
        v2 = Y[t] ** (1 - s) + bs * theta * Pin[t + 1] ** (e - 1) * v2
        R[t] = np.log(Z[t]) - np.log(e / (e - 1) * v1 / v2)
        y_next = Y[t]
    return R, Pi, np.log(Y / y0), S


def calvo_economy_path(T, rbar, p=BASE, th0=0.99, step=0.002, min_step=1e-7,
                       tol=1e-10):
    """Follow the deflationary branch from theta = th0 downward until it ends.

    Returns a list of (theta, Pi, x, S, u); the last entry is the boundary estimate.
    u holds the solver's unknowns, so a path point can warm-start a solve elsewhere.
    """
    e = p['eps']
    th = th0
    u = np.log(1 - zfloor(th, e)) * np.ones(T)  # start at Z = 1
    out = []
    with np.errstate(all='ignore'):
        while th > 1e-4:
            f = lambda v: resid_econ(v, th, T, rbar, p)[0]
            sol = root(f, u, method='hybr', options={'xtol': 1e-13})
            ok = sol.success and np.all(np.isfinite(sol.x)) and np.max(np.abs(f(sol.x))) < tol
            if not ok:
                if step > min_step:
                    th += step
                    step /= 4
                    th -= step
                    continue
                break
            u = sol.x
            _, Pi, x, S = resid_econ(u, th, T, rbar, p)
            out.append((th, Pi, x, S, u.copy()))
            th -= step
    return out


def calvo_economy_at(theta, T, rbar, u0, p=BASE, tol=1e-10):
    """Solve at one theta from the warm start u0; None if the solver fails."""
    with np.errstate(all='ignore'):
        f = lambda v: resid_econ(v, theta, T, rbar, p)[0]
        sol = root(f, u0, method='hybr', options={'xtol': 1e-13})
        if not (sol.success and np.max(np.abs(f(sol.x))) < tol):
            return None
        _, Pi, x, S = resid_econ(sol.x, theta, T, rbar, p)
    return dict(Pi=Pi, x=x, S=S)


def nearest_above(path, theta):
    """Path point with the smallest theta at or above the target (for warm starts)."""
    cands = [q for q in path if q[0] >= theta - 1e-12]
    return min(cands, key=lambda q: q[0])


def refine_boundary(T, rbar, p=BASE, target=-10.0, steps=40, **path_kw):
    """Boundary of the economy-wide deflationary branch, located where Pi_0 -> 0.

    Natural continuation in theta stalls a little before the boundary because
    Pi_0 falls very fast there. From the last continuation point we instead fix
    log Pi_0 along a sequence of increasingly negative values and solve for
    (theta, u_1, ..., u_{T-1}). theta at log Pi_0 = target is the boundary
    estimate; stopping the sequence at log Pi_0 = -3 instead of -10 changes it by less than 1e-7.
    """
    e = p['eps']
    th, Pi, _, _, u = calvo_economy_path(T, rbar, p=p, **path_kw)[-1]
    x = np.concatenate([[np.log(th / (1 - th))], u[1:]])
    best = th
    with np.errstate(all='ignore'):
        for lp in np.linspace(float(np.log(Pi[0])), target, steps):
            def G(v, lp=lp):
                th_ = 1 / (1 + np.exp(-v[0]))
                q = th_ * np.exp((e - 1) * lp)                     # theta * Pi_0^(eps-1)
                gap = zfloor(th_, e) * np.expm1(-np.log1p(-q) / (e - 1))  # Z_0 - floor, computed stably
                return resid_econ(np.concatenate([[np.log(gap)], v[1:]]), th_, T, rbar, p)[0]
            sol = root(G, x, method='hybr')
            if not (np.all(np.isfinite(sol.x)) and np.max(np.abs(G(sol.x))) < 1e-10):
                break
            x = sol.x
            best = float(1 / (1 + np.exp(-x[0])))
    return best


def calvo_economy_T1_roots(theta, rbar, p=BASE, n=200001):
    """All equilibria of a one-period trap (T = 1), found by scanning Z_0.

    Returns a list of (log Pi_0, S_0). Used for the interval check in part `checks`.
    """
    e = p['eps']
    zmin = zfloor(theta, e)
    u = np.linspace(np.log(1e-9), np.log(np.exp(3.0) - zmin), n)
    with np.errstate(all='ignore'):
        r = np.array([resid_econ(np.array([ui]), theta, 1, rbar, p)[0][0] for ui in u[::20]])
    ug = u[::20]
    roots = []
    for i in np.where(np.isfinite(r[:-1]) & np.isfinite(r[1:]) & (np.sign(r[:-1]) != np.sign(r[1:])))[0]:
        lo, hi = ug[i], ug[i + 1]
        flo = r[i]
        for _ in range(80):
            m = 0.5 * (lo + hi)
            fm = resid_econ(np.array([m]), theta, 1, rbar, p)[0][0]
            if np.sign(fm) == np.sign(flo):
                lo, flo = m, fm
            else:
                hi = m
        _, Pi, _, S = resid_econ(np.array([0.5 * (lo + hi)]), theta, 1, rbar, p)
        roots.append((float(np.log(Pi[0])), float(S[0])))
    return roots


# ---------------------------------------------------------------------------
# Rotemberg with the log-linear slope matched to economy-wide Calvo
# ---------------------------------------------------------------------------
def _bisect(F, lo, hi, iters=80):
    flo = F(lo)
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        fm = F(m)
        if np.sign(fm) == np.sign(flo):
            lo, flo = m, fm
        else:
            hi = m
    return 0.5 * (lo + hi)


def all_roots(F, g):
    """All roots of F found on the sorted grid g (possibly empty).

    Sign changes on the grid catch isolated roots. A pair of roots that falls
    between two grid points shows up as a local extremum of F that does not
    cross zero on the grid, so every such extremum is refined with a bounded
    scalar optimizer and checked for a hidden crossing.
    """
    from scipy.optimize import minimize_scalar
    with np.errstate(all='ignore'):
        v = np.array(F(g), dtype=float)
    ok = np.isfinite(v)
    roots = []
    for i in np.where(ok[:-1] & ok[1:] & (np.sign(v[:-1]) != np.sign(v[1:])))[0]:
        roots.append(_bisect(F, g[i], g[i + 1]))
    mid = ok[1:-1] & ok[:-2] & ok[2:]
    is_min = mid & (v[1:-1] <= v[:-2]) & (v[1:-1] <= v[2:]) & (v[1:-1] > 0)
    is_max = mid & (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] < 0)
    for i in np.where(is_min | is_max)[0] + 1:
        sgn = 1.0 if is_min[i - 1] else -1.0
        with np.errstate(all='ignore'):
            res = minimize_scalar(lambda z: sgn * F(z), bounds=(g[i - 1], g[i + 1]), method='bounded',
                                  options={'xatol': 1e-14})
        if sgn * res.fun < 0:  # the extremum crosses zero: two roots inside the cell
            roots.append(_bisect(F, g[i - 1], res.x))
            roots.append(_bisect(F, res.x, g[i + 1]))
    return roots


def largest_root(F, g):
    """Largest root of F on the sorted grid g, or None."""
    roots = all_roots(F, g)
    return max(roots) if roots else None


LOGK_MIN = -700.0  # smallest log slack that double precision can represent safely


def rotemberg(theta, T, rbar, p=BASE, grid_n=2001, full=False, counts=None):
    """Backward recursion for the Rotemberg model under the exact terminal regime.

    The unknown at each date is L = log k, where k = 1 - (psi/2)(Pi-1)^2 is the
    share of output left after price-adjustment costs, so C = k Y. Deflation
    Pi in (0, 1] maps one to one into L, increasing. Parameterizing by L keeps
    solutions accurate when adjustment costs absorb almost all output (k close to 0),
    which happens along long deflationary traps when psi > 2. At each date we take
    the largest root, the one connected to Pi = 1.
    Returns None if no root exists (or if the root would need L < LOGK_MIN).
    If counts is a list, the number of genuine solutions at each date (roots with |F| < 1e-8, which excludes spurious sign changes where Pi is numerically zero) is appended to it.
    """
    s, f, e, b = p['sigma'], p['phi'], p['eps'], p['beta']
    y0 = yss(p)
    psi = (e - 1) * theta / ((1 - theta) * (1 - b * theta))
    bs = 1 / (1 + rbar)
    # lower end of the domain: Pi = 0 if psi < 2, otherwise the representable limit
    lmin = np.log(1 - psi / 2) if psi < 2 else LOGK_MIN
    span = -lmin
    g = np.unique(np.concatenate([
        np.linspace(lmin, 0.0, grid_n),
        -span * np.geomspace(1e-12, 1.0, grid_n // 2),          # dense near L = 0 (mild deflation)
        lmin + span * np.geomspace(1e-14, 1e-2, grid_n // 4),   # dense near the lower end
    ]))
    g = g[(g > lmin) & (g <= 0.0)]
    pi_n, c_n, y_n = 1.0, y0, y0
    Pis, xs, shares = [], [], []
    for t in range(T - 1, -1, -1):
        C = (bs * c_n ** (-s) / pi_n) ** (-1 / s)
        fut_over_y_n = bs * (c_n / C) ** (-s) * psi * (pi_n - 1) * pi_n  # future term divided by Y_{t+1}

        def pi_of(L):
            return 1 - np.sqrt(np.maximum(2 * (1 - np.exp(L)) / psi, 0.0))

        def F(L):
            Pi = pi_of(L)
            k = np.exp(L)
            # Rotemberg condition with Y = C/k and the future term scaled by Y_{t+1}/Y_t
            return psi * (Pi - 1) * Pi - ((1 - e) + e * (C / k) ** f * C ** s + fut_over_y_n * y_n * k / C)

        if counts is not None:
            counts.append(sum(1 for r in all_roots(F, g) if abs(F(r)) < 1e-8))
        L = largest_root(F, g)
        if L is None:
            return None
        Pi = float(pi_of(L))
        Pis.append(Pi)
        xs.append(np.log(C / y0))
        shares.append(L)
        pi_n, c_n, y_n = Pi, C, C / np.exp(L)
    out = dict(Pi=np.array(Pis[::-1]), x=np.array(xs[::-1]))
    if full:
        out['log_k'] = np.array(shares[::-1])            # log of output share left after costs
        out['cost_share'] = -np.expm1(out['log_k'])      # adjustment costs / output
    return out


def rotemberg_resource_threshold(T, rbar, share=0.5, p=BASE):
    """Smallest theta at which adjustment costs stay below `share` of output at every date."""
    def ok(th, T_, rb):
        o = rotemberg(th, T_, rb, p=p, full=True)
        return None if o is None or o['cost_share'].max() >= share else o
    return threshold(ok, T, rbar)


# ---------------------------------------------------------------------------
# Log-linear benchmark
# ---------------------------------------------------------------------------
def loglin(theta, T, rbar, labor='economy', p=BASE):
    s, f, e, b = p['sigma'], p['phi'], p['eps'], p['beta']
    k = (1 - theta) * (1 - b * theta) / theta * (s + f)
    if labor == 'industry':
        k /= 1 + e * f
    r = np.log(1 + rbar)
    x = pi = 0.0
    X, Pp = [], []
    for _ in range(T):
        x = x + (pi + r) / s
        pi = b * pi + k * x
        X.append(x)
        Pp.append(pi)
    return dict(Pi=np.exp(np.array(Pp[::-1])), x=np.array(X[::-1]))


def threshold(fun, T, rbar, lo=1e-5, hi=0.995, iters=50, **kw):
    """Bisection for the smallest theta with a solution (assumes an interval)."""
    if fun(hi, T, rbar, **kw) is None:
        return float('nan')
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if fun(m, T, rbar, **kw) is None:
            lo = m
        else:
            hi = m
    return hi


def first_order(rbar, p=BASE):
    """First-order threshold formulas of Propositions 1, 2 and 4 (last trap period)."""
    s, f, e = p['sigma'], p['phi'], p['eps']
    zeta = s + f
    xbar = np.log(1 + rbar) / s
    return dict(economy=(e - 1) * zeta * abs(xbar) / (1 + f),
                industry=(e - 1) * zeta * abs(xbar) / (1 + e * f),
                rotemberg=2 * zeta * abs(xbar) / ((e - 1) * f))


def provable_bound(rbar, p=BASE, theta_max=0.5, n=500001):
    """Explicit lower bound of Proposition 3 (economy-wide labor, sigma = 1).

    Below the returned theta, eta(theta) (1 - mbar(theta)) >= 1 - (1-theta)^(1/(eps-1))
    holds, so no equilibrium with Pi_{T-1} <= 1 exists. eta(theta) is in closed
    form only for sigma = 1, and mbar uses the dispersion bound B(theta) of
    Lemma 3. The bound concerns the last trap period, so it holds for every T.
    """
    s, f, e, b = p['sigma'], p['phi'], p['eps'], p['beta']
    assert s == 1.0, 'closed form of eta(theta) requires sigma = 1'
    zeta, xbar, bT1 = s + f, np.log(1 + rbar) / s, 1 / (1 + rbar)

    def G(th):
        eta = 1 / (1 + bT1 * th / (1 - b * th))
        B = (1 - th) ** (-e / (e - 1))
        mbar = np.exp(zeta * xbar) * (B / (1 - th + th * B)) ** f
        return eta * (1 - mbar) - (1 - zfloor(th, e))

    g = np.linspace(1e-9, theta_max, n)
    v = G(g)
    i = int(np.argmax(v < 0))
    assert i > 0 and np.all(v[:i] >= 0), 'sufficient condition must hold on (0, bound)'
    return brentq(G, g[i - 1], g[i], xtol=1e-14)


# ---------------------------------------------------------------------------
# Tables and checks
# ---------------------------------------------------------------------------
TS, RS = [1, 2, 4, 8], [-0.0025, -0.005, -0.01]


def t1():
    """Table 1: existence thresholds."""
    out = {}
    for T in TS:
        for r in RS:
            out[f'T{T}_r{r}'] = dict(economy=refine_boundary(T, r),
                                     industry=threshold(calvo_industry, T, r),
                                     rotemberg=threshold(rotemberg, T, r),
                                     rotemberg_cost50=rotemberg_resource_threshold(T, r))
            v = out[f'T{T}_r{r}']
            print(f"T={T} rbar={100*r:.2f}%  econ {v['economy']:.4f}  ind {v['industry']:.4f}  "
                  f"rot {v['rotemberg']:.4f}  rot(costs<50%) {v['rotemberg_cost50']:.4f}", flush=True)
    return out


def t2():
    """Table 2: first-period outcomes, T = 8, rbar = -0.5%."""
    path = calvo_economy_path(8, -0.005)
    g = lambda o: None if o is None else [100 * float(np.log(o['Pi'][0])), 100 * float(o['x'][0])]
    out = {}
    for th in [0.9, 0.85, 0.8, 0.75, 0.7]:
        oe = calvo_economy_at(th, 8, -0.005, nearest_above(path, th)[4])
        out[str(th)] = dict(loglin_e=g(loglin(th, 8, -0.005)),
                            economy=g(oe), S0=float(oe['S'][0]),
                            rotemberg=g(rotemberg(th, 8, -0.005)),
                            loglin_i=g(loglin(th, 8, -0.005, 'industry')),
                            industry=g(calvo_industry(th, 8, -0.005)))
        print(th, out[str(th)], flush=True)
    return out


def t3():
    """Table 3: approach to the existence boundary, T = 8, rbar = -0.5%, with log-linear inflation at the same theta."""
    path = calvo_economy_path(8, -0.005)
    end = refine_boundary(8, -0.005)
    th_i, th_r = threshold(calvo_industry, 8, -0.005), threshold(rotemberg, 8, -0.005)
    out = dict(theta_star=dict(economy=end, industry=th_i, rotemberg=th_r), rows={})
    for dd in [0.02, 1e-3, 1e-5]:
        oe = calvo_economy_at(end + dd, 8, -0.005, nearest_above(path, end + dd)[4])
        oi, orr = calvo_industry(th_i + dd, 8, -0.005), rotemberg(th_r + dd, 8, -0.005, full=True)
        out['rows'][str(dd)] = dict(economy=[100 * float(np.log(oe['Pi'][0])), 100 * float(oe['x'][0])],
                                    industry=[100 * float(np.log(oi['Pi'][0])), 100 * float(oi['x'][0])],
                                    rotemberg=[100 * float(np.log(orr['Pi'][0])), 100 * float(orr['x'][0]),
                                               float(orr['log_k'].min())],
                                    loglin_e=100 * float(np.log(loglin(end + dd, 8, -0.005)['Pi'][0])),
                                    loglin_i=100 * float(np.log(loglin(th_i + dd, 8, -0.005, 'industry')['Pi'][0])))
        print(dd, out['rows'][str(dd)], flush=True)
    return out


def t4():
    """Table 4: time aggregation, T = 8 quarters, rbar = -0.5% per quarter."""
    out = {}
    for n in [1, 3, 6, 13]:
        p = dict(BASE, beta=0.99 ** (1 / n))
        r = (1 - 0.005) ** (1 / n) - 1
        ti = threshold(lambda th, T, rb: calvo_industry(th, T, rb, p=p), 8 * n, r)
        tr = threshold(lambda th, T, rb: rotemberg(th, T, rb, p=p), 8 * n, r)
        te = refine_boundary(8 * n, r, p=p, th0=0.999, step=0.002 / n) if n <= 6 else float('nan')
        tc = rotemberg_resource_threshold(8 * n, r, p=p)
        out[str(n)] = dict(industry=ti ** n, rotemberg=tr ** n, rotemberg_cost50=tc ** n, economy=te ** n)
        print(n, out[str(n)], flush=True)
    return out


def t5():
    """Table 5: first-order formulas against exact thresholds, T = 1."""
    out = {}
    for r in RS:
        fo = first_order(r)
        ex = dict(economy=refine_boundary(1, r),
                  industry=threshold(calvo_industry, 1, r),
                  rotemberg=threshold(rotemberg, 1, r))
        out[str(r)] = dict(first_order=fo, exact=ex)
        print(r, out[str(r)], flush=True)
    return out


def t6():
    """Table 6: sensitivity of T = 8 thresholds (rbar = -0.5%) to calibration."""
    cases = {'baseline': {}, 'eps=4': dict(eps=4.0), 'eps=11': dict(eps=11.0),
             'phi=0.5': dict(phi=0.5), 'phi=2': dict(phi=2.0), 'sigma=2': dict(sigma=2.0)}
    out = {}
    for name, ch in cases.items():
        p = dict(BASE, **ch)
        out[name] = dict(economy=refine_boundary(8, -0.005, p=p),
                         industry=threshold(calvo_industry, 8, -0.005, p=p),
                         rotemberg=threshold(rotemberg, 8, -0.005, p=p),
                         rotemberg_cost50=rotemberg_resource_threshold(8, -0.005, p=p))
        print(name, out[name], flush=True)
    return out


def bound():
    """Numerical illustration of Proposition 3 (not tabulated in the paper): the explicit
    lower bound, its first-order approximation and the computed theta_C*, T = 1."""
    out = {}
    for r in RS:
        lb, tc = provable_bound(r), refine_boundary(1, r)
        fo = (BASE['eps'] - 1) * (BASE['sigma'] + BASE['phi']) * abs(np.log(1 + r)) / (1 + BASE['eps'] * BASE['phi'])
        out[str(r)] = dict(provable_bound=lb, first_order=fo, theta_C=tc, ratio=lb / tc)
        print(r, out[str(r)], flush=True)
    return out


def checks():
    """Computational validation reported in Appendix C."""
    out = {}
    # (a) existence is an interval in theta: scan a grid and compare with the bisection threshold
    grid = np.round(np.arange(0.005, 0.996, 0.005), 3)
    for name, fun in [('industry', calvo_industry), ('rotemberg', rotemberg)]:
        for T in [1, 8]:
            exists = [fun(th, T, -0.005) is not None for th in grid]
            first = grid[exists.index(True)] if any(exists) else None
            gaps = any(exists[i] and not exists[i + 1] for i in range(len(exists) - 1))
            out[f'interval_{name}_T{T}'] = dict(first_theta_on_grid=float(first), gap_found=bool(gaps))
    # economy-wide, T = 1: deflationary root (log Pi <= 0) exists on the grid iff theta > theta*
    th_star = refine_boundary(1, -0.005)
    ex = [any(lp <= 0 for lp, _ in calvo_economy_T1_roots(th, -0.005, n=40001))
          for th in np.round(np.arange(0.005, 0.2, 0.005), 3)]
    out['interval_economy_T1'] = dict(theta_star=th_star, exists_on_grid=ex)
    # (a') Rotemberg: largest number of genuine solutions at any date, over all trap lengths and natural rates of Table 1 and the theta grid above
    nmax = {}
    for T in [1, 2, 4, 8]:
        for rb in [-0.0025, -0.005, -0.01]:
            c = []
            for th in grid:
                rotemberg(th, T, rb, counts=c)
            nmax[f'T{T}_rbar{rb}'] = max(c)
    out['rotemberg_max_solutions_per_date'] = nmax
    # (b) Rotemberg grid resolution
    out['rotemberg_grid'] = {str(g): threshold(rotemberg, 8, -0.005, grid_n=g) for g in [1001, 2001, 8001]}
    # (c) economy-wide solver: continuation step and residual tolerance
    out['economy_solver'] = {
        'continuation end, baseline': calvo_economy_path(8, -0.005)[-1][0],
        'continuation end, step=0.001': calvo_economy_path(8, -0.005, step=0.001)[-1][0],
        'continuation end, tol=1e-12': calvo_economy_path(8, -0.005, tol=1e-12)[-1][0],
        'refined boundary': refine_boundary(8, -0.005),
    }
    # (d) initial dispersion S_{-1}: inherited dispersion enters S_0 through theta*Pi_0^eps*S_{-1},
    # which vanishes at the boundary, so the economy-wide thresholds should not depend on it
    global S_INIT
    out['initial_dispersion'] = {}
    for s0 in [1.0, 1.0005, 1.005]:
        S_INIT = s0
        out['initial_dispersion'][str(s0)] = dict(T1=refine_boundary(1, -0.005), T8=refine_boundary(8, -0.005))
    S_INIT = 1.0
    print(json.dumps(out, indent=1), flush=True)
    return out


PARTS = dict(t1=t1, t2=t2, t3=t3, t4=t4, t5=t5, t6=t6, bound=bound, checks=checks)

if __name__ == '__main__':
    parts = sys.argv[1:] or list(PARTS)
    try:
        results = json.load(open('results.json'))
    except (OSError, ValueError):
        results = {}
    for part in parts:
        t0 = time.time()
        print(f'--- {part}: {PARTS[part].__doc__.strip().splitlines()[0]}', flush=True)
        results[part] = PARTS[part]()
        print(f'    ({time.time() - t0:.0f} s)', flush=True)
        json.dump(results, open('results.json', 'w'), indent=1, default=float)
