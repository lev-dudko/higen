"""Statistical uncertainty of the weighted ROC AUC and a train/test gap test.

Companion to the weighted-loss early-stopping criterion: the AUC is a two-sample
U-statistic (Mann-Whitney), so its sampling variance is *not* the variance of a
per-event mean.  We use the DeLong decomposition generalised to event weights:
the AUC is the weighted mean of per-event *placement values*, and its variance
splits into a signal-sample term plus a background-sample term.  Each term reuses
the same unbiased "error of a weighted mean" estimator as the loss criterion
(unbiased weighted variance / Kish effective N).

Conventions match `train.py::_eval_auc`:
    y_true  -- int array, 1 = signal (positive), 0 = background (negative)
    y_score -- float array, classifier score (logit; monotone is all that matters)
    weight  -- float array, per-event weight (assumed >= 0)

The train and test sets are disjoint event samples, so their AUC estimates are
statistically independent -> the gap uncertainty is the geometric sum.

Reference: DeLong, DeLong & Clarke-Pearson, Biometrics 44 (1988) 837;
weighted generalisation as used e.g. in survey-weighted ROC analyses.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "weighted_auc_var",
    "auc_train_test_gap",
    "bootstrap_auc_std",
]


def _wsplit(query: np.ndarray, ref_scores: np.ndarray, ref_weights: np.ndarray):
    """For every q in `query`, split the weighted reference mass by score.

    Returns (w_less, w_equal, w_greater, total) where
        w_less[k]    = sum of ref weights with ref_score <  query[k]
        w_equal[k]   = sum of ref weights with ref_score == query[k]
        w_greater[k] = sum of ref weights with ref_score >  query[k]
    Ties are handled exactly (no random jitter), cost O((|ref|+|query|) log|ref|).
    """
    order = np.argsort(ref_scores, kind="mergesort")
    rs = ref_scores[order]
    rw = ref_weights[order]
    cum = np.concatenate(([0.0], np.cumsum(rw)))   # cum[k] = sum of first k weights
    total = cum[-1]
    lo = np.searchsorted(rs, query, side="left")
    hi = np.searchsorted(rs, query, side="right")
    w_less = cum[lo]
    w_equal = cum[hi] - cum[lo]
    w_greater = total - cum[hi]
    return w_less, w_equal, w_greater, total


def _var_of_weighted_mean(vals: np.ndarray, w: np.ndarray):
    """Unbiased estimate of Var(weighted mean) -- same estimator as the loss case.

    s^2     = sum w (v - vbar)^2 / (W - sum w^2 / W)      (unbiased per-item var)
    N_eff   = W^2 / sum w^2                                (Kish effective sample)
    Var     = s^2 / N_eff
    """
    W = w.sum()
    sw2 = (w * w).sum()
    mean = (w * vals).sum() / W
    denom = W - sw2 / W
    if denom <= 0:
        return float("nan"), float(mean), float("nan")
    s2 = (w * (vals - mean) ** 2).sum() / denom
    n_eff = W * W / sw2
    return float(s2 / n_eff), float(mean), float(n_eff)


def weighted_auc_var(y_true, y_score, weight):
    """Weighted ROC AUC and its (weighted DeLong) sampling variance.

    Returns dict with keys:
        auc, var, sigma, n_eff_pos, n_eff_neg, w_pos, w_neg
    `auc` matches sklearn.metrics.roc_auc_score(..., sample_weight=weight).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    w = np.asarray(weight, dtype=float)

    pos = y == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        raise ValueError("need both classes present to define AUC")

    Xp, wp = s[pos], w[pos]      # signal scores / weights
    Yn, wn = s[neg], w[neg]      # background scores / weights
    Wp, Wn = wp.sum(), wn.sum()

    # placement of each SIGNAL among BACKGROUND: fraction of bkg it beats
    less_p, eq_p, _gr_p, _ = _wsplit(Xp, Yn, wn)
    V10 = (less_p + 0.5 * eq_p) / Wn

    # placement of each BACKGROUND among SIGNAL: fraction of signal that beats it
    _ls_n, eq_n, gr_n, _ = _wsplit(Yn, Xp, wp)
    V01 = (gr_n + 0.5 * eq_n) / Wp

    var_pos, auc_p, neff_p = _var_of_weighted_mean(V10, wp)
    var_neg, auc_n, neff_n = _var_of_weighted_mean(V01, wn)

    auc = 0.5 * (auc_p + auc_n)          # identical up to rounding; average for safety
    var = var_pos + var_neg
    return {
        "auc": auc,
        "var": var,
        "sigma": float(np.sqrt(var)) if var >= 0 else float("nan"),
        "n_eff_pos": neff_p,
        "n_eff_neg": neff_n,
        "w_pos": float(Wp),
        "w_neg": float(Wn),
    }


def _logit(a: float) -> float:
    a = min(max(a, 1e-12), 1.0 - 1e-12)
    return np.log(a / (1.0 - a))


def auc_train_test_gap(train, test, k: float = 2.0, transform: str = "none"):
    """Overfitting test on the train/test AUC gap, weighted.

    train, test : tuples (y_true, y_score, weight) for each split.
    k           : threshold in units of the combined sigma (2 or 3 typically).
    transform   : "none" compares AUC directly; "logit" compares on the
                  logit-AUC scale (delta method) -- recommended when AUC -> 1,
                  where the AUC distribution is skewed and Gaussian K-sigma
                  coverage on the raw scale is unreliable.

    Returns dict with auc_train/auc_test, sigmas, signed gap, combined sigma,
    z = gap / sigma_gap, and `overfit` = (z > k).  The gap is signed
    (train - test): genuine overfitting makes train AUC the larger one, so only a
    positive excess should trigger a stop.
    """
    rt = weighted_auc_var(*train)
    re = weighted_auc_var(*test)

    if transform == "logit":
        a_tr, a_te = rt["auc"], re["auc"]
        g_tr, g_te = _logit(a_tr), _logit(a_te)
        # delta method: Var(logit A) = Var(A) / (A (1-A))^2
        s_tr = np.sqrt(rt["var"]) / (a_tr * (1 - a_tr))
        s_te = np.sqrt(re["var"]) / (a_te * (1 - a_te))
        gap = g_tr - g_te
        sigma_gap = float(np.sqrt(s_tr ** 2 + s_te ** 2))
        sigma_train, sigma_test = float(s_tr), float(s_te)
    else:
        gap = rt["auc"] - re["auc"]
        sigma_train, sigma_test = rt["sigma"], re["sigma"]
        sigma_gap = float(np.sqrt(rt["var"] + re["var"]))

    z = gap / sigma_gap if sigma_gap > 0 else float("inf")
    return {
        "auc_train": rt["auc"],
        "auc_test": re["auc"],
        "sigma_train": sigma_train,
        "sigma_test": sigma_test,
        "gap": float(gap),
        "sigma_gap": sigma_gap,
        "z": float(z),
        "k": k,
        "overfit": bool(z > k),
        "transform": transform,
        "n_eff_pos_train": rt["n_eff_pos"], "n_eff_neg_train": rt["n_eff_neg"],
        "n_eff_pos_test": re["n_eff_pos"], "n_eff_neg_test": re["n_eff_neg"],
    }


def bootstrap_auc_std(y_true, y_score, weight, n_boot: int = 500, seed: int = 0):
    """Stratified weighted-bootstrap std of the AUC -- cross-check for the analytic
    sigma and a fallback when weights are heavy-tailed / AUC is non-Gaussian.

    Resamples signal and background indices independently with replacement
    (weights carried along), recomputing the weighted AUC each time.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    w = np.asarray(weight, dtype=float)
    ipos = np.where(y == 1)[0]
    ineg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        bp = rng.choice(ipos, size=ipos.size, replace=True)
        bn = rng.choice(ineg, size=ineg.size, replace=True)
        idx = np.concatenate([bp, bn])
        aucs[b] = weighted_auc_var(y[idx], s[idx], w[idx])["auc"]
    return float(aucs.std(ddof=1))


# --------------------------------------------------------------------------- #
#  Self-test: verify (1) AUC == sklearn, (2) analytic sigma ~ bootstrap sigma  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 4000
    y = (rng.random(n) < 0.5).astype(int)
    # separable-ish scores + heavy-tailed weights (log-normal) to stress N_eff
    score = rng.normal(loc=y * 0.8, scale=1.0)
    wt = rng.lognormal(mean=0.0, sigma=1.0, size=n)

    res = weighted_auc_var(y, score, wt)
    print(f"weighted AUC      = {res['auc']:.6f}")
    print(f"analytic sigma    = {res['sigma']:.6f}")
    print(f"N_eff pos/neg     = {res['n_eff_pos']:.1f} / {res['n_eff_neg']:.1f}  "
          f"(raw {int((y==1).sum())} / {int((y==0).sum())})")

    try:
        from sklearn.metrics import roc_auc_score
        sk = roc_auc_score(y, score, sample_weight=wt)
        print(f"sklearn AUC       = {sk:.6f}   (delta {abs(sk-res['auc']):.2e})")
    except Exception as e:  # noqa: BLE001
        print(f"sklearn check skipped: {e}")

    boot = bootstrap_auc_std(y, score, wt, n_boot=400, seed=7)
    print(f"bootstrap sigma   = {boot:.6f}   (ratio analytic/boot = {res['sigma']/boot:.3f})")
