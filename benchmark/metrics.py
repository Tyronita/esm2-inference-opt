"""
All ProteinGym-relevant metrics for variant effect prediction.

Primary:
  spearman_rho    Rank correlation — standard ProteinGym metric
  pearson_r       Linear correlation
  kendall_tau     Non-parametric rank correlation

Ranking:
  ndcg_10         NDCG@10% — top-decile precision
  ndcg_50         NDCG@50%
  top5_recall     Fraction of top-5% experimental variants in model's top-5%
  top10_recall    Fraction of top-10% experimental variants in model's top-10%

Binary (where DMS_score_bin available):
  auc             Area under ROC curve
  mcc             Matthews correlation coefficient

Agreement:
  fraction_correct  Fraction of variants where sign(score) == sign(fitness)
"""

import math

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def compute_all(
    scores: list[float],
    fitness: list[float],
    fitness_bin: list[int] | None = None,
) -> dict:
    s = np.array(scores, dtype=float)
    f = np.array(fitness, dtype=float)

    # Drop NaN
    valid = np.isfinite(s) & np.isfinite(f)
    s, f = s[valid], f[valid]
    n = len(s)

    if n < 4:
        return {"n_valid": n, "error": "too few valid variants"}

    results: dict = {"n_valid": n}

    # ── Correlation ──────────────────────────────────────────────────────────
    results["spearman_rho"], results["spearman_p"] = spearmanr(s, f)
    results["pearson_r"],    results["pearson_p"]  = pearsonr(s, f)
    results["kendall_tau"],  results["kendall_p"]  = kendalltau(s, f)

    # ── NDCG ─────────────────────────────────────────────────────────────────
    # Shift fitness to [0, inf) for NDCG (requires non-negative relevance)
    f_shifted = f - f.min()
    results["ndcg_10"] = _ndcg(s, f_shifted, k=max(1, int(n * 0.10)))
    results["ndcg_50"] = _ndcg(s, f_shifted, k=max(1, int(n * 0.50)))

    # ── Top-K recall ──────────────────────────────────────────────────────────
    results["top5_recall"]  = _topk_recall(s, f, k=max(1, int(n * 0.05)))
    results["top10_recall"] = _topk_recall(s, f, k=max(1, int(n * 0.10)))

    # ── Binary metrics ────────────────────────────────────────────────────────
    if fitness_bin is not None:
        fb = np.array(fitness_bin, dtype=int)[valid]
        if len(np.unique(fb)) == 2:
            from sklearn.metrics import matthews_corrcoef, roc_auc_score
            results["auc"] = roc_auc_score(fb, s)
            results["mcc"] = matthews_corrcoef(fb, (s > np.median(s)).astype(int))
        else:
            results["auc"] = float("nan")
            results["mcc"] = float("nan")

    # ── Agreement ─────────────────────────────────────────────────────────────
    f_median = np.median(f)
    results["fraction_correct"] = float(np.mean(np.sign(s - np.median(s)) == np.sign(f - f_median)))

    return results


def _ndcg(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """NDCG@k where relevance is non-negative fitness."""
    order = np.argsort(scores)[::-1][:k]
    ideal_order = np.argsort(relevance)[::-1][:k]

    def dcg(idxs):
        return sum(relevance[i] / math.log2(r + 2) for r, i in enumerate(idxs))

    dcg_val = dcg(order)
    idcg_val = dcg(ideal_order)
    return float(dcg_val / idcg_val) if idcg_val > 0 else 0.0


def _topk_recall(scores: np.ndarray, fitness: np.ndarray, k: int) -> float:
    """Fraction of true top-k variants that appear in predicted top-k."""
    true_top  = set(np.argsort(fitness)[::-1][:k])
    pred_top  = set(np.argsort(scores)[::-1][:k])
    return len(true_top & pred_top) / k


METRIC_LABELS = {
    "spearman_rho":    "Spearman ρ",
    "pearson_r":       "Pearson r",
    "kendall_tau":     "Kendall τ",
    "ndcg_10":         "NDCG@10%",
    "ndcg_50":         "NDCG@50%",
    "top5_recall":     "Top-5% recall",
    "top10_recall":    "Top-10% recall",
    "auc":             "AUC-ROC",
    "mcc":             "MCC",
    "fraction_correct":"Fraction correct",
}
