"""
Evaluation — Walk-Forward Validation & Calibration
------------------------------------------------------
This is where "quantify your uncertainty" becomes concrete. We evaluate
on log loss and Brier score (calibration-sensitive metrics), not accuracy
alone — a model that says "60% home win" should be right about 60% of the
time across all such predictions, not just "mostly right."
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: list[str] = ["home_win", "draw", "away_win"],
) -> dict:
    """
    Compute log loss and per-class Brier score for a set of predictions.

    Args:
        y_true: (n,) integer array, true class index for each match
        y_pred_proba: (n, 3) array, predicted probability for each class
        class_names: names for the 3 outcome classes

    Returns:
        dict with overall log_loss, per-class brier scores, and accuracy
    """
    n_classes = y_pred_proba.shape[1]

    ll = log_loss(y_true, y_pred_proba, labels=list(range(n_classes)))

    brier_scores = {}
    for i, name in enumerate(class_names):
        binary_true = (y_true == i).astype(int)
        brier_scores[name] = brier_score_loss(binary_true, y_pred_proba[:, i])

    predicted_class = np.argmax(y_pred_proba, axis=1)
    accuracy = (predicted_class == y_true).mean()

    return {
        "log_loss": ll,
        "brier_scores": brier_scores,
        "mean_brier": np.mean(list(brier_scores.values())),
        "accuracy": accuracy,
        "n_matches": len(y_true),
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_idx: int,
    class_name: str,
    n_bins: int = 10,
    ax=None,
):
    """
    Plot a calibration curve: for predictions binned by confidence,
    does the actual outcome rate match the predicted probability?

    A well-calibrated model's points lie on the diagonal. Points above
    the diagonal mean the model is UNDERCONFIDENT for that bin; below
    means OVERCONFIDENT.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    binary_true = (y_true == class_idx).astype(int)
    probs = y_pred_proba[:, class_idx]

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_true_rates = []
    bin_pred_means = []
    bin_counts = []

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_true_rates.append(binary_true[mask].mean())
            bin_pred_means.append(probs[mask].mean())
            bin_counts.append(mask.sum())

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
    sizes = [max(20, min(200, c * 3)) for c in bin_counts]
    ax.scatter(bin_pred_means, bin_true_rates, s=sizes, alpha=0.7, label=class_name)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Actual outcome rate')
    ax.set_title(f'Calibration: {class_name}')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def walk_forward_summary(results_by_tournament: dict[int, dict]) -> None:
    """
    Print a clean summary table of walk-forward validation results
    across multiple held-out tournaments.

    Args:
        results_by_tournament: {year: evaluate_predictions() output dict}
    """
    print(f"\n{'='*70}")
    print(f"{'Tournament':<12}{'N Matches':<12}{'Log Loss':<12}{'Mean Brier':<12}{'Accuracy':<10}")
    print(f"{'-'*70}")
    for year, r in sorted(results_by_tournament.items()):
        print(
            f"{year:<12}{r['n_matches']:<12}{r['log_loss']:<12.4f}"
            f"{r['mean_brier']:<12.4f}{r['accuracy']:<10.3f}"
        )
    print(f"{'='*70}")

    avg_log_loss = np.mean([r["log_loss"] for r in results_by_tournament.values()])
    avg_brier = np.mean([r["mean_brier"] for r in results_by_tournament.values()])
    avg_acc = np.mean([r["accuracy"] for r in results_by_tournament.values()])
    print(f"\nAverage across tournaments: log_loss={avg_log_loss:.4f}, "
          f"mean_brier={avg_brier:.4f}, accuracy={avg_acc:.3f}")

    # Compare against a naive baseline: always predict the historical
    # base rates (home_win ~46%, draw ~25%, away_win ~29% is typical
    # for international football)
    print(f"\nNaive baseline (constant historical base rates) would give:")
    print(f"  log_loss ≈ 1.02-1.05 (typical for 3-class with these base rates)")
    print(f"  Beat this baseline meaningfully -> model has learned real signal.")


def baseline_constant_prediction(y_true: np.ndarray) -> dict:
    """
    Compute the naive baseline: predict the empirical class distribution
    for every single match (no team-specific information at all).
    This is the bar the model MUST clear to be worth anything.
    """
    n_classes = 3
    class_rates = np.array([(y_true == i).mean() for i in range(n_classes)])
    y_pred_proba = np.tile(class_rates, (len(y_true), 1))
    return evaluate_predictions(y_true, y_pred_proba)


if __name__ == "__main__":
    # Smoke test with synthetic data
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1, 2], size=n, p=[0.46, 0.25, 0.29])

    # Simulate a "decent" model: correct class gets higher prob on average
    y_pred_proba = np.random.dirichlet([2, 1, 1.5], size=n)
    for i in range(n):
        # Bias the prediction toward the true class somewhat
        y_pred_proba[i, y_true[i]] += 0.3
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

    results = evaluate_predictions(y_true, y_pred_proba)
    print("Simulated model evaluation:")
    print(results)

    baseline = baseline_constant_prediction(y_true)
    print("\nNaive baseline evaluation:")
    print(baseline)

    print(f"\nModel log_loss ({results['log_loss']:.4f}) vs "
          f"baseline log_loss ({baseline['log_loss']:.4f})")
    if results['log_loss'] < baseline['log_loss']:
        print("Model beats naive baseline.")
    else:
        print("Model does NOT beat naive baseline — needs work.")
