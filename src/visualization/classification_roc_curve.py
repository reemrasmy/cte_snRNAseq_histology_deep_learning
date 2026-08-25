#!/usr/bin/env python3

"""
ROC utilities for binary Low-vs-High CTE classification.

Expected prediction columns:
    true_label: 0 = Low CTE, 1 = High CTE
    prob_high: predicted probability of High CTE

The ROC curve is calculated from pooled out-of-fold predictions,
with High CTE (class 1) treated as the positive class.

This module is called automatically by the ABMIL stage-classification
script after cross-validation predictions are generated.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def validate_roc_predictions(predictions_df: pd.DataFrame) -> None:
    """Check that held-out predictions can be used for ROC analysis."""

    required = {"true_label", "prob_high"}
    missing = required.difference(predictions_df.columns)

    if missing:
        raise ValueError(f"Prediction dataframe is missing columns: {sorted(missing)}")

    if predictions_df.empty:
        raise ValueError("Prediction dataframe contains no rows.")

    if predictions_df["true_label"].isna().any():
        raise ValueError("true_label contains missing values.")

    if predictions_df["prob_high"].isna().any():
        raise ValueError("prob_high contains missing values.")

    invalid_labels = set(predictions_df["true_label"].astype(int).unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            f"true_label must contain only 0 and 1. Found: {sorted(invalid_labels)}"
        )

    if predictions_df["true_label"].nunique() != 2:
        raise ValueError("ROC-AUC requires both Low and High samples.")

    if not predictions_df["prob_high"].between(0, 1).all():
        raise ValueError("prob_high contains values outside [0, 1].")


def calculate_roc(predictions_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Calculate pooled out-of-fold ROC coordinates and ROC-AUC."""

    validate_roc_predictions(predictions_df)

    y_true = predictions_df["true_label"].to_numpy()
    y_score = predictions_df["prob_high"].to_numpy()

    pooled_auc = roc_auc_score(y_true, y_score)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true,
        y_score,
        pos_label=1,
    )

    roc_points_df = pd.DataFrame({
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "threshold": thresholds,
    })

    return pooled_auc, roc_points_df


def calculate_fold_auc(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ROC-AUC separately for each held-out fold."""

    if "fold" not in predictions_df.columns:
        print("No fold column found. Skipping fold-level ROC-AUC.")
        return pd.DataFrame()

    fold_auc_rows = []

    for fold_number, fold_df in predictions_df.groupby("fold"):
        if fold_df["true_label"].nunique() != 2:
            fold_auc = float("nan")
            print(
                f"Warning: fold {fold_number} contains only one class. "
                "Fold AUC is undefined."
            )
        else:
            fold_auc = roc_auc_score(
                fold_df["true_label"],
                fold_df["prob_high"],
            )

        fold_auc_rows.append({
            "fold": fold_number,
            "n_samples": len(fold_df),
            "n_low": int((fold_df["true_label"] == 0).sum()),
            "n_high": int((fold_df["true_label"] == 1).sum()),
            "roc_auc": fold_auc,
        })

    return pd.DataFrame(fold_auc_rows)


def plot_roc_curve(
    pooled_auc: float,
    roc_points_df: pd.DataFrame,
    output_path: Path,
    stain: str,
) -> None:
    """Save the pooled out-of-fold ROC curve."""

    plt.figure(figsize=(6, 6))

    plt.plot(
        roc_points_df["false_positive_rate"],
        roc_points_df["true_positive_rate"],
        linewidth=2,
        label=f"{stain} ABMIL (AUC = {pooled_auc:.3f})",
    )

    # Random-classifier reference line.
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Chance (AUC = 0.500)",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"{stain} ABMIL: Low vs High CTE\n"
        "Pooled Out-of-Fold ROC Curve"
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_roc_results(
    pooled_auc: float,
    roc_points_df: pd.DataFrame,
    fold_auc_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save ROC coordinates and pooled/fold-level AUC results."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roc_points_df.to_csv(
        output_dir / "roc_curve_points.csv",
        index=False,
    )

    valid_fold_auc = pd.Series(dtype=float)

    if not fold_auc_df.empty:
        fold_auc_df.to_csv(
            output_dir / "fold_roc_auc.csv",
            index=False,
        )
        valid_fold_auc = fold_auc_df["roc_auc"].dropna()

    summary_rows = [{
        "metric": "pooled_out_of_fold_roc_auc",
        "value": pooled_auc,
    }]

    if len(valid_fold_auc) > 0:
        summary_rows.extend([
            {
                "metric": "mean_fold_roc_auc",
                "value": valid_fold_auc.mean(),
            },
            {
                "metric": "std_fold_roc_auc",
                "value": valid_fold_auc.std(),
            },
        ])

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "roc_auc_summary.csv",
        index=False,
    )


def save_roc_outputs(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    stain: str,
) -> float:
    """
    Calculate and save all ROC outputs from pooled OOF predictions.

    Returns the pooled ROC-AUC.
    """
    # Pooled OOF includes the prediction cases from all 5 cross-validation folds into one set to calcuate 
    # one overall ROC curve and AUC from them. Takes into account all donors.  
    pooled_auc, roc_points_df = calculate_roc(predictions_df)
    fold_auc_df = calculate_fold_auc(predictions_df)

    plot_roc_curve(
        pooled_auc,
        roc_points_df,
        Path(output_dir) / "roc_curve.png",
        stain,
    )

    save_roc_results(
        pooled_auc,
        roc_points_df,
        fold_auc_df,
        output_dir,
    )

    print(f"\nPooled out-of-fold ROC-AUC: {pooled_auc:.4f}")

    if not fold_auc_df.empty:
        print("\nFold-level ROC-AUC:")
        print(fold_auc_df.to_string(index=False))

        valid_fold_auc = fold_auc_df["roc_auc"].dropna()
        if len(valid_fold_auc) > 0:
            print(
                f"\nMean fold ROC-AUC: "
                f"{valid_fold_auc.mean():.4f} ± "
                f"{valid_fold_auc.std():.4f}"
            )

    return pooled_auc