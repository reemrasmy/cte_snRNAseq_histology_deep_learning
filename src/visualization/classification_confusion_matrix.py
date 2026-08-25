#!/usr/bin/env python3

"""
Create the pooled out-of-fold confusion matrix and classification report
for binary Low-vs-High CTE stage classification.

Expected columns:
    true_label
    pred_label

Class mapping:
    0 = Low CTE
    1 = High CTE
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def save_confusion_matrix(
    predictions_df: pd.DataFrame,
    output_path: Path,
    class_names: list[str] | None = None,
) -> None:
    """Save pooled out-of-fold confusion matrix."""

    if class_names is None:
        class_names = ["Low", "High"]

    required = {"true_label", "pred_label"}
    missing = required.difference(predictions_df.columns)

    if missing:
        raise ValueError(
            f"Prediction dataframe is missing required columns: {sorted(missing)}"
        )

    cm = confusion_matrix(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(values_format="d")
    plt.title("ABMIL Stage Classification Confusion Matrix")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.close()


def save_classification_report(
    predictions_df: pd.DataFrame,
    output_path: Path,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """Save pooled out-of-fold classification report."""

    if class_names is None:
        class_names = ["Low", "High"]

    report = classification_report(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path)

    return report_df


def save_classification_summary(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    class_names: list[str] | None = None,
) -> None:
    """Save the same pooled classification outputs used in the original model."""

    if class_names is None:
        class_names = ["Low", "High"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(
        predictions_df,
        output_dir / "confusion_matrix.png",
        class_names,
    )

    save_classification_report(
        predictions_df,
        output_dir / "classification_report.csv",
        class_names,
    )

    print("\nAggregate cross-validation classification report:")
    print(
        classification_report(
            predictions_df["true_label"],
            predictions_df["pred_label"],
            labels=[0, 1],
            target_names=class_names,
            zero_division=0,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pooled classification summary outputs."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Predictions file was not found:\n{args.predictions}"
        )

    predictions_df = pd.read_csv(args.predictions)

    save_classification_summary(
        predictions_df,
        args.output_dir,
    )


if __name__ == "__main__":
    main()