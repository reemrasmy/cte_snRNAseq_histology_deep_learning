#!/usr/bin/env python3

"""
Generate observed-versus-predicted plots for continuous targets.

This module can either be imported by the ABMIL regression model
or run independently from the command line using a saved predictions
CSV.

Expected prediction columns follow the naming:

    true_<TARGET>
    pred_<TARGET>

Example:
    true_PC1
    pred_PC1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_prediction_plot(
    predictions_df: pd.DataFrame,
    target_name: str,
    output_path: Path,
) -> None:
    """
    Save an observed-versus-predicted scatter plot.

    This preserves the plotting methodology used by the
    original regression script.
    """

    true_column = (
        f"true_{target_name}"
    )

    pred_column = (
        f"pred_{target_name}"
    )

    required_columns = {
        true_column,
        pred_column,
    }

    missing = (
        required_columns
        .difference(
            predictions_df.columns
        )
    )

    if missing:
        raise ValueError(
            "Predictions dataframe is "
            "missing required columns: "
            f"{sorted(missing)}"
        )

    # Creating the scatter plot dataframe using true_* and pred_* column values
    plot_df = (
        predictions_df[
            [
                true_column,
                pred_column,
            ]
        ]
        .dropna()
        .copy()
    )

    if plot_df.empty:
        raise ValueError(
            f"No usable predictions for "
            f"target '{target_name}'."
        )

    y_true = (
        plot_df[
            true_column
        ].to_numpy()
    )

    y_pred = (
        plot_df[
            pred_column
        ].to_numpy()
    )

    lower = min(
        y_true.min(),
        y_pred.min(),
    )

    upper = max(
        y_true.max(),
        y_pred.max(),
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.75,
    )

    # Identity line:
    # perfect predictions would fall on y = x.
    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
    )

    plt.xlabel(
        f"Observed {target_name}"
    )

    plt.ylabel(
        f"Predicted {target_name}"
    )

    plt.title(
        f"Observed vs Predicted "
        f"{target_name}"
    )

    plt.tight_layout()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the figure to the output path (results directory)
    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

########################################## Command-line Arguments ##########################################

def parse_args() -> argparse.Namespace:


    parser = argparse.ArgumentParser(
        description=(
            "Generate observed-versus-"
            "predicted regression plots."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help=(
            "CSV containing true_<TARGET> "
            "and pred_<TARGET> columns."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where plots "
            "will be saved."
        ),
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        default=[
            "PC1",
            "PC2",
        ],
        help=(
            "Target names to plot. "
            "Default: PC1 PC2"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point.
    """

    args = parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(
            "Predictions CSV was not "
            f"found:\n{args.predictions}"
        )

    predictions_df = pd.read_csv(
        args.predictions
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nRegression plots"
    )

    print(
        "----------------"
    )

    print(
        f"Predictions: "
        f"{args.predictions}"
    )

    print(
        "Targets: "
        f"{args.targets}"
    )

    for target_name in args.targets:

        output_path = (
            args.output_dir
            / (
                f"{target_name.lower()}"
                "_observed_vs_predicted.png"
            )
        )

        save_prediction_plot(
            predictions_df=(
                predictions_df
            ),
            target_name=target_name,
            output_path=output_path,
        )

        print(
            f"Saved: {output_path}"
        )

    print(
        "\nRegression plotting complete."
    )


if __name__ == "__main__":
    main()