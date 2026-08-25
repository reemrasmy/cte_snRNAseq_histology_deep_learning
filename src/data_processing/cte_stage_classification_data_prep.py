#!/usr/bin/env python3

"""
Merge embedding metadata with donor-level CTE stage metadata. This is the metadata that is inputted into the model.

The script joins the embedding index to stage metadata using donor_id,
reports embedding donors that do not have stage information, removes those
unmatched rows, and saves the final stage-classification metadata CSV.

Expected embedding metadata:
    donor_id
    embedding_file
    ... other embedding/sample columns

Expected stage metadata:
    donor_id
    cte_stage
    cte_label

Example:
    python cte_stage_classification_data_prep.py \
        --embeddings-metadata /path/to/embedding_index.csv \
        --stage-metadata /path/to/sample_stage_metadata.csv \
        --output-csv /path/to/embedding_stage_metadata.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge embedding metadata with donor-level CTE stage metadata."
    )

    parser.add_argument(
        "--embeddings-metadata",
        type=Path,
        required=True,
        help="CSV containing slide/embedding metadata.",
    )

    parser.add_argument(
        "--stage-metadata",
        type=Path,
        required=True,
        help="CSV containing donor-level CTE stage information.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path where the merged stage-classification metadata will be saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.embeddings_metadata.exists():
        raise FileNotFoundError(
            f"Embedding metadata was not found:\n{args.embeddings_metadata}"
        )

    if not args.stage_metadata.exists():
        raise FileNotFoundError(
            f"Stage metadata was not found:\n{args.stage_metadata}"
        )

    embeddings_meta = pd.read_csv(args.embeddings_metadata)
    stage_meta = pd.read_csv(args.stage_metadata)

    # donor_id is the shared identifier linking histology and donor-level stage data.
    if "donor_id" not in embeddings_meta.columns:
        raise ValueError("Embedding metadata is missing required column: donor_id")

    if "donor_id" not in stage_meta.columns:
        raise ValueError("Stage metadata is missing required column: donor_id")

    if "cte_stage" not in stage_meta.columns:
        raise ValueError("Stage metadata is missing required column: cte_stage")

    merged = embeddings_meta.merge(
        stage_meta,
        on="donor_id",
        how="left",
    )

    # Report embedding donors that could not be matched to stage metadata.
    missing = merged[merged["cte_stage"].isna()]

    if not missing.empty:
        print("\nEmbedding donors missing stage metadata:")
        print(
            missing[["donor_id"]]
            .drop_duplicates()
            .to_string(index=False)
        )

    # Keep only samples with available stage information 
    # It is expected to have more snRNA-seq donors than image donors
    merged = merged.dropna(subset=["cte_stage"]).copy()

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        args.output_csv,
        index=False,
    )

    print("\nStage metadata merge summary")
    print("----------------------------")
    print(f"Embedding rows: {len(embeddings_meta)}")
    print(f"Stage metadata rows: {len(stage_meta)}")
    print(f"Merged rows retained: {len(merged)}")
    print(f"Unique donors retained: {merged['donor_id'].nunique()}")
    print(f"Saved metadata to:\n{args.output_csv}")


if __name__ == "__main__":
    main()