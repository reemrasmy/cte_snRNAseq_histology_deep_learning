#!/usr/bin/env python3

"""
Create regression modeling metadata by merging histology embedding metadata
with donor-level snRNA-seq PC targets.

The script:
1. Loads an embedding metadata CSV.
2. Loads donor-level PC targets.
3. Selects the requested cell type.
4. Merges the two datasets on donor_id.
5. Reports donors missing transcriptomic targets.
6. Removes rows without usable targets.
7. Adds optional experiment provenance.
8. Saves the final regression metadata CSV.

Expected embedding metadata columns:
    slide_id
    donor_id
    embedding_file
    coord_csv
    n_tiles

Expected PC target columns:
    donor_id
    cell_type
    pc1_mean
    pc2_mean
    n_cells

Sample Usage:
    python data_processing/make_regression_metadata.py \
        --embeddings-metadata /path/to/embedding_metadata.csv \
        --pc-targets /path/to/donor_pc_targets_by_celltype.csv \
        --cell-type Micro \
        --output-csv /path/to/micro_pc_regression_metadata.csv \
        --coordinate-source transferred_from_lhe \
        --filter-type white_filter \
        --white-threshold 0.75
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

########################################## Command-line Arguments ##########################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge embedding metadata with donor-level snRNA-seq PC targets."
    )

    parser.add_argument(
        "--embeddings-metadata",
        type=Path,
        required=True,
        help="CSV containing slide/embedding metadata.",
    )

    parser.add_argument(
        "--pc-targets",
        type=Path,
        required=True,
        help="CSV containing donor-level PC targets by cell type.",
    )

    parser.add_argument(
        "--cell-type",
        required=True,
        help="Cell type to select, e.g. Micro, Exc, Astro, Inh, or Oligo.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path where regression modeling metadata will be saved.",
    )

    # Optional provenance fields
    parser.add_argument(
        "--coordinate-source",
        default=None,
        help="Optional coordinate source, e.g. native or transferred_from_lhe.",
    )

    parser.add_argument(
        "--filter-type",
        default=None,
        help="Optional filtering description, e.g. white_filter.",
    )

    parser.add_argument(
        "--white-threshold",
        type=float,
        default=None,
        help="Optional white-fraction threshold, e.g. 0.75.",
    )

    # Used if a sample was found to be a bad quality image 
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing output CSV to be replaced.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.pc_targets.exists():
        raise FileNotFoundError(
            f"PC target metadata was not found:\n{args.pc_targets}"
        )

    if not args.embeddings_metadata.exists():
        raise FileNotFoundError(
            f"Embedding metadata was not found:\n{args.embeddings_metadata}"
        )

    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output metadata already exists:\n{args.output_csv}\n\n"
            "Use --overwrite if you intend to replace it."
        )

    pc_targets = pd.read_csv(args.pc_targets)
    embeddings_meta = pd.read_csv(args.embeddings_metadata)

    required_embedding_columns = {
        "slide_id",
        "donor_id",
        "embedding_file",
        "coord_csv",
        "n_tiles",
    }

    missing_embedding_columns = (
        required_embedding_columns
        .difference(embeddings_meta.columns)
    )

    if missing_embedding_columns:
        raise ValueError(
            "Embedding metadata is missing columns: "
            f"{sorted(missing_embedding_columns)}"
        )

    required_target_columns = {
        "donor_id",
        "cell_type",
        "pc1_mean",
        "pc2_mean",
        "n_cells",
    }

    missing_target_columns = (
        required_target_columns
        .difference(pc_targets.columns)
    )

    if missing_target_columns:
        raise ValueError(
            "PC target metadata is missing columns: "
            f"{sorted(missing_target_columns)}"
        )

    # Keep only donor targets for the requested cell type.
    pc_targets = pc_targets.loc[
        pc_targets["cell_type"] == args.cell_type
    ].copy()

    if pc_targets.empty:
        raise ValueError(
            f"No target rows were found for cell type: {args.cell_type}"
        )

    # Donor-level targets should contain exactly one row per donor/cell type.
    duplicated_target_donors = pc_targets.loc[
        pc_targets["donor_id"].duplicated(keep=False),
        "donor_id",
    ].astype(str).unique()

    if len(duplicated_target_donors) > 0:
        raise ValueError(
            "The selected cell type contains duplicate donor targets. "
            "Expected one target row per donor.\n"
            f"Examples: {duplicated_target_donors[:10]}"
        )

    merged = embeddings_meta.merge(
        pc_targets[
            [
                "donor_id",
                "cell_type",
                "pc1_mean",
                "pc2_mean",
                "n_cells",
            ]
        ],
        on="donor_id",
        how="left",
        validate="many_to_one",
    )

    target_columns = [
        "pc1_mean",
        "pc2_mean",
    ]

    missing = merged.loc[
        merged[target_columns]
        .isna()
        .any(axis=1)
    ].copy()

    print("\nMissing transcriptomic targets:")

    if missing.empty:
        print("None")
    else:
        print(
            missing[
                [
                    "slide_id",
                    "donor_id",
                ]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

    # Keep only samples with complete regression targets.
    merged = merged.dropna(
        subset=target_columns
    ).copy()

    merged["regression_cell_type"] = args.cell_type
    merged["source_embedding_metadata"] = str(args.embeddings_metadata)

    if args.coordinate_source is not None:
        merged["coordinate_source"] = args.coordinate_source

    if args.filter_type is not None:
        merged["filter_type"] = args.filter_type

    if args.white_threshold is not None:
        merged["white_fraction_threshold"] = args.white_threshold

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        args.output_csv,
        index=False,
    )

    print("\n" + "=" * 70)
    print("Regression metadata creation complete")
    print("=" * 70)

    print(f"Cell type:          {args.cell_type}")
    print(f"Embedding rows:     {len(embeddings_meta)}")
    print(f"Target donors:      {pc_targets['donor_id'].nunique()}")
    print(f"Merged usable rows: {len(merged)}")
    print(f"Usable donors:      {merged['donor_id'].nunique()}")

    if args.coordinate_source is not None:
        print(f"Coordinate source:  {args.coordinate_source}")

    if args.filter_type is not None:
        print(f"Filter type:        {args.filter_type}")

    if args.white_threshold is not None:
        print(f"White threshold:    {args.white_threshold}")

    print(f"\nEmbedding metadata:\n{args.embeddings_metadata}")
    print(f"\nSaved regression metadata:\n{args.output_csv}")


if __name__ == "__main__":
    main()