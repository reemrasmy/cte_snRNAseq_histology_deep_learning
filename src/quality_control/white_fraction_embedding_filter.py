#!/usr/bin/env python3

"""
Filter UNI2 tile embeddings using precomputed tile-level white fractions.

For each slide, this script:

1. Loads the original UNI2 .pt file.
2. Loads the corresponding tile white-fraction CSV.
3. Verifies that QC rows align with embedding rows.
4. Retains tiles satisfying:
       shape_valid == True
       white_fraction <= threshold
5. Saves a new filtered .pt file without overwriting the original.
6. Updates an output metadata CSV and filtering summary CSV after
   every successfully processed slide.

The  WSI-reading and UNI2 inference steps are not repeated.

Sample Run: 
python src.quality_control.white_fraction_embedding_filter.py \
    --metadata /path/to/embedding_metadata.csv
    --qc-dir /path/to/white/fraction/generation/output/directory
    --output-root /filtered/embedding/output/directory
    --white-fraction-threshold 0.75 \
    --slide-id K1322_DLFC_7_IBA1_001
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


############################## Arguments ##############################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter tile embeddings using precomputed white-fraction QC CSVs."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help=(
            "Input metadata CSV containing at least slide_id and "
            "embedding_file."
        ),
    )

    parser.add_argument(
        "--qc-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing per-slide files named "
            "<slide_id>_tile_white_fraction.csv."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Parent output directory, such as .../white_filter/native. "
            "A threshold_XXX directory is created automatically below it."
        ),
    )

    parser.add_argument(
        "--white-fraction-threshold",
        type=float,
        required=True,
        help=(
            "Maximum allowed tile white fraction. For example, 0.90 "
            "keeps tiles with white_fraction <= 0.90."
        ),
    )

    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help=(
            "Optional filtering summary CSV path. By default, it is "
            "saved inside --output-root."
        ),
    )

    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help=(
            "Optional updated metadata CSV path. The embedding_file "
            "column will point to filtered files."
        ),
    )

    parser.add_argument(
        "--slide-id",
        type=str,
        default=None,
        help="Optional single-slide smoke test.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of metadata rows to process.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing filtered embedding files.",
    )

    return parser.parse_args()


############################## Utilities ##############################


def parse_boolean_series(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    """Convert Boolean or string Boolean values to bool."""

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)

    normalized = (
        values.astype(str)
        .str.strip()
        .str.lower()
    )

    valid = normalized.isin(["true", "false"])

    if not valid.all():
        examples = (
            values.loc[~valid]
            .astype(str)
            .unique()[:5]
        )

        raise ValueError(
            f"Column '{column_name}' contains invalid Boolean "
            f"values: {examples}"
        )

    return normalized.eq("true")


def atomic_write_csv(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write a CSV through a temporary file and then replace the target.

    This reduces the chance of leaving a partially written CSV if the
    process is interrupted during writing.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    os.replace(
        temporary_path,
        output_path,
    )


def upsert_csv_record(
    record: dict[str, Any],
    output_path: Path,
    key_column: str = "slide_id",
) -> None:
    """
    Insert or replace one CSV row using key_column as the unique key.
    """

    new_df = pd.DataFrame([record])

    if output_path.exists():
        existing_df = pd.read_csv(output_path)

        if key_column in existing_df.columns:
            existing_df = existing_df.loc[
                existing_df[key_column].astype(str)
                != str(record[key_column])
            ].copy()

        combined_df = pd.concat(
            [existing_df, new_df],
            ignore_index=True,
        )
    else:
        combined_df = new_df

    atomic_write_csv(
        dataframe=combined_df,
        output_path=output_path,
    )


def remove_csv_record(
    output_path: Path,
    key_value: str,
    key_column: str = "slide_id",
) -> None:
    """Remove one row from a CSV if it exists."""

    if not output_path.exists():
        return

    dataframe = pd.read_csv(output_path)

    if key_column not in dataframe.columns:
        return

    filtered = dataframe.loc[
        dataframe[key_column].astype(str)
        != str(key_value)
    ].copy()

    atomic_write_csv(
        dataframe=filtered,
        output_path=output_path,
    )


def safe_torch_load(path: Path) -> Any:
    """
    Load a PyTorch file on CPU.

    weights_only=False is required for payloads that contain objects
    such as pandas DataFrames, lists, or dictionaries.
    """

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(
            path,
            map_location="cpu",
        )


def filter_sequence(
    sequence: Any,
    keep_indices: np.ndarray,
    original_n_tiles: int,
    field_name: str,
) -> Any:
    """
    Filter a per-tile object while preserving its original type when
    possible.

    Supported values:
      - pandas DataFrame
      - pandas Series
      - torch Tensor
      - numpy array
      - list
      - tuple

    If the object's first dimension does not equal original_n_tiles,
    it is treated as non-tile-level metadata and returned unchanged.
    """

    if isinstance(sequence, pd.DataFrame):
        if len(sequence) != original_n_tiles:
            return sequence

        filtered = (
            sequence.iloc[keep_indices]
            .copy()
            .reset_index(drop=True)
        )

        return filtered

    if isinstance(sequence, pd.Series):
        if len(sequence) != original_n_tiles:
            return sequence

        return (
            sequence.iloc[keep_indices]
            .copy()
            .reset_index(drop=True)
        )

    if torch.is_tensor(sequence):
        if sequence.ndim == 0 or sequence.shape[0] != original_n_tiles:
            return sequence

        index_tensor = torch.as_tensor(
            keep_indices,
            dtype=torch.long,
        )

        return sequence.index_select(
            0,
            index_tensor,
        )

    if isinstance(sequence, np.ndarray):
        if sequence.ndim == 0 or sequence.shape[0] != original_n_tiles:
            return sequence

        return sequence[keep_indices]

    if isinstance(sequence, list):
        if len(sequence) != original_n_tiles:
            return sequence

        return [
            sequence[index]
            for index in keep_indices.tolist()
        ]

    if isinstance(sequence, tuple):
        if len(sequence) != original_n_tiles:
            return sequence

        return tuple(
            sequence[index]
            for index in keep_indices.tolist()
        )

    return sequence


############################## Validation ##############################


def validate_qc_alignment(
    slide_id: str,
    qc_df: pd.DataFrame,
    embeddings: torch.Tensor,
) -> None:
    """Confirm one-to-one alignment between QC rows and embeddings."""

    required_columns = {
        "tile_index",
        "shape_valid",
        "white_fraction",
        "white_pixel_threshold",
    }

    missing_columns = required_columns.difference(
        qc_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"{slide_id}: QC CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if embeddings.ndim != 2:
        raise ValueError(
            f"{slide_id}: expected embeddings with shape "
            f"[n_tiles, embedding_dim], received "
            f"{tuple(embeddings.shape)}"
        )

    if len(qc_df) != embeddings.shape[0]:
        raise ValueError(
            f"{slide_id}: QC/embedding row mismatch.\n"
            f"QC rows:        {len(qc_df)}\n"
            f"Embedding rows: {embeddings.shape[0]}"
        )

    tile_indices = pd.to_numeric(
        qc_df["tile_index"],
        errors="coerce",
    )

    if tile_indices.isna().any():
        raise ValueError(
            f"{slide_id}: tile_index contains missing or "
            "non-numeric values."
        )

    tile_indices = tile_indices.astype(int).to_numpy()
    expected = np.arange(len(qc_df))

    if not np.array_equal(
        tile_indices,
        expected,
    ):
        raise ValueError(
            f"{slide_id}: tile_index is not exactly "
            "0, 1, 2, ..., N-1. Filtering cannot safely continue."
        )

    stored_pixel_thresholds = (
        pd.to_numeric(
            qc_df["white_pixel_threshold"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    if len(stored_pixel_thresholds) != 1:
        raise ValueError(
            f"{slide_id}: expected one white_pixel_threshold in "
            f"the QC CSV, found {stored_pixel_thresholds}."
        )


############################## Process Slide ##############################


def process_slide(
    row: pd.Series,
    qc_dir: Path,
    output_root: Path,
    white_fraction_threshold: float,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Filter one slide's embeddings and return:
      1. filtering summary record
      2. updated metadata record
    """

    slide_id = str(row["slide_id"])
    embedding_path = Path(
        str(row["embedding_file"])
    )
    coordinate_path = Path(
        str(row["coord_csv"])
    )

    qc_path = (
        qc_dir
        / f"{slide_id}_tile_white_fraction.csv"
    )

    embeddings_output_dir = output_root / "embeddings"
    coordinates_output_dir = output_root / "coordinates"

    output_path = (
        embeddings_output_dir
        / f"{embedding_path.stem}_white_filtered.pt"
    )

    filtered_coordinate_path = (
        coordinates_output_dir
        / f"{slide_id}_white_filtered_coords.csv"
    )

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"{slide_id}: embedding file was not found:\n"
            f"{embedding_path}"
        )

    if not coordinate_path.exists():
        raise FileNotFoundError(
            f"{slide_id}: coordinate CSV was not found:\n"
            f"{coordinate_path}"
        )

    if not qc_path.exists():
        raise FileNotFoundError(
            f"{slide_id}: white-fraction QC CSV was not found:\n"
            f"{qc_path}"
        )

    if (
        output_path.exists()
        and filtered_coordinate_path.exists()
        and not overwrite
    ):
        print(
            f"Skipping {slide_id}: filtered embedding already exists."
        )

        payload = safe_torch_load(output_path)

        if not isinstance(payload, dict):
            raise TypeError(
                f"{slide_id}: existing filtered file is not a "
                "dictionary payload."
            )

        if "embeddings" not in payload:
            raise KeyError(
                f"{slide_id}: existing filtered file has no "
                "'embeddings' key."
            )

        filter_info = payload.get(
            "white_fraction_filter",
            {},
        )

        stored_threshold = filter_info.get(
            "white_fraction_threshold"
        )

        if (
            stored_threshold is not None
            and not np.isclose(
                float(stored_threshold),
                white_fraction_threshold,
            )
        ):
            raise ValueError(
                f"{slide_id}: existing output used threshold "
                f"{stored_threshold}, but this run requested "
                f"{white_fraction_threshold}. Use --overwrite or a "
                "different output directory."
            )

        retained_n_tiles = int(
            payload["embeddings"].shape[0]
        )

        original_n_tiles = int(
            filter_info.get(
                "original_n_tiles",
                retained_n_tiles,
            )
        )

        removed_n_tiles = (
            original_n_tiles - retained_n_tiles
        )

        summary_record = {
            "slide_id": slide_id,
            "donor_id": str(
                row.get("donor_id", "")
            ),
            "status": "existing",
            "original_n_tiles": original_n_tiles,
            "retained_n_tiles": retained_n_tiles,
            "removed_n_tiles": removed_n_tiles,
            "fraction_retained": (
                retained_n_tiles / original_n_tiles
                if original_n_tiles > 0
                else np.nan
            ),
            "fraction_removed": (
                removed_n_tiles / original_n_tiles
                if original_n_tiles > 0
                else np.nan
            ),
            "white_fraction_threshold": (
                white_fraction_threshold
            ),
            "white_pixel_threshold": filter_info.get(
                "white_pixel_threshold",
                np.nan,
            ),
            "original_embedding_file": str(
                embedding_path
            ),
            "filtered_embedding_file": str(
                output_path
            ),
            "white_fraction_qc_file": str(
                qc_path
            ),
        }

        updated_metadata = row.to_dict()
        updated_metadata["original_embedding_file"] = str(
            embedding_path
        )
        updated_metadata["embedding_file"] = str(
            output_path
        )
        updated_metadata["coord_csv"] = str(
            filtered_coordinate_path
        )
        updated_metadata["n_tiles_original"] = (
            original_n_tiles
        )
        updated_metadata["n_tiles"] = (
            retained_n_tiles
        )
        updated_metadata["white_fraction_threshold"] = (
            white_fraction_threshold
        )
        updated_metadata["white_fraction_qc_file"] = str(
            qc_path
        )
        updated_metadata["original_coord_csv"] = str(
            coordinate_path
        )
        updated_metadata["status"] = "white_filtered_existing"

        summary_record["original_coord_csv"] = str(
            coordinate_path
        )
        summary_record["filtered_coord_csv"] = str(
            filtered_coordinate_path
        )

        return summary_record, updated_metadata

    payload = safe_torch_load(
        embedding_path
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"{slide_id}: expected embedding file to contain a "
            "dictionary payload, received "
            f"{type(payload).__name__}."
        )

    if "embeddings" not in payload:
        raise KeyError(
            f"{slide_id}: payload is missing the 'embeddings' key."
        )

    embeddings = payload["embeddings"]

    if not torch.is_tensor(embeddings):
        raise TypeError(
            f"{slide_id}: payload['embeddings'] must be a tensor, "
            f"received {type(embeddings).__name__}."
        )

    qc_df = pd.read_csv(
        qc_path
    )
    coordinate_df = pd.read_csv(
        coordinate_path
    )

    validate_qc_alignment(
        slide_id=slide_id,
        qc_df=qc_df,
        embeddings=embeddings,
    )

    if len(coordinate_df) != embeddings.shape[0]:
        raise ValueError(
            f"{slide_id}: coordinate/embedding row mismatch.\n"
            f"Coordinate rows: {len(coordinate_df)}\n"
            f"Embedding rows:  {embeddings.shape[0]}"
        )

    shape_valid = parse_boolean_series(
        qc_df["shape_valid"],
        column_name="shape_valid",
    )

    white_fraction = pd.to_numeric(
        qc_df["white_fraction"],
        errors="coerce",
    )

    keep_mask = (
        shape_valid
        & white_fraction.notna()
        & (
            white_fraction
            <= white_fraction_threshold
        )
    )

    keep_indices = np.flatnonzero(
        keep_mask.to_numpy()
    )

    original_n_tiles = int(
        embeddings.shape[0]
    )

    retained_n_tiles = int(
        len(keep_indices)
    )

    removed_n_tiles = (
        original_n_tiles - retained_n_tiles
    )

    if retained_n_tiles == 0:
        raise ValueError(
            f"{slide_id}: threshold "
            f"{white_fraction_threshold} removed every tile."
        )

    index_tensor = torch.as_tensor(
        keep_indices,
        dtype=torch.long,
    )

    filtered_embeddings = embeddings.index_select(
        0,
        index_tensor,
    )

    filtered_payload = dict(payload)
    filtered_payload["embeddings"] = (
        filtered_embeddings
    )

    # Preserve original row indices so filtered embeddings can always
    # be traced back to the original embeddings and coordinate rows.
    filtered_payload["original_tile_indices"] = (
        index_tensor
    )

    # Filter common per-tile payload fields when their length matches
    # the number of original embeddings.
    possible_tile_fields = [
        "tile_metadata",
        "coordinates",
        "coords",
        "tile_coords",
    ]

    for field_name in possible_tile_fields:
        if field_name in filtered_payload:
            filtered_payload[field_name] = filter_sequence(
                sequence=filtered_payload[field_name],
                keep_indices=keep_indices,
                original_n_tiles=original_n_tiles,
                field_name=field_name,
            )

    filtered_qc = (
        qc_df.loc[keep_mask]
        .copy()
        .reset_index(drop=True)
    )

    filtered_qc["original_tile_index"] = (
        filtered_qc["tile_index"].astype(int)
    )

    filtered_qc["tile_index"] = np.arange(
        len(filtered_qc)
    )

    filtered_payload["white_fraction_qc"] = (
        filtered_qc
    )

    white_pixel_threshold = int(
        pd.to_numeric(
            qc_df["white_pixel_threshold"],
            errors="coerce",
        )
        .dropna()
        .iloc[0]
    )

    filtered_payload["white_fraction_filter"] = {
        "white_fraction_threshold": float(
            white_fraction_threshold
        ),
        "white_pixel_threshold": (
            white_pixel_threshold
        ),
        "original_n_tiles": original_n_tiles,
        "retained_n_tiles": retained_n_tiles,
        "removed_n_tiles": removed_n_tiles,
        "fraction_retained": (
            retained_n_tiles / original_n_tiles
        ),
        "fraction_removed": (
            removed_n_tiles / original_n_tiles
        ),
        "source_embedding_file": str(
            embedding_path
        ),
        "source_coord_csv": str(
            coordinate_path
        ),
        "filtered_coord_csv": str(
            filtered_coordinate_path
        ),
        "white_fraction_qc_file": str(
            qc_path
        ),
    }

    filtered_coordinate_df = (
        coordinate_df.iloc[keep_indices]
        .copy()
        .reset_index(drop=True)
    )
    filtered_coordinate_df.insert(
        0,
        "original_tile_index",
        keep_indices,
    )
    filtered_coordinate_df.insert(
        1,
        "filtered_tile_index",
        np.arange(retained_n_tiles),
    )

    embeddings_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    coordinates_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_coordinate_path = filtered_coordinate_path.with_suffix(
        filtered_coordinate_path.suffix + ".tmp"
    )
    filtered_coordinate_df.to_csv(
        temporary_coordinate_path,
        index=False,
    )
    os.replace(
        temporary_coordinate_path,
        filtered_coordinate_path,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    torch.save(
        filtered_payload,
        temporary_path,
    )

    os.replace(
        temporary_path,
        output_path,
    )

    print("\n" + "=" * 70)
    print(f"Completed: {slide_id}")
    print(f"Original tiles: {original_n_tiles}")
    print(f"Retained tiles: {retained_n_tiles}")
    print(f"Removed tiles:  {removed_n_tiles}")
    print(
        f"Fraction removed: "
        f"{removed_n_tiles / original_n_tiles:.2%}"
    )
    print(f"Saved: {output_path}")
    print("=" * 70)

    summary_record = {
        "slide_id": slide_id,
        "donor_id": str(
            row.get("donor_id", "")
        ),
        "status": "completed",
        "original_n_tiles": original_n_tiles,
        "retained_n_tiles": retained_n_tiles,
        "removed_n_tiles": removed_n_tiles,
        "fraction_retained": (
            retained_n_tiles / original_n_tiles
        ),
        "fraction_removed": (
            removed_n_tiles / original_n_tiles
        ),
        "white_fraction_threshold": (
            white_fraction_threshold
        ),
        "white_pixel_threshold": (
            white_pixel_threshold
        ),
        "original_embedding_file": str(
            embedding_path
        ),
        "filtered_embedding_file": str(
            output_path
        ),
        "original_coord_csv": str(
            coordinate_path
        ),
        "filtered_coord_csv": str(
            filtered_coordinate_path
        ),
        "white_fraction_qc_file": str(
            qc_path
        ),
    }

    updated_metadata = row.to_dict()
    updated_metadata["original_embedding_file"] = str(
        embedding_path
    )
    updated_metadata["embedding_file"] = str(
        output_path
    )
    updated_metadata["coord_csv"] = str(
        filtered_coordinate_path
    )
    updated_metadata["original_coord_csv"] = str(
        coordinate_path
    )
    updated_metadata["status"] = "white_filtered"
    updated_metadata["embedding_dim"] = int(
        filtered_embeddings.shape[1]
    )
    updated_metadata["n_tiles_original"] = (
        original_n_tiles
    )
    updated_metadata["n_tiles"] = (
        retained_n_tiles
    )
    updated_metadata["white_fraction_threshold"] = (
        white_fraction_threshold
    )
    updated_metadata["white_fraction_qc_file"] = str(
        qc_path
    )

    return summary_record, updated_metadata


############################## Main ##############################


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.white_fraction_threshold <= 1.0:
        raise ValueError(
            "--white-fraction-threshold must be between 0 and 1."
        )

    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Metadata CSV was not found:\n{args.metadata}"
        )

    if not args.qc_dir.exists():
        raise FileNotFoundError(
            f"QC directory was not found:\n{args.qc_dir}"
        )

    metadata_df = pd.read_csv(
        args.metadata
    )

    required_columns = {
        "slide_id",
        "embedding_file",
        "coord_csv",
    }

    missing_columns = required_columns.difference(
        metadata_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    metadata_df = metadata_df.dropna(
        subset=[
            "slide_id",
            "embedding_file",
            "coord_csv",
        ]
    ).copy()

    if metadata_df.empty:
        raise ValueError(
            "No usable metadata rows remain."
        )

    duplicated_slide_ids = metadata_df.loc[
        metadata_df["slide_id"].duplicated(
            keep=False
        ),
        "slide_id",
    ].astype(str).unique()

    if len(duplicated_slide_ids) > 0:
        raise ValueError(
            "Metadata contains duplicate slide IDs. "
            "Each slide must appear once.\n"
            f"Examples: {duplicated_slide_ids[:10]}"
        )

    if args.slide_id is not None:
        metadata_df = metadata_df.loc[
            metadata_df["slide_id"].astype(str)
            == args.slide_id
        ].copy()

        if metadata_df.empty:
            raise ValueError(
                f"Slide ID was not found: {args.slide_id}"
            )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than zero."
            )

        metadata_df = (
            metadata_df
            .head(args.limit)
            .copy()
        )

    threshold_name = (
        f"threshold_{round(args.white_fraction_threshold * 100):03d}"
    )
    run_output_root = (
        args.output_root
        / threshold_name
    )
    run_output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        args.summary_out
        if args.summary_out is not None
        else run_output_root / "filtering_summary.csv"
    )

    metadata_out_path = (
        args.metadata_out
        if args.metadata_out is not None
        else run_output_root / "filtered_embedding_metadata.csv"
    )

    failure_path = (
        run_output_root / "failures.csv"
    )

    print("\nWhite-fraction embedding filtering")
    print("----------------------------------")
    print(f"Metadata rows: {len(metadata_df)}")
    print(f"QC directory: {args.qc_dir}")
    print(f"Threshold output root: {run_output_root}")
    print(
        f"Keep rule: shape_valid == True and "
        f"white_fraction <= "
        f"{args.white_fraction_threshold}"
    )
    print(f"Summary: {summary_path}")
    print(f"Updated metadata: {metadata_out_path}")

    successful = 0
    failed = 0

    for _, row in tqdm(
        metadata_df.iterrows(),
        total=len(metadata_df),
        desc="Slides",
    ):
        slide_id = str(
            row["slide_id"]
        )

        try:
            summary_record, updated_metadata = (
                process_slide(
                    row=row,
                    qc_dir=args.qc_dir,
                    output_root=run_output_root,
                    white_fraction_threshold=(
                        args.white_fraction_threshold
                    ),
                    overwrite=args.overwrite,
                )
            )

            # Save progress immediately after this slide finishes.
            upsert_csv_record(
                record=summary_record,
                output_path=summary_path,
                key_column="slide_id",
            )

            abmil_columns = [
                "slide_id",
                "donor_id",
                "region",
                "stain",
                "magnification",
                "embedding_file",
                "coord_csv",
                "slide_path",
                "n_tiles",
                "embedding_dim",
                "status",
                "cte_label",
                "cte_stage",
            ]
            updated_metadata = {
                **{
                    column: updated_metadata.get(column, np.nan)
                    for column in abmil_columns
                },
                **{
                    key: value
                    for key, value in updated_metadata.items()
                    if key not in abmil_columns
                },
            }

            upsert_csv_record(
                record=updated_metadata,
                output_path=metadata_out_path,
                key_column="slide_id",
            )

            # Remove a stale failure if this slide succeeded on retry.
            remove_csv_record(
                output_path=failure_path,
                key_value=slide_id,
                key_column="slide_id",
            )

            successful += 1

        except Exception as error:
            print(
                f"\nFAILED: {slide_id}\n"
                f"{type(error).__name__}: {error}"
            )

            failure_record = {
                "slide_id": slide_id,
                "donor_id": str(
                    row.get("donor_id", "")
                ),
                "embedding_file": str(
                    row.get("embedding_file", "")
                ),
                "coord_csv": str(
                    row.get("coord_csv", "")
                ),
                "white_fraction_qc_file": str(
                    args.qc_dir
                    / f"{slide_id}_tile_white_fraction.csv"
                ),
                "white_fraction_threshold": (
                    args.white_fraction_threshold
                ),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            upsert_csv_record(
                record=failure_record,
                output_path=failure_path,
                key_column="slide_id",
            )

            failed += 1

    print("\n" + "=" * 70)
    print("Filtering complete")
    print("=" * 70)
    print(f"Successful slides: {successful}")
    print(f"Failed slides:     {failed}")
    print(f"Summary:           {summary_path}")
    print(f"Updated metadata:  {metadata_out_path}")

    if failure_path.exists():
        print(f"Failures:          {failure_path}")


if __name__ == "__main__":
    main()