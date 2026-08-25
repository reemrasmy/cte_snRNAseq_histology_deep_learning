#!/usr/bin/env python3

"""
Calculate tile-level white-background fractions for existing WSI
coordinate CSVs.

No tile PNGs, Boolean masks, or filtered embedding files are saved.

For each slide, this script:

    1. Reads every tile from the WSI using the existing coordinate CSV.
    2. Calculates the fraction of pixels classified as white.
    3. Records tile dimensions and whether extraction returned the
       expected shape.
    4. Saves:
        - one per-tile white-fraction CSV per slide;
        - one slide-level summary CSV;
        - one failure CSV when errors occur.

The threshold used to retain or remove tiles is intentionally not
applied here. It should be selected later during modeling.

*** After generating white tile fraction csvs, run white_fraction_embedding_filter.py on the new coordinate csv to set a 
white threshold and filter the existing embeddings ***

python -m src.quality_control.generate_white_tile_fractions.py \
    --metadata /path/to/embedding/metadata \
    --qc-root /path/to/output/directory \
    --coordinate-source native \
    --slide-id /path/to/one/slide-id/   (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tiatoolbox.wsicore.wsireader import WSIReader
from tqdm import tqdm


############################## Defaults ##############################

DEFAULT_WHITE_PIXEL_THRESHOLD = 220

############################## Arguments ##############################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the white-pixel fraction for every tile "
            "without saving tile images or applying a keep threshold."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help=(
            "CSV containing at least slide_id, coord_csv, and "
            "slide_path."
        ),
    )

    parser.add_argument(
        "--qc-root",
        type=Path,
        required=True,
        help=(
            "Root directory for white-fraction QC outputs. "
            "The script creates STAIN/COORDINATE_SOURCE below it."
        ),
    )

    parser.add_argument(
        "--coordinate-source",
        type=str,
        required=True,
        choices=[
            "native",
            "transferred_from_lhe",
        ],
        help=(
            "Origin of the tile coordinates. Examples: native or "
            "transferred_from_lhe."
        ),
    )

    parser.add_argument(
        "--stain",
        type=str,
        default=None,
        help=(
            "Optional stain override, such as IBA1, AT8, or LHE. "
            "When omitted, the stain is inferred from the metadata."
        ),
    )

    parser.add_argument(
        "--white-pixel-threshold",
        type=int,
        default=DEFAULT_WHITE_PIXEL_THRESHOLD,
        help=(
            "A pixel is classified as white when all RGB channels "
            "are greater than or equal to this value. Default: 220."
        ),
    )

    parser.add_argument(
        "--slide-id",
        type=str,
        default=None,
        help=(
            "Optional slide ID for a single-slide smoke test."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of metadata rows to process."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Recalculate slides whose per-tile QC CSV already exists."
        ),
    )

    return parser.parse_args()


############################## White Fraction ##############################

def calculate_white_fraction(
    tile_array: np.ndarray,
    white_pixel_threshold: int,
) -> float:
    """
    Calculate the fraction of pixels classified as white.

    A pixel is considered white when all three RGB channels are
    greater than or equal to white_pixel_threshold.

    Examples:
        0.04 = 4% white pixels
        0.80 = 80% white pixels
        1.00 = 100% white pixels
    """

    if tile_array.ndim != 3:
        raise ValueError(
            f"Expected a 3D image array, received "
            f"{tile_array.shape}"
        )

    if tile_array.shape[2] < 3:
        raise ValueError(
            f"Expected at least 3 image channels, received "
            f"{tile_array.shape}"
        )

    rgb = tile_array[:, :, :3]

    white_mask = np.all(
        rgb >= white_pixel_threshold,
        axis=2,
    )

    return float(white_mask.mean())


############################## Slide Path ##############################

def get_slide_path(
    metadata_row: pd.Series,
    coord_df: pd.DataFrame,
) -> Path:
    """
    Obtain and validate the WSI path.

    The metadata slide_path is preferred. If it is unavailable,
    slide_path is read from the coordinate CSV.

    When both sources provide a path, they must agree.
    """

    metadata_slide_path = None

    if (
        "slide_path" in metadata_row.index
        and pd.notna(metadata_row["slide_path"])
    ):
        metadata_slide_path = Path(
            str(metadata_row["slide_path"])
        )

    coord_slide_path = None

    if "slide_path" in coord_df.columns:
        coord_paths = (
            coord_df["slide_path"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(coord_paths) > 1:
            raise ValueError(
                "The coordinate CSV contains multiple slide paths: "
                f"{coord_paths[:5]}"
            )

        if len(coord_paths) == 1:
            coord_slide_path = Path(coord_paths[0])

    if metadata_slide_path is not None:
        slide_path = metadata_slide_path

    elif coord_slide_path is not None:
        slide_path = coord_slide_path

    else:
        raise ValueError(
            "No slide_path was found in either the metadata row "
            "or coordinate CSV."
        )

    if (
        metadata_slide_path is not None
        and coord_slide_path is not None
        and metadata_slide_path.resolve()
        != coord_slide_path.resolve()
    ):
        raise ValueError(
            "Metadata and coordinate CSV contain different "
            f"slide paths:\n"
            f"Metadata:    {metadata_slide_path}\n"
            f"Coordinates: {coord_slide_path}"
        )

    if not slide_path.exists():
        raise FileNotFoundError(
            f"WSI was not found:\n{slide_path}"
        )

    return slide_path


############################## Boolean Parsing ##############################

def parse_boolean_series(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convert Boolean or string Boolean values into a Boolean Series.
    """

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_values = normalized.isin(["true", "false"])

    if not valid_values.all():
        invalid_examples = (
            values.loc[~valid_values]
            .astype(str)
            .unique()[:5]
        )

        raise ValueError(
            f"Column '{column_name}' contains invalid Boolean "
            f"values: {invalid_examples}"
        )

    return normalized.eq("true")


############################## Existing QC Summary ##############################

def summarize_existing_qc(
    row: pd.Series,
    qc_path: Path,
    coordinate_source: str,
    white_pixel_threshold: int,
) -> dict:
    """
    Reconstruct the slide-level summary from an existing QC CSV.
    """

    qc_df = pd.read_csv(qc_path)

    required_columns = {
        "white_fraction",
        "shape_valid",
        "white_pixel_threshold",
    }

    missing_columns = required_columns.difference(qc_df.columns)

    if missing_columns:
        raise ValueError(
            f"Existing QC CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    shape_valid = parse_boolean_series(
        qc_df["shape_valid"],
        column_name="shape_valid",
    )

    stored_thresholds = (
        pd.to_numeric(
            qc_df["white_pixel_threshold"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    if len(stored_thresholds) != 1:
        raise ValueError(
            f"{qc_path}: expected exactly one stored white-pixel "
            f"threshold, found {stored_thresholds}."
        )

    stored_threshold = int(stored_thresholds[0])

    if stored_threshold != white_pixel_threshold:
        raise ValueError(
            f"{qc_path}: existing QC used white-pixel threshold "
            f"{stored_threshold}, but this run requested "
            f"{white_pixel_threshold}. Use --overwrite to recalculate."
        )

    valid_white = qc_df.loc[
        shape_valid,
        "white_fraction",
    ].dropna()

    return {
        "slide_id": str(row["slide_id"]),
        "donor_id": str(row.get("donor_id", "")),
        "region": str(row.get("region", "")),
        "stain": str(row.get("stain", "")),
        "coordinate_source": coordinate_source,
        "magnification": str(row.get("magnification", "")),
        "status": "existing",
        "total_tiles": len(qc_df),
        "valid_shape_tiles": int(shape_valid.sum()),
        "invalid_shape_tiles": int((~shape_valid).sum()),
        "mean_white_fraction": (
            float(valid_white.mean())
            if len(valid_white) > 0
            else np.nan
        ),
        "median_white_fraction": (
            float(valid_white.median())
            if len(valid_white) > 0
            else np.nan
        ),
        "minimum_white_fraction": (
            float(valid_white.min())
            if len(valid_white) > 0
            else np.nan
        ),
        "maximum_white_fraction": (
            float(valid_white.max())
            if len(valid_white) > 0
            else np.nan
        ),
        "fraction_above_080": (
            float((valid_white > 0.80).mean())
            if len(valid_white) > 0
            else np.nan
        ),
        "fraction_above_090": (
            float((valid_white > 0.90).mean())
            if len(valid_white) > 0
            else np.nan
        ),
        "fraction_above_095": (
            float((valid_white > 0.95).mean())
            if len(valid_white) > 0
            else np.nan
        ),
        "tile_qc_file": str(qc_path),
        "coord_csv": str(row["coord_csv"]),
        "slide_path": str(row.get("slide_path", "")),
        "embedding_file": str(row.get("embedding_file", "")),
        "white_pixel_threshold": white_pixel_threshold,
    }


############################## Process One Slide ##############################

def process_slide(
    row: pd.Series,
    output_dir: Path,
    coordinate_source: str,
    white_pixel_threshold: int,
    overwrite: bool,
) -> dict:
    """
    Calculate white fractions for all coordinate rows from one slide.
    """

    slide_id = str(row["slide_id"])
    coord_csv = Path(str(row["coord_csv"]))

    qc_path = (
        output_dir
        / f"{slide_id}_tile_white_fraction.csv"
    )

    if not coord_csv.exists():
        raise FileNotFoundError(
            f"Coordinate CSV was not found:\n{coord_csv}"
        )

    if qc_path.exists() and not overwrite:
        print(
            f"Skipping {slide_id}: QC CSV already exists."
        )

        return summarize_existing_qc(
            row=row,
            qc_path=qc_path,
            coordinate_source=coordinate_source,
            white_pixel_threshold=white_pixel_threshold,
        )

    coord_df = pd.read_csv(coord_csv)

    if coord_df.empty:
        raise ValueError(
            f"{slide_id}: coordinate CSV contains no rows."
        )

    required_coord_columns = {
        "x_start",
        "y_start",
        "tile_width",
        "tile_height",
        "resolution",
        "units",
    }

    missing_columns = required_coord_columns.difference(
        coord_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"{slide_id}: coordinate CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # Optional sanity check against metadata without opening embeddings.
    if "n_tiles" in row.index and pd.notna(row["n_tiles"]):
        metadata_n_tiles = int(row["n_tiles"])

        if metadata_n_tiles != len(coord_df):
            raise ValueError(
                f"{slide_id}: metadata/coordinate mismatch.\n"
                f"Metadata n_tiles: {metadata_n_tiles}\n"
                f"Coordinate rows:  {len(coord_df)}"
            )

    slide_path = get_slide_path(
        metadata_row=row,
        coord_df=coord_df,
    )

    print("\n" + "=" * 70)
    print(f"Processing: {slide_id}")
    print(f"Coordinate rows: {len(coord_df)}")
    print(f"Slide: {slide_path}")
    print(
        f"White pixel threshold: "
        f"{white_pixel_threshold}"
    )
    print("=" * 70)

    wsi = WSIReader.open(slide_path)

    qc_records = []

    for tile_position, (_, coord_row) in enumerate(
        tqdm(
            coord_df.iterrows(),
            total=len(coord_df),
            desc=slide_id,
        )
    ):
        requested_width = int(
            coord_row["tile_width"]
        )

        requested_height = int(
            coord_row["tile_height"]
        )

        tile = wsi.read_rect(
            location=(
                int(coord_row["x_start"]),
                int(coord_row["y_start"]),
            ),
            size=(
                requested_width,
                requested_height,
            ),
            resolution=float(
                coord_row["resolution"]
            ),
            units=str(
                coord_row["units"]
            ),
        )

        tile_array = np.asarray(tile)

        expected_shape = (
            requested_height,
            requested_width,
        )

        shape_valid = (
            tile_array.ndim == 3
            and tile_array.shape[:2] == expected_shape
            and tile_array.shape[2] >= 3
        )

        if shape_valid:
            white_fraction = calculate_white_fraction(
                tile_array=tile_array,
                white_pixel_threshold=white_pixel_threshold,
            )
        else:
            white_fraction = np.nan

        qc_records.append({
            # This positional index must align with embedding rows.
            "tile_index": tile_position,

            "x_start": int(coord_row["x_start"]),
            "y_start": int(coord_row["y_start"]),
            "tile_width": requested_width,
            "tile_height": requested_height,
            "resolution": float(
                coord_row["resolution"]
            ),
            "units": str(coord_row["units"]),

            "returned_height": (
                int(tile_array.shape[0])
                if tile_array.ndim >= 2
                else np.nan
            ),
            "returned_width": (
                int(tile_array.shape[1])
                if tile_array.ndim >= 2
                else np.nan
            ),
            "returned_channels": (
                int(tile_array.shape[2])
                if tile_array.ndim == 3
                else np.nan
            ),

            "shape_valid": bool(shape_valid),
            "white_pixel_threshold": white_pixel_threshold,
            "white_fraction": white_fraction,
        })

        # The tile is measured and then discarded from memory.
        del tile
        del tile_array

    qc_df = pd.DataFrame(qc_records)

    qc_df.to_csv(
        qc_path,
        index=False,
    )

    valid_white = qc_df.loc[
        qc_df["shape_valid"],
        "white_fraction",
    ].dropna()

    valid_shape_tiles = int(
        qc_df["shape_valid"].sum()
    )

    invalid_shape_tiles = (
        len(qc_df) - valid_shape_tiles
    )

    print(f"\nCompleted {slide_id}")
    print(f"Total tiles:         {len(qc_df)}")
    print(f"Valid-shape tiles:   {valid_shape_tiles}")
    print(f"Invalid-shape tiles: {invalid_shape_tiles}")

    if len(valid_white) > 0:
        print(
            f"Mean white fraction:   "
            f"{valid_white.mean():.4f}"
        )
        print(
            f"Median white fraction: "
            f"{valid_white.median():.4f}"
        )
        print(
            f"Tiles above 0.80:      "
            f"{(valid_white > 0.80).mean():.2%}"
        )

    return {
        "slide_id": slide_id,
        "donor_id": str(row.get("donor_id", "")),
        "region": str(row.get("region", "")),
        "stain": str(row.get("stain", "")),
        "coordinate_source": coordinate_source,
        "magnification": str(
            row.get("magnification", "")
        ),
        "status": "completed",
        "total_tiles": len(qc_df),
        "valid_shape_tiles": valid_shape_tiles,
        "invalid_shape_tiles": invalid_shape_tiles,
        "mean_white_fraction": (
            float(valid_white.mean())
            if len(valid_white) > 0
            else np.nan
        ),
        "median_white_fraction": (
            float(valid_white.median())
            if len(valid_white) > 0
            else np.nan
        ),
        "minimum_white_fraction": (
            float(valid_white.min())
            if len(valid_white) > 0
            else np.nan
        ),
        "maximum_white_fraction": (
            float(valid_white.max())
            if len(valid_white) > 0
            else np.nan
        ),
        "fraction_above_080": (
            float((valid_white > 0.80).mean())
            if len(valid_white) > 0
            else np.nan
        ),
        "fraction_above_090": (
            float((valid_white > 0.90).mean())
            if len(valid_white) > 0
            else np.nan
        ),
        "fraction_above_095": (
            float((valid_white > 0.95).mean())
            if len(valid_white) > 0
            else np.nan
        ),
        "tile_qc_file": str(qc_path),
        "coord_csv": str(coord_csv),
        "slide_path": str(slide_path),
        "embedding_file": str(
            row.get("embedding_file", "")
        ),
        "white_pixel_threshold": white_pixel_threshold,
    }

# Determining the stain using either the supplied argument in command-line or infer from the required columns in csv
def resolve_stain(
    metadata_df: pd.DataFrame,
    requested_stain: str | None,
) -> str:
    """
    Determine the stain for the current run.

    Prefer an explicitly supplied --stain value. Otherwise, infer
    the stain from the metadata. One QC run must contain one stain.
    """

    if requested_stain is not None:
        stain = requested_stain.strip().upper()

        if not stain:
            raise ValueError(
                "--stain cannot be an empty string."
            )


        if "stain" in metadata_df.columns:
            metadata_stains = (
                metadata_df["stain"]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .unique()
            )

            conflicting_stains = [
                value
                for value in metadata_stains
                if value != stain
            ]

            if conflicting_stains:
                raise ValueError(
                    "The requested stain does not agree with the "
                    "metadata.\n"
                    f"Requested stain: {stain}\n"
                    f"Metadata stains: "
                    f"{sorted(metadata_stains.tolist())}"
                )

        return stain

    if "stain" not in metadata_df.columns:
        raise ValueError(
            "The metadata does not contain a 'stain' column. "
            "Supply the stain using --stain."
        )

    metadata_stains = (
        metadata_df["stain"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
    )

    # Raise ValueError if the stain is not found in metadata
    if len(metadata_stains) == 0:
        raise ValueError(
            "No valid stain values were found in the metadata. "
            "Supply the stain using --stain."
        )

    if len(metadata_stains) > 1:
        raise ValueError(
            "The metadata contains multiple stains. Run this script "
            "separately for each stain.\n"
            f"Stains found: {sorted(metadata_stains.tolist())}"
        )

    return str(metadata_stains[0])


############################## Incremental CSV Writing ##############################

def upsert_csv_record(
    record: dict,
    csv_path: Path,
    key_columns: list[str],
) -> None:
    """
    Replaces an older record with
    the same key if one already exists.

    The CSV is written to a temporary file and then atomically moved
    into place, reducing the chance of leaving a partially written
    summary if the process is interrupted during the write.
    """

    new_df = pd.DataFrame([record])

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)

        missing_keys = [
            column
            for column in key_columns
            if column not in existing_df.columns
        ]

        if missing_keys:
            raise ValueError(
                f"Existing CSV is missing key columns "
                f"{missing_keys}: {csv_path}"
            )

        combined_df = pd.concat(
            [existing_df, new_df],
            ignore_index=True,
            sort=False,
        )
    else:
        combined_df = new_df

    combined_df = combined_df.drop_duplicates(
        subset=key_columns,
        keep="last",
    )

    temporary_path = csv_path.with_suffix(
        csv_path.suffix + ".tmp"
    )

    combined_df.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(csv_path)


def remove_csv_record(
    csv_path: Path,
    key_column: str,
    key_value: str,
) -> None:
    """Remove a stale record, such as an old failure after success exists."""

    if not csv_path.exists():
        return

    existing_df = pd.read_csv(csv_path)

    if key_column not in existing_df.columns:
        return

    keep_mask = (
        existing_df[key_column].astype(str)
        != str(key_value)
    )

    updated_df = existing_df.loc[keep_mask].copy()

    temporary_path = csv_path.with_suffix(
        csv_path.suffix + ".tmp"
    )

    updated_df.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(csv_path)


############################## Main ##############################

def main() -> None:
    args = parse_args()

    if not 0 <= args.white_pixel_threshold <= 255:
        raise ValueError(
            "--white-pixel-threshold must be between 0 and 255."
        )

    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Metadata CSV was not found:\n{args.metadata}"
        )


    metadata_df = pd.read_csv(args.metadata)

    required_metadata_columns = {
        "slide_id",
        "coord_csv",
        "slide_path",
    }

    missing_columns = required_metadata_columns.difference(
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
            "coord_csv",
            "slide_path",
        ]
    ).copy()

    if metadata_df.empty:
        raise ValueError(
            "No usable metadata rows remain after removing rows "
            "with missing required values."
        )

    stain = resolve_stain(
        metadata_df=metadata_df,
        requested_stain=args.stain,
    )

    metadata_df["stain"] = stain

    coordinate_source = (
        args.coordinate_source
        .strip()
        .lower()
    )

    output_dir = (
        args.qc_root
        / stain
        / coordinate_source
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Prevent accidentally processing a duplicated metadata row twice.
    duplicated_slide_ids = metadata_df.loc[
        metadata_df["slide_id"].duplicated(
            keep=False
        ),
        "slide_id",
    ].astype(str).unique()

    if len(duplicated_slide_ids) > 0:
        raise ValueError(
            "Metadata contains duplicate slide IDs. "
            "Each slide should be processed once.\n"
            f"Examples: {duplicated_slide_ids[:10]}"
        )

    if args.slide_id is not None:
        metadata_df = metadata_df.loc[
            metadata_df["slide_id"].astype(str)
            == args.slide_id
        ].copy()

        if metadata_df.empty:
            raise ValueError(
                "Slide ID was not found in metadata: "
                f"{args.slide_id}"
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

    print("\nWhite-fraction QC run")
    print("---------------------")
    print(f"Metadata rows: {len(metadata_df)}")
    print(f"Stain: {stain}")
    print(f"Coordinate source: {coordinate_source}")
    print(f"Output directory: {output_dir}")
    print(
        f"White pixel threshold: "
        f"{args.white_pixel_threshold}"
    )
    print(
        "No tile-retention threshold is applied in this script."
    )

    summary_path = (
        output_dir
        / (
            f"{stain.lower()}_"
            f"{coordinate_source}_"
            f"white_fraction_summary.csv"
        )
    )

    failure_path = (
        output_dir
        / (
            f"{stain.lower()}_"
            f"{coordinate_source}_"
            f"white_fraction_failures.csv"
        )
    )

    successful_slides = 0
    failed_slides = 0

    for _, row in metadata_df.iterrows():
        slide_id = str(row["slide_id"])

        try:
            summary = process_slide(
                row=row,
                output_dir=output_dir,
                coordinate_source=coordinate_source,
                white_pixel_threshold=(
                    args.white_pixel_threshold
                ),
                overwrite=args.overwrite,
            )

            # Persist this slide immediately. On a restarted run, an
            # existing row for the same slide is replaced rather than
            # duplicated.
            upsert_csv_record(
                record=summary,
                csv_path=summary_path,
                key_columns=["slide_id"],
            )

            # If this slide failed in an earlier run but now succeeds,
            # remove its stale failure entry.
            remove_csv_record(
                csv_path=failure_path,
                key_column="slide_id",
                key_value=slide_id,
            )

            successful_slides += 1

            print(
                f"Summary updated immediately: {summary_path}"
            )

        except Exception as error:
            print(
                f"\nFAILED: {slide_id}\n"
                f"{type(error).__name__}: {error}"
            )

            failure = {
                "slide_id": slide_id,
                "donor_id": str(
                    row.get("donor_id", "")
                ),
                "region": str(
                    row.get("region", "")
                ),
                "stain": stain,
                "coordinate_source": coordinate_source,
                "coord_csv": str(
                    row.get("coord_csv", "")
                ),
                "slide_path": str(
                    row.get("slide_path", "")
                ),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            # Failures are also persisted immediately so they survive
            # a later crash.
            upsert_csv_record(
                record=failure,
                csv_path=failure_path,
                key_columns=["slide_id"],
            )

            failed_slides += 1

            print(
                f"Failure recorded immediately: {failure_path}"
            )

    print("\n" + "=" * 70)
    print("White-fraction calculation complete")
    print("=" * 70)
    print(
        f"Successful slides this run: "
        f"{successful_slides}"
    )
    print(
        f"Failed slides this run: "
        f"{failed_slides}"
    )
    print(f"Summary: {summary_path}")

    if failure_path.exists():
        print(f"Failures: {failure_path}")


if __name__ == "__main__":
    main()


