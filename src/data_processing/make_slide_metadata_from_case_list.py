#!/usr/bin/env python3

"""
Builds a slide-image metadata csv from a case/scan list and WSI directory.

This script is intended for slide collections where the .svs filename does not contain all required metadata.

Used for IBA1 and AT8 (in this project) staining using the provided case list excel sheets
    - at8_case_list.csv
    - iba1_case_list.csv

Expected case-list columns:
    caseid, block_id_simple, stainid, mag, file_name

Output columns:
    donor_id, region, section, stain, magnification, slide_id, slide_file, slide_path

The standardized slide ID is generated as:
    donor_region_section_stain_###

Example Command-line Run:
    python -m src.data_processing.make_slide_metadata_from_case_list \
        --input-csv /path/to/case_list.csv \
        --slides-dir /path/to/slide_image/directory \
        --region DLFC \
        --output-csv /path/to/output/slide_metadata.csv \
        --missing-csv /path/to/missing_slides.csv   

The missing-file report is optional. If --missing-csv is not provided, the
missing or skipped rows the rows are still reported in the terminal but are not
saved to a CSV.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd


############################## Arguments ##############################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build slide metadata from a case/scan list and WSI directory."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Case/scan list CSV containing slide information.",
    )

    parser.add_argument(
        "--slides-dir",
        type=Path,
        required=True,
        help="Directory containing the referenced .svs files.",
    )

    parser.add_argument(
        "--region",
        required=True,
        help="Brain region represented by these slides, e.g. DLFC.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path where the generated slide metadata will be saved.",
    )

    parser.add_argument(
        "--missing-csv",
        type=Path,
        default=None,
        help="Optional path for missing or skipped slide records.",
    )

    return parser.parse_args()


############################## Helpers ##############################

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize case-list column names before processing."""

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df

# Making slide_ids because some donors have more than one slide image for a given stain
def make_slide_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a unique standardized slide ID within each donor/section/stain group.

    Example:
        K0038_DLFC_7_AT8_001
        K0038_DLFC_7_AT8_002
    """

    df = df.copy()

    # Number slides separately within each donor/region/section/stain group.
    df["slide_number"] = (
        df.groupby(["donor_id", "region", "section", "stain"])
        .cumcount()
        .add(1)
    )

    df["slide_id"] = (
        df["donor_id"].astype(str)
        + "_"
        + df["region"].astype(str)
        + "_"
        + df["section"].astype(str)
        + "_"
        + df["stain"].astype(str)
        + "_"
        + df["slide_number"].astype(str).str.zfill(3)
    )

    return df


############################## Metadata Creation ##############################

def build_metadata(
    input_csv: Path,
    slides_dir: Path,
    region: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match case-list rows to existing SVS files and create slide metadata.
    """

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input case-list CSV was not found:\n{input_csv}"
        )

    if not slides_dir.exists():
        raise FileNotFoundError(
            f"Slides directory was not found:\n{slides_dir}"
        )

    if not slides_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory for --slides-dir:\n{slides_dir}"
        )

    scan_log = clean_columns(pd.read_csv(input_csv))

    required_columns = {
        "caseid",
        "block_id_simple",
        "stainid",
        "mag",
        "file_name",
    }

    missing_columns = required_columns.difference(scan_log.columns)

    if missing_columns:
        raise ValueError(
            "Case list is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    rows = []
    missing = []

    for _, row in scan_log.iterrows():
        donor_id = str(row["caseid"]).strip()
        section = str(row["block_id_simple"]).strip()
        stain = str(row["stainid"]).strip()
        magnification = str(row["mag"]).strip()
        slide_file = str(row["file_name"]).strip()

        # Only rows that point to a single SVS file can be used directly.
        if not slide_file.lower().endswith(".svs"):
            missing.append({
                "donor_id": donor_id,
                "section": section,
                "stain": stain,
                "slide_file": slide_file,
                "reason": "not_single_svs_file",
            })
            continue

        slide_path = slides_dir / slide_file

        if not slide_path.exists():
            missing.append({
                "donor_id": donor_id,
                "section": section,
                "stain": stain,
                "slide_file": slide_file,
                "reason": "file_not_found",
            })
            continue

        rows.append({
            "donor_id": donor_id,
            "region": region,
            "section": section,
            "stain": stain,
            "magnification": magnification,
            "slide_file": slide_file,
            "slide_path": str(slide_path.resolve()),
        })

    if not rows:
        raise ValueError(
            "No case-list rows matched valid .svs files."
        )

    metadata = pd.DataFrame(rows)
    metadata = make_slide_ids(metadata)

    metadata = metadata[
        [
            "donor_id",
            "region",
            "section",
            "stain",
            "magnification",
            "slide_id",
            "slide_file",
            "slide_path",
        ]
    ]

    metadata = (
        metadata
        .sort_values(
            ["region", "stain", "donor_id", "section", "slide_id"]
        )
        .reset_index(drop=True)
    )

    missing_df = pd.DataFrame(missing)

    return metadata, missing_df


############################## Save Outputs ##############################

def save_outputs(
    metadata: pd.DataFrame,
    missing_df: pd.DataFrame,
    output_csv: Path,
    missing_csv: Path | None,
) -> None:
    """Save matched metadata and, optionally, missing/skipped rows."""

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        output_csv,
        index=False,
    )

    if missing_csv is not None:
        missing_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        missing_df.to_csv(
            missing_csv,
            index=False,
        )


############################## Main ##############################

def main() -> None:
    args = parse_args()

    metadata, missing_df = build_metadata(
        input_csv=args.input_csv,
        slides_dir=args.slides_dir,
        region=args.region,
    )

    save_outputs(
        metadata=metadata,
        missing_df=missing_df,
        output_csv=args.output_csv,
        missing_csv=args.missing_csv,
    )

    print("\nSlide metadata summary")
    print("----------------------")
    print(f"Matched slides: {len(metadata)}")
    print(f"Missing/skipped rows: {len(missing_df)}")
    print(f"Region: {args.region}")
    print(f"Saved metadata to:\n{args.output_csv}")

    if args.missing_csv is not None:
        print(f"Saved missing-file report to:\n{args.missing_csv}")

    print("\nSlides by stain:")
    print(metadata["stain"].value_counts().sort_index())

    print("\nPreview:")
    print(metadata.head())


if __name__ == "__main__":
    main()


"""
Test Run: 

python -m src.data_processing.make_slide_image_metadata \
    --input-csv /restricted/projectnb/cteseq/users/rrasmy/cte_snRNAseq_image_transcriptomics_model/data_processing/at8_case_list.csv \
    --slides-dir "/restricted/projectnb/cteseq/data/CTE_Single_Cell/Whole Slide Images/DLFC/AT8/AT8 b7" \
    --region DLFC \
    --output-csv /restricted/projectnb/cteseq/users/rrasmy/cte_snRNAseq_image_transcriptomics_model/metadata/test_at8_metadata.csv \
    --missing-csv /restricted/projectnb/cteseq/users/rrasmy/cte_snRNAseq_image_transcriptomics_model/metadata/test_at8_missing_files.csv

"""