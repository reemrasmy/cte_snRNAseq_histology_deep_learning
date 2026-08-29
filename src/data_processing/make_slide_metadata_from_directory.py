#!/usr/bin/env python3

"""
Generate slide metadata directly from informative .svs filenames.

Expected filename format:
    <donor>_<section>_<stain>_<magnification>.svs --> K0038_7_LHE_20.svs

Output columns:
    donor_id, region, section, stain, magnification, slide_id, slide_file, slide_path

Slide IDs are generated as:
    donor_region_section_stain_magnification_### --> K0038_DLFC_7_LHE_20_001

Example command:
    python -m src.data_processing.make_slide_metadata_from_directory \
        --slides-dir /path/to/slide_image/storage/directory \
        --region DLFC \
        --output-csv /path/to/output/metadata
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


############################## Arguments ##############################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate slide metadata from informative .svs filenames."
    )

    parser.add_argument(
        "--slides-dir",
        type=Path,
        required=True,
        help="Directory containing .svs whole-slide images.",
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
        help="Path where slide metadata will be saved.",
    )

    return parser.parse_args()


############################## Filename Parsing ##############################

def parse_slide_name(slide_path: Path, region: str) -> dict:
    """Extract metadata fields from the slide filename."""

    # Example: K0038_7_LHE_20.svs
    parts = slide_path.stem.split("_")

    if len(parts) != 4:
        raise ValueError(
            f"Unexpected slide filename: {slide_path.name}\n"
            "Expected format: donor_section_stain_magnification.svs"
        )

    donor_id, section, stain, magnification = parts

    return {
        "donor_id": donor_id,
        "region": region,
        "section": section,
        "stain": stain,
        "magnification": magnification,
        "slide_file": slide_path.name,
        "slide_path": str(slide_path.resolve()),
    }


############################## Slide IDs ##############################

def make_slide_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a standardized unique slide ID.

    Slides sharing the same donor, region, section, stain, and
    magnification are numbered 001, 002, etc.
    """

    df = df.copy()

    df["slide_number"] = (
        df.groupby(
            [
                "donor_id",
                "region",
                "section",
                "stain",
                "magnification",
            ]
        )
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
        + df["magnification"].astype(str)
        + "_"
        + df["slide_number"].astype(str).str.zfill(3)
    )

    return df


############################## Metadata Creation ##############################

def build_metadata(
    slides_dir: Path,
    region: str,
) -> pd.DataFrame:
    """Create one metadata row for each .svs file."""

    if not slides_dir.exists():
        raise FileNotFoundError(
            f"Slides directory was not found:\n{slides_dir}"
        )

    if not slides_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory:\n{slides_dir}"
        )

    slides = sorted(slides_dir.glob("*.svs"))

    if not slides:
        raise ValueError(
            f"No .svs files found in:\n{slides_dir}"
        )

    rows = [
        parse_slide_name(slide, region)
        for slide in slides
    ]

    metadata = pd.DataFrame(rows)

    # Add standardized slide IDs after all slides are collected.
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
            [
                "region",
                "stain",
                "donor_id",
                "section",
                "magnification",
                "slide_id",
            ]
        )
        .reset_index(drop=True)
    )

    return metadata


############################## Save Output ##############################

def save_metadata(
    metadata: pd.DataFrame,
    output_csv: Path,
) -> None:
    """Save slide metadata to CSV."""

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        output_csv,
        index=False,
    )


############################## Main ##############################

def main() -> None:
    args = parse_args()

    metadata = build_metadata(
        slides_dir=args.slides_dir,
        region=args.region,
    )

    save_metadata(
        metadata=metadata,
        output_csv=args.output_csv,
    )

    print("\nSlide metadata summary")
    print("----------------------")
    print(f"Slides found: {len(metadata)}")
    print(f"Region: {args.region}")
    print(f"Saved metadata to:\n{args.output_csv}")

    print("\nSlides by stain:")
    print(metadata["stain"].value_counts().sort_index())

    print("\nPreview:")
    print(metadata.head())


if __name__ == "__main__":
    main()


"""
Test Run: 

python -m .data_processing.make_slide_metadata_from_directory \
    --slides-dir "/restricted/projectnb/cteseq/data/CTE_Single_Cell/Whole Slide Images/DLFC/LHE" \
    --region DLFC \
    --output-csv /restricted/projectnb/cteseq/users/rrasmy/cte_snRNAseq_image_transcriptomics_model/metadata/test_lhe_metadata.csv

"""