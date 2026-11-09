from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from tiatoolbox.wsicore.wsireader import WSIReader

"""
-------------------------------- Tile Coordinate QC --------------------------------

This script performs visual quality control (QC) on generated WSI tile coordinates.

For each selected slide, the script:
1. Reads the tile coordinate CSV and corresponding whole-slide image (WSI).
2. Converts tile coordinates to the WSI level-0 coordinate space (the raw image).
3. Creates a thumbnail showing raw image and the overall tissue section.
4. Overlays all generated tile coordinates to visualize tissue coverage done by generate_tile_coords.py.
5. Randomly samples tile locations and extracts larger context patches for
   visual inspection of the tissue represented by the coordinates.
6. Saves slide-level QC information and a summary of the full QC run.

Slides can be selected using donor IDs, individual slide IDs, or --all-slides. 
The output allows the user to visually inspect the quality of tiles, tissue pigmentation, and visualize how well tissue segmentation worked 

Example Usage:

python src/quality_control/tissue_segmentation_qc.py \
    --coordinate-summary metadata/DLFC/IBA1_coordinate_summary.csv \
    --slides-dir /path/to/IBA1/slides \
    --output-dir qc/tile_coordinates \
    --donor-ids K0038 K0125 \
    --n-patches 5 \
    --context-size 1024

The default context patch size is 1024 x 1024 pixels, extracted at the same
MPP used to generate the original tile coordinates.
------------------------------------------------------------------------------------
"""

########################## Command-line Arguments ##########################
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "QC tile coordinates by creating a coordinate coverage overlay "
            "and random context patches for selected slides."
        )
    )

    # The coordinate summary containing the WSI coordinate csv information. Outputted by src/data_processing/generate_tile_coords.py 
    parser.add_argument(
        "--coordinate-summary",
        required=True,
        type=Path,
        help="Coordinate summary CSV.",
    )

    # The directorr containing all the WSI that are meant to be processed. Should contain the slides in the coordinate_summary.csv 
    parser.add_argument(
        "--slides-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing the whole-slide image files. "
            "The WSI filename is read from the 'slide_file' column "
            "of the coordinate summary CSV."
        ),
    )

    # Root location where all the QC results will be written. The script later organizes by the stain, donor_id, and slide_id, so just the general qc location
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root directory for QC outputs.",
    )

    selection = parser.add_mutually_exclusive_group(
        required=True
    )

    # Give the donor ids to process all slides belonging to that donor (used to determine which slide from a specific donor are the best quality)
    selection.add_argument(
        "--donor-ids",
        nargs="+",
        help=(
            "Process all slides belonging to these donor IDs."
        ),
    )

    selection.add_argument(
        "--slide-ids",
        nargs="+",
        help="Process only these slide IDs.",
    )

    # Use this tag if you would like to process all slides in the given coordinate summary
    selection.add_argument(
        "--all-slides",
        action="store_true",
        help="Process every slide in the coordinate summary.",
    )

    parser.add_argument(
        "--n-patches",
        type=int,
        default=5,
        help="Number of random context patches per slide. Default: 5.",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=1024,
        help=(
            "Context patch width/height in output pixels. "
            "Default: 1024."
        ),
    )

    parser.add_argument(
        "--thumbnail-max-dim",
        type=int,
        default=1800,
        help="Maximum thumbnail dimension. Default: 1800.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=5,
        help="Random seed for coordinate sampling. Default: 5.",
    )

    return parser.parse_args()



########################## Slide/Donor Selection ##########################

# Selecting summary rows based on donor IDs, slide IDs, or --all-slides.
def select_slides(summary, args):


    # If donor IDs are supplied, keep every slide belonging to those donors
    if args.donor_ids:
        selected = summary[
            summary["donor_id"].astype(str).isin(args.donor_ids)
        ].copy()

    # If slide ids are supplied, keep only those exact slides
    elif args.slide_ids:
        selected = summary[
            summary["slide_id"].astype(str).isin(args.slide_ids)
        ].copy()

    # Last remaining option is the tag --all-slides
    else:
        selected = summary.copy()

    # Raise error if non of the requested IDs were found
    if selected.empty:
        raise ValueError(
            "No slides matched the requested selection."
        )

    return selected


######################### Coordinate / MPP (microns per pixel) handling #########################

# Reading the resolution given in each coordinate summary column (ex. 0.5 mpp) 
def get_coordinate_mpp(coords):

    # handle missing resolution or units
    if "resolution" not in coords.columns:
        raise ValueError(
            "Coordinate CSV is missing 'resolution'."
        )

    if "units" not in coords.columns:
        raise ValueError(
            "Coordinate CSV is missing 'units'."
        )

    units = (
        coords["units"]
        .astype(str)
        .str.lower()
        .str.strip()
        .unique()
    )

    if len(units) != 1 or units[0] != "mpp":
        raise ValueError(
            f"Expected units='mpp', found {units}."
        )

    values = (
        pd.to_numeric(
            coords["resolution"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    if len(values) != 1:
        raise ValueError(
            "Expected exactly one coordinate resolution "
            f"but found {values}."
        )

    return float(values[0])


def get_slide_mpp(slide):
    """
    Read native level-0 MPP from OpenSlide metadata.
    """

    mpp_x = slide.properties.get(
        openslide.PROPERTY_NAME_MPP_X
    )

    mpp_y = slide.properties.get(
        openslide.PROPERTY_NAME_MPP_Y
    )

    if mpp_x is None or mpp_y is None:
        raise ValueError(
            "Slide does not contain OpenSlide MPP metadata."
        )

    return float(mpp_x), float(mpp_y)


def coordinates_to_level0(
    coords,
    coordinate_mpp,
    slide_mpp_x,
    slide_mpp_y,
):
    """
    Convert coordinate-space locations to OpenSlide level-0 space.

    This follows the same MPP conversion tested by s previous
    coordinate audit:

        level0_x = x * coordinate_mpp / slide_mpp_x
        level0_y = y * coordinate_mpp / slide_mpp_y
    """

    scale_x = coordinate_mpp / slide_mpp_x
    scale_y = coordinate_mpp / slide_mpp_y

    x_level0 = (
        coords["x_start"].to_numpy(dtype=float)
        * scale_x
    )

    y_level0 = (
        coords["y_start"].to_numpy(dtype=float)
        * scale_y
    )

    return x_level0, y_level0, scale_x, scale_y

########################### Thumbnail and Coordinate Coverage Figure  ##########################

# Essentially just ouputting the raw image of the WSI tissue sample 
def make_thumbnail(
    slide,
    max_dim,
):
    slide_width, slide_height = slide.dimensions

    scale = min(
        max_dim / slide_width,
        max_dim / slide_height,
        1.0,
    )

    size = (
        max(1, int(round(slide_width * scale))),
        max(1, int(round(slide_height * scale))),
    )

    return slide.get_thumbnail(size).convert("RGB")


def save_coordinate_overlay(
    thumbnail,
    x_level0,
    y_level0,
    slide_width,
    slide_height,
    output_path,
    title,
):
    """
    Plot all coordinate centers over the already-created thumbnail.

    No WSI patches are read here.
    """

    thumbnail_x = (
        x_level0
        * thumbnail.width
        / slide_width
    )

    thumbnail_y = (
        y_level0
        * thumbnail.height
        / slide_height
    )

    plt.figure(
        figsize=(
            max(6, thumbnail.width / 250),
            max(6, thumbnail.height / 250),
        )
    )

    plt.imshow(thumbnail)

    plt.scatter(
        thumbnail_x,
        thumbnail_y,
        s=2,
        alpha=0.35,
        linewidths=0,
    )

    plt.xlim(0, thumbnail.width)
    plt.ylim(thumbnail.height, 0)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Random context patches
# ---------------------------------------------------------------------

def save_random_context_patches(
    coords,
    x_level0,
    y_level0,
    wsi,
    coordinate_mpp,
    output_dir,
    n_patches,
    context_size,
    random_seed,
):
    """
    Randomly select coordinates and read a larger context region.

    The context patch is returned at the SAME MPP used to create the
    original coordinate tiles.

    Example:
        1024 px × 0.5 MPP = 512 µm field of view.
    """

    rng = np.random.default_rng(random_seed)

    n_to_sample = min(
        n_patches,
        len(coords),
    )

    indices = rng.choice(
        len(coords),
        size=n_to_sample,
        replace=False,
    )

    records = []

    # Half-width of the output context patch in microns.
    context_half_microns = (
        context_size
        * coordinate_mpp
        / 2.0
    )

    for patch_number, index in enumerate(
        indices,
        start=1,
    ):
        row = coords.iloc[index]

        tile_width = float(
            row.get("tile_width", 256)
        )

        tile_height = float(
            row.get("tile_height", 256)
        )

        # x_level0/y_level0 represent the top-left of the original tile.
        # Convert the tile's physical half-size into level-0 pixels so
        # that we can identify the tile center.
        #
        # We derive local scale from the already converted coordinate.
        
        # if float(row["x_start"]) != 0:
        #     scale_x = (
        #         x_level0[index]
        #         / float(row["x_start"])
        #     )
        # else:
        #     scale_x = None

        # if float(row["y_start"]) != 0:
        #     scale_y = (
        #         y_level0[index]
        #         / float(row["y_start"])
        #     )
        # else:
        #     scale_y = None

        # For coordinates at zero, infer scale from coordinate arrays
        # is inconvenient, so use the WSI metadata directly below.
        slide_mpp_x = float(
            wsi.info.mpp[0]
        )

        slide_mpp_y = float(
            wsi.info.mpp[1]
        )

        tile_width_level0 = (
            tile_width
            * coordinate_mpp
            / slide_mpp_x
        )

        tile_height_level0 = (
            tile_height
            * coordinate_mpp
            / slide_mpp_y
        )

        center_x_level0 = (
            x_level0[index]
            + tile_width_level0 / 2
        )

        center_y_level0 = (
            y_level0[index]
            + tile_height_level0 / 2
        )

        # read_rect location is in baseline / level-0 coordinates.
        #
        # Determine the desired physical context width and convert
        # half of it into level-0 pixels.
        context_half_level0_x = (
            context_half_microns
            / slide_mpp_x
        )

        context_half_level0_y = (
            context_half_microns
            / slide_mpp_y
        )

        context_x = int(
            round(
                center_x_level0
                - context_half_level0_x
            )
        )

        context_y = int(
            round(
                center_y_level0
                - context_half_level0_y
            )
        )

        # Clamp the origin to zero.
        context_x = max(0, context_x)
        context_y = max(0, context_y)

        patch = wsi.read_rect(
            location=(
                context_x,
                context_y,
            ),
            size=(
                context_size,
                context_size,
            ),
            resolution=coordinate_mpp,
            units="mpp",
        )

        patch_array = np.asarray(patch)

        if isinstance(patch, Image.Image):
            patch_image = patch.convert("RGB")
        else:
            patch_image = Image.fromarray(
                patch_array[:, :, :3]
            ).convert("RGB")

        patch_path = (
            output_dir
            / f"patch_{patch_number:02d}"
              f"_row_{index}.png"
        )

        patch_image.save(patch_path)

        records.append(
            {
                "patch_number": patch_number,
                "coordinate_row": int(index),
                "tile_x_start": float(
                    row["x_start"]
                ),
                "tile_y_start": float(
                    row["y_start"]
                ),
                "tile_x_level0": float(
                    x_level0[index]
                ),
                "tile_y_level0": float(
                    y_level0[index]
                ),
                "context_x_level0": context_x,
                "context_y_level0": context_y,
                "context_size_pixels": (
                    context_size
                ),
                "context_resolution_mpp": (
                    coordinate_mpp
                ),
                "saved_patch": str(
                    patch_path
                ),
            }
        )

    return records


# ---------------------------------------------------------------------
# One slide
# ---------------------------------------------------------------------

def process_slide(
    row,
    args,
):
    donor_id = str(row["donor_id"])
    slide_id = str(row["slide_id"])
    stain = str(row["stain"])
    slide_file = str(row["slide_file"])

    coord_path = Path(
        str(row["coord_file"])
    )

    slide_path = (
        args.slides_dir
        / slide_file
    )

    print("\n" + "=" * 80)
    print(f"Donor: {donor_id}")
    print(f"Slide: {slide_id}")
    print(f"Coordinates: {coord_path}")
    print(f"Slide file: {slide_file}")
    print(f"WSI path: {slide_path}")

    if not coord_path.exists():
        raise FileNotFoundError(
            f"Coordinate CSV not found:\n{coord_path}"
        )

    if not slide_path.exists():
        raise FileNotFoundError(
            f"WSI not found for {slide_id}:\n"
            f"{slide_path}"
        )

    print("\n" + "=" * 80)
    print(f"Donor: {donor_id}")
    print(f"Slide: {slide_id}")
    print(f"Coordinates: {coord_path}")

    if not coord_path.exists():
        raise FileNotFoundError(
            f"Coordinate CSV not found:\n{coord_path}"
        )
    coords = pd.read_csv(coord_path)

    if coords.empty:
        raise ValueError(
            f"Coordinate CSV is empty: {coord_path}"
        )

    required = {
        "x_start",
        "y_start",
        "resolution",
        "units",
    }

    missing = required - set(coords.columns)

    if missing:
        raise ValueError(
            f"{coord_path.name} is missing: "
            f"{sorted(missing)}"
        )


    slide_output_dir = (
        args.output_dir
        / stain
        / donor_id
        / slide_id
    )

    patch_output_dir = (
        slide_output_dir
        / "context_patches"
    )

    patch_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coordinate_mpp = get_coordinate_mpp(
        coords
    )

    slide = openslide.OpenSlide(
        str(slide_path)
    )

    try:
        slide_width, slide_height = (
            slide.dimensions
        )

        slide_mpp_x, slide_mpp_y = (
            get_slide_mpp(slide)
        )

        thumbnail = make_thumbnail(
            slide,
            args.thumbnail_max_dim,
        )

    finally:
        slide.close()

    # --------------------------------------------------------------
    # Convert coordinate locations into level-0 WSI coordinates.
    # --------------------------------------------------------------

    (
        x_level0,
        y_level0,
        scale_x,
        scale_y,
    ) = coordinates_to_level0(
        coords=coords,
        coordinate_mpp=coordinate_mpp,
        slide_mpp_x=slide_mpp_x,
        slide_mpp_y=slide_mpp_y,
    )

    # --------------------------------------------------------------
    # Save the single thumbnail.
    # --------------------------------------------------------------

    thumbnail_path = (
        slide_output_dir
        / "01_thumbnail.png"
    )

    thumbnail.save(
        thumbnail_path
    )

    # --------------------------------------------------------------
    # Save coverage overlay.
    # --------------------------------------------------------------

    overlay_path = (
        slide_output_dir
        / "02_coordinate_coverage.png"
    )

    save_coordinate_overlay(
        thumbnail=thumbnail,
        x_level0=x_level0,
        y_level0=y_level0,
        slide_width=slide_width,
        slide_height=slide_height,
        output_path=overlay_path,
        title=(
            f"{slide_id} | "
            f"coordinate coverage"
        ),
    )

    # --------------------------------------------------------------
    # Save only a small random sample of actual WSI regions.
    # --------------------------------------------------------------

    wsi = WSIReader.open(
        str(slide_path)
    )

    patch_records = (
        save_random_context_patches(
            coords=coords,
            x_level0=x_level0,
            y_level0=y_level0,
            wsi=wsi,
            coordinate_mpp=coordinate_mpp,
            output_dir=patch_output_dir,
            n_patches=args.n_patches,
            context_size=args.context_size,
            random_seed=args.random_seed,
        )
    )

    pd.DataFrame(
        patch_records
    ).to_csv(
        slide_output_dir
        / "context_patch_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Small slide-level QC summary.
    # --------------------------------------------------------------

    summary = {
        "donor_id": donor_id,
        "slide_id": slide_id,
        "stain": stain,
        "slide_file": str(slide_file),
        "slide_path": str(slide_path),
        "coord_file": str(coord_path),
        "coordinate_rows": len(coords),
        "slide_width_level0": slide_width,
        "slide_height_level0": slide_height,
        "slide_mpp_x": slide_mpp_x,
        "slide_mpp_y": slide_mpp_y,
        "coordinate_mpp": coordinate_mpp,
        "coordinate_to_level0_scale_x": (
            scale_x
        ),
        "coordinate_to_level0_scale_y": (
            scale_y
        ),
        "n_context_patches": len(
            patch_records
        ),
        "thumbnail_path": str(
            thumbnail_path
        ),
        "coverage_map_path": str(
            overlay_path
        ),
    }

    print(
        f"Level-0 dimensions: "
        f"{slide_width:,} × {slide_height:,}"
    )

    print(
        f"Slide MPP: "
        f"x={slide_mpp_x:.4f}, "
        f"y={slide_mpp_y:.4f}"
    )

    print(
        f"Coordinate MPP: "
        f"{coordinate_mpp:.4f}"
    )

    print(
        f"Coordinate → level-0 scale: "
        f"x={scale_x:.4f}, "
        f"y={scale_y:.4f}"
    )

    print(
        f"Saved {len(patch_records)} "
        "context patches."
    )

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.coordinate_summary.exists():
        raise FileNotFoundError(
            args.coordinate_summary
        )

    if args.n_patches <= 0:
        raise ValueError(
            "--n-patches must be greater than 0."
        )

    if args.context_size <= 0:
        raise ValueError(
            "--context-size must be greater than 0."
        )

    summary = pd.read_csv(
        args.coordinate_summary
    )

    required_summary_columns = {
        "donor_id",
        "slide_id",
        "slide_file",
        "coord_file",
        "stain"
    }

    missing = (
        required_summary_columns
        - set(summary.columns)
    )

    if missing:
        raise ValueError(
            "Coordinate summary is missing required "
            f"columns: {sorted(missing)}"
        )

    selected = select_slides(
        summary,
        args,
    )

    print("=" * 80)
    print("Tile coordinate QC")
    print("=" * 80)

    print(
        f"Slides selected: {len(selected)}"
    )

    print(
        f"Context patches per slide: "
        f"{args.n_patches}"
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_records = []

    for _, row in selected.iterrows():
        try:
            record = process_slide(
                row=row,
                args=args,
            )

            record["status"] = "completed"

        except Exception as exc:
            print(
                f"ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

            record = {
                "donor_id": row["donor_id"],
                "slide_id": row["slide_id"],
                "slide_file": row["slide_file"],
                "coord_file": row["coord_file"],
                "status": "failed",
                "error": str(exc),
            }

        run_records.append(record)

    run_summary_path = (
        args.output_dir
        / "qc_run_summary.csv"
    )

    pd.DataFrame(
        run_records
    ).to_csv(
        run_summary_path,
        index=False,
    )

    completed = sum(
        record["status"] == "completed"
        for record in run_records
    )

    print("\n" + "=" * 80)
    print("QC complete")
    print("=" * 80)

    print(
        f"Completed: {completed}/{len(run_records)}"
    )

    print(
        f"Run summary: {run_summary_path}"
    )


if __name__ == "__main__":
    main()

