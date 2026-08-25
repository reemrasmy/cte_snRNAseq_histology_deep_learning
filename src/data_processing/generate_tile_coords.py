from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from tiatoolbox.tools.patchextraction import get_patch_extractor


"""
Generate tissue tile coordinates from whole-slide images (WSIs) using TIAToolbox.

This script reads slide-level metadata and generates a coordinate CSV for each
WSI using sliding-window patch extraction and tissue masking. Only tile
coordinates are saved; image tiles are not written to disk.

For each slide, the script:
1. Reads the WSI path and sample information from the metadata CSV.
2. Applies a tissue mask to the WSI.
3. Identifies valid tile locations using the requested patch size, stride,
   resolution, and minimum tissue coverage.
4. Saves the retained tile coordinates to a slide-specific CSV.
5. Creates a summary CSV containing the coordinate file path, number of
   retained tiles, extraction parameters, and processing status for each slide.

Coordinate files are automatically organized by region and stain:

    <coords-out>/
    └── <region>/
        └── <stain>/
            └── <slide_id>_<parameters>_coords.csv

The input metadata must contain:
    donor_id, region, section, stain, magnification,
    slide_id, slide_file, slide_path

Default extraction settings:
    Patch size:       256 x 256 pixels
    Stride:           256 pixels
    Resolution:       0.5 mpp
    Tissue mask:      Otsu
    Min. mask ratio:  0.8

Example output filename:
    K0038_DLFC_7_LHE_20_001_256px_0.5mpp_tissue80_coords.csv

Example Usage: 

Usage:
    python src/generate_tile_coords.py \
        --metadata /path/to/lhe_slide_metadata.csv \
        --coords-out /path/to/tile_coords \
        --summary-out /path/to/tile_coordinate_summary.csv
"""


####################### Command-line Arguments ####################### 
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate TIAToolbox tissue tile coordinates from WSI metadata."
        )
    )

    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="CSV containing slide-image metadata to process.",
    )

    parser.add_argument(
        "--coords-out",
        type=Path,
        required = True,
        help="Directory where generated coordinate CSVs will be stored.",
    )

    parser.add_argument(
        "--summary-out",
        required=True,
        type=Path,
        help="Output CSV containing one summary row per slide.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help=(
            "Requested extraction resolution. "
            "With --units mpp, 0.5 means 0.5 microns/pixel."
        ),
    )

    parser.add_argument(
        "--units",
        type=str,
        default="mpp",
        choices=["mpp"],
    )

    parser.add_argument(
        "--min-mask-ratio",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--mask-method",
        type=str,
        default="otsu",
        choices=["otsu", "morphological"],
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate coordinate CSVs even when they already exist.",
    )

    return parser.parse_args()


def generate_coords_for_slide(
    row,
    coords_out,
    patch_size,
    stride,
    resolution,
    units,
    min_mask_ratio,
    mask_method,
    overwrite=False,
):
    slide_path = Path(row["slide_path"])
    slide_file = row["slide_file"]
    slide_id = row["slide_id"]

    donor_id = row["donor_id"]
    region = row["region"]
    section = row["section"]
    stain = row["stain"]
    magnification = row["magnification"]

    print("\n" + "=" * 70)
    print(f"Processing: {slide_id}")
    print("=" * 70)

    print(
        f"Donor: {donor_id} | "
        f"Region: {region} | "
        f"Section: {section} | "
        f"Stain: {stain} | "
        f"Metadata magnification: {magnification}x"
    )

    print(f"WSI: {slide_path}")
    print(
        f"Requested coordinate resolution: "
        f"{resolution} {units}"
    )
    print(
        f"Patch size: {patch_size} x {patch_size} pixels "
        f"at requested resolution"
    )

    if units == "mpp":
        physical_patch_size = (
            patch_size * resolution
        )

        print(
            f"Approximate physical field of view: "
            f"{physical_patch_size:.2f} x "
            f"{physical_patch_size:.2f} microns"
        )

    if not slide_path.exists():
        raise FileNotFoundError(
            f"WSI does not exist:\n{slide_path}"
        )

    # A parameter tag of the tissue tile-extraction settings, used to name the coordinate csv 
        # 256px = each tile is 256 x 256 pixels
        # 0.5mpp = tiles are defined at a resolution of 0.5 microns per pixel
        # tissue80 = a tile must have at least 80% of its area as tissue coverage
    param_tag = (
        f"{patch_size}px_"
        f"{resolution:g}mpp_"
        f"tissue{int(min_mask_ratio * 100)}"
    )

    # This makes the tile_coords directory organized by the region and respective stain. 
    # The region and stain are inferred from the required metadata columns.
    outdir = (
        coords_out
        / str(region)
        / str(stain)
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coord_path = (
        outdir
        / f"{slide_id}_{param_tag}_coords.csv"
    )

    if coord_path.exists() and not overwrite:
        print(
            f"Coordinate file already exists; skipping:\n"
            f"{coord_path}"
        )

        existing_df = pd.read_csv(
            coord_path
        )

        return {
            "donor_id": donor_id,
            "region": region,
            "section": section,
            "stain": stain,
            "magnification": magnification,
            "slide_file": slide_file,
            "slide_id": slide_id,
            "n_tiles": len(existing_df),
            "coord_file": str(coord_path),
            "patch_size": patch_size,
            "stride": stride,
            "resolution": resolution,
            "units": units,
            "coordinate_space": "requested_resolution",
            "mask_method": mask_method,
            "min_mask_ratio": min_mask_ratio,
            "status": "skipped_existing",
            "error": None,
        }


    # TIAToolbox creates the sliding grid at the requested resolution/units. 
    patch_extractor = get_patch_extractor(
        input_img=slide_path,
        method_name="slidingwindow",
        patch_size=(
            patch_size,
            patch_size,
        ),
        stride=(
            stride,
            stride,
        ),
        input_mask=mask_method,
        resolution=resolution,
        units=units,
        min_mask_ratio=min_mask_ratio,
    )

    coords = []

    for tile_index, coord in enumerate(
        patch_extractor.coordinate_list
    ):
        x_start, y_start, x_end, y_end = (
            int(value)
            for value in coord
        )

        tile_width = x_end - x_start
        tile_height = y_end - y_start

        tile_id = (
            f"{slide_id}_"
            f"tile{tile_index:06d}_"
            f"x{x_start}_y{y_start}"
        )

        coords.append(
            {
                "tile_id": tile_id,

                "donor_id": donor_id,
                "region": region,
                "section": section,
                "stain": stain,
                "magnification": magnification,

                "slide_id": slide_id,
                "slide_file": slide_file,
                "slide_path": str(slide_path),

                "x_start": x_start,
                "y_start": y_start,
                "x_end": x_end,
                "y_end": y_end,
                "tile_width": tile_width,
                "tile_height": tile_height,

                "patch_size": patch_size,
                "stride": stride,
                "resolution": resolution,
                "units": units,

                # Explicitly record how x/y must be interpreted later.
                "coordinate_space": "requested_resolution",

                "mask_method": mask_method,
                "min_mask_ratio": min_mask_ratio,
            }
        )

    coord_df = pd.DataFrame(
        coords
    )

    if coord_df.empty:
        raise ValueError(
            f"No coordinates were generated for {slide_id}."
        )

    coord_df.to_csv(
        coord_path,
        index=False,
    )

    print(
        f"Saved {len(coord_df):,} coordinates to:\n"
        f"{coord_path}"
    )

    print(
        "Coordinate extent:"
    )
    print(
        f"  x: {coord_df['x_start'].min():,} "
        f"to {coord_df['x_start'].max():,}"
    )
    print(
        f"  y: {coord_df['y_start'].min():,} "
        f"to {coord_df['y_start'].max():,}"
    )

    return {
        "donor_id": donor_id,
        "region": region,
        "section": section,
        "stain": stain,
        "magnification": magnification,
        "slide_file": slide_file,
        "slide_id": slide_id,
        "n_tiles": len(coord_df),
        "coord_file": str(coord_path),
        "patch_size": patch_size,
        "stride": stride,
        "resolution": resolution,
        "units": units,
        "coordinate_space": "requested_resolution",
        "mask_method": mask_method,
        "min_mask_ratio": min_mask_ratio,
        "status": "completed",
        "error": None,
    }


def main():
    args = parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Metadata CSV does not exist:\n"
            f"{args.metadata}"
        )

    args.coords_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(
        args.metadata
    )

    print("\nSlide IDs:")
    print(
        metadata[
            [
                "slide_file",
                "slide_id",
            ]
        ].to_string(index=False)
    )
    
    required_columns = {
        "donor_id",
        "region",
        "section",
        "stain",
        "magnification",
        "slide_id",
        "slide_file",
        "slide_path",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print("\nCoordinate extraction configuration")
    print("-----------------------------------")
    print(f"Metadata: {args.metadata}")
    print(f"Slides: {len(metadata)}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {args.stride}")
    print(
        f"Resolution: "
        f"{args.resolution} {args.units}"
    )
    print(
        f"Mask: {args.mask_method}"
    )
    print(
        f"Minimum mask ratio: "
        f"{args.min_mask_ratio}"
    )
    print(
        f"Coordinate root: "
        f"{args.coords_out}"
    )
    print(
        f"Summary output: "
        f"{args.summary_out}"
    )
    print(
        f"Overwrite: {args.overwrite}"
    )

    summaries = []

    for _, row in metadata.iterrows():
        try:
            result = generate_coords_for_slide(
                row=row,
                coords_out=args.coords_out,
                patch_size=args.patch_size,
                stride=args.stride,
                resolution=args.resolution,
                units=args.units,
                min_mask_ratio=args.min_mask_ratio,
                mask_method=args.mask_method,
                overwrite=args.overwrite,
            )

            summaries.append(
                result
            )

        except Exception as error:
            slide_file = row.get(
                "slide_file",
                "unknown",
            )

            slide_id = row.get(
                "slide_id",
                Path(str(slide_file)).stem,
            )

            print(
                f"\nERROR processing {slide_id}"
            )
            print(error)

            summaries.append(
                {
                    "donor_id": row.get(
                        "donor_id"
                    ),
                    "region": row.get(
                        "region"
                    ),
                    "section": row.get(
                        "section"
                    ),
                    "stain": row.get(
                        "stain"
                    ),
                    "magnification": row.get(
                        "magnification"
                    ),
                    "slide_file": slide_file,
                    "slide_id": slide_id,
                    "n_tiles": None,
                    "coord_file": None,
                    "patch_size": args.patch_size,
                    "stride": args.stride,
                    "resolution": args.resolution,
                    "units": args.units,
                    "coordinate_space": (
                        "requested_resolution"
                    ),
                    "mask_method": args.mask_method,
                    "min_mask_ratio": (
                        args.min_mask_ratio
                    ),
                    "status": "error",
                    "error": str(error),
                }
            )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        args.summary_out,
        index=False,
    )

    print("\nCoordinate extraction complete")
    print("==============================")
    print(
        f"Slides requested: "
        f"{len(metadata)}"
    )
    print(
        f"Completed: "
        f"{(summary_df['status'] == 'completed').sum()}"
    )
    print(
        f"Skipped existing: "
        f"{(summary_df['status'] == 'skipped_existing').sum()}"
    )
    print(
        f"Errors: "
        f"{(summary_df['status'] == 'error').sum()}"
    )
    print(
        f"Summary saved to:\n"
        f"{args.summary_out}"
    )


if __name__ == "__main__":
    main()