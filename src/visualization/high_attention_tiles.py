#!/usr/bin/env python3

"""
Utilities for extracting the highest-attention WSI tiles.

This module is task independent and can be used with attention
generated from either classification or regression models.

Two coordinate-reading modes are supported:

    level0:
        Convert coordinate-frame positions into OpenSlide level-0
        positions before calling WSIReader.read_rect().

    resolution:
        Pass the original coordinate-frame positions directly to
        WSIReader.read_rect() with coord_space="resolution".

The caller explicitly selects the coordinate mode so that the
visualization code does not depend on whether the model task is
classification or regression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image
from tiatoolbox.wsicore.wsireader import WSIReader


DEFAULT_TOP_K = 20
DEFAULT_WHITE_PIXEL_THRESHOLD = 240

VALID_COORDINATE_MODES = {
    "level0",
    "resolution",
}


def save_top_attention_tiles(
    attention_df: pd.DataFrame,
    slide_path: str,
    tile_output_dir: Path,
    summary_output_path: Path,
    top_k: int = DEFAULT_TOP_K,
    white_pixel_threshold: int = DEFAULT_WHITE_PIXEL_THRESHOLD,
    coordinate_mode: str = "level0",
) -> Path:
    """
    Save the highest-attention WSI tiles and a summary CSV.

    Parameters
    ----------
    attention_df:
        Tile-level dataframe containing coordinates and
        attention scores.

    slide_path:
        Path to the WSI.

    tile_output_dir:
        Directory where top-attention tile PNGs will be saved.

    summary_output_path:
        Path where the top-tile summary CSV will be saved.

    top_k:
        Number of highest-attention tiles to save.

    white_pixel_threshold:
        RGB threshold used to calculate white fraction for
        diagnostic purposes.

    coordinate_mode:
        Method used to interpret x_start/y_start when reading
        tiles from the WSI.

        "level0":
            Convert coordinate-frame locations into level-0
            WSI pixels using coordinate MPP / slide MPP.

        "resolution":
            Pass the original coordinate-frame locations to
            WSIReader.read_rect() and specify
            coord_space="resolution".

    Returns
    -------
    Path
        Path to the saved top-tile summary CSV.
    """

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------

    if coordinate_mode not in VALID_COORDINATE_MODES:
        raise ValueError(
            f"coordinate_mode must be one of "
            f"{sorted(VALID_COORDINATE_MODES)}, "
            f"but received '{coordinate_mode}'."
        )

    required_columns = {
        "x_start",
        "y_start",
        "tile_width",
        "tile_height",
        "resolution",
        "units",
        "attention_score",
    }

    missing = required_columns.difference(
        attention_df.columns
    )

    if missing:
        raise ValueError(
            "Attention dataframe is missing "
            "required columns: "
            f"{sorted(missing)}"
        )

    if len(attention_df) == 0:
        raise ValueError(
            "Attention dataframe is empty."
        )

    if not Path(slide_path).exists():
        raise FileNotFoundError(
            f"WSI was not found:\n{slide_path}"
        )

    # ------------------------------------------------------------------
    # Retrieve slide identifier for diagnostic output
    # ------------------------------------------------------------------

    if "slide_id" in attention_df.columns:
        slide_id = str(
            attention_df["slide_id"].iloc[0]
        )
    else:
        slide_id = Path(slide_path).stem

    # ------------------------------------------------------------------
    # Select highest-attention tiles
    # ------------------------------------------------------------------

    top_tiles = (
        attention_df
        .sort_values(
            "attention_score",
            ascending=False,
        )
        .head(top_k)
        .copy()
    )

    tile_output_dir = Path(
        tile_output_dir
    )

    tile_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve behavior of the existing implementation:
    # clear old PNGs from this slide's tile directory.
    for old_tile in tile_output_dir.glob("*.png"):
        old_tile.unlink()

    # ------------------------------------------------------------------
    # Open WSI
    # ------------------------------------------------------------------

    wsi = WSIReader.open(
        slide_path
    )

    # ------------------------------------------------------------------
    # Read native level-0 MPP
    # ------------------------------------------------------------------

    slide_os = openslide.OpenSlide(
        str(slide_path)
    )

    try:
        mpp_x_value = slide_os.properties.get(
            openslide.PROPERTY_NAME_MPP_X
        )

        mpp_y_value = slide_os.properties.get(
            openslide.PROPERTY_NAME_MPP_Y
        )

        if (
            mpp_x_value is None
            or mpp_y_value is None
        ):
            raise ValueError(
                f"{slide_id}: OpenSlide MPP "
                "metadata is missing."
            )

        slide_mpp_x = float(
            mpp_x_value
        )

        slide_mpp_y = float(
            mpp_y_value
        )

    finally:
        slide_os.close()

    tile_summary_rows = []

    # ------------------------------------------------------------------
    # Extract each highest-attention tile
    # ------------------------------------------------------------------

    for rank, (_, row) in enumerate(
        top_tiles.iterrows(),
        start=1,
    ):

        coordinate_units = (
            str(row["units"])
            .lower()
            .strip()
        )

        if coordinate_units != "mpp":
            raise ValueError(
                f"{slide_id}: expected "
                "coordinate units='mpp', "
                f"but found '{coordinate_units}'."
            )

        coordinate_mpp = float(
            row["resolution"]
        )

        # --------------------------------------------------------------
        # Calculate level-0 equivalents.
        #
        # These are calculated in BOTH modes so that diagnostics and
        # output summaries remain directly comparable.
        # --------------------------------------------------------------

        coordinate_to_level0_x = (
            coordinate_mpp
            / slide_mpp_x
        )

        coordinate_to_level0_y = (
            coordinate_mpp
            / slide_mpp_y
        )

        x_level0 = int(
            round(
                float(row["x_start"])
                * coordinate_to_level0_x
            )
        )

        y_level0 = int(
            round(
                float(row["y_start"])
                * coordinate_to_level0_y
            )
        )

        # --------------------------------------------------------------
        # Read tile.
        #
        # The two modes preserve the two coordinate conventions used
        # by the existing working modeling scripts.
        # --------------------------------------------------------------

        if coordinate_mode == "level0":

            tile = wsi.read_rect(
                location=(
                    x_level0,
                    y_level0,
                ),
                size=(
                    int(row["tile_width"]),
                    int(row["tile_height"]),
                ),
                resolution=coordinate_mpp,
                units=coordinate_units,
            )

        elif coordinate_mode == "resolution":

            tile = wsi.read_rect(
                location=(
                    int(row["x_start"]),
                    int(row["y_start"]),
                ),
                size=(
                    int(row["tile_width"]),
                    int(row["tile_height"]),
                ),
                resolution=coordinate_mpp,
                units=coordinate_units,
                coord_space="resolution",
            )

        # --------------------------------------------------------------
        # Convert tile to array and calculate diagnostic white fraction
        # --------------------------------------------------------------

        tile_array = np.asarray(
            tile
        )

        white_fraction = float(
            np.mean(
                np.all(
                    tile_array[:, :, :3]
                    >= white_pixel_threshold,
                    axis=2,
                )
            )
        )

        # --------------------------------------------------------------
        # Save tile
        # --------------------------------------------------------------

        tile_path = (
            tile_output_dir
            / (
                f"rank{rank:02d}_"
                f"attention"
                f"{float(row['attention_score']):.6f}_"
                f"white"
                f"{white_fraction:.3f}_"
                f"coordx"
                f"{int(row['x_start'])}_"
                f"coordy"
                f"{int(row['y_start'])}_"
                f"level0x"
                f"{x_level0}_"
                f"level0y"
                f"{y_level0}.png"
            )
        )

        if isinstance(
            tile,
            Image.Image,
        ):
            tile.save(
                tile_path
            )

        else:
            Image.fromarray(
                tile_array
            ).save(
                tile_path
            )

        # --------------------------------------------------------------
        # Build tile summary
        # --------------------------------------------------------------

        tile_summary = {
            "rank": rank,

            "attention_score": float(
                row["attention_score"]
            ),

            "white_fraction": (
                white_fraction
            ),

            "coordinate_mode": (
                coordinate_mode
            ),

            "coordinate_x_start": int(
                row["x_start"]
            ),

            "coordinate_y_start": int(
                row["y_start"]
            ),

            "level0_x_start": (
                x_level0
            ),

            "level0_y_start": (
                y_level0
            ),

            "coordinate_mpp": (
                coordinate_mpp
            ),

            "slide_mpp_x": (
                slide_mpp_x
            ),

            "slide_mpp_y": (
                slide_mpp_y
            ),

            "tile_file": str(
                tile_path
            ),
        }

        # --------------------------------------------------------------
        # Preserve task-specific/run metadata when available.
        #
        # No classification/regression branching is required.
        # --------------------------------------------------------------

        optional_columns = [
            "donor_id",
            "slide_id",
            "fold",
            "checkpoint_type",
            "selected_epoch",

            # Regression
            "true_PC1",
            "pred_PC1",
            "true_PC2",
            "pred_PC2",

            # Classification
            "true_label",
            "pred_label",
            "prob_low",
            "prob_high",
        ]

        for column in optional_columns:

            if column in row.index:
                tile_summary[
                    column
                ] = row[column]

        tile_summary_rows.append(
            tile_summary
        )

        # --------------------------------------------------------------
        # Diagnostic output for highest-attention tile
        # --------------------------------------------------------------

        if rank == 1:

            print(
                "\nTop-attention tile diagnostics"
            )

            print(
                "------------------------------"
            )

            print(
                f"Slide: {slide_id}"
            )

            print(
                f"Coordinate mode: "
                f"{coordinate_mode}"
            )

            print(
                "Level-0 MPP: "
                f"x={slide_mpp_x:.6f}, "
                f"y={slide_mpp_y:.6f}"
            )

            print(
                "Coordinate MPP: "
                f"{coordinate_mpp:.6f}"
            )

            print(
                "Coordinate-to-level-0 scale: "
                f"x={coordinate_to_level0_x:.4f}, "
                f"y={coordinate_to_level0_y:.4f}"
            )

            print(
                "Original coordinate: "
                f"({int(row['x_start'])}, "
                f"{int(row['y_start'])})"
            )

            print(
                "Equivalent level-0 coordinate: "
                f"({x_level0}, "
                f"{y_level0})"
            )

    # ------------------------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------------------------

    summary_output_path = Path(
        summary_output_path
    )

    summary_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        tile_summary_rows
    ).to_csv(
        summary_output_path,
        index=False,
    )

    return summary_output_path