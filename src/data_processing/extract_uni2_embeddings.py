from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import openslide
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from tiatoolbox.wsicore.wsireader import WSIReader

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from huggingface_hub import login

"""
Extract UNI2 embeddings from whole-slide image (WSI) tile coordinates.

This script converts tissue tile locations from previously generated coordinate
CSVs into pretrained UNI2-h feature embeddings for downstream modeling.

For each slide, the script:
1. Reads the coordinate CSV path from the supplied metadata/coordinate summary.
2. Uses TIAToolbox to read each image patch directly from the original WSI.
3. Interprets tile coordinates allowing the same extraction
   logic to be used for WSIs scanned at different native magnifications.
4. Passes tiles through UNI2-h in batches to generate a 1536-dimensional
   embedding for every retained tissue tile.
5. Saves the embeddings and corresponding tile metadata as a .pt file.
6. Creates or updates an embedding index CSV with one row per processed slide.

The input metadata must contain a column identifying each coordinate CSV.
Accepted column names are:
    coord_file
    coord_csv
    output_coord_file

Sample Usage: 
    python src/extract_uni2_embeddings.py \
        --metadata /path/to/tile_coordinate_summary.csv \
        --embedding-root /path/to/output/embeddings \
        --index-out /path/to/output/embedding_index.csv    

Sample Usage smoke test (process only the first slide in the supplied metadata):
    python src/extract_uni2_embeddings.py \
        --metadata /path/to/tile_coordinate_summary.csv \
        --embedding-root /path/to/output/embeddings \
        --index-out /path/to/output/embedding_index.csv \
        --smoke-test
"""    

########################### UNI2 Settings ###########################

EMBEDDING_DIM = 1536

########################### Command-line Arguments ###########################

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract UNI2 embeddings from WSI tile-coordinate CSVs. "
            "Coordinates are interpreted in the requested-resolution "
            "coordinate space used during TIAToolbox patch extraction."
        )
    )

    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help=(
            "CSV containing slides/coordinate files to process. "
            "Must contain either coord_file or coord_csv."
        ),
    )

    parser.add_argument(
        "--embedding-root",
        required=True,
        type=Path,
        help="Directory where UNI2 .pt embedding files will be written.",
    )

    parser.add_argument(
        "--index-out",
        required=True,
        type=Path,
        help="CSV index describing generated embedding files.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of tiles processed by UNI2 per GPU batch.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "PyTorch DataLoader workers. Default 0 is safest for "
            "whole-slide readers."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing embedding .pt file.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process only the first slide in the supplied metadata.",
    )

    return parser.parse_args()

########################### Load UNI2 Foundation Model (from hugging-face) ###########################

def load_uni2_model(device):
    """
    Load pretrained UNI2-h.

    Authentication should be supplied through the Hugging Face cache
    or the HF_TOKEN environment variable rather than hard-coding a token.
    """

    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        login(token=hf_token)
        print("Using Hugging Face token from HF_TOKEN environment variable.")
    else:
        print(
            "HF_TOKEN environment variable not set. "
            "Using existing Hugging Face authentication/cache."
        )

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": EMBEDDING_DIM,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        **timm_kwargs,
    )

    transform = create_transform(
        **resolve_data_config(
            model.pretrained_cfg,
            model=model,
        )
    )

    model.eval()
    model.to(device)

    return model, transform


# Reading the coordinates from slide-coordinate csvs as images 
class WSICoordDataset(Dataset):
    """
    Convert coordinate rows into image patches for UNI2.

    IMPORTANT:
    x_start/y_start are interpreted in the coordinate CSV's requested
    resolution space.

    Example:
        resolution = 0.5
        units = mpp

    This is intentionally independent of whether the original WSI was
    scanned at 20x or 40x.
    """

    def __init__(
        self,
        coord_csv: Path,
        transform,
    ):
        self.coord_csv = Path(coord_csv)

        if not self.coord_csv.exists():
            raise FileNotFoundError(
                f"Coordinate CSV does not exist:\n"
                f"{self.coord_csv}"
            )

        self.coords = pd.read_csv(
            self.coord_csv
        ).reset_index(drop=True)

        self.coords["original_tile_index"] = (
            self.coords.index
        )

        if self.coords.empty:
            raise ValueError(
                f"Coordinate CSV is empty:\n"
                f"{self.coord_csv}"
            )

        required_columns = {
            "slide_path",
            "x_start",
            "y_start",
            "tile_width",
            "tile_height",
            "resolution",
            "units",
        }

        missing = (
            required_columns
            - set(self.coords.columns)
        )

        if missing:
            raise ValueError(
                f"{self.coord_csv.name} is missing required columns: "
                f"{sorted(missing)}"
            )

        self.transform = transform

        self.slide_path = str(
            self.coords.loc[0, "slide_path"]
        )

        if not Path(self.slide_path).exists():
            raise FileNotFoundError(
                f"WSI does not exist:\n"
                f"{self.slide_path}"
            )

        # Confirm one resolution/unit is used throughout the CSV.
        resolutions = (
            pd.to_numeric(
                self.coords["resolution"],
                errors="coerce",
            )
            .dropna()
            .unique()
        )

        if len(resolutions) != 1:
            raise ValueError(
                "Expected exactly one coordinate resolution, "
                f"but found: {resolutions}"
            )

        units = (
            self.coords["units"]
            .astype(str)
            .str.lower()
            .str.strip()
            .unique()
        )

        if len(units) != 1:
            raise ValueError(
                "Expected exactly one coordinate unit, "
                f"but found: {units}"
            )

        self.coordinate_resolution = float(
            resolutions[0]
        )

        self.coordinate_units = str(
            units[0]
        )

        # Opening the slide image 
        self.reader = WSIReader.open(
            self.slide_path
        )

        self.print_coordinate_diagnostics()

    def print_coordinate_diagnostics(self):
        """
        Show how the coordinate system compares with the raw level-0 WSI.
        This is diagnostic only; no manual coordinate conversion is
        performed because read_rect receives coord_space='resolution'.
        """

        slide = openslide.OpenSlide(
            self.slide_path
        )

        try:
            width, height = slide.dimensions

            mpp_x_value = slide.properties.get(
                openslide.PROPERTY_NAME_MPP_X
            )

            mpp_y_value = slide.properties.get(
                openslide.PROPERTY_NAME_MPP_Y
            )

            objective_power = slide.properties.get(
                openslide.PROPERTY_NAME_OBJECTIVE_POWER
            )

            print("\nCoordinate-system check")
            print("-----------------------")
            print(
                f"Slide: "
                f"{Path(self.slide_path).name}"
            )
            print(
                f"Level-0 dimensions: "
                f"{width:,} x {height:,}"
            )
            print(
                f"Objective power: "
                f"{objective_power}"
            )
            print(
                f"Coordinate resolution: "
                f"{self.coordinate_resolution} "
                f"{self.coordinate_units}"
            )
            print(
                "Coordinate interpretation: "
                "requested-resolution space"
            )
            print(
                "WSI read_rect coord_space: "
                "resolution"
            )

            if (
                mpp_x_value is not None
                and mpp_y_value is not None
                and self.coordinate_units == "mpp"
            ):
                slide_mpp_x = float(
                    mpp_x_value
                )
                slide_mpp_y = float(
                    mpp_y_value
                )

                scale_x = (
                    self.coordinate_resolution
                    / slide_mpp_x
                )

                scale_y = (
                    self.coordinate_resolution
                    / slide_mpp_y
                )

                print(
                    f"Level-0 MPP: "
                    f"x={slide_mpp_x:.6f}, "
                    f"y={slide_mpp_y:.6f}"
                )

                print(
                    "Equivalent coordinate-to-level0 scale: "
                    f"x={scale_x:.4f}, "
                    f"y={scale_y:.4f}"
                )

        finally:
            slide.close()

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]

        ################## 40x Magnification Issue Fix ##################
        # The coordinate CSV was generated at a requested resolution
        # such as 0.5 mpp. Therefore x_start/y_start must be interpreted
        # in THAT coordinate space, rather than baseline/level-0 pixels.

        patch = self.reader.read_rect(
            location=(
                int(row["x_start"]),
                int(row["y_start"]),
            ),
            size=(
                int(row["tile_width"]),
                int(row["tile_height"]),
            ),
            resolution=float(
                row["resolution"]
            ),
            units=str(
                row["units"]
            ),
            coord_space="resolution",
        )

        if not isinstance(
            patch,
            Image.Image,
        ):
            patch = Image.fromarray(
                patch
            )

        patch = patch.convert(
            "RGB"
        )

        image_tensor = self.transform(
            patch
        )

        return image_tensor


# -------------------------------------------------------------------------
# Extract one slide
# -------------------------------------------------------------------------

def extract_embeddings_for_csv(
    coord_csv,
    model,
    transform,
    embedding_root,
    batch_size,
    num_workers,
    device,
    overwrite=False,
):
    coord_csv = Path(
        coord_csv
    )

    dataset = WSICoordDataset(
        coord_csv=coord_csv,
        transform=transform,
    )

    first_row = dataset.coords.iloc[0]

    required_metadata_columns = {
        "slide_id",
        "donor_id",
        "region",
        "stain",
        "magnification",
        "slide_path",
    }

    missing = (
        required_metadata_columns
        - set(dataset.coords.columns)
    )

    if missing:
        raise ValueError(
            f"{coord_csv.name} is missing sample metadata columns: "
            f"{sorted(missing)}"
        )

    slide_id = str(
        first_row["slide_id"]
    )

    donor_id = str(
        first_row["donor_id"]
    )

    region = str(
        first_row["region"]
    )

    stain = str(
        first_row["stain"]
    )

    magnification = first_row[
        "magnification"
    ]

    slide_path = str(
        first_row["slide_path"]
    )

    embedding_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    outpath = (
        embedding_root
        / f"{slide_id}_uni2.pt"
    )

    if (
        outpath.exists()
        and not overwrite
    ):
        print(
            f"\nSkipped: {slide_id} "
            "embedding already exists."
        )

        return {
            "slide_id": slide_id,
            "donor_id": donor_id,
            "region": region,
            "stain": stain,
            "magnification": magnification,
            "embedding_file": str(
                outpath
            ),
            "coord_csv": str(
                coord_csv
            ),
            "slide_path": slide_path,
            "n_tiles": len(
                dataset
            ),
            "embedding_dim": (
                EMBEDDING_DIM
            ),
            "embedding_model": (
                "UNI2-h"
            ),
            "coordinate_source": (
                "native"
            ),
            "read_coord_space": (
                "resolution"
            ),
            "status": (
                "skipped_existing"
            ),
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_embeddings = []

    for batch_idx, image_batch in enumerate(
        dataloader,
        start=1,
    ):
        image_batch = image_batch.to(
            device
        )

        with torch.inference_mode():
            embeddings = model(
                image_batch
            )

        all_embeddings.append(
            embeddings.cpu()
        )

        if (
            batch_idx % 100 == 0
            or batch_idx == len(dataloader)
        ):
            tiles_processed = min(
                batch_idx * batch_size,
                len(dataset),
            )

            print(
                f"  {slide_id}: "
                f"{tiles_processed:,}/"
                f"{len(dataset):,} tiles"
            )

    embedding_tensor = torch.cat(
        all_embeddings,
        dim=0,
    )

    if (
        embedding_tensor.shape[0]
        != len(dataset)
    ):
        raise ValueError(
            f"Embedding/tile mismatch for "
            f"{slide_id}: "
            f"{embedding_tensor.shape[0]} "
            f"embeddings vs "
            f"{len(dataset)} coordinates"
        )

    if (
        embedding_tensor.shape[1]
        != EMBEDDING_DIM
    ):
        raise ValueError(
            f"Unexpected embedding dimension "
            f"for {slide_id}: "
            f"{embedding_tensor.shape[1]}"
        )

    # Since the DataLoader shuffle=False and every CSV row produces exactly
    # one image tensor, the original coordinate dataframe is the exact
    # tile_metadata ordering corresponding to embedding_tensor.
    #
    # Keeping ALL coordinate metadata is safer than keeping only x/y.
    tile_metadata_df = (
        dataset.coords
        .reset_index(drop=True)
        .copy()
    )

    if (
        len(tile_metadata_df)
        != embedding_tensor.shape[0]
    ):
        raise ValueError(
            f"tile_metadata length mismatch "
            f"for {slide_id}."
        )

    torch.save(
        {
            "slide_id": slide_id,
            "donor_id": donor_id,
            "region": region,
            "stain": stain,
            "magnification": magnification,

            "embedding_model": (
                "UNI2-h"
            ),
            "embedding_dim": int(
                embedding_tensor.shape[1]
            ),

            "embeddings": (
                embedding_tensor
            ),

            # Full coordinate metadata, in exact embedding order.
            "tile_metadata": (
                tile_metadata_df
            ),

            "coord_csv": str(
                coord_csv
            ),

            "slide_path": (
                slide_path
            ),

            "coordinate_source": "native",

            # Permanently record the coordinate interpretation.
            "coordinate_resolution": (
                dataset.coordinate_resolution
            ),

            "coordinate_units": (
                dataset.coordinate_units
            ),

            "coordinate_space": (
                "requested_resolution"
            ),

            "read_coord_space": (
                "resolution"
            ),
        },
        outpath,
    )

    print(
        f"\nCompleted: {slide_id}"
    )
    print(
        f"  Tiles: "
        f"{embedding_tensor.shape[0]:,}"
    )
    print(
        f"  Features: "
        f"{embedding_tensor.shape[1]:,}"
    )
    print(
        f"  Saved: "
        f"{outpath}"
    )

    return {
        "slide_id": slide_id,
        "donor_id": donor_id,
        "region": region,
        "stain": stain,
        "magnification": magnification,
        "embedding_file": str(
            outpath
        ),
        "coord_csv": str(
            coord_csv
        ),
        "slide_path": (
            slide_path
        ),
        "n_tiles": int(
            embedding_tensor.shape[0]
        ),
        "embedding_dim": int(
            embedding_tensor.shape[1]
        ),
        "embedding_model": (
            "UNI2-h"
        ),
        "coordinate_source": (
            "native"
        ),
        "read_coord_space": (
            "resolution"
        ),
        "status": (
            "completed"
        ),
    }



# Handling possible differences in metadata column naming for reprdoucibility
def get_coordinate_column(
    metadata,
):
    """
    Allow either coordinate-summary naming convention.
    """

    if "coord_file" in metadata.columns:
        return "coord_file"

    if "coord_csv" in metadata.columns:
        return "coord_csv"

    if "output_coord_file" in metadata.columns:
        return "output_coord_file"

    raise ValueError(
        "Input metadata must contain one of:\n"
        "  coord_file\n"
        "  coord_csv\n"
        "  output_coord_file"
    )


def load_coordinate_files(
    metadata_path,
    smoke_test=False,
):
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata does not exist:\n"
            f"{metadata_path}"
        )

    metadata = pd.read_csv(
        metadata_path
    )

    coord_column = get_coordinate_column(
        metadata
    )

    metadata = metadata.dropna(
        subset=[
            coord_column
        ]
    ).copy()

    if smoke_test:
        metadata = (
            metadata
            .head(1)
            .copy()
        )

    coordinate_files = []

    for _, row in metadata.iterrows():
        coord_path = Path(
            str(row[coord_column])
        )

        if coord_path.exists():
            coordinate_files.append(
                coord_path
            )

        else:
            print(
                f"WARNING: coordinate CSV "
                f"does not exist:\n"
                f"{coord_path}"
            )

    if len(coordinate_files) == 0:
        raise ValueError(
            "No valid coordinate CSV files "
            "were found."
        )

    return coordinate_files


# If a slide's embeddings exist, update it instead of creating a duplicate
def append_row_to_index(
    row,
    index_path,
):
    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_df = pd.DataFrame(
        [row]
    )

    if not index_path.exists():
        row_df.to_csv(
            index_path,
            index=False,
        )

        return

    existing = pd.read_csv(
        index_path
    )

    # Update an existing slide rather than creating duplicate rows.
    if (
        row["slide_id"]
        in existing["slide_id"].astype(str).values
    ):
        existing = existing.loc[
            existing["slide_id"].astype(str)
            != str(row["slide_id"])
        ].copy()

        existing = pd.concat(
            [
                existing,
                row_df,
            ],
            ignore_index=True,
        )

        existing.to_csv(
            index_path,
            index=False,
        )

        print(
            f"Updated index row: "
            f"{row['slide_id']}"
        )

    else:
        row_df.to_csv(
            index_path,
            mode="a",
            header=False,
            index=False,
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nUNI2 embedding extraction")
    print("=========================")
    print(
        f"Device: {device}"
    )
    print(
        f"Metadata: "
        f"{args.metadata}"
    )
    print(
        f"Embedding root: "
        f"{args.embedding_root}"
    )
    print(
        f"Index output: "
        f"{args.index_out}"
    )
    print(
        f"Batch size: "
        f"{args.batch_size}"
    )
    print(
        f"Num workers: "
        f"{args.num_workers}"
    )
    print(
        "Coordinate interpretation: "
        "resolution"
    )
    print(
        f"Overwrite: "
        f"{args.overwrite}"
    )
    print(
        f"Smoke test: "
        f"{args.smoke_test}"
    )

    args.embedding_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.index_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coordinate_files = (
        load_coordinate_files(
            metadata_path=args.metadata,
            smoke_test=args.smoke_test,
        )
    )

    print(
        f"\nFound "
        f"{len(coordinate_files)} "
        f"coordinate CSV(s)."
    )

    model, transform = (
        load_uni2_model(
            device=device
        )
    )

    for slide_number, coord_csv in enumerate(
        coordinate_files,
        start=1,
    ):
        print(
            "\n"
            + "=" * 70
        )
        print(
            f"[{slide_number}/"
            f"{len(coordinate_files)}] "
            f"{coord_csv.name}"
        )
        print(
            "=" * 70
        )

        try:
            result = (
                extract_embeddings_for_csv(
                    coord_csv=coord_csv,
                    model=model,
                    transform=transform,
                    embedding_root=(
                        args.embedding_root
                    ),
                    batch_size=(
                        args.batch_size
                    ),
                    num_workers=(
                        args.num_workers
                    ),
                    device=device,
                    overwrite=(
                        args.overwrite
                    ),
                )
            )

        except Exception as error:
            print(
                f"\nERROR processing "
                f"{coord_csv}"
            )
            print(error)

            result = {
                "slide_id": (
                    coord_csv.stem
                ),
                "donor_id": None,
                "region": None,
                "stain": None,
                "magnification": None,
                "embedding_file": None,
                "coord_csv": str(
                    coord_csv
                ),
                "slide_path": None,
                "n_tiles": None,
                "embedding_dim": None,
                "embedding_model": (
                    "UNI2-h"
                ),
                "coordinate_source": (
                    "native"
                ),
                "read_coord_space": (
                    "resolution"
                ),
                "status": (
                    f"error: {error}"
                ),
            }

        append_row_to_index(
            result,
            args.index_out,
        )

    print("\nEmbedding extraction complete")
    print("=============================")
    print(
        f"Embedding index:\n"
        f"{args.index_out}"
    )


if __name__ == "__main__":
    main()
