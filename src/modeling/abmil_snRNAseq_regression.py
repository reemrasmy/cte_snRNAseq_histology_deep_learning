#!/usr/bin/env python3

"""
ABMIL regression for predicting donor-level snRNA-seq continuous targets
from precomputed UNI2 whole-slide histology embeddings.

Handles metadata loading, target scaling, model training, cross-validation,
evaluation, and attention-to-tile alignment.

Visualization is handled by:
    - src/visualization/attention_heatmaps.py
    - src/visualization/high_attention_tiles.py
    - src/visualization/regression_plots.py

Current Implementation: 
    - Supports one embedding file / WSI per donor.
    - Supports one or more continuous snRNA-seq target columns.
    - Does not yet support combining multiple embedding files or stains
      for the same donor in a single model input.

Example Run: 
    python -m src.modeling.abmil_snRNAseq_regression \
        --metadata /path/to/embedding_target_metadata.csv \     -- can be created using src/data_processing/regression_data_prep.py
        --output-dir /path/to/results/regression/... \
        --stains IBA1 \
        --coordinate-source native \    --- optional
        --filter-name white075 \        --- optional 
        --target-name microglia_pc1_pc2 \
        --targets pc1_mean pc2_mean

Run with --help to see all available options.

"""

############################## Imports ##############################

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from final_src.modeling.cte_abmil import ABMILRegressor
from final_src.visualization.attention_heatmaps import save_attention_heatmap
from final_src.visualization.high_attention_tiles import save_top_attention_tiles
from final_src.visualization.regression_plots import save_prediction_plot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
HEAD_TYPE = "mlp"

############################## Command-line Arguments ##############################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train donor-level ABMIL regression from precomputed WSI embeddings."
    )

    # Input/output
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Donor-level regression metadata CSV.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Base directory for regression results.")

    # Experiment description
    parser.add_argument("--stains", nargs="+", required=True,
                        help="Histology stain(s), e.g. IBA1 or IBA1 AT8.")
    parser.add_argument("--target-name", required=True,
                        help="Descriptive name for the biological target set used for organizing results, e.g. microglia_pc1_pc2.")
    parser.add_argument("--coordinate-source", default=None,
                        help="Optional coordinate source, e.g. native or transferred_from_lhe.")
    parser.add_argument("--filter-name", default=None,
                        help="Optional filtering description, e.g. white075.")
    parser.add_argument("--run-name", default="baseline",
                        help="Optional experiment description. Default: baseline.")

    # Targets: allowing for 1 or more predictive columns
    parser.add_argument("--targets", nargs="+", required=True,
        help="One or more continuous snRNA-seq target columns from the metadata.",
    )


    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--input-dim", type=int, default=1536)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)

    # Attention
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of highest-attention tiles to save.")
    parser.add_argument("--no-attention", action="store_true",
                        help="Skip attention CSVs, top tiles, and heatmaps.")

    return parser.parse_args()

############################## Output Setup ##############################

def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)

def create_output_dirs(args: argparse.Namespace) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stain_label = safe_name("_".join(stain.upper() for stain in args.stains))
    results_dir = args.output_dir / stain_label

    if args.coordinate_source:
        results_dir /= safe_name(args.coordinate_source)
    if args.filter_name:
        results_dir /= safe_name(args.filter_name)

    results_dir /= safe_name(args.target_name)
    results_dir /= f"{safe_name(args.run_name)}_{timestamp}"

    paths = {
        "results": results_dir,
        "checkpoints": results_dir / "checkpoints",
        "plots": results_dir / "plots",
        "attention": results_dir / "attention_scores",
        "top_tiles": results_dir / "top_attention_tiles",
        "heatmaps": results_dir / "attention_heatmaps",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths

def save_run_config(args: argparse.Namespace, results_dir: Path) -> None:
    config = vars(args).copy()
    config["metadata"] = str(config["metadata"])
    config["output_dir"] = str(config["output_dir"])
    config["command"] = " ".join(sys.argv)
    config["device"] = str(DEVICE)

    with open(results_dir / "run_config.json", "w") as handle:
        json.dump(config, handle, indent=4)

############################## Reproducibility ##############################

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

############################## Metadata ##############################

def load_metadata(metadata_csv: Path, target_columns: list[str]) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file was not found:\n{metadata_csv}")

    df = pd.read_csv(metadata_csv)
    required = {"donor_id", "slide_id", "embedding_file", "coord_csv", "slide_path", *target_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=list(required)).copy()
    for target in target_columns:
        df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=target_columns).copy()

    for column in ["embedding_file", "coord_csv", "slide_path"]:
        df[column] = df[column].astype(str)

    embedding_exists = df["embedding_file"].apply(lambda path: Path(path).exists())
    if not embedding_exists.all():
        missing_paths = df.loc[~embedding_exists, "embedding_file"].tolist()
        print(f"\nWarning: {len(missing_paths)} embedding files were not found.")
        for path in missing_paths[:10]:
            print(f"  Missing: {path}")
        df = df.loc[embedding_exists].copy()

    df = (df.sort_values(["donor_id", "slide_id"])
          .drop_duplicates(subset="donor_id", keep="first")
          .reset_index(drop=True))

    duplicated = df.loc[df["donor_id"].duplicated(keep=False), "donor_id"].unique()
    if len(duplicated) > 0:
        raise ValueError(
            "Regression metadata contains multiple rows for some donors. "
            f"Example donors: {duplicated[:10]}"
        )

    print("\nRegression metadata summary")
    print("---------------------------")
    print(f"Rows retained: {len(df)}")
    print(f"Unique donors: {df['donor_id'].nunique()}")
    print("\nTarget summary:")
    print(df[target_columns].describe())
    return df

############################## Dataset ##############################

class RegressionBagDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, scaled_target_columns: list[str], input_dim: int):
        self.df = metadata_df.reset_index(drop=True)
        self.scaled_target_columns = scaled_target_columns
        self.input_dim = input_dim

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        embedding_path = Path(row["embedding_file"])
        loaded = torch.load(embedding_path, map_location="cpu")

        if "embeddings" not in loaded:
            raise KeyError(f"'embeddings' was not found in {embedding_path}")

        tile_embeddings = loaded["embeddings"].float()
        if tile_embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings for {row['donor_id']}, "
                f"received {tuple(tile_embeddings.shape)}"
            )
        if tile_embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected embedding dimension {self.input_dim}, "
                f"but {row['donor_id']} has {tile_embeddings.shape[1]}"
            )

        target = torch.tensor(
            row[self.scaled_target_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        return (
            tile_embeddings, target, str(row["donor_id"]), str(row["slide_id"]),
            str(row["embedding_file"]), str(row["coord_csv"]), str(row["slide_path"]),
        )

############################## Training ##############################

def train_one_epoch(model, dataloader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for tile_embeddings, target, *_ in dataloader:
        tile_embeddings = tile_embeddings.squeeze(0).to(device)
        target = target.squeeze(0).to(device)
        optimizer.zero_grad()
        prediction, _ = model(tile_embeddings)

        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} does not match target shape {target.shape}"
            )

        loss = criterion(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_scaled_loss(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for tile_embeddings, target, *_ in dataloader:
            tile_embeddings = tile_embeddings.squeeze(0).to(device)
            target = target.squeeze(0).to(device)
            prediction, _ = model(tile_embeddings)
            total_loss += criterion(prediction, target).item()

    return total_loss / len(dataloader)

############################## Attention Metadata ##############################

def _tile_metadata_from_embedding(embedding_file: str, coord_csv: str) -> pd.DataFrame:
    payload = torch.load(embedding_file, map_location="cpu")
    tile_metadata = payload.get("tile_metadata")

    if tile_metadata is None:
        coord_df = pd.read_csv(coord_csv)
    elif isinstance(tile_metadata, pd.DataFrame):
        coord_df = tile_metadata.copy()
    elif isinstance(tile_metadata, dict):
        coord_df = pd.DataFrame(tile_metadata)
    else:
        coord_df = pd.DataFrame(list(tile_metadata))

    coord_df = coord_df.reset_index(drop=True)
    defaults = {"tile_width": 256, "tile_height": 256, "resolution": 0.5, "units": "mpp"}
    for column, value in defaults.items():
        if column not in coord_df.columns:
            coord_df[column] = value
    return coord_df

def save_attention_outputs(
    donor_id: str, slide_id: str, embedding_file: str, coord_csv: str,
    slide_path: str, attention_scores: np.ndarray, true_values: np.ndarray,
    pred_values: np.ndarray, target_names: list[str], fold_number: int,
    selected_epoch: int, output_dirs: dict[str, Path],
    checkpoint_type: str = "lowest_val_loss", top_k: int = 20,
) -> dict:

    coord_df = _tile_metadata_from_embedding(embedding_file, coord_csv)
    attention_scores = np.asarray(attention_scores).reshape(-1)

    if len(coord_df) != len(attention_scores):
        raise ValueError(
            f"{slide_id}: {len(coord_df)} tile rows but {len(attention_scores)} attention scores."
        )

    required = {"x_start", "y_start", "tile_width", "tile_height", "resolution", "units"}
    missing = required.difference(coord_df.columns)
    if missing:
        raise ValueError(f"{slide_id} tile metadata is missing: {sorted(missing)}")

    coord_df["attention_score"] = attention_scores
    coord_df["donor_id"] = donor_id
    coord_df["slide_id"] = slide_id
    coord_df["fold"] = fold_number
    coord_df["checkpoint_type"] = checkpoint_type
    coord_df["selected_epoch"] = selected_epoch
    coord_df["slide_path"] = slide_path
    coord_df["embedding_file"] = embedding_file
    coord_df["coord_csv"] = coord_csv

    for i, target_name in enumerate(target_names):
        coord_df[f"true_{target_name}"] = float(true_values[i])
        coord_df[f"pred_{target_name}"] = float(pred_values[i])

    fold_attention_dir = output_dirs["attention"] / f"fold_{fold_number}" / checkpoint_type
    fold_attention_dir.mkdir(parents=True, exist_ok=True)

    attention_csv = fold_attention_dir / f"{donor_id}_{slide_id}_attention_scores.csv"
    coord_df.to_csv(attention_csv, index=False)

    tile_dir = (
        output_dirs["top_tiles"] / f"fold_{fold_number}" /
        checkpoint_type / donor_id / slide_id
    )
    top_tile_summary = fold_attention_dir / f"{donor_id}_{slide_id}_top_tiles_summary.csv"

    save_top_attention_tiles(
        attention_df=coord_df,
        slide_path=slide_path,
        tile_output_dir=tile_dir,
        summary_output_path=top_tile_summary,
        top_k=top_k,
        coordinate_mode="level0",
    )

    heatmap_dir = (
        output_dirs["heatmaps"] / f"fold_{fold_number}" /
        checkpoint_type / donor_id
    )
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    raw_thumbnail = heatmap_dir / f"{slide_id}_raw_thumbnail.png"
    heatmap = heatmap_dir / f"{slide_id}_attention_heatmap.png"

    save_attention_heatmap(
        attention_df=coord_df,
        slide_path=slide_path,
        heatmap_output_path=heatmap,
        raw_thumbnail_output_path=raw_thumbnail,
        title=f"{donor_id} | {slide_id} | Fold {fold_number} | Epoch {selected_epoch}",
    )

    return {
        "attention_csv": str(attention_csv),
        "top_tile_summary_csv": str(top_tile_summary),
        "raw_thumbnail_png": str(raw_thumbnail),
        "heatmap_png": str(heatmap),
    }

############################## Predictions ##############################

def collect_predictions(
    model, dataloader, target_scaler, device, fold_number: int,
    selected_epoch: int, target_names: list[str], output_dirs: dict[str, Path],
    top_k: int, save_attention: bool = True,
    checkpoint_type: str = "lowest_val_loss",
) -> pd.DataFrame:

    model.eval()
    records = []

    with torch.no_grad():
        for (
            tile_embeddings, scaled_target, donor_id, slide_id,
            embedding_file, coord_csv, slide_path,
        ) in dataloader:

            tile_embeddings = tile_embeddings.squeeze(0).to(device)
            scaled_prediction, attention_weights = model(tile_embeddings)

            scaled_prediction_np = scaled_prediction.detach().cpu().numpy().reshape(1, -1)
            scaled_target_np = scaled_target.squeeze(0).cpu().numpy().reshape(1, -1)
            prediction = target_scaler.inverse_transform(scaled_prediction_np)[0]
            target = target_scaler.inverse_transform(scaled_target_np)[0]

            attention_paths = {
                "attention_csv": None, "top_tile_summary_csv": None,
                "raw_thumbnail_png": None, "heatmap_png": None,
            }

            if save_attention:
                attention_scores = attention_weights.detach().cpu().numpy().reshape(-1)
                attention_paths = save_attention_outputs(
                    donor_id=donor_id[0],
                    slide_id=slide_id[0],
                    embedding_file=embedding_file[0],
                    coord_csv=coord_csv[0],
                    slide_path=slide_path[0],
                    attention_scores=attention_scores,
                    true_values=target,
                    pred_values=prediction,
                    target_names=target_names,
                    fold_number=fold_number,
                    selected_epoch=selected_epoch,
                    output_dirs=output_dirs,
                    checkpoint_type=checkpoint_type,
                    top_k=top_k,
                )

            record = {
                "donor_id": donor_id[0],
                "slide_id": slide_id[0],
                "embedding_file": embedding_file[0],
                "coord_csv": coord_csv[0],
                "slide_path": slide_path[0],
                "max_attention": attention_weights.max().item(),
                "attention_sum": attention_weights.sum().item(),
                **attention_paths,
            }

            for i, target_name in enumerate(target_names):
                record[f"true_{target_name}"] = float(target[i])
                record[f"pred_{target_name}"] = float(prediction[i])

            records.append(record)

    return pd.DataFrame(records)

############################## Metrics ##############################

def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(pearsonr(y_true, y_pred).statistic)

def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(spearmanr(y_true, y_pred).statistic)

def calculate_regression_metrics(predictions_df: pd.DataFrame, target_name: str) -> dict:
    y_true = predictions_df[f"true_{target_name}"].to_numpy()
    y_pred = predictions_df[f"pred_{target_name}"].to_numpy()
    mse = mean_squared_error(y_true, y_pred)

    return {
        f"{target_name}_MAE": mean_absolute_error(y_true, y_pred),
        f"{target_name}_RMSE": np.sqrt(mse),
        f"{target_name}_R2": r2_score(y_true, y_pred),
        f"{target_name}_Pearson": safe_pearson(y_true, y_pred),
        f"{target_name}_Spearman": safe_spearman(y_true, y_pred),
    }

############################## Cross Validation ##############################

def run_cross_validation(
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dirs: dict[str, Path],
):
    splitter = GroupKFold(n_splits=args.n_splits)
    scaled_target_columns = [f"target_{i + 1}_scaled" for i in range(len(args.targets))]
    fold_metrics, all_oof_predictions = [], []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X=df, groups=df["donor_id"]), start=1
    ):
        print("\n" + "=" * 60)
        print(f"Starting fold {fold}")
        print("=" * 60)

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        print(f"Training donors: {len(train_df)}")
        print(f"Validation donors: {len(val_df)}")

        target_scaler = StandardScaler()
        train_df[scaled_target_columns] = target_scaler.fit_transform(train_df[args.targets])
        val_df[scaled_target_columns] = target_scaler.transform(val_df[args.targets])

        train_dataset = RegressionBagDataset(train_df, scaled_target_columns, args.input_dim)
        val_dataset = RegressionBagDataset(val_df, scaled_target_columns, args.input_dim)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = ABMILRegressor(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            output_dim=len(args.targets),
            dropout=args.dropout,
        ).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state = None
        epochs_without_improvement = 0
        training_history = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss = evaluate_scaled_loss(model, val_loader, criterion, DEVICE)
            training_history.append({
                "fold": fold, "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
            })

            print(
                f"Fold {fold} | Epoch {epoch:03d}/{args.epochs} | "
                f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

        if best_model_state is None:
            raise RuntimeError(f"No model checkpoint was created for fold {fold}")

        model.load_state_dict(best_model_state)

        fold_predictions = collect_predictions(
            model=model,
            dataloader=val_loader,
            target_scaler=target_scaler,
            device=DEVICE,
            fold_number=fold,
            selected_epoch=best_epoch,
            target_names=args.targets,
            output_dirs=output_dirs,
            top_k=args.top_k,
            save_attention=not args.no_attention,
            checkpoint_type="lowest_val_loss",
        )

        fold_predictions["fold"] = fold
        fold_predictions["best_epoch"] = best_epoch

        target_metrics = {}

        for target_name in args.targets:
            target_metrics.update(
                calculate_regression_metrics(fold_predictions, target_name)
            )

        current_metrics = {
            "fold": fold,
            "best_epoch": best_epoch,
            "best_scaled_val_mse": best_val_loss,
            **target_metrics,
        }

        fold_metrics.append(current_metrics)
        all_oof_predictions.append(fold_predictions)

        print(f"\nFold {fold} metrics")
        for name, value in current_metrics.items():
            if name not in {"fold", "best_epoch"}:
                print(f"{name}: {value:.4f}")

        pd.DataFrame(training_history).to_csv(
            output_dirs["results"] / f"fold_{fold}_training_history.csv", index=False
        )

        checkpoint = {
            "fold": fold,
            "best_epoch": best_epoch,
            "model_state_dict": best_model_state,
            "target_mean": target_scaler.mean_.tolist(),
            "target_scale": target_scaler.scale_.tolist(),
            "target_columns": args.targets,
            "model_config": {
                "input_dim": args.input_dim,
                "hidden_dim": args.hidden_dim,
                "mlp_hidden_dim": args.mlp_hidden_dim,
                "output_dim": len(args.targets),
                "dropout": args.dropout,
                "head_type": HEAD_TYPE,
            },
        }

        torch.save(
            checkpoint,
            output_dirs["checkpoints"] / f"fold_{fold}_best_model.pt",
        )

    return (
        pd.DataFrame(fold_metrics),
        pd.concat(all_oof_predictions, ignore_index=True),
    )

############################## Save Results ##############################

def save_results(
    fold_metrics_df: pd.DataFrame,
    oof_predictions_df: pd.DataFrame,
    target_names: list[str],
    output_dirs: dict[str, Path],
) -> None:

    results_dir = output_dirs["results"]
    fold_metrics_path = results_dir / "fold_metrics.csv"
    predictions_path = results_dir / "oof_predictions.csv"
    pooled_metrics_path = results_dir / "pooled_oof_metrics.json"

    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    oof_predictions_df.to_csv(predictions_path, index=False)

    pooled_metrics = {}
    for target_name in target_names:
        pooled_metrics.update(
            calculate_regression_metrics(oof_predictions_df, target_name)
        )

        save_prediction_plot(
            oof_predictions_df,
            target_name,
            output_dirs["plots"]
            / f"{safe_name(target_name.lower())}_observed_vs_predicted.png",
        )

    with open(pooled_metrics_path, "w") as handle:
        json.dump(pooled_metrics, handle, indent=4)

    print("\n" + "=" * 60)
    print("Pooled out-of-fold metrics")
    print("=" * 60)
    for name, value in pooled_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nSaved:")
    print(f"  {fold_metrics_path}")
    print(f"  {predictions_path}")
    print(f"  {pooled_metrics_path}")
    print(f"  {output_dirs['plots']}")

############################## Main ##############################

def main() -> None:
    args = parse_args()

    # Current implementation expects one embedding bag per donor.
    if len(args.stains) > 1:
        raise ValueError(
            "Multiple stains were specified, but the current regression dataset "
            "supports one embedding_file per donor. Multi-stain modeling is not yet implemented."
        )

    set_seed(args.seed)
    output_dirs = create_output_dirs(args)
    save_run_config(args, output_dirs["results"])

    print(f"Using device: {DEVICE}")
    print(f"Stain(s): {', '.join(args.stains)}")
    print(f"Targets: {args.targets}")
    print(f"Results directory:\n{output_dirs['results']}")

    metadata_df = load_metadata(args.metadata, args.targets)
    fold_metrics_df, oof_predictions_df = run_cross_validation(
        metadata_df, args, output_dirs
    )
    save_results(
        fold_metrics_df, oof_predictions_df,
        args.targets, output_dirs
    )

    print("\nRegression modeling complete.")

if __name__ == "__main__":
    main()


"""
Test run: 

python -m final_src.modeling.abmil_snRNAseq_regression \
    --metadata /restricted/projectnb/cteseq/users/rrasmy/cte_snRNAseq_image_transcriptomics_model/metadata/regression/IBA1/native/white_filter/threshold_075/iba1_micro_pc_regression_metadata_w_lhe_and_002_replacements.csv \
    --output-dir /restricted/projectnb/cteseq/users/rrasmy/cte_snRNAseq_image_transcriptomics_model/results/modeling_results/pc_regression \
    --stains IBA1 \
    --coordinate-source native \
    --filter-name white075 \
    --target-name microglia_pc1_pc2 \
    --targets pc1_mean pc2_mean \
    --run-name argparse_test
"""


