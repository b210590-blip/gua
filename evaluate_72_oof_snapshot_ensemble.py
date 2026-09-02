from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG, apply_runtime_profile
from data_pipeline import (
    ColdStartStationDataset,
    build_or_load_hourly_cube,
    haversine_matrix,
    load_static,
    standardize_static,
)
from evaluate_outer import load_checkpoint, scaler_from_checkpoint
from evaluate_validation_only_snapshot_ensemble import EPOCHS, aligned_predictions, station_rmse
from model import TCNTargetCrossAttention
from sanity_check import audit_dataset
from train_formal import (
    DeviceFeatureBuilder,
    make_index_loader,
    make_vectorized_loader,
    regression_metrics,
    resolve_target,
)


def find_metrics_root() -> Path:
    explicit = os.environ.get("DL_TCN_EPOCH_SELECTOR_72_ROOT", "").strip()
    candidates = [Path(explicit)] if explicit else []
    candidates.extend([
        Path("/content/epoch_selector_72_minimal"),
        Path("/content/epoch_selector_72_minimal/crossfit"),
    ])
    for candidate in candidates:
        root = candidate / "crossfit" if (candidate / "crossfit").is_dir() else candidate
        files = [root / f"fold_{fold:02d}" / "validation_station_metrics_all_epochs.csv" for fold in range(6)]
        if all(path.is_file() for path in files):
            return root
    raise FileNotFoundError(
        "找不到epoch_selector_72_minimal解壓後的6個metrics CSV；"
        "請用DL_TCN_EPOCH_SELECTOR_72_ROOT指定資料夾。"
    )


def load_72_metric_curves(root: Path) -> pd.DataFrame:
    parts = []
    for fold in range(6):
        path = root / f"fold_{fold:02d}" / "validation_station_metrics_all_epochs.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["fold"] = fold
        parts.append(frame)
    history = pd.concat(parts, ignore_index=True)
    history["station_index"] = history.station_index.astype(int)
    history["epoch"] = history.epoch.astype(int)
    if history.station_index.nunique() != 72:
        raise RuntimeError("metrics ZIP不是72個唯一測站")
    if not (history.groupby("station_index").epoch.nunique() == 15).all():
        raise RuntimeError("不是每站都有epoch 1..15 metrics")
    return history


def candidate_weights_from_60(history: pd.DataFrame) -> list[dict]:
    training = history[history.fold != 0]
    if training.station_index.nunique() != 60:
        raise RuntimeError("fold 1..5合併後不是60個meta-training stations")
    pivot = training.pivot(index="station_index", columns="epoch", values="rmse").loc[:, list(EPOCHS)]
    values = pivot.to_numpy(float)
    regret = values - values.min(axis=1, keepdims=True)
    macro_regret = regret.mean(axis=0)
    ranked = np.argsort(macro_regret, kind="stable")
    candidates = []

    for top_k in (2, 3, 5, 8, 15):
        weight = np.zeros(len(EPOCHS), dtype=float)
        weight[ranked[:top_k]] = 1.0 / top_k
        candidates.append({"family": "uniform_top_k", "parameter": float(top_k), "weight": weight})

    positive = macro_regret[macro_regret > 1e-12]
    scale = float(np.median(positive)) if len(positive) else 1.0
    for multiplier in (0.25, 0.5, 1.0, 2.0):
        temperature = max(scale * multiplier, 1e-8)
        logits = -macro_regret / temperature
        logits -= logits.max()
        weight = np.exp(logits)
        weight /= weight.sum()
        candidates.append({"family": "softmax_macro_regret", "parameter": multiplier, "weight": weight})

    best_epoch_position = np.argmin(values, axis=1)
    frequency = np.bincount(best_epoch_position, minlength=len(EPOCHS)).astype(float)
    frequency /= frequency.sum()
    candidates.append({"family": "best_epoch_frequency", "parameter": 0.0, "weight": frequency})
    return candidates


def build_context(root: Path):
    paths = [root / "fold_00" / "epoch_checkpoints" / f"epoch_{epoch:03d}.pt" for epoch in EPOCHS]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"fold_00仍缺少{len(missing)}個checkpoint；目前只需要這15個，不需要其他fold。"
        )
    static, _, static_cols = load_static()
    outer = resolve_target(static, CFG.target_site)
    first = load_checkpoint(paths[0])
    train_idx = np.asarray(first["train_indices"], dtype=int)
    validation_idx = np.asarray(first["validation_indices"], dtype=int)
    if len(train_idx) != 60 or len(validation_idx) != 12:
        raise RuntimeError("fold_00 checkpoint不是60 train / 12 validation")
    if outer in train_idx or outer in validation_idx:
        raise RuntimeError("桃園混入fold_00 train/validation")

    scaler = scaler_from_checkpoint(first)
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)
    static_scaled = standardize_static(static, static_cols, scaler)
    validation_ds = ColdStartStationDataset(
        validation_idx, train_idx, CFG.train_start, CFG.train_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    outer_ds = ColdStartStationDataset(
        [outer], train_idx, CFG.test_start, CFG.test_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    if audit_dataset(validation_ds, max_samples=32)["sampled_donor_counts"] != [60]:
        raise RuntimeError("fold_00 validation donors不是60")
    if audit_dataset(outer_ds, max_samples=32)["sampled_donor_counts"] != [60]:
        raise RuntimeError("桃園 donors不是60")

    validation_builder = outer_builder = None
    if CFG.device.type == "cuda":
        max_time = int(max(validation_ds.row_times.max(), outer_ds.row_times.max()))
        validation_builder = DeviceFeatureBuilder(
            train_idx, cube, max_time, timestamps, static, static_scaled,
            distance, scaler, outer, CFG.device,
        )
        outer_builder = validation_builder
        validation_loader = make_index_loader(validation_ds, False)
        outer_loader = make_index_loader(outer_ds, False)
    else:
        validation_loader = make_vectorized_loader(
            validation_ds, train_idx, cube, timestamps, static,
            static_scaled, distance, scaler, False,
        )
        outer_loader = make_vectorized_loader(
            outer_ds, train_idx, cube, timestamps, static,
            static_scaled, distance, scaler, False,
        )
    return (
        paths, static, static_cols, timestamps, outer, validation_idx,
        validation_loader, outer_loader, validation_builder, outer_builder,
    )


def main() -> None:
    apply_runtime_profile(CFG)
    metrics_root = find_metrics_root()
    history = load_72_metric_curves(metrics_root)
    candidates = candidate_weights_from_60(history)
    checkpoint_root = Path(os.environ.get(
        "DL_TCN_CROSSFIT_ROOT",
        str(CFG.formal_output_root / "crossfit_target_conditioned_snapshots"),
    ))
    (
        paths, static, static_cols, timestamps, outer, validation_idx,
        validation_loader, outer_loader, validation_builder, outer_builder,
    ) = build_context(checkpoint_root)

    expected_fold0 = set(history.loc[history.fold == 0, "station_index"].unique().astype(int))
    if set(validation_idx.astype(int)) != expected_fold0:
        raise RuntimeError("metrics ZIP的fold_00與checkpoint的12 validation stations不一致")

    model = TCNTargetCrossAttention(static_dim=len(static_cols)).to(CFG.device)
    print("Infer fold_00 validation snapshots; 桃園truth尚未讀取。", flush=True)
    validation_base, validation_matrix = aligned_predictions(
        model, paths, validation_loader, CFG.device, static, timestamps, validation_builder
    )
    validation_stations = np.unique(validation_base.station_index.to_numpy(int))
    scored = []
    for candidate in candidates:
        prediction = validation_matrix @ candidate["weight"]
        held_rmse = [
            station_rmse(validation_base, prediction, int(station))
            for station in validation_stations
        ]
        scored.append({**candidate, "held12_macro_rmse": float(np.mean(held_rmse))})
    selected = min(scored, key=lambda row: row["held12_macro_rmse"])
    weights = selected["weight"]
    lock = {
        "protocol": "60 stations from metric folds 1..5 create weights; fold_00 12 stations select candidate; lock before Taoyuan truth",
        "selection_used_taoyuan_truth": False,
        "meta_training_stations": 60,
        "meta_validation_stations": 12,
        "candidate_scores": [
            {"family": row["family"], "parameter": row["parameter"], "held12_macro_rmse": row["held12_macro_rmse"]}
            for row in sorted(scored, key=lambda row: row["held12_macro_rmse"])
        ],
        "selected_method": {
            "family": selected["family"], "parameter": selected["parameter"],
            "held12_macro_rmse": selected["held12_macro_rmse"],
        },
        "weights": [
            {"epoch": epoch, "weight": float(weight)}
            for epoch, weight in zip(EPOCHS, weights) if weight > 1e-8
        ],
    }
    print("\nLOCKED 60+12 ENSEMBLE BEFORE TAOYUAN TRUTH")
    print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)

    print("\nOuter inference after weights are locked.", flush=True)
    outer_base, outer_matrix = aligned_predictions(
        model, paths, outer_loader, CFG.device, static, timestamps, outer_builder
    )
    prediction = outer_matrix @ weights
    result = {
        "target_station_index": int(outer),
        "siteid": str(static.loc[outer, "siteid"]),
        "sitename": str(static.loc[outer, "sitename"]),
        "selection_used_taoyuan_truth": False,
        "metrics": regression_metrics(outer_base.y_true.to_numpy(float), prediction),
        "selected_method": lock["selected_method"],
        "weights": lock["weights"],
        "files_written": 0,
    }
    print("\nTAOYUAN 72-STATION-INFORMED ENSEMBLE RESULT")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
