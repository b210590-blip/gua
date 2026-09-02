from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CFG, apply_runtime_profile
from data_pipeline import (
    ColdStartStationDataset,
    build_or_load_hourly_cube,
    haversine_matrix,
    load_static,
    standardize_static,
)
from evaluate_outer import load_checkpoint, scaler_from_checkpoint
from evaluate_validation_only_snapshot_ensemble import (
    EPOCHS,
    RIDGE_CANDIDATES,
    TOP_K_CANDIDATES,
    aligned_predictions,
    fit_convex_weights,
    station_rmse_by_epoch,
)
from model import TCNTargetCrossAttention
from sanity_check import audit_dataset
from train_formal import (
    DeviceFeatureBuilder,
    make_index_loader,
    make_vectorized_loader,
    regression_metrics,
    resolve_target,
)


def prediction_files(fold_dir: Path) -> list[Path]:
    return [fold_dir / "epoch_predictions" / f"epoch_{epoch:03d}.npz" for epoch in EPOCHS]


def checkpoint_files(fold_dir: Path) -> list[Path]:
    return [fold_dir / "epoch_checkpoints" / f"epoch_{epoch:03d}.pt" for epoch in EPOCHS]


def load_saved_prediction_matrix(paths: list[Path]) -> tuple[pd.DataFrame, np.ndarray]:
    base = None
    columns = []
    for epoch, path in zip(EPOCHS, paths):
        payload = np.load(path)
        current = pd.DataFrame({
            "station_index": np.asarray(payload["station_index"], dtype=int),
            "timestamp": pd.to_datetime(np.asarray(payload["timestamp_ns"], dtype="int64")),
            "y_true": np.asarray(payload["y_true"], dtype="float64"),
        })
        prediction = np.asarray(payload["y_pred"], dtype="float64")
        if base is None:
            base = current
        else:
            if not np.array_equal(base.station_index.to_numpy(), current.station_index.to_numpy()):
                raise RuntimeError(f"epoch {epoch}: station alignment不同")
            if not np.array_equal(
                pd.to_datetime(base.timestamp).astype("int64").to_numpy(),
                pd.to_datetime(current.timestamp).astype("int64").to_numpy(),
            ):
                raise RuntimeError(f"epoch {epoch}: timestamp alignment不同")
            if not np.allclose(base.y_true, current.y_true, rtol=0, atol=1e-6):
                raise RuntimeError(f"epoch {epoch}: truth alignment不同")
        columns.append(prediction)
    assert base is not None
    return base, np.stack(columns, axis=1)


def infer_fold(
    fold: int,
    root: Path,
    static: pd.DataFrame,
    static_cols: list[str],
    cube,
    timestamps,
    distance,
    outer: int,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    fold_dir = root / f"fold_{fold:02d}"
    saved = prediction_files(fold_dir)
    if all(path.is_file() for path in saved):
        print(f"fold {fold}: load 15 saved validation prediction files", flush=True)
        return load_saved_prediction_matrix(saved)

    checkpoints = checkpoint_files(fold_dir)
    missing = [path.name for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"fold_{fold:02d} 沒有完整逐epoch predictions，也缺少{len(missing)}個checkpoint。"
            "無法只靠RMSE表計算ensemble。"
        )
    first = load_checkpoint(checkpoints[0])
    train_idx = np.asarray(first["train_indices"], dtype=int)
    validation_idx = np.asarray(first["validation_indices"], dtype=int)
    if len(train_idx) != 60 or len(validation_idx) != 12:
        raise RuntimeError(f"fold {fold}不是60 train / 12 validation")
    if outer in train_idx or outer in validation_idx:
        raise RuntimeError(f"fold {fold}混入outer target")
    if list(first["static_columns"]) != list(static_cols):
        raise RuntimeError(f"fold {fold} static columns不同")

    scaler = scaler_from_checkpoint(first)
    static_scaled = standardize_static(static, static_cols, scaler)
    dataset = ColdStartStationDataset(
        validation_idx,
        train_idx,
        CFG.train_start,
        CFG.train_end,
        cube,
        timestamps,
        static,
        static_scaled,
        distance,
        scaler,
    )
    if audit_dataset(dataset, max_samples=32)["sampled_donor_counts"] != [60]:
        raise RuntimeError(f"fold {fold} validation donors不是60")
    builder = None
    if device.type == "cuda":
        builder = DeviceFeatureBuilder(
            train_idx,
            cube,
            int(dataset.row_times.max()),
            timestamps,
            static,
            static_scaled,
            distance,
            scaler,
            outer,
            device,
        )
        loader = make_index_loader(dataset, False)
    else:
        loader = make_vectorized_loader(
            dataset,
            train_idx,
            cube,
            timestamps,
            static,
            static_scaled,
            distance,
            scaler,
            False,
        )
    model = TCNTargetCrossAttention(static_dim=len(static_cols)).to(device)
    print(f"fold {fold}: infer validation snapshots", flush=True)
    return aligned_predictions(
        model, checkpoints, loader, device, static, timestamps, builder
    )


def combine(folds: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    base_parts = []
    matrix_parts = []
    for item in folds:
        base_parts.append(item["base"])
        matrix_parts.append(item["matrix"])
    return pd.concat(base_parts, ignore_index=True), np.concatenate(matrix_parts, axis=0)


def weights_for_candidate(
    base: pd.DataFrame,
    matrix: np.ndarray,
    candidate: tuple[str, float],
) -> np.ndarray:
    family, parameter = candidate
    stations = np.unique(base.station_index.to_numpy(int))
    if family == "uniform_top_k":
        rmse_curve = station_rmse_by_epoch(base, matrix, stations)
        selected = np.argsort(rmse_curve.mean(axis=0), kind="stable")[: int(parameter)]
        weights = np.zeros(len(EPOCHS), dtype=float)
        weights[selected] = 1.0 / len(selected)
        return weights
    if family == "convex_ridge":
        return fit_convex_weights(base, matrix, stations, float(parameter))
    if family == "hard_epoch":
        rmse_curve = station_rmse_by_epoch(base, matrix, stations)
        selected = int(np.argmin(rmse_curve.mean(axis=0)))
        weights = np.zeros(len(EPOCHS), dtype=float)
        weights[selected] = 1.0
        return weights
    raise ValueError(candidate)


def macro_station_rmse(base: pd.DataFrame, prediction: np.ndarray) -> float:
    values = []
    station = base.station_index.to_numpy(int)
    truth = base.y_true.to_numpy(float)
    for index in np.unique(station):
        mask = station == index
        values.append(float(np.sqrt(np.mean((prediction[mask] - truth[mask]) ** 2))))
    return float(np.mean(values))


def choose_candidate_inner(training_folds: list[dict], candidates: list[tuple[str, float]]):
    scores = []
    for candidate in candidates:
        held_scores = []
        for inner in range(len(training_folds)):
            fit_folds = [item for position, item in enumerate(training_folds) if position != inner]
            fit_base, fit_matrix = combine(fit_folds)
            weights = weights_for_candidate(fit_base, fit_matrix, candidate)
            held = training_folds[inner]
            held_scores.append(macro_station_rmse(held["base"], held["matrix"] @ weights))
        scores.append((float(np.mean(held_scores)), candidate))
    return min(scores, key=lambda row: row[0]), scores


def summarize(base: pd.DataFrame, prediction: np.ndarray) -> dict:
    station_rows = []
    station = base.station_index.to_numpy(int)
    truth = base.y_true.to_numpy(float)
    for index in np.unique(station):
        mask = station == index
        metrics = regression_metrics(truth[mask], prediction[mask])
        station_rows.append(metrics)
    pooled = regression_metrics(truth, prediction)
    return {
        "stations": int(len(station_rows)),
        "macro_mae": float(np.mean([row["mae"] for row in station_rows])),
        "macro_rmse": float(np.mean([row["rmse"] for row in station_rows])),
        "macro_r2": float(np.mean([row["r2"] for row in station_rows])),
        "macro_bias": float(np.mean([row["bias"] for row in station_rows])),
        "pooled_mae": float(pooled["mae"]),
        "pooled_rmse": float(pooled["rmse"]),
        "pooled_r2": float(pooled["r2"]),
        "pooled_bias": float(pooled["bias"]),
    }


def main() -> None:
    apply_runtime_profile(CFG)
    device = CFG.device
    root = Path(os.environ.get(
        "DL_TCN_CROSSFIT_ROOT",
        str(CFG.formal_output_root / "crossfit_target_conditioned_snapshots"),
    ))
    static, _, static_cols = load_static()
    outer = resolve_target(static, CFG.target_site)
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)

    fold_data = []
    for fold in range(6):
        base, matrix = infer_fold(
            fold, root, static, static_cols, cube, timestamps, distance, outer, device
        )
        fold_data.append({"fold": fold, "base": base, "matrix": matrix})
    all_base, all_matrix = combine(fold_data)
    if all_base.station_index.nunique() != 72:
        raise RuntimeError("六個fold合併後不是72個唯一validation stations")

    candidates = [
        *(('uniform_top_k', float(k)) for k in TOP_K_CANDIDATES),
        *(('convex_ridge', float(value)) for value in RIDGE_CANDIDATES),
    ]
    nested_prediction_parts = []
    hard_prediction_parts = []
    fold_audit = []
    for held_fold in range(6):
        training = [item for item in fold_data if item["fold"] != held_fold]
        held = fold_data[held_fold]
        selected, inner_scores = choose_candidate_inner(training, candidates)
        _, candidate = selected
        fit_base, fit_matrix = combine(training)
        weights = weights_for_candidate(fit_base, fit_matrix, candidate)
        hard_weights = weights_for_candidate(fit_base, fit_matrix, ("hard_epoch", 1.0))
        nested_prediction_parts.append(held["matrix"] @ weights)
        hard_prediction_parts.append(held["matrix"] @ hard_weights)
        fold_audit.append({
            "held_fold": held_fold,
            "selected_family": candidate[0],
            "selected_parameter": candidate[1],
            "inner_macro_rmse": selected[0],
            "selected_epochs": [
                epoch for epoch, weight in zip(EPOCHS, weights) if weight > 1e-8
            ],
            "hard_epoch": int(np.argmax(hard_weights)) + 1,
        })

    nested_prediction = np.concatenate(nested_prediction_parts)
    hard_prediction = np.concatenate(hard_prediction_parts)
    # combine() preserves fold order, matching the prediction concatenation.
    nested_summary = summarize(all_base, nested_prediction)
    hard_summary = summarize(all_base, hard_prediction)

    # Choose one deployable ensemble family/parameter using six-fold OOF scores.
    deployment_scores = []
    for candidate in candidates:
        predictions = []
        for held_fold in range(6):
            training = [item for item in fold_data if item["fold"] != held_fold]
            fit_base, fit_matrix = combine(training)
            weights = weights_for_candidate(fit_base, fit_matrix, candidate)
            held = fold_data[held_fold]
            predictions.append(held["matrix"] @ weights)
        full_prediction = np.concatenate(predictions)
        deployment_scores.append({
            "family": candidate[0],
            "parameter": candidate[1],
            "oof72_macro_rmse": macro_station_rmse(all_base, full_prediction),
        })
    selected_deployment = min(deployment_scores, key=lambda row: row["oof72_macro_rmse"])
    deployable_weights = weights_for_candidate(
        all_base,
        all_matrix,
        (selected_deployment["family"], selected_deployment["parameter"]),
    )

    result = {
        "protocol": "six outer folds; each held 12 stations; ensemble choice and weights use other 60 stations only",
        "selection_used_taoyuan_truth": False,
        "files_written": 0,
        "nested_72_station_ensemble": nested_summary,
        "crossfit_hard_single_epoch_reference": hard_summary,
        "fold_audit": fold_audit,
        "selected_deployment_method": selected_deployment,
        "deployable_weights_for_taoyuan": [
            {"epoch": epoch, "weight": float(weight)}
            for epoch, weight in zip(EPOCHS, deployable_weights)
            if weight > 1e-8
        ],
        "deployment_candidate_scores": sorted(
            deployment_scores, key=lambda row: row["oof72_macro_rmse"]
        ),
    }
    print("\n72-STATION CROSS-FITTED SNAPSHOT ENSEMBLE")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
