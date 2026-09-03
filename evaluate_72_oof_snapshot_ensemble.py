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


def meta_features(base: pd.DataFrame, matrix: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Only model predictions and known clock time; never target observations."""
    ensemble = matrix @ prior
    disagreement = matrix - ensemble[:, None]
    timestamp = pd.to_datetime(base["timestamp"], errors="raise")
    hour_angle = 2.0 * np.pi * timestamp.dt.hour.to_numpy(float) / 24.0
    year_angle = 2.0 * np.pi * (timestamp.dt.dayofyear.to_numpy(float) - 1.0) / 365.25
    return np.column_stack([
        ensemble,
        disagreement,
        matrix.std(axis=1),
        matrix.max(axis=1) - matrix.min(axis=1),
        np.sin(hour_angle), np.cos(hour_angle),
        np.sin(year_angle), np.cos(year_angle),
    ])


def fit_ridge_residual(
    features: np.ndarray,
    residual: np.ndarray,
    station: np.ndarray,
    alpha: float,
) -> dict:
    # Each station contributes total weight 1, regardless of timestamp coverage.
    unique, counts = np.unique(station, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    sample_weight = np.asarray([1.0 / count_map[int(s)] for s in station], dtype=float)
    weight_sum = sample_weight.sum()
    mean_x = (sample_weight[:, None] * features).sum(axis=0) / weight_sum
    scale_x = np.sqrt(
        (sample_weight[:, None] * (features - mean_x) ** 2).sum(axis=0) / weight_sum
    )
    scale_x = np.where(scale_x > 1e-8, scale_x, 1.0)
    x = (features - mean_x) / scale_x
    mean_y = float(np.sum(sample_weight * residual) / weight_sum)
    yc = residual - mean_y
    gram = x.T @ (sample_weight[:, None] * x)
    rhs = x.T @ (sample_weight * yc)
    coefficient = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), rhs)
    return {
        "mean_x": mean_x,
        "scale_x": scale_x,
        "mean_y": mean_y,
        "coefficient": coefficient,
    }


def predict_ridge_residual(model: dict, features: np.ndarray) -> np.ndarray:
    x = (features - model["mean_x"]) / model["scale_x"]
    return model["mean_y"] + x @ model["coefficient"]


def nested_station_ridge(
    base: pd.DataFrame,
    matrix: np.ndarray,
    candidates: list[dict],
    final_prior: np.ndarray,
) -> tuple[dict, list[dict], float, list[dict]]:
    truth = base.y_true.to_numpy(float)
    station = base.station_index.to_numpy(int)
    stations = np.unique(station)
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    ridge_rmse = {alpha: [] for alpha in alphas}
    base_rmse = []
    fold_audit = []

    # Fair paired comparison: for each held station, the base candidate is
    # selected using only the other 11 stations. Both base and ridge are then
    # evaluated on exactly the same unseen station.
    for held in stations:
        train = station != held
        test = ~train
        train_stations = np.unique(station[train])
        candidate_scores = []
        for candidate in candidates:
            prediction = matrix @ candidate["weight"]
            rmse = [
                float(np.sqrt(np.mean((truth[(station == s)] - prediction[(station == s)]) ** 2)))
                for s in train_stations
            ]
            candidate_scores.append(float(np.mean(rmse)))
        chosen_index = int(np.argmin(candidate_scores))
        chosen = candidates[chosen_index]
        prior = chosen["weight"]
        prior_prediction = matrix @ prior
        features = meta_features(base, matrix, prior)
        residual = truth - prior_prediction
        held_base_rmse = float(np.sqrt(np.mean((truth[test] - prior_prediction[test]) ** 2)))
        base_rmse.append(held_base_rmse)
        fold_audit.append({
            "held_station": int(held),
            "base_family_selected_on_other_11": chosen["family"],
            "base_parameter": chosen["parameter"],
            "held_base_rmse": held_base_rmse,
        })
        for alpha in alphas:
            model = fit_ridge_residual(features[train], residual[train], station[train], alpha)
            prediction = prior_prediction[test] + predict_ridge_residual(model, features[test])
            ridge_rmse[alpha].append(float(np.sqrt(np.mean((truth[test] - prediction) ** 2))))
    scores = [
        {"alpha": alpha, "station_loso_macro_rmse": float(np.mean(ridge_rmse[alpha]))}
        for alpha in alphas
    ]
    selected = min(scores, key=lambda row: row["station_loso_macro_rmse"])
    features = meta_features(base, matrix, final_prior)
    prior_prediction = matrix @ final_prior
    residual = truth - prior_prediction
    final_model = fit_ridge_residual(features, residual, station, selected["alpha"])
    return final_model, scores, float(np.mean(base_rmse)), fold_audit


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
    ridge_model, ridge_scores, base_loso_rmse, fold_audit = nested_station_ridge(
        validation_base, validation_matrix, candidates, weights
    )
    ridge_selected = min(ridge_scores, key=lambda row: row["station_loso_macro_rmse"])
    deploy_stacker = (
        ridge_selected["station_loso_macro_rmse"] < base_loso_rmse
    )
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
        "ridge_residual_stacker": {
            "selection": "leave-one-validation-station-out",
            "uses_taoyuan_truth": False,
            "candidate_scores": ridge_scores,
            "selected_alpha": ridge_selected["alpha"],
            "nested_macro_rmse": ridge_selected["station_loso_macro_rmse"],
            "paired_base_station_loso_macro_rmse": base_loso_rmse,
            "accepted_before_taoyuan_truth": deploy_stacker,
            "acceptance_rule": "paired station-LOSO ridge RMSE must beat paired station-LOSO base RMSE",
            "fold_audit": fold_audit,
        },
    }
    print("\nLOCKED 60+12 ENSEMBLE BEFORE TAOYUAN TRUTH")
    print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)

    print("\nOuter inference after weights are locked.", flush=True)
    outer_base, outer_matrix = aligned_predictions(
        model, paths, outer_loader, CFG.device, static, timestamps, outer_builder
    )
    base_prediction = outer_matrix @ weights
    outer_features = meta_features(outer_base, outer_matrix, weights)
    stacked_prediction = base_prediction + predict_ridge_residual(ridge_model, outer_features)
    deployed_prediction = stacked_prediction if deploy_stacker else base_prediction
    result = {
        "target_station_index": int(outer),
        "siteid": str(static.loc[outer, "siteid"]),
        "sitename": str(static.loc[outer, "sitename"]),
        "selection_used_taoyuan_truth": False,
        "base_ensemble_metrics": regression_metrics(
            outer_base.y_true.to_numpy(float), base_prediction
        ),
        "ridge_residual_stacker_metrics": regression_metrics(
            outer_base.y_true.to_numpy(float), stacked_prediction
        ),
        "deployed_method": "ridge_residual_stacker" if deploy_stacker else "base_ensemble",
        "deployed_metrics": regression_metrics(
            outer_base.y_true.to_numpy(float), deployed_prediction
        ),
        "selected_method": lock["selected_method"],
        "weights": lock["weights"],
        "files_written": 0,
    }
    print("\nTAOYUAN 72-STATION-INFORMED ENSEMBLE RESULT")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
