from __future__ import annotations

import json
import os
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
from model import TCNTargetCrossAttention
from sanity_check import audit_dataset
from train_formal import (
    DeviceFeatureBuilder,
    make_index_loader,
    make_vectorized_loader,
    resolve_target,
    validate_epoch,
)


def same_timestamps(left: pd.Series, right: pd.Series) -> bool:
    return np.array_equal(
        pd.to_datetime(left).astype("int64").to_numpy(),
        pd.to_datetime(right).astype("int64").to_numpy(),
    )


def main() -> None:
    apply_runtime_profile(CFG)
    device = CFG.device
    root = Path(os.environ.get(
        "DL_TCN_CROSSFIT_ROOT",
        str(CFG.formal_output_root / "crossfit_target_conditioned_snapshots"),
    ))
    refined_dir = Path(os.environ.get("DL_TCN_REFINED_RULE_OUTPUT", str(root / "refined_epoch_rule")))
    lock_path = refined_dir / "LOCKED_OUTER_REFINED_EPOCH_BEFORE_TRUTH.json"
    if not lock_path.is_file():
        raise FileNotFoundError("請先執行 evaluate_refined_epoch_rule.py，先鎖定outer epoch再讀truth")
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    if locked.get("outer_truth_read") is not False:
        raise RuntimeError("outer epoch lock未明確標示為truth-free")

    epochs = list(range(1, int(os.environ.get("DL_TCN_MAX_EPOCHS", "15")) + 1))
    checkpoint_paths = [
        root / "fold_00" / "epoch_checkpoints" / f"epoch_{epoch:03d}.pt"
        for epoch in epochs
    ]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("fold_00缺少epoch snapshots：\n" + "\n".join(missing))

    static, _, static_cols = load_static()
    outer = resolve_target(static, CFG.target_site)
    if outer != int(locked["target_station_index"]):
        raise RuntimeError("refined lock的outer target與目前CFG.target_site不一致")
    locked_epoch = int(locked["locked_epoch"])
    if locked_epoch not in epochs:
        raise RuntimeError(f"鎖定epoch {locked_epoch}不在待評估範圍")

    first = load_checkpoint(checkpoint_paths[0])
    train_idx = np.asarray(first["train_indices"], dtype=int)
    validation_idx = np.asarray(first["validation_indices"], dtype=int)
    if len(train_idx) != 60 or len(validation_idx) != 12:
        raise RuntimeError("fold_00 snapshot不是60 train / 12 validation")
    if outer in train_idx or outer in validation_idx:
        raise RuntimeError("outer target混入fold_00 train/validation")
    if len(np.union1d(train_idx, validation_idx)) != 72:
        raise RuntimeError("fold_00的60/12沒有覆蓋完整known-72")
    if list(first["static_columns"]) != list(static_cols):
        raise RuntimeError("checkpoint static欄位與目前資料不一致")

    scaler = scaler_from_checkpoint(first)
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)
    static_scaled = standardize_static(static, static_cols, scaler)
    outer_ds = ColdStartStationDataset(
        [outer], train_idx, CFG.test_start, CFG.test_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    sanity = audit_dataset(outer_ds, max_samples=min(64, len(outer_ds)))
    if sanity["sampled_donor_counts"] != [60]:
        raise RuntimeError("outer epoch curve donors不是60")

    feature_builder = None
    if device.type == "cuda":
        feature_builder = DeviceFeatureBuilder(
            train_idx, cube, int(outer_ds.row_times.max()), timestamps,
            static, static_scaled, distance, scaler, outer, device,
        )
        loader = make_index_loader(outer_ds, False)
    else:
        loader = make_vectorized_loader(
            outer_ds, train_idx, cube, timestamps, static,
            static_scaled, distance, scaler, False,
        )

    output = Path(os.environ.get("DL_TCN_OUTER_CURVE_OUTPUT", str(root / "outer_epoch_curve")))
    output.mkdir(parents=True, exist_ok=True)
    model = TCNTargetCrossAttention(static_dim=len(static_cols)).to(device)
    curve_rows: list[dict] = []
    prediction_columns: list[np.ndarray] = []
    base: pd.DataFrame | None = None

    for epoch, checkpoint_path in zip(epochs, checkpoint_paths):
        checkpoint = load_checkpoint(checkpoint_path)
        if int(checkpoint["epoch"]) != epoch:
            raise RuntimeError(f"epoch {epoch} checkpoint metadata錯誤")
        if not np.array_equal(np.asarray(checkpoint["train_indices"], dtype=int), train_idx):
            raise RuntimeError(f"epoch {epoch} train split不一致")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        metrics, _, predictions = validate_epoch(
            model, loader, device, static, timestamps, epoch, feature_builder,
        )
        if base is None:
            base = predictions[["station_index", "timestamp", "y_true"]].copy()
        else:
            if not np.array_equal(base.station_index.to_numpy(), predictions.station_index.to_numpy()):
                raise RuntimeError(f"epoch {epoch} station alignment不同")
            if not same_timestamps(base.timestamp, predictions.timestamp):
                raise RuntimeError(f"epoch {epoch} timestamp alignment不同")
            if not np.allclose(base.y_true, predictions.y_true, rtol=0, atol=1e-6):
                raise RuntimeError(f"epoch {epoch} truth alignment不同")
        prediction_columns.append(predictions.y_pred.to_numpy("float32"))
        curve_rows.append({
            "epoch": epoch, "is_locked_epoch": epoch == locked_epoch,
            "n": len(predictions), "coverage": len(predictions) / max(outer_ds.truth_rows, 1),
            "mae": float(metrics["mae"]), "rmse": float(metrics["rmse"]),
            "r2": float(metrics["r2"]), "bias": float(metrics["bias"]),
        })
        print(f"outer epoch curve {epoch}/{epochs[-1]} RMSE={metrics['rmse']:.4f}", flush=True)

    assert base is not None
    curve = pd.DataFrame(curve_rows)
    oracle_epoch = int(curve.loc[curve.rmse.idxmin(), "epoch"])
    locked_row = curve[curve.epoch == locked_epoch].iloc[0]
    oracle_row = curve[curve.epoch == oracle_epoch].iloc[0]
    summary = {
        "target_station_index": outer,
        "siteid": str(static.loc[outer, "siteid"]),
        "sitename": str(static.loc[outer, "sitename"]),
        "locked_epoch_before_truth": locked_epoch,
        "oracle_epoch_after_unblinding": oracle_epoch,
        "absolute_epoch_error": abs(locked_epoch - oracle_epoch),
        "locked_rmse": float(locked_row.rmse),
        "oracle_rmse": float(oracle_row.rmse),
        "rmse_regret": float(locked_row.rmse - oracle_row.rmse),
        "locked_mae": float(locked_row.mae),
        "locked_r2": float(locked_row.r2),
        "locked_bias": float(locked_row.bias),
        "truth_timestamps": int(outer_ds.truth_rows),
        "predicted_timestamps": int(len(base)),
        "donors": 60,
        "selection_used_outer_truth": False,
    }
    curve.to_csv(output / "outer_epoch_curve_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output / "outer_epoch_curve_predictions.npz",
        epochs=np.asarray(epochs, dtype=np.int16),
        timestamp_ns=pd.to_datetime(base.timestamp).astype("int64").to_numpy(),
        y_true=base.y_true.to_numpy("float32"),
        y_pred=np.stack(prediction_columns, axis=1),
    )
    (output / "outer_locked_vs_oracle.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(curve.epoch, curve.rmse, marker="o", linewidth=1.8)
        axis.axvline(locked_epoch, color="tab:orange", linestyle="--", label=f"locked={locked_epoch}")
        axis.scatter([oracle_epoch], [oracle_row.rmse], color="tab:red", zorder=3, label=f"oracle={oracle_epoch}")
        axis.set(xlabel="Epoch", ylabel="Outer RMSE", title=f"{static.loc[outer, 'sitename']} outer epoch curve")
        axis.set_xticks(epochs)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "outer_epoch_curve_rmse.png", dpi=180)
        plt.close(figure)
    except ImportError:
        print("matplotlib未安裝，略過PNG；CSV與NPZ已完整保存", flush=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Output: {output}", flush=True)


if __name__ == "__main__":
    main()
