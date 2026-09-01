from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from data_pipeline import load_static


# Fixed transparent tree optimized on the 72 known OOF stations.
# Each tuple is (feature, operator, threshold).
RULES = (
    ((("transportation_ratio_ring_0_1km", "<=", 0.15737653), ("distance_to_major_road_m", "<=", 878.58069), ("forest_ratio_ring_1_5km", "<=", 0.21636209), ("longitude", "<=", 121.02537)), -1),
    ((("transportation_ratio_ring_0_1km", "<=", 0.15737653), ("distance_to_major_road_m", "<=", 878.58069), ("forest_ratio_ring_1_5km", "<=", 0.21636209), ("longitude", ">", 121.02537)), 10),
    ((("transportation_ratio_ring_0_1km", "<=", 0.15737653), ("distance_to_major_road_m", "<=", 878.58069), ("forest_ratio_ring_1_5km", ">", 0.21636209)), -10),
    ((("transportation_ratio_ring_0_1km", "<=", 0.15737653), ("distance_to_major_road_m", ">", 878.58069), ("industrial_ratio_ring_10_20km", "<=", 0.019137823)), -11),
    ((("transportation_ratio_ring_0_1km", "<=", 0.15737653), ("distance_to_major_road_m", ">", 878.58069), ("industrial_ratio_ring_10_20km", ">", 0.019137823)), -6),
    ((("transportation_ratio_ring_0_1km", ">", 0.15737653), ("latitude", "<=", 25.061682), ("forest_ratio_ring_10_20km", "<=", 0.11156811), ("distance_to_industrial_park_m", "<=", 1941.349)), -6),
    ((("transportation_ratio_ring_0_1km", ">", 0.15737653), ("latitude", "<=", 25.061682), ("forest_ratio_ring_10_20km", "<=", 0.11156811), ("distance_to_industrial_park_m", ">", 1941.349), ("ndvi_mean_1km", "<=", 0.35838397)), -13),
    ((("transportation_ratio_ring_0_1km", ">", 0.15737653), ("latitude", "<=", 25.061682), ("forest_ratio_ring_10_20km", "<=", 0.11156811), ("distance_to_industrial_park_m", ">", 1941.349), ("ndvi_mean_1km", ">", 0.35838397)), -2),
    ((("transportation_ratio_ring_0_1km", ">", 0.15737653), ("latitude", "<=", 25.061682), ("forest_ratio_ring_10_20km", ">", 0.11156811), ("elev_std_20km", "<=", 217.43793), ("elev_std_5km", "<=", 22.667489)), 3),
    ((("transportation_ratio_ring_0_1km", ">", 0.15737653), ("latitude", "<=", 25.061682), ("forest_ratio_ring_10_20km", ">", 0.11156811), ("elev_std_20km", "<=", 217.43793), ("elev_std_5km", ">", 22.667489)), 10),
    ((("transportation_ratio_ring_0_1km", ">", 0.15737653), ("latitude", "<=", 25.061682), ("forest_ratio_ring_10_20km", ">", 0.11156811), ("elev_std_20km", ">", 217.43793), ("ndvi_seasonal_amplitude_3km", "<=", 0.083990335)), -4),
    ((("transportation_ratio_ring_0_1km", ">", 0.15737653), ("latitude", "<=", 25.061682), ("forest_ratio_ring_10_20km", ">", 0.11156811), ("elev_std_20km", ">", 217.43793), ("ndvi_seasonal_amplitude_3km", ">", 0.083990335)), 1),
    ((("transportation_ratio_ring_0_1km", ">", 0.15737653), ("latitude", ">", 25.061682)), -11),
)


def condition_matches(row: pd.Series, condition: tuple[str, str, float]) -> bool:
    feature, operator, threshold = condition
    value = float(pd.to_numeric(row[feature], errors="coerce"))
    if not np.isfinite(value):
        return False
    return value <= threshold if operator == "<=" else value > threshold


def select_rule(row: pd.Series) -> tuple[int, int, str]:
    for index, (conditions, offset) in enumerate(RULES):
        if all(condition_matches(row, condition) for condition in conditions):
            text = " AND ".join(
                f"{feature} {operator} {threshold:.8g}"
                for feature, operator, threshold in conditions
            )
            return index, int(offset), text
    raise RuntimeError("static row沒有匹配任何固定規則")


def load_history(root: Path) -> pd.DataFrame:
    parts = []
    for fold in range(6):
        path = root / f"fold_{fold:02d}" / "validation_station_metrics_all_epochs.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["fold"] = fold
        parts.append(frame)
    history = pd.concat(parts, ignore_index=True)
    history["station_index"] = history.station_index.astype(int)
    history["epoch"] = history.epoch.astype(int)
    return history


def oracle_table(history: pd.DataFrame) -> pd.DataFrame:
    return (
        history.sort_values(["station_index", "rmse", "epoch"])
        .groupby("station_index", as_index=False).first()
        [["station_index", "siteid", "sitename", "fold", "epoch", "rmse"]]
        .rename(columns={"epoch": "true_epoch", "rmse": "oracle_rmse"})
        .sort_values("station_index").reset_index(drop=True)
    )


def reference_epoch(history: pd.DataFrame, fold: int, excluded_station: int | None) -> int:
    rows = history[history.fold == fold]
    if excluded_station is not None:
        rows = rows[rows.station_index != excluded_station]
    return int(rows.groupby("epoch").rmse.mean().idxmin())


def epoch_regime(epoch: np.ndarray) -> np.ndarray:
    epoch = np.asarray(epoch)
    return np.where(epoch <= 5, 0, np.where(epoch <= 10, 1, 2))


def main() -> None:
    root = Path(os.environ.get("DL_TCN_CROSSFIT_ROOT", "/content/crossfit_target_conditioned_snapshots"))
    output = Path(os.environ.get("DL_TCN_OPTIMIZED_RULE_OUTPUT", str(root / "optimized_epoch_rule")))
    output.mkdir(parents=True, exist_ok=True)
    history = load_history(root)
    oracle = oracle_table(history)
    static, _, _ = load_static()
    indexed = history.set_index(["station_index", "epoch"])

    rows = []
    for _, item in oracle.iterrows():
        station = int(item.station_index)
        fold = int(item.fold)
        ref_epoch = reference_epoch(history, fold, station)
        leaf, offset, rule_text = select_rule(static.loc[station])
        selected_epoch = int(np.clip(ref_epoch + offset, 1, 15))
        metric = indexed.loc[(station, selected_epoch)]
        rows.append({
            "station_index": station, "siteid": str(item.siteid), "sitename": str(item.sitename),
            "fold": fold, "leaf": leaf, "rule": rule_text, "offset": offset,
            "reference_epoch": ref_epoch, "selected_epoch": selected_epoch,
            "true_epoch": int(item.true_epoch), "epoch_error": abs(selected_epoch - int(item.true_epoch)),
            "classification_correct": bool(epoch_regime([selected_epoch])[0] == epoch_regime([int(item.true_epoch)])[0]),
            "n": int(metric.n), "mae": float(metric.mae), "rmse": float(metric.rmse),
            "r2": float(metric.r2), "bias": float(metric.bias),
            "oracle_rmse": float(item.oracle_rmse), "rmse_regret": float(metric.rmse - item.oracle_rmse),
        })
    result = pd.DataFrame(rows)

    sum_y = sum_y2 = 0.0
    total_truth = 0
    for fold in range(6):
        data = np.load(root / f"fold_{fold:02d}" / "epoch_predictions" / "epoch_001.npz")
        truth = np.asarray(data["y_true"], dtype="float64")
        sum_y += float(truth.sum()); sum_y2 += float(np.square(truth).sum()); total_truth += len(truth)
    sse = float(np.sum(result.n * np.square(result.rmse)))
    sst = float(sum_y2 - sum_y * sum_y / total_truth)
    summary = {
        "stations": len(result),
        "epoch_mae": float(result.epoch_error.mean()),
        "classification_accuracy": float(result.classification_correct.mean()),
        "exact_epoch_accuracy": float((result.epoch_error == 0).mean()),
        "within_1_epoch_accuracy": float((result.epoch_error <= 1).mean()),
        "macro_rmse": float(result.rmse.mean()),
        "mean_rmse_regret": float(result.rmse_regret.mean()),
        "pooled_rmse": float(np.sqrt(sse / result.n.sum())),
        "pooled_r2_exact": float(1.0 - sse / sst),
    }
    result.to_csv(output / "known72_rule_results.csv", index=False, encoding="utf-8-sig")
    (output / "known72_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    outer_matches = static.index[static.sitename.astype(str) == str(CFG.target_site)].to_numpy()
    if len(outer_matches) != 1:
        raise RuntimeError(f"outer target {CFG.target_site}無法唯一定位")
    outer = int(outer_matches[0])
    outer_leaf, outer_offset, outer_rule = select_rule(static.loc[outer])
    fold0_reference = reference_epoch(history, 0, None)
    locked = {
        "target_site": CFG.target_site, "target_station_index": outer,
        "leaf": outer_leaf, "matched_rule": outer_rule, "offset": outer_offset,
        "fold0_reference_epoch": fold0_reference,
        "locked_epoch": int(np.clip(fold0_reference + outer_offset, 1, 15)),
        "outer_truth_read": False,
    }
    (output / "LOCKED_OUTER_EPOCH_BEFORE_TRUTH.json").write_text(
        json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("KNOWN 72 SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nLOCKED OUTER")
    print(json.dumps(locked, ensure_ascii=False, indent=2))
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
