from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from data_pipeline import load_static


RULE_FILE = Path(__file__).with_name("refined_epoch_rules.csv")


def parse_rule(text: str) -> tuple[tuple[str, str, float], ...]:
    conditions = []
    for part in str(text).split(" AND "):
        match = re.fullmatch(r"(.+?) (<=|>) ([-+0-9.eE]+)", part.strip())
        if match is None:
            raise ValueError(f"無法解析規則條件：{part}")
        conditions.append((match.group(1), match.group(2), float(match.group(3))))
    return tuple(conditions)


def load_rules() -> list[dict]:
    frame = pd.read_csv(RULE_FILE, encoding="utf-8-sig").sort_values("leaf")
    return [
        {
            "leaf": int(row.leaf), "offset": int(row.offset),
            "rule": str(row.rule), "conditions": parse_rule(row.rule),
        }
        for row in frame.itertuples(index=False)
    ]


def condition_matches(row: pd.Series, condition: tuple[str, str, float]) -> bool:
    feature, operator, threshold = condition
    value = float(pd.to_numeric(row[feature], errors="coerce"))
    if not np.isfinite(value):
        return False
    return value <= threshold if operator == "<=" else value > threshold


def select_rule(row: pd.Series, rules: list[dict]) -> tuple[int, int, str]:
    for rule in rules:
        if all(condition_matches(row, condition) for condition in rule["conditions"]):
            return int(rule["leaf"]), int(rule["offset"]), str(rule["rule"])
    raise RuntimeError("static row沒有匹配任何 refined epoch 規則")


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


def epoch_regime(epoch) -> np.ndarray:
    epoch = np.asarray(epoch)
    return np.where(epoch <= 5, 0, np.where(epoch <= 10, 1, 2))


def main() -> None:
    root = Path(os.environ.get("DL_TCN_CROSSFIT_ROOT", "/content/crossfit_target_conditioned_snapshots"))
    output = Path(os.environ.get("DL_TCN_REFINED_RULE_OUTPUT", str(root / "refined_epoch_rule")))
    output.mkdir(parents=True, exist_ok=True)
    rules = load_rules()
    history = load_history(root)
    oracle = oracle_table(history)
    static, _, _ = load_static()
    indexed = history.set_index(["station_index", "epoch"])

    rows = []
    for _, item in oracle.iterrows():
        station = int(item.station_index)
        fold = int(item.fold)
        ref_epoch = reference_epoch(history, fold, station)
        leaf, offset, rule_text = select_rule(static.loc[station], rules)
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
    error = result.epoch_error.to_numpy(float)
    summary = {
        "stations": len(result), "leaves": len(rules),
        "epoch_mae": float(error.mean()),
        "classification_accuracy": float(result.classification_correct.mean()),
        "exact_epoch_accuracy": float((error == 0).mean()),
        "within_1_epoch_accuracy": float((error <= 1).mean()),
        "macro_rmse": float(result.rmse.mean()),
        "mean_rmse_regret": float(result.rmse_regret.mean()),
    }
    result.to_csv(output / "known72_refined_rule_results.csv", index=False, encoding="utf-8-sig")
    (output / "known72_refined_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    outer_matches = static.index[static.sitename.astype(str) == str(CFG.target_site)].to_numpy()
    if len(outer_matches) != 1:
        raise RuntimeError(f"outer target {CFG.target_site}無法唯一定位")
    outer = int(outer_matches[0])
    outer_leaf, outer_offset, outer_rule = select_rule(static.loc[outer], rules)
    fold0_reference = reference_epoch(history, 0, None)
    locked = {
        "target_site": CFG.target_site, "target_station_index": outer,
        "leaf": outer_leaf, "matched_rule": outer_rule, "offset": outer_offset,
        "fold0_reference_epoch": fold0_reference,
        "locked_epoch": int(np.clip(fold0_reference + outer_offset, 1, 15)),
        "outer_truth_read": False,
    }
    (output / "LOCKED_OUTER_REFINED_EPOCH_BEFORE_TRUTH.json").write_text(
        json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("KNOWN 72 REFINED SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nLOCKED OUTER")
    print(json.dumps(locked, ensure_ascii=False, indent=2))
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
