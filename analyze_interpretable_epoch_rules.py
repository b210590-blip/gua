from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from data_pipeline import load_static


GREEN_FEATURES = (
    "forest_ratio_ring_0_1km",
    "forest_ratio_ring_1_5km",
    "forest_ratio_ring_5_10km",
    "ndvi_mean_1km",
    "ndvi_mean_3km",
)
URBAN_FEATURES = (
    "transportation_ratio_ring_0_1km",
    "commercial_ratio_ring_0_1km",
    "built_up_ratio_ring_1_5km",
    "residential_ratio_ring_0_1km",
)
RULE_FEATURES = GREEN_FEATURES + URBAN_FEATURES
LOWER_QUANTILE = 0.35
UPPER_QUANTILE = 0.70


def epoch_regime(epoch: int) -> str:
    if 1 <= epoch <= 5:
        return "early"
    if 6 <= epoch <= 10:
        return "middle"
    if 11 <= epoch <= 15:
        return "late"
    raise ValueError(f"epoch超出1..15: {epoch}")


def load_history(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("fold_*/validation_station_metrics_all_epochs.csv"))
    if len(files) != 6:
        raise FileNotFoundError(f"應有6個fold，實際找到{len(files)}個: {root}")
    frames = []
    for path in files:
        part = pd.read_csv(path, encoding="utf-8-sig")
        part["fold"] = int(path.parent.name.split("_")[-1])
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "station_index", "siteid", "sitename", "fold", "epoch",
        "n", "mae", "rmse", "r2", "bias",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"epoch history缺少欄位: {sorted(missing)}")
    frame["station_index"] = frame["station_index"].astype(int)
    frame["epoch"] = frame["epoch"].astype(int)
    if frame.station_index.nunique() != 72:
        raise RuntimeError("history必須包含72個OOF測站")
    counts = frame.groupby("station_index").epoch.nunique()
    if not (counts == 15).all():
        raise RuntimeError("每站必須都有epoch 1..15")
    return frame


def build_oracle(history: pd.DataFrame) -> pd.DataFrame:
    oracle = (
        history.sort_values(["station_index", "rmse", "epoch"])
        .groupby("station_index", as_index=False)
        .first()
    )
    oracle = oracle[
        ["station_index", "siteid", "sitename", "fold", "epoch", "rmse", "r2"]
    ].rename(
        columns={
            "epoch": "true_best_epoch",
            "rmse": "oracle_rmse",
            "r2": "oracle_r2",
        }
    )
    oracle["true_regime"] = oracle.true_best_epoch.map(epoch_regime)
    return oracle


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mid-rank empirical CDF of values relative to reference, NaN-safe."""
    reference = np.asarray(reference, dtype="float64")
    values = np.asarray(values, dtype="float64")
    finite = np.sort(reference[np.isfinite(reference)])
    if not len(finite):
        return np.full(values.shape, 0.5, dtype="float64")
    left = np.searchsorted(finite, values, side="left")
    right = np.searchsorted(finite, values, side="right")
    result = (left + right) / (2.0 * len(finite))
    result[~np.isfinite(values)] = 0.5
    return result


def score_against_reference(
    reference: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    percentiles = pd.DataFrame(index=target.index)
    for feature in RULE_FEATURES:
        percentiles[f"pct_{feature}"] = empirical_percentile(
            pd.to_numeric(reference[feature], errors="coerce").to_numpy(),
            pd.to_numeric(target[feature], errors="coerce").to_numpy(),
        )
    green_columns = [f"pct_{feature}" for feature in GREEN_FEATURES]
    urban_columns = [f"pct_{feature}" for feature in URBAN_FEATURES]
    percentiles["green_score"] = percentiles[green_columns].mean(axis=1)
    percentiles["urban_score"] = percentiles[urban_columns].mean(axis=1)
    percentiles["green_minus_urban"] = (
        percentiles.green_score - percentiles.urban_score
    )

    reference_scores = pd.DataFrame(index=reference.index)
    for feature in RULE_FEATURES:
        reference_scores[f"pct_{feature}"] = empirical_percentile(
            pd.to_numeric(reference[feature], errors="coerce").to_numpy(),
            pd.to_numeric(reference[feature], errors="coerce").to_numpy(),
        )
    reference_scores["green_score"] = reference_scores[
        [f"pct_{feature}" for feature in GREEN_FEATURES]
    ].mean(axis=1)
    reference_scores["urban_score"] = reference_scores[
        [f"pct_{feature}" for feature in URBAN_FEATURES]
    ].mean(axis=1)
    contrast = reference_scores.green_score - reference_scores.urban_score
    return percentiles, contrast


def classify(contrast: float, lower: float, upper: float) -> str:
    if contrast >= upper:
        return "early"
    if contrast <= lower:
        return "late"
    return "middle"


def fold_reference_epoch_excluding_station(
    history: pd.DataFrame,
    station: int,
    additionally_excluded: int | None = None,
) -> int:
    folds = history.loc[history.station_index == station, "fold"].unique()
    if len(folds) != 1:
        raise RuntimeError(f"station={station} fold不唯一")
    fold = int(folds[0])
    keep = (history.fold == fold) & (history.station_index != station)
    if additionally_excluded is not None:
        keep &= history.station_index != int(additionally_excluded)
    peers = history.loc[keep]
    excluded_is_same_fold = (
        additionally_excluded is not None
        and bool(
            (history.loc[history.station_index == additionally_excluded, "fold"] == fold)
            .any()
        )
    )
    expected = 10 if excluded_is_same_fold else 11
    if peers.station_index.nunique() != expected:
        raise RuntimeError(
            f"station={station} fold reference應有{expected}站，"
            f"實際{peers.station_index.nunique()}站"
        )
    return int(peers.groupby("epoch").rmse.mean().idxmin())


def choose_group_offsets(
    history: pd.DataFrame,
    oracle: pd.DataFrame,
    reference_epochs: dict[int, int],
    reference_ids: np.ndarray,
    reference_groups: np.ndarray,
    target_group: str,
) -> tuple[int, int, int]:
    """Choose fold-relative offsets from same-static-group reference stations."""
    members = reference_ids[reference_groups == target_group]
    if not len(members):
        raise RuntimeError(f"reference中沒有{target_group}群")
    oracle_indexed = oracle.set_index("station_index")
    history_indexed = history.set_index(["station_index", "epoch"])
    true_offsets = np.asarray(
        [
            int(oracle_indexed.loc[int(member), "true_best_epoch"])
            - int(reference_epochs[int(member)])
            for member in members
        ],
        dtype="float64",
    )
    median_offset = int(np.floor(np.median(true_offsets) + 0.5))

    candidates = []
    for offset in range(-14, 15):
        regrets = []
        for member in members:
            member = int(member)
            epoch = int(np.clip(reference_epochs[member] + offset, 1, 15))
            rmse = float(history_indexed.loc[(member, epoch), "rmse"])
            oracle_rmse = float(oracle_indexed.loc[member, "oracle_rmse"])
            regrets.append(rmse - oracle_rmse)
        candidates.append((float(np.mean(regrets)), abs(offset), offset))
    regret_offset = int(min(candidates)[2])
    return regret_offset, median_offset, int(len(members))


def evidence_text(percentile_row: pd.Series) -> str:
    evidence = []
    for feature in GREEN_FEATURES:
        value = float(percentile_row[f"pct_{feature}"])
        evidence.append((abs(value - 0.5), value - 0.5, feature, value))
    for feature in URBAN_FEATURES:
        value = float(percentile_row[f"pct_{feature}"])
        # High urban values support late; use negative sign for the early axis.
        evidence.append((abs(value - 0.5), 0.5 - value, feature, value))
    strongest = sorted(evidence, reverse=True)[:4]
    parts = []
    for _, signed, feature, value in strongest:
        direction = "supports early" if signed > 0 else "supports late"
        parts.append(f"{feature} pct={value:.2f} {direction}")
    return "; ".join(parts)


def select_metric(history: pd.DataFrame, station: int, epoch: int) -> dict:
    row = history.loc[
        (history.station_index == station) & (history.epoch == epoch)
    ]
    if len(row) != 1:
        raise RuntimeError(f"station={station}, epoch={epoch}不是唯一一列")
    return row.iloc[0].to_dict()


def truth_sufficient_statistics(root: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for fold in range(6):
        path = root / f"fold_{fold:02d}" / "epoch_predictions" / "epoch_001.npz"
        data = np.load(path)
        station = np.asarray(data["station_index"], dtype=int)
        truth = np.asarray(data["y_true"], dtype="float64")
        for idx in np.unique(station):
            values = truth[station == idx]
            result[int(idx)] = {
                "n": int(len(values)),
                "sum_y": float(values.sum()),
                "sum_y2": float(np.square(values).sum()),
            }
    return result


def performance_summary(
    rows: pd.DataFrame,
    truth_stats: dict[int, dict[str, float]],
    method: str,
) -> dict:
    n = rows.n.to_numpy("float64")
    total_n = float(n.sum())
    expected_n = sum(truth_stats[int(i)]["n"] for i in rows.station_index)
    if int(total_n) != int(expected_n):
        raise RuntimeError(f"{method}: coverage分母不一致")
    sse = float(np.sum(n * np.square(rows.rmse.to_numpy("float64"))))
    sum_y = sum(truth_stats[int(i)]["sum_y"] for i in rows.station_index)
    sum_y2 = sum(truth_stats[int(i)]["sum_y2"] for i in rows.station_index)
    sst = float(sum_y2 - sum_y * sum_y / total_n)
    result = {
        "method": method,
        "stations": int(len(rows)),
        "regime_accuracy": float(rows.regime_correct.mean()),
        "mean_absolute_epoch_error": float(rows.epoch_error_abs.mean()),
        "macro_rmse": float(rows.rmse.mean()),
        "macro_mae": float(rows.mae.mean()),
        "macro_r2": float(rows.r2.mean()),
        "macro_bias": float(rows.bias.mean()),
        "pooled_rmse": float(np.sqrt(sse / total_n)),
        "pooled_mae": float(np.sum(n * rows.mae) / total_n),
        "pooled_bias": float(np.sum(n * rows.bias) / total_n),
        "pooled_r2_exact": float(1.0 - sse / sst),
        "mean_rmse_regret_vs_oracle": float(rows.rmse_regret_vs_oracle.mean()),
    }
    if "offset_error_abs" in rows.columns:
        result["mean_absolute_offset_error"] = float(rows.offset_error_abs.mean())
    return result


def static_association_table(
    static: pd.DataFrame,
    static_cols: list[str],
    oracle: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    epoch = oracle.set_index("station_index").true_best_epoch
    regime = oracle.set_index("station_index").true_regime
    for feature in static_cols:
        values = pd.to_numeric(static.loc[oracle.station_index, feature], errors="coerce")
        values.index = oracle.station_index
        ranked_correlation = values.rank().corr(epoch.rank())
        row = {
            "feature": feature,
            "spearman_vs_best_epoch": ranked_correlation,
        }
        for label in ("early", "middle", "late"):
            selected = values[regime == label]
            row[f"{label}_mean"] = selected.mean()
            row[f"{label}_median"] = selected.median()
        rows.append(row)
    result = pd.DataFrame(rows)
    result["absolute_spearman"] = result.spearman_vs_best_epoch.abs()
    return result.sort_values("absolute_spearman", ascending=False)


def main() -> None:
    if CFG.max_epochs != 15:
        raise RuntimeError("fold-relative規則需要完整epoch 1..15，請設定DL_TCN_MAX_EPOCHS=15")
    root = Path(
        os.environ.get(
            "DL_TCN_CROSSFIT_ROOT",
            str(CFG.formal_output_root / "crossfit_target_conditioned_snapshots"),
        )
    )
    output = Path(
        os.environ.get(
            "DL_TCN_RULE_OUTPUT",
            str(root / "interpretable_epoch_rule_analysis"),
        )
    )
    output.mkdir(parents=True, exist_ok=True)

    history = load_history(root)
    oracle = build_oracle(history)
    static, _, static_cols = load_static()
    missing_features = set(RULE_FEATURES) - set(static.columns)
    if missing_features:
        raise RuntimeError(f"static缺少規則特徵: {sorted(missing_features)}")

    station_ids = oracle.station_index.to_numpy(int)
    station_rows = []
    metric_rows = []
    print("Transparent static rule LOSO: 0/72", flush=True)
    for position, station in enumerate(station_ids):
        reference_ids = np.delete(station_ids, position)
        reference = static.loc[reference_ids, list(RULE_FEATURES)]
        target = static.loc[[station], list(RULE_FEATURES)]
        target_scores, reference_contrast = score_against_reference(reference, target)
        lower = float(reference_contrast.quantile(LOWER_QUANTILE))
        upper = float(reference_contrast.quantile(UPPER_QUANTILE))
        score_row = target_scores.iloc[0]
        predicted_regime = classify(
            float(score_row.green_minus_urban), lower, upper
        )
        reference_groups = np.asarray(
            [classify(float(value), lower, upper) for value in reference_contrast],
            dtype=str,
        )
        # Strict outer LOSO: target station is excluded not only from its own
        # fold reference, but also from every calibration station's reference.
        reference_epochs = {
            int(member): fold_reference_epoch_excluding_station(
                history, int(member), additionally_excluded=int(station)
            )
            for member in reference_ids
        }
        regret_offset, median_offset, group_reference_stations = choose_group_offsets(
            history,
            oracle,
            reference_epochs,
            reference_ids,
            reference_groups,
            predicted_regime,
        )
        oracle_row = oracle.loc[oracle.station_index == station].iloc[0]
        target_reference_epoch = fold_reference_epoch_excluding_station(
            history, int(station)
        )
        curve_epoch = int(np.clip(target_reference_epoch + regret_offset, 1, 15))
        median_epoch = int(np.clip(target_reference_epoch + median_offset, 1, 15))
        true_offset = int(oracle_row.true_best_epoch) - target_reference_epoch
        common = {
            "station_index": int(station),
            "siteid": str(oracle_row.siteid),
            "sitename": str(oracle_row.sitename),
            "true_best_epoch": int(oracle_row.true_best_epoch),
            "true_regime": str(oracle_row.true_regime),
            "predicted_regime": predicted_regime,
            "regime_correct": predicted_regime == oracle_row.true_regime,
            "fold": int(oracle_row.fold),
            "fold_reference_epoch_excluding_target": target_reference_epoch,
            "true_fold_relative_offset": true_offset,
            "curve_selected_epoch": curve_epoch,
            "curve_selected_offset": regret_offset,
            "median_selected_epoch": median_epoch,
            "median_selected_offset": median_offset,
            "group_reference_stations": group_reference_stations,
            "green_score": float(score_row.green_score),
            "urban_score": float(score_row.urban_score),
            "green_minus_urban": float(score_row.green_minus_urban),
            "loso_lower_threshold": lower,
            "loso_upper_threshold": upper,
            "evidence": evidence_text(score_row),
            **{
                column: float(score_row[column])
                for column in score_row.index
                if column.startswith("pct_")
            },
        }
        station_rows.append(common)
        for method, selected_epoch in (
            ("fold_relative_same_group_regret_loso", curve_epoch),
            ("fold_relative_same_group_median_loso", median_epoch),
        ):
            selected_offset = selected_epoch - target_reference_epoch
            metric = select_metric(history, int(station), selected_epoch)
            metric_rows.append(
                {
                    **common,
                    "method": method,
                    "selected_epoch": selected_epoch,
                    "selected_fold_relative_offset": selected_offset,
                    "offset_error_abs": abs(selected_offset - true_offset),
                    "epoch_error_abs": abs(
                        selected_epoch - int(oracle_row.true_best_epoch)
                    ),
                    **{
                        key: metric[key]
                        for key in ("n", "mae", "rmse", "r2", "bias")
                    },
                    "oracle_rmse": float(oracle_row.oracle_rmse),
                    "rmse_regret_vs_oracle": float(
                        metric["rmse"] - oracle_row.oracle_rmse
                    ),
                }
            )
        if (position + 1) % 12 == 0:
            print(f"Transparent static rule LOSO: {position + 1}/72", flush=True)

    station_table = pd.DataFrame(station_rows)
    metrics = pd.DataFrame(metric_rows)
    truth_stats = truth_sufficient_statistics(root)
    summaries = []
    for method, rows in metrics.groupby("method", sort=False):
        summaries.append(performance_summary(rows, truth_stats, str(method)))

    oracle_metrics = []
    for _, row in oracle.iterrows():
        metric = select_metric(history, int(row.station_index), int(row.true_best_epoch))
        oracle_metrics.append(
            {
                "station_index": int(row.station_index),
                "n": metric["n"],
                "mae": metric["mae"],
                "rmse": metric["rmse"],
                "r2": metric["r2"],
                "bias": metric["bias"],
                "regime_correct": True,
                "epoch_error_abs": 0,
                "rmse_regret_vs_oracle": 0.0,
            }
        )
    summaries.append(
        performance_summary(
            pd.DataFrame(oracle_metrics), truth_stats, "station_oracle_reference"
        )
    )

    group_audit = (
        metrics.groupby(["method", "predicted_regime"], as_index=False)
        .agg(
            stations=("station_index", "size"),
            regime_accuracy=("regime_correct", "mean"),
            true_epoch_mean=("true_best_epoch", "mean"),
            true_epoch_median=("true_best_epoch", "median"),
            true_epoch_min=("true_best_epoch", "min"),
            true_epoch_max=("true_best_epoch", "max"),
            reference_epoch_mean=("fold_reference_epoch_excluding_target", "mean"),
            true_offset_mean=("true_fold_relative_offset", "mean"),
            true_offset_median=("true_fold_relative_offset", "median"),
            selected_offset_mean=("selected_fold_relative_offset", "mean"),
            selected_offset_min=("selected_fold_relative_offset", "min"),
            selected_offset_max=("selected_fold_relative_offset", "max"),
            selected_epoch_mean=("selected_epoch", "mean"),
            selected_epoch_min=("selected_epoch", "min"),
            selected_epoch_max=("selected_epoch", "max"),
            macro_rmse=("rmse", "mean"),
            mean_regret=("rmse_regret_vs_oracle", "mean"),
        )
    )
    confusion = pd.crosstab(
        station_table.true_regime,
        station_table.predicted_regime,
        rownames=["true_regime"],
        colnames=["predicted_regime"],
        dropna=False,
    ).reindex(index=["early", "middle", "late"], columns=["early", "middle", "late"], fill_value=0)

    association = static_association_table(static, static_cols, oracle)
    station_table.to_csv(
        output / "transparent_rule_station_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        output / "transparent_rule_selected_epoch_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summaries).to_csv(
        output / "transparent_rule_performance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_audit.to_csv(
        output / "transparent_rule_group_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    confusion.to_csv(
        output / "transparent_rule_confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    association.to_csv(
        output / "static_vs_best_epoch_association.csv",
        index=False,
        encoding="utf-8-sig",
    )
    station_table[
        [
            "station_index", "siteid", "sitename", "fold",
            "fold_reference_epoch_excluding_target", "true_best_epoch",
            "true_fold_relative_offset",
        ]
    ].to_csv(
        output / "fold_relative_epoch_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )
    definition = {
        "green_features": list(GREEN_FEATURES),
        "urban_features": list(URBAN_FEATURES),
        "score": "mean(green empirical percentiles) - mean(urban empirical percentiles)",
        "thresholds": {
            "late": f"score <= reference {LOWER_QUANTILE:.0%} quantile",
            "middle": "between thresholds",
            "early": f"score >= reference {UPPER_QUANTILE:.0%} quantile",
        },
        "epoch_calibration": {
            "fold_reference": (
                "For each pseudo-target, the reference epoch is chosen from the "
                "other 11 validation stations in the same independently trained fold."
            ),
            "fold_relative_same_group_regret_loso": (
                "Choose a fold-relative offset minimizing mean station regret among "
                "same-static-group reference stations, then add it to the target fold reference."
            ),
            "fold_relative_same_group_median_loso": (
                "Use the median fold-relative best-epoch offset among same-static-group "
                "reference stations, then add it to the target fold reference."
            ),
        },
        "evaluation": (
            "Each station is excluded from static percentiles, thresholds, its fold "
            "reference epoch, and offset calibration until final scoring."
        ),
        "important_limitation": (
            "The nine rule features were chosen after inspecting all 72 development stations. "
            "This is an interpretable development analysis, not a fully independent outer evaluation."
        ),
        "uses_classifier": False,
        "tcn_retraining": False,
    }
    (output / "transparent_rule_definition.json").write_text(
        json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nPerformance:", flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print("\nGroup audit:", flush=True)
    print(group_audit.to_string(index=False), flush=True)
    print("\nConfusion matrix:", flush=True)
    print(confusion.to_string(), flush=True)
    print(f"\nOutput: {output}", flush=True)


if __name__ == "__main__":
    main()
