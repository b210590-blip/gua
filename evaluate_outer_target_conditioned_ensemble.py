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
    regression_metrics,
    resolve_target,
    validate_epoch,
)


def target_weights(gate:dict,static:pd.DataFrame,outer:int) -> np.ndarray:
    cols=list(gate["static_columns"])
    raw=static.loc[[outer],cols].apply(pd.to_numeric,errors="coerce").to_numpy("float64")
    median=np.asarray(gate["static_missing_median"],dtype="float64")
    raw=np.where(np.isfinite(raw),raw,median)
    dim=int(gate["pca_dim"])
    if dim:
        mean=np.asarray(gate["mean"],dtype="float64")
        std=np.asarray(gate["std"],dtype="float64")
        components=np.asarray(gate["components"],dtype="float64")
        z=((raw-mean)/std)@components.T
    else:
        z=np.zeros((1,0),dtype="float64")
    intercept=np.asarray(gate["intercept"],dtype="float64")
    slope=np.asarray(gate["slope"],dtype="float64").reshape(dim,len(intercept))
    logits=intercept[None,:]+z@slope; logits-=logits.max(axis=1,keepdims=True)
    weights=np.exp(logits[0]); weights/=weights.sum()
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(),1.0,atol=1e-9):
        raise RuntimeError("outer snapshot weights非finite或總和不為1")
    return weights


def main():
    apply_runtime_profile(CFG)
    device=CFG.device
    root=Path(os.environ.get("DL_TCN_CROSSFIT_ROOT",str(CFG.formal_output_root/"crossfit_target_conditioned_snapshots")))
    ensemble_dir=Path(os.environ.get("DL_TCN_ENSEMBLE_OUTPUT",str(root/"target_conditioned_ensemble")))
    output=Path(os.environ.get("DL_TCN_OUTER_ENSEMBLE_OUTPUT",str(root/"outer_target_conditioned_ensemble")))
    output.mkdir(parents=True,exist_ok=True)
    gate=json.loads((ensemble_dir/"deployable_target_conditioned_gate.json").read_text(encoding="utf-8"))
    epochs=[int(x) for x in gate["epochs"]]
    checkpoints=[root/"fold_00"/"epoch_checkpoints"/f"epoch_{epoch:03d}.pt" for epoch in epochs]
    if any(not path.exists() for path in checkpoints):
        raise FileNotFoundError("fold_00缺少完整epoch snapshots")

    static,_,static_cols=load_static(); outer=resolve_target(static,CFG.target_site)
    if outer!=int(gate["outer_index_excluded"]): raise RuntimeError("gate outer與目前target不一致")
    if list(static_cols)!=list(gate["static_columns"]): raise RuntimeError("gate static欄位不一致")
    weights=target_weights(gate,static,outer)
    # Weights are now locked before the outer dataset/truth is constructed.
    locked_weight_payload={
        "outer_index":outer,"siteid":str(static.loc[outer,"siteid"]),"sitename":str(static.loc[outer,"sitename"]),
        "weights":[{"epoch":epoch,"weight":float(weight)} for epoch,weight in zip(epochs,weights)],
        "uses_outer_pm25":False,
    }
    (output/"LOCKED_WEIGHTS_BEFORE_OUTER_TRUTH.json").write_text(json.dumps(locked_weight_payload,ensure_ascii=False,indent=2),encoding="utf-8")

    first=load_checkpoint(checkpoints[0]); train_idx=np.asarray(first["train_indices"],dtype=int)
    validation_idx=np.asarray(first["validation_indices"],dtype=int)
    if len(train_idx)!=60 or len(validation_idx)!=12 or outer in train_idx or outer in validation_idx:
        raise RuntimeError("fold_00 snapshot protocol不是60/12/1")
    scaler=scaler_from_checkpoint(first)
    cube,timestamps=build_or_load_hourly_cube(static)
    distance=haversine_matrix(static.longitude,static.latitude)
    static_scaled=standardize_static(static,static_cols,scaler)
    outer_ds=ColdStartStationDataset(
        [outer],train_idx,CFG.test_start,CFG.test_end,cube,timestamps,static,static_scaled,distance,scaler,
    )
    sanity=audit_dataset(outer_ds,max_samples=min(64,len(outer_ds)))
    if sanity["sampled_donor_counts"]!=[60]: raise RuntimeError("outer ensemble donors不是60")
    feature_builder=None
    if device.type=="cuda":
        feature_builder=DeviceFeatureBuilder(
            train_idx,cube,int(outer_ds.row_times.max()),timestamps,static,static_scaled,distance,scaler,outer,device,
        )
        loader=make_index_loader(outer_ds,False)
    else:
        loader=make_vectorized_loader(outer_ds,train_idx,cube,timestamps,static,static_scaled,distance,scaler,False)

    epoch_predictions=[]; base=None
    model=TCNTargetCrossAttention(static_dim=len(static_cols)).to(device)
    for epoch,path in zip(epochs,checkpoints):
        checkpoint=load_checkpoint(path)
        if int(checkpoint["epoch"])!=epoch or not np.array_equal(np.asarray(checkpoint["train_indices"]),train_idx):
            raise RuntimeError(f"epoch {epoch} snapshot metadata不一致")
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        _,_,predictions=validate_epoch(model,loader,device,static,timestamps,epoch,feature_builder)
        if base is None:
            base=predictions[["station_index","timestamp","y_true"]].copy()
        else:
            if not np.array_equal(base.station_index.to_numpy(),predictions.station_index.to_numpy()):
                raise RuntimeError("outer snapshots station alignment不同")
            if not np.array_equal(pd.to_datetime(base.timestamp).astype("int64"),pd.to_datetime(predictions.timestamp).astype("int64")):
                raise RuntimeError("outer snapshots timestamp alignment不同")
            if not np.allclose(base.y_true,predictions.y_true,rtol=0,atol=1e-6):
                raise RuntimeError("outer snapshots truth alignment不同")
        epoch_predictions.append(predictions.y_pred.to_numpy("float64"))
        print(f"outer ensemble inference epoch {epoch}/{epochs[-1]}",flush=True)
    matrix=np.stack(epoch_predictions,axis=1)
    base["y_pred"]=(matrix@weights).astype("float32")
    metrics=regression_metrics(base.y_true.to_numpy(),base.y_pred.to_numpy())
    coverage=float(len(base)/max(outer_ds.truth_rows,1))
    summary={
        **locked_weight_payload,"metrics":metrics,"truth_timestamps":int(outer_ds.truth_rows),
        "predicted_timestamps":len(base),"coverage":coverage,"donors":60,
        "validation_donors_used":0,"selection":"target static -> learned snapshot weights; outer truth not used",
    }
    base.insert(1,"siteid",str(static.loc[outer,"siteid"]))
    base.insert(2,"sitename",str(static.loc[outer,"sitename"]))
    base.to_csv(output/"outer_target_conditioned_predictions.csv",index=False,encoding="utf-8-sig",date_format="%Y-%m-%d %H:%M:%S")
    pd.DataFrame([{**metrics,"n":len(base),"coverage":coverage}]).to_csv(output/"outer_target_conditioned_metrics.csv",index=False,encoding="utf-8-sig")
    (output/"outer_target_conditioned_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
