from __future__ import annotations

import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CFG, apply_runtime_profile
from data_pipeline import (
    ColdStartStationDataset,
    build_or_load_hourly_cube,
    fit_train_only_scaler,
    haversine_matrix,
    load_static,
    make_meta_crossfit_folds,
    standardize_static,
)
from model import TCNTargetCrossAttention
from sanity_check import audit_dataset, smoke_forward
from train_formal import (
    DeviceFeatureBuilder,
    amp_dtype_for,
    cpu_state_dict,
    make_grad_scaler,
    make_index_loader,
    make_loader,
    make_optimizer,
    make_station_sample_weights,
    make_vectorized_loader,
    resolve_target,
    seed_all,
    serializable_config,
    train_epoch,
    validate_epoch,
)


def requested_folds() -> list[int]:
    raw=os.environ.get("DL_TCN_CROSSFIT_FOLDS","all").strip().lower()
    if raw in {"","all"}:
        return list(range(6))
    folds=sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not folds or any(x<0 or x>5 for x in folds):
        raise ValueError("DL_TCN_CROSSFIT_FOLDS只能是all或0..5的逗號清單")
    return folds


def save_epoch_predictions() -> bool:
    return os.environ.get("DL_TCN_SAVE_EPOCH_PREDICTIONS", "1").strip() != "0"


def scaler_payload(scaler) -> dict:
    return {
        "dynamic_mean":scaler.dynamic_mean,"dynamic_std":scaler.dynamic_std,
        "static_mean":scaler.static_mean,"static_std":scaler.static_std,
        "static_median":scaler.static_median,
    }


def rng_payload() -> dict:
    payload={
        "python":random.getstate(),"numpy":np.random.get_state(),
        "torch":torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"]=torch.cuda.get_rng_state_all()
    return payload


def restore_rng(payload: dict) -> None:
    random.setstate(payload["python"]); np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state_all(payload["cuda"])


def run_fold(
    fold_id:int,train_idx:np.ndarray,val_idx:np.ndarray,outer:int,
    static:pd.DataFrame,clusters:np.ndarray,static_cols:list[str],cube,timestamps,distance,
    root:Path,device:torch.device,
) -> dict:
    # A fold must be reproducible whether it is run alone or after another
    # fold. Resume later restores the exact post-epoch RNG state.
    seed_all(CFG.seed+fold_id*1009)
    fold_dir=root/f"fold_{fold_id:02d}"
    epoch_dir=fold_dir/"epoch_checkpoints"; prediction_dir=fold_dir/"epoch_predictions"
    station_metric_path=fold_dir/"validation_station_metrics_all_epochs.csv"
    paths=[fold_dir,epoch_dir]
    if save_epoch_predictions(): paths.append(prediction_dir)
    for path in paths: path.mkdir(parents=True,exist_ok=True)
    complete_path=fold_dir/"COMPLETE.json"
    if complete_path.exists():
        print(f"fold {fold_id}: COMPLETE exists, skip",flush=True)
        return json.loads(complete_path.read_text(encoding="utf-8"))

    if len(train_idx)!=60 or len(val_idx)!=12 or outer in train_idx or outer in val_idx:
        raise RuntimeError(f"fold {fold_id} protocol不是60/12且outer排除")
    scaler=fit_train_only_scaler(cube,timestamps,static,static_cols,train_idx)
    static_scaled=standardize_static(static,static_cols,scaler)
    train_ds=ColdStartStationDataset(
        train_idx,train_idx,CFG.train_start,CFG.train_end,cube,timestamps,
        static,static_scaled,distance,scaler,
    )
    val_ds=ColdStartStationDataset(
        val_idx,train_idx,CFG.train_start,CFG.train_end,cube,timestamps,
        static,static_scaled,distance,scaler,
    )
    if audit_dataset(train_ds,max_samples=32)["sampled_donor_counts"]!=[59]:
        raise RuntimeError(f"fold {fold_id} training donors不是59")
    if audit_dataset(val_ds,max_samples=32)["sampled_donor_counts"]!=[60]:
        raise RuntimeError(f"fold {fold_id} validation donors不是60")

    base_model=TCNTargetCrossAttention(static_dim=len(static_cols)).to(device)
    model=base_model
    feature_builder=None
    if device.type=="cuda":
        max_time_index=int(max(train_ds.row_times.max(),val_ds.row_times.max()))
        feature_builder=DeviceFeatureBuilder(
            train_idx,cube,max_time_index,timestamps,static,static_scaled,distance,scaler,outer,device,
        )
        train_loader=make_index_loader(train_ds,True); val_loader=make_index_loader(val_ds,False)
    else:
        train_loader=make_vectorized_loader(train_ds,train_idx,cube,timestamps,static,static_scaled,distance,scaler,True)
        val_loader=make_vectorized_loader(val_ds,train_idx,cube,timestamps,static,static_scaled,distance,scaler,False)
    smoke=smoke_forward(base_model,make_loader(train_ds,False,CFG.smoke_batch_size),device)
    base_model.zero_grad(set_to_none=True)
    optimizer,fused=make_optimizer(base_model,device)
    if CFG.compile_mode != "off":
        if not hasattr(torch, "compile"):
            raise RuntimeError("目前PyTorch沒有torch.compile，請設DL_TCN_COMPILE_MODE=off")
        model=torch.compile(
            base_model,mode=CFG.compile_mode,fullgraph=False,dynamic=True,
        )
    amp_dtype=amp_dtype_for(device); grad_scaler=make_grad_scaler(amp_dtype==torch.float16)
    station_weights=make_station_sample_weights(train_ds,len(static),device)

    split={
        "fold":fold_id,"outer_index":int(outer),"outer_siteid":str(static.loc[outer,"siteid"]),
        "train_indices":train_idx.tolist(),"validation_indices":val_idx.tolist(),
        "train_siteids":static.loc[train_idx,"siteid"].astype(str).tolist(),
        "validation_siteids":static.loc[val_idx,"siteid"].astype(str).tolist(),
        "train_clusters":sorted(np.unique(clusters[train_idx]).astype(int).tolist()),
        "validation_clusters":sorted(np.unique(clusters[val_idx]).astype(int).tolist()),
        "training_donors":59,"validation_donors":60,
    }
    (fold_dir/"split.json").write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding="utf-8")
    (fold_dir/"sanity.json").write_text(json.dumps({"split":split,"model":smoke},ensure_ascii=False,indent=2),encoding="utf-8")

    resume_path=fold_dir/"last_training_state.pt"
    start_epoch=1; history=[]; fold_started=time.perf_counter()
    if resume_path.exists():
        resume=torch.load(resume_path,map_location=device,weights_only=False)
        if not np.array_equal(np.asarray(resume["train_indices"]),train_idx) or not np.array_equal(np.asarray(resume["validation_indices"]),val_idx):
            raise RuntimeError(f"fold {fold_id} resume split不一致")
        base_model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        grad_scaler.load_state_dict(resume["grad_scaler_state_dict"])
        restore_rng(resume["rng_state"])
        start_epoch=int(resume["epoch"])+1
        history=list(resume.get("history",[]))
        print(f"fold {fold_id}: resume from epoch {start_epoch}",flush=True)
    print(
        f"fold {fold_id}: train={len(train_ds):,} valid={len(val_ds):,} "
        f"batch={CFG.batch_size} epochs={CFG.max_epochs} device={device}",flush=True,
    )

    for epoch in range(start_epoch,CFG.max_epochs+1):
        epoch_started=time.perf_counter()
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
        train_started=time.perf_counter()
        train_loss=train_epoch(model,train_loader,optimizer,grad_scaler,device,epoch,feature_builder,station_weights)
        if device.type=="cuda": torch.cuda.synchronize(device)
        train_seconds=time.perf_counter()-train_started
        validation_started=time.perf_counter()
        metrics,station_metrics,predictions=validate_epoch(model,val_loader,device,static,timestamps,epoch,feature_builder)
        if device.type=="cuda": torch.cuda.synchronize(device)
        validation_seconds=time.perf_counter()-validation_started
        runtime=time.perf_counter()-epoch_started
        peak=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else 0.0
        row={
            "fold":fold_id,"epoch":epoch,"train_loss":train_loss,
            **{f"validation_{k}":v for k,v in metrics.items()},
            "train_seconds":train_seconds,"validation_seconds":validation_seconds,
            "epoch_runtime_seconds":runtime,"gpu_peak_vram_mb":peak,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(fold_dir/"training_history.csv",index=False,encoding="utf-8-sig")
        station_metrics.insert(0,"fold",fold_id); station_metrics.insert(1,"epoch",epoch)
        if station_metric_path.exists():
            previous_station_metrics=pd.read_csv(station_metric_path,encoding="utf-8-sig")
            previous_station_metrics=previous_station_metrics[previous_station_metrics["epoch"]!=epoch]
            station_history=pd.concat([previous_station_metrics,station_metrics],ignore_index=True)
        else:
            station_history=station_metrics
        station_history.sort_values(["epoch","station_index"]).to_csv(
            station_metric_path,index=False,encoding="utf-8-sig"
        )
        if save_epoch_predictions():
            np.savez_compressed(
                prediction_dir/f"epoch_{epoch:03d}.npz",
                station_index=predictions["station_index"].to_numpy("int16"),
                timestamp_ns=pd.to_datetime(predictions["timestamp"]).astype("int64").to_numpy(),
                y_true=predictions["y_true"].to_numpy("float32"),
                y_pred=predictions["y_pred"].to_numpy("float32"),
            )
        checkpoint={
            "fold":fold_id,"epoch":epoch,"model_state_dict":cpu_state_dict(base_model),
            "validation_metrics":metrics,"train_indices":train_idx,"validation_indices":val_idx,
            "outer_index_excluded":outer,"static_columns":static_cols,"scaler":scaler_payload(scaler),
            "config":serializable_config(),"torch_version":torch.__version__,
        }
        torch.save(checkpoint,epoch_dir/f"epoch_{epoch:03d}.pt")
        torch.save({
            **checkpoint,"optimizer_state_dict":optimizer.state_dict(),
            "grad_scaler_state_dict":grad_scaler.state_dict(),"rng_state":rng_payload(),"history":history,
        },resume_path)
        print(
            f"fold {fold_id} epoch {epoch}: MacroRMSE={metrics['macro_station_rmse']:.4f} "
            f"pooledR2={metrics['r2']:.4f} train={train_seconds:.1f}s "
            f"valid={validation_seconds:.1f}s total={runtime:.1f}s",flush=True,
        )

    summary={
        **split,"epochs_completed":CFG.max_epochs,"runtime_seconds":time.perf_counter()-fold_started,
        "snapshot_directory":str(epoch_dir),"prediction_directory":str(prediction_dir),
        "epoch_predictions_saved":save_epoch_predictions(),
        "fused_adamw":fused,"device":str(device),"compile_mode":CFG.compile_mode,
    }
    complete_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    if resume_path.exists():
        resume_path.unlink()
    del model,base_model,optimizer,grad_scaler,feature_builder,train_loader,val_loader,train_ds,val_ds
    gc.collect()
    if device.type=="cuda": torch.cuda.empty_cache()
    return summary


def main() -> None:
    seed_all(CFG.seed)
    runtime=apply_runtime_profile(CFG)
    device=CFG.device
    torch.set_num_threads(min(os.cpu_count() or 1,12))
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=CFG.enable_tf32
        torch.backends.cudnn.allow_tf32=CFG.enable_tf32
        torch.backends.cudnn.benchmark=True
        torch.set_float32_matmul_precision("high")
    root=Path(os.environ.get("DL_TCN_CROSSFIT_ROOT",str(CFG.formal_output_root/"crossfit_target_conditioned_snapshots")))
    root.mkdir(parents=True,exist_ok=True)
    static,clusters,static_cols=load_static(); outer=resolve_target(static,CFG.target_site)
    folds=make_meta_crossfit_folds(clusters,outer)
    manifest={
        "protocol":"outer excluded; 6 disjoint folds; each fold 60 train / 12 unseen validation",
        "outer_index":outer,"outer_siteid":str(static.loc[outer,"siteid"]),"outer_sitename":str(static.loc[outer,"sitename"]),
        "epochs":CFG.max_epochs,"requested_folds":requested_folds(),"runtime_profile":runtime,
        "folds":[{"fold":i,"train":tr.tolist(),"validation":va.tolist()} for i,(tr,va) in enumerate(folds)],
    }
    (root/"crossfit_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    cube,timestamps=build_or_load_hourly_cube(static); distance=haversine_matrix(static.longitude,static.latitude)
    summaries=[]
    for fold_id in requested_folds():
        train_idx,val_idx=folds[fold_id]
        summaries.append(run_fold(fold_id,train_idx,val_idx,outer,static,clusters,static_cols,cube,timestamps,distance,root,device))
    print(json.dumps({"status":"done","root":str(root),"folds":[x["fold"] for x in summaries]},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
