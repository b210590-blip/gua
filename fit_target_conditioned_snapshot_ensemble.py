from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CFG
from data_pipeline import load_static
from train_formal import regression_metrics, resolve_target


def load_oof(root:Path,station_count:int,max_epochs:int):
    records={}
    station_fold={}
    for fold in range(6):
        fold_dir=root/f"fold_{fold:02d}"
        if not (fold_dir/"COMPLETE.json").exists():
            raise FileNotFoundError(f"fold {fold}尚未完成: {fold_dir}")
        arrays=[]
        for epoch in range(1,max_epochs+1):
            path=fold_dir/"epoch_predictions"/f"epoch_{epoch:03d}.npz"
            if not path.exists(): raise FileNotFoundError(path)
            arrays.append(np.load(path))
        base=arrays[0]
        station=np.asarray(base["station_index"],dtype=int)
        timestamp=np.asarray(base["timestamp_ns"],dtype="int64")
        truth=np.asarray(base["y_true"],dtype="float32")
        pred=[]
        for epoch,array in enumerate(arrays,start=1):
            if not np.array_equal(station,array["station_index"]) or not np.array_equal(timestamp,array["timestamp_ns"]):
                raise RuntimeError(f"fold {fold} epoch {epoch} prediction row alignment不同")
            if not np.allclose(truth,array["y_true"],rtol=0,atol=1e-6):
                raise RuntimeError(f"fold {fold} epoch {epoch} truth不同")
            pred.append(np.asarray(array["y_pred"],dtype="float32"))
        pred=np.stack(pred,axis=1)
        for idx in np.unique(station):
            if int(idx) in records: raise RuntimeError(f"station {idx}重複出現在OOF validation")
            keep=station==idx
            records[int(idx)]={"timestamp_ns":timestamp[keep],"y":truth[keep],"p":pred[keep]}
            station_fold[int(idx)]=fold
    if len(records)!=station_count-1:
        raise RuntimeError(f"OOF應有72站，實際{len(records)}")
    return records,station_fold


def station_error_covariance(record:dict) -> np.ndarray:
    error=record["p"].astype("float64")-record["y"].astype("float64")[:,None]
    return (error.T@error)/len(error)


def make_representation(x_train:np.ndarray,x_apply:np.ndarray,dim:int):
    if dim==0:
        return np.zeros((len(x_train),0)),np.zeros((len(x_apply),0)),{
            "mean":[],"std":[],"components":[],
        }
    mean=x_train.mean(axis=0); std=x_train.std(axis=0); std[std<1e-8]=1.0
    train_z=(x_train-mean)/std; apply_z=(x_apply-mean)/std
    _,_,vt=np.linalg.svd(train_z,full_matrices=False)
    components=vt[:min(dim,vt.shape[0])]
    return train_z@components.T,apply_z@components.T,{
        "mean":mean.tolist(),"std":std.tolist(),"components":components.tolist(),
    }


def fit_gate(covariances:np.ndarray,z:np.ndarray,l2:float,steps:int=1200):
    torch.manual_seed(20260901)
    c=torch.as_tensor(covariances,dtype=torch.float64)
    x=torch.as_tensor(z,dtype=torch.float64)
    epochs=c.shape[-1]
    intercept=torch.zeros(epochs,dtype=torch.float64,requires_grad=True)
    slope=torch.zeros((x.shape[1],epochs),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([intercept,slope],lr=0.05)
    best=None; best_loss=float("inf"); stale=0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits=intercept[None,:]+x@slope
        weights=torch.softmax(logits,dim=1)
        mse=torch.einsum("se,sef,sf->s",weights,c,weights)
        penalty=l2*torch.mean(torch.square(slope)) if slope.numel() else torch.zeros((),dtype=torch.float64)
        # Match the checkpoint criterion: every station has equal weight and
        # the optimized quantity is Macro station RMSE, not pooled sample MSE.
        loss=torch.sqrt(torch.clamp(mse,min=1e-12)).mean()+penalty
        loss.backward(); optimizer.step()
        value=float(loss.detach())
        if value<best_loss-1e-10:
            best_loss=value; best=(intercept.detach().clone(),slope.detach().clone()); stale=0
        else:
            stale+=1
            if stale>=150: break
    return best[0].numpy(),best[1].numpy()


def apply_gate(intercept:np.ndarray,slope:np.ndarray,z:np.ndarray) -> np.ndarray:
    logits=intercept[None,:]+z@slope
    logits-=logits.max(axis=1,keepdims=True)
    weights=np.exp(logits); weights/=weights.sum(axis=1,keepdims=True)
    return weights


def macro_rmse_from_cov(covariances,weights):
    mse=np.einsum("se,sef,sf->s",weights,covariances,weights)
    return float(np.mean(np.sqrt(np.maximum(mse,0.0))))


def select_hyperparameters(x,covariances,fold_ids,candidates):
    rows=[]
    for dim,l2 in candidates:
        scores=[]
        for fold in sorted(np.unique(fold_ids)):
            train=fold_ids!=fold; valid=~train
            z_train,z_valid,_=make_representation(x[train],x[valid],dim)
            intercept,slope=fit_gate(covariances[train],z_train,l2)
            weights=apply_gate(intercept,slope,z_valid)
            scores.append(macro_rmse_from_cov(covariances[valid],weights))
        rows.append({"pca_dim":dim,"l2":l2,"cv_macro_rmse":float(np.mean(scores)),"fold_macro_rmse":scores})
    return min(rows,key=lambda row:row["cv_macro_rmse"]),rows


def predictions_for_weights(records,station_ids,weights):
    frames=[]
    for station,w in zip(station_ids,weights):
        record=records[int(station)]
        pred=record["p"].astype("float64")@w
        frames.append(pd.DataFrame({
            "station_index":int(station),"timestamp_ns":record["timestamp_ns"],
            "y_true":record["y"],"y_pred":pred.astype("float32"),
        }))
    return pd.concat(frames,ignore_index=True)


def summarize_method(name,frame,static):
    station_rows=[]
    for station,group in frame.groupby("station_index",sort=True):
        metrics=regression_metrics(group.y_true.to_numpy(),group.y_pred.to_numpy())
        station_rows.append({
            "method":name,"station_index":int(station),"siteid":str(static.loc[station,"siteid"]),
            "sitename":str(static.loc[station,"sitename"]),"n":len(group),**metrics,
        })
    rows=pd.DataFrame(station_rows); pooled=regression_metrics(frame.y_true.to_numpy(),frame.y_pred.to_numpy())
    summary={
        "method":name,"stations":len(rows),"macro_rmse":float(rows.rmse.mean()),
        "macro_mae":float(rows.mae.mean()),"macro_r2":float(rows.r2.mean()),
        "macro_bias":float(rows.bias.mean()),"pooled_rmse":pooled["rmse"],
        "pooled_mae":pooled["mae"],"pooled_r2_exact":pooled["r2"],"pooled_bias":pooled["bias"],
    }
    return summary,rows


def main():
    root=Path(os.environ.get("DL_TCN_CROSSFIT_ROOT",str(CFG.formal_output_root/"crossfit_target_conditioned_snapshots")))
    output=Path(os.environ.get("DL_TCN_ENSEMBLE_OUTPUT",str(root/"target_conditioned_ensemble")))
    output.mkdir(parents=True,exist_ok=True)
    static,_,static_cols=load_static(); outer=resolve_target(static,CFG.target_site)
    records,station_fold=load_oof(root,len(static),CFG.max_epochs)
    station_ids=np.array(sorted(records),dtype=int)
    if outer in station_ids: raise RuntimeError("outer target不應出現在OOF selector labels")
    x=static.loc[station_ids,static_cols].apply(pd.to_numeric,errors="coerce").to_numpy("float64")
    # Imputation statistics use only the 72 known stations; outer static never
    # contributes to selector fitting.
    med=np.nanmedian(x,axis=0); x=np.where(np.isfinite(x),x,med)
    covariances=np.stack([station_error_covariance(records[int(i)]) for i in station_ids])
    fold_ids=np.array([station_fold[int(i)] for i in station_ids],dtype=int)
    candidates=[(dim,l2) for dim in (0,3,5,10) for l2 in (0.01,0.1,1.0,10.0)]
    selected,cv_rows=select_hyperparameters(x,covariances,fold_ids,candidates)

    # Strict gate-level six-fold predictions: each station's PM error curve is
    # excluded while its static features are transformed by the other folds.
    target_weights=np.zeros((len(station_ids),CFG.max_epochs),dtype="float64")
    global_soft_weights=np.zeros_like(target_weights)
    global_hard_weights=np.zeros_like(target_weights)
    nested_selected=[]
    for fold in sorted(np.unique(fold_ids)):
        train=fold_ids!=fold; valid=~train
        # Nested selection: this fold is absent not only from gate fitting but
        # also from PCA-dimension/L2 choice.
        inner_selected,_=select_hyperparameters(x[train],covariances[train],fold_ids[train],candidates)
        nested_selected.append({"heldout_fold":int(fold),**inner_selected})
        z_train,z_valid,_=make_representation(x[train],x[valid],int(inner_selected["pca_dim"]))
        intercept,slope=fit_gate(covariances[train],z_train,float(inner_selected["l2"]))
        target_weights[valid]=apply_gate(intercept,slope,z_valid)
        i0,s0=fit_gate(covariances[train],np.zeros((train.sum(),0)),0.0)
        global_soft_weights[valid]=apply_gate(i0,s0,np.zeros((valid.sum(),0)))
        macro_epoch_rmse=np.sqrt(np.maximum(np.diagonal(covariances[train],axis1=1,axis2=2),0.0)).mean(axis=0)
        global_hard_weights[valid,int(np.argmin(macro_epoch_rmse))]=1.0
    oracle_weights=np.zeros_like(target_weights)
    for row,cov in enumerate(covariances): oracle_weights[row,int(np.argmin(np.diag(cov)))]=1.0

    method_weights={
        "global_macro_hard_epoch_crossfit":global_hard_weights,
        "global_soft_snapshot_ensemble":global_soft_weights,
        "target_conditioned_snapshot_ensemble":target_weights,
        "station_oracle_hard_epoch":oracle_weights,
    }
    summaries=[]; station_metric_frames=[]
    for name,weights in method_weights.items():
        frame=predictions_for_weights(records,station_ids,weights)
        summary,station_rows=summarize_method(name,frame,static)
        summaries.append(summary); station_metric_frames.append(station_rows)
        frame.to_csv(output/f"{name}_predictions.csv.gz",index=False,compression="gzip")
    pd.DataFrame(summaries).to_csv(output/"performance_summary.csv",index=False,encoding="utf-8-sig")
    pd.concat(station_metric_frames,ignore_index=True).to_csv(output/"station_performance.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(target_weights,columns=[f"epoch_{i:03d}" for i in range(1,CFG.max_epochs+1)]).assign(
        station_index=station_ids,siteid=static.loc[station_ids,"siteid"].astype(str).to_numpy(),
        sitename=static.loc[station_ids,"sitename"].astype(str).to_numpy(),fold=fold_ids,
    ).to_csv(output/"crossvalidated_target_epoch_weights.csv",index=False,encoding="utf-8-sig")
    (output/"hyperparameter_cv.json").write_text(json.dumps({
        "deployable_selected_on_all_72":selected,"all_72_candidates":cv_rows,
        "nested_selection_for_oof_performance":nested_selected,
    },ensure_ascii=False,indent=2),encoding="utf-8")

    # Fit the deployable gate on all 72 OOF station curves.  Only known-station
    # static and OOF PM errors are used; outer PM2.5 is never read.
    dim=int(selected["pca_dim"]); z_all,_,rep=make_representation(x,x[:0],dim)
    intercept,slope=fit_gate(covariances,z_all,float(selected["l2"]))
    gate={
        "method":"target_conditioned_snapshot_ensemble","epochs":list(range(1,CFG.max_epochs+1)),
        "pca_dim":dim,"l2":float(selected["l2"]),"static_columns":static_cols,
        "known_station_indices":station_ids.tolist(),"outer_index_excluded":int(outer),
        "static_missing_median":med.tolist(),**rep,
        "intercept":intercept.tolist(),"slope":slope.tolist(),
        "warning":"Pseudo-target performance is gate-level cross-validation. Final outer use is leakage-free because outer PM2.5 was excluded from all six base trajectories and selector fitting.",
    }
    (output/"deployable_target_conditioned_gate.json").write_text(json.dumps(gate,ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(f"selected gate: PCA{dim}, l2={selected['l2']} | output={output}",flush=True)


if __name__=="__main__":
    main()
