from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from data_pipeline import load_static
from evaluate_outer import load_checkpoint


REQUIRED_STATION_COLUMNS=("epoch","station_index","siteid","sitename","n","mae","rmse","r2","bias")


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def analysis_paths():
    raw=os.environ.get("DL_TCN_ANALYSIS_DIR","").strip()
    if not raw:
        raise RuntimeError("請設定DL_TCN_ANALYSIS_DIR，指向dropout-only training輸出資料夾")
    source=Path(raw)
    output=Path(os.environ.get("DL_TCN_ANALYSIS_OUTPUT",str(source/"target_aware_epoch_analysis")))
    required={
        "station_history":source/"validation_station_metrics_all_epochs.csv",
        "training_history":source/"training_history.csv",
        "checkpoint":source/"best_checkpoint.pt",
    }
    missing=[str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少分析輸入:\n"+"\n".join(missing))
    output.mkdir(parents=True,exist_ok=True)
    return source,output,required


def validate_histories(station_history,training_history,validation_indices):
    missing=[column for column in REQUIRED_STATION_COLUMNS if column not in station_history.columns]
    if missing:
        raise RuntimeError(f"station history缺少欄位: {missing}")
    for column in ("epoch","validation_rmse","validation_macro_station_rmse"):
        if column not in training_history.columns:
            raise RuntimeError(f"training history缺少欄位: {column}")
    station_history=station_history.copy()
    station_history["station_index"]=station_history.station_index.astype(int)
    station_history["epoch"]=station_history.epoch.astype(int)
    expected=set(np.asarray(validation_indices,dtype=int).tolist())
    actual=set(station_history.station_index.unique().tolist())
    if actual!=expected:
        raise RuntimeError(f"epoch history stations與checkpoint validation split不一致: history={sorted(actual)}, checkpoint={sorted(expected)}")
    duplicated=station_history.duplicated(["station_index","epoch"])
    if duplicated.any():
        raise RuntimeError("station history有重複station×epoch rows")
    epoch_sets=station_history.groupby("station_index").epoch.apply(lambda x:tuple(sorted(x))).tolist()
    if len(set(epoch_sets))!=1:
        raise RuntimeError("不同validation stations的epoch coverage不一致")
    numeric=("n","mae","rmse","r2","bias")
    for column in numeric:
        station_history[column]=pd.to_numeric(station_history[column],errors="coerce")
    if station_history[["n","mae","rmse","bias"]].isna().any().any():
        raise RuntimeError("station history必要指標含NaN")
    return station_history


def static_matrix_from_checkpoint(static,static_cols,checkpoint):
    saved=checkpoint.get("scaler",{})
    required=("static_mean","static_std","static_median")
    if any(key not in saved for key in required):
        raise RuntimeError("checkpoint缺少train-only static scaler")
    if list(checkpoint.get("static_columns",[]))!=list(static_cols):
        raise RuntimeError("checkpoint與目前static欄位名稱/順序不一致")
    mean=np.asarray(saved["static_mean"],dtype="float64")
    std=np.asarray(saved["static_std"],dtype="float64")
    median=np.asarray(saved["static_median"],dtype="float64")
    x=static[static_cols].apply(pd.to_numeric,errors="coerce").to_numpy("float64")
    x=np.where(np.isfinite(x),x,median)
    x=(x-mean)/std
    if x.shape[1]!=49 or not np.isfinite(x).all():
        raise RuntimeError(f"static standardized matrix錯誤: shape={x.shape}")
    return x


def similarity_matrix(x,method):
    if method=="rbf_euclidean":
        delta=x[:,None,:]-x[None,:,:]
        distance=np.sqrt(np.sum(np.square(delta),axis=-1))
        off=distance[np.triu_indices(len(x),1)]
        sigma=float(np.median(off[off>0])) if np.any(off>0) else 1.0
        similarity=np.exp(-0.5*np.square(distance/sigma))
        return similarity,distance,{"method":method,"rbf_sigma":sigma}
    if method=="cosine":
        norm=np.linalg.norm(x,axis=1,keepdims=True)
        unit=x/np.maximum(norm,1e-12)
        cosine=np.clip(unit@unit.T,-1.0,1.0)
        similarity=np.clip((cosine+1.0)/2.0,0.0,1.0)
        distance=1.0-cosine
        return similarity,distance,{"method":method}
    raise ValueError("DL_TCN_SIMILARITY只支援rbf_euclidean或cosine")


def nearest_available_epoch(value,available):
    return int(min(available,key=lambda epoch:(abs(epoch-value),epoch)))


def leave_one_out_epoch_predictions(similarity,true_best,available_epochs):
    station_indices=true_best.index.to_numpy(int)
    best_values=true_best.to_numpy(float)
    predicted={}
    for i,station in enumerate(station_indices):
        weights=similarity[i].copy()
        weights[i]=0.0
        if not np.isfinite(weights).all() or weights.sum()<=0:
            weights=np.ones(len(station_indices),dtype="float64"); weights[i]=0.0
        raw=float(np.sum(weights*best_values)/np.sum(weights))
        rounded=int(np.floor(raw+0.5))
        predicted[int(station)]=nearest_available_epoch(rounded,available_epochs)
    return predicted


def select_station_rows(station_history,epochs_by_station,method,true_best):
    rows=[]
    indexed=station_history.set_index(["station_index","epoch"],drop=False)
    for station,epoch in epochs_by_station.items():
        row=indexed.loc[(int(station),int(epoch))]
        oracle_epoch=int(true_best.loc[int(station)])
        oracle_rmse=float(indexed.loc[(int(station),oracle_epoch),"rmse"])
        rows.append({
            "method":method,"selected_epoch":int(epoch),"true_best_epoch":oracle_epoch,
            "epoch_error":int(epoch)-oracle_epoch,"absolute_epoch_error":abs(int(epoch)-oracle_epoch),
            "rmse_regret_vs_station_oracle":float(row.rmse)-oracle_rmse,
            **{column:row[column] for column in REQUIRED_STATION_COLUMNS if column!="epoch"},
        })
    return pd.DataFrame(rows)


def summarize_performance(selected):
    rows=[]
    for method,frame in selected.groupby("method",sort=False):
        n=frame.n.to_numpy(float); total=n.sum()
        rows.append({
            "method":method,
            "stations":int(len(frame)),
            "macro_rmse":float(frame.rmse.mean()),
            "pooled_rmse_from_station_sse":float(np.sqrt(np.sum(n*np.square(frame.rmse))/total)),
            "macro_mae":float(frame.mae.mean()),
            "sample_weighted_mae":float(np.sum(n*frame.mae)/total),
            "macro_r2":float(frame.r2.mean()),
            "sample_weighted_station_r2":float(np.sum(n*frame.r2)/total),
            "macro_bias":float(frame.bias.mean()),
            "sample_weighted_bias":float(np.sum(n*frame.bias)/total),
            "mean_absolute_epoch_error":float(frame.absolute_epoch_error.mean()),
            "median_absolute_epoch_error":float(frame.absolute_epoch_error.median()),
            "exact_epoch_rate":float((frame.absolute_epoch_error==0).mean()),
            "within_one_epoch_rate":float((frame.absolute_epoch_error<=1).mean()),
            "mean_rmse_regret_vs_station_oracle":float(frame.rmse_regret_vs_station_oracle.mean()),
        })
    return pd.DataFrame(rows)


def save_similarity(similarity,validation_indices,static,output):
    ids=static.loc[validation_indices,"siteid"].astype(str).tolist()
    matrix=pd.DataFrame(similarity,index=ids,columns=ids)
    matrix.index.name="siteid"
    matrix.to_csv(output/"station_similarity_matrix.csv",encoding="utf-8-sig")
    long=[]
    for i,source in enumerate(validation_indices):
        for j,reference in enumerate(validation_indices):
            long.append({
                "station_index":int(source),"siteid":str(static.loc[source,"siteid"]),"sitename":str(static.loc[source,"sitename"]),
                "reference_station_index":int(reference),"reference_siteid":str(static.loc[reference,"siteid"]),
                "reference_sitename":str(static.loc[reference,"sitename"]),"similarity":float(similarity[i,j]),
            })
    pd.DataFrame(long).to_csv(output/"station_similarity_long.csv",index=False,encoding="utf-8-sig")


def save_heatmap(similarity,validation_indices,static,output):
    plt=pyplot()
    labels=static.loc[validation_indices,"siteid"].astype(str).tolist()
    fig,axis=plt.subplots(figsize=(9,8))
    image=axis.imshow(similarity,vmin=0,vmax=1,cmap="viridis")
    axis.set_xticks(range(len(labels)),labels,rotation=45,ha="right")
    axis.set_yticks(range(len(labels)),labels)
    axis.set_xlabel("Reference validation station siteid")
    axis.set_ylabel("Pseudo-target validation station siteid")
    axis.set_title("Static/environment similarity (labels excluded)")
    fig.colorbar(image,ax=axis,label="similarity")
    fig.tight_layout(); fig.savefig(output/"station_similarity_heatmap.png",dpi=180); plt.close(fig)


def save_pca(x,train_indices,validation_indices,clusters,static,true_best,output):
    plt=pyplot()
    train=x[train_indices]
    center=train.mean(axis=0)
    _,_,vt=np.linalg.svd(train-center,full_matrices=False)
    coordinates=(x-center)@vt[:2].T
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    axes[0].scatter(coordinates[train_indices,0],coordinates[train_indices,1],c=clusters[train_indices],cmap="tab10",s=25,alpha=.30,label="60 train")
    scatter=axes[0].scatter(coordinates[validation_indices,0],coordinates[validation_indices,1],c=clusters[validation_indices],cmap="tab10",s=80,edgecolor="black",label="12 validation")
    for station in validation_indices:
        axes[0].annotate(str(static.loc[station,"siteid"]),coordinates[station],fontsize=8,xytext=(3,3),textcoords="offset points")
    axes[0].set_title("Train-fitted static PCA, colored by existing cluster")
    axes[0].legend()
    epoch_values=np.array([true_best.loc[int(station)] for station in validation_indices])
    epoch_scatter=axes[1].scatter(coordinates[validation_indices,0],coordinates[validation_indices,1],c=epoch_values,cmap="plasma",s=90,edgecolor="black")
    for station in validation_indices:
        axes[1].annotate(str(static.loc[station,"siteid"]),coordinates[station],fontsize=8,xytext=(3,3),textcoords="offset points")
    axes[1].set_title("Same PCA, colored by true station-best epoch")
    fig.colorbar(epoch_scatter,ax=axes[1],label="true best epoch")
    for axis in axes:
        axis.set_xlabel("PC1"); axis.set_ylabel("PC2"); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"station_static_pca.png",dpi=180); plt.close(fig)


def main():
    _,output,paths=analysis_paths()
    checkpoint=load_checkpoint(paths["checkpoint"])
    static,clusters,static_cols=load_static()
    train_indices=np.asarray(checkpoint.get("train_indices"),dtype=int)
    validation_indices=np.asarray(checkpoint.get("validation_indices"),dtype=int)
    if len(train_indices)!=60 or len(validation_indices)!=12:
        raise RuntimeError("checkpoint不是60 train / 12 validation")
    station_history=pd.read_csv(paths["station_history"],encoding="utf-8-sig")
    training_history=pd.read_csv(paths["training_history"],encoding="utf-8-sig")
    station_history=validate_histories(station_history,training_history,validation_indices)
    available_epochs=sorted(station_history.epoch.unique().astype(int).tolist())

    best_rows=(station_history.sort_values(["station_index","rmse","epoch"])
               .groupby("station_index",as_index=False).first())
    true_best=(best_rows.set_index("station_index").epoch.astype(int)
               .reindex(validation_indices))
    if true_best.isna().any():
        raise RuntimeError("部分validation station缺少true best epoch")
    global_epoch=int(training_history.loc[training_history.validation_rmse.idxmin(),"epoch"])
    macro_epoch=int(training_history.loc[training_history.validation_macro_station_rmse.idxmin(),"epoch"])

    x=static_matrix_from_checkpoint(static,static_cols,checkpoint)
    validation_x=x[validation_indices]
    method=os.environ.get("DL_TCN_SIMILARITY","rbf_euclidean").strip().lower()
    similarity,_,similarity_meta=similarity_matrix(validation_x,method)
    predicted=leave_one_out_epoch_predictions(similarity,true_best,available_epochs)
    global_selection={int(station):global_epoch for station in validation_indices}
    macro_selection={int(station):macro_epoch for station in validation_indices}
    oracle_selection={int(station):int(true_best.loc[int(station)]) for station in validation_indices}

    selected=pd.concat([
        select_station_rows(station_history,global_selection,"global_validation_rmse",true_best),
        select_station_rows(station_history,macro_selection,"macro_station_rmse",true_best),
        select_station_rows(station_history,predicted,"target_aware_static_similarity_loso",true_best),
        select_station_rows(station_history,oracle_selection,"station_oracle",true_best),
    ],ignore_index=True)
    performance=summarize_performance(selected)

    comparison=best_rows[["station_index","siteid","sitename"]].copy()
    comparison["true_best_epoch"]=comparison.station_index.map(true_best)
    comparison["global_epoch"]=global_epoch
    comparison["macro_epoch"]=macro_epoch
    comparison["predicted_epoch"]=comparison.station_index.map(predicted)
    comparison["epoch_error"]=comparison.predicted_epoch-comparison.true_best_epoch
    comparison["absolute_epoch_error"]=comparison.epoch_error.abs()
    comparison["global_absolute_epoch_error"]=(comparison.global_epoch-comparison.true_best_epoch).abs()
    comparison["macro_absolute_epoch_error"]=(comparison.macro_epoch-comparison.true_best_epoch).abs()
    comparison=comparison.sort_values("station_index")

    station_history[list(REQUIRED_STATION_COLUMNS)].sort_values(["station_index","epoch"]).to_csv(
        output/"station_epoch_matrix.csv",index=False,encoding="utf-8-sig"
    )
    comparison.to_csv(output/"epoch_selection_by_station.csv",index=False,encoding="utf-8-sig")
    selected.to_csv(output/"selected_epoch_station_performance.csv",index=False,encoding="utf-8-sig")
    performance.to_csv(output/"epoch_selection_performance_summary.csv",index=False,encoding="utf-8-sig")
    save_similarity(similarity,validation_indices,static,output)
    save_heatmap(similarity,validation_indices,static,output)
    save_pca(x,train_indices,validation_indices,clusters,static,true_best,output)

    perf=performance.set_index("method")
    target=perf.loc["target_aware_static_similarity_loso"]
    global_result=perf.loc["global_validation_rmse"]
    macro_result=perf.loc["macro_station_rmse"]
    conclusion={
        "global_epoch":global_epoch,"macro_epoch":macro_epoch,
        "available_epochs":available_epochs,"similarity":similarity_meta,
        "epoch_label_sources":"other 11 validation pseudo-targets only; held-out target excluded",
        "train_stations_used_for":"checkpoint train-only static normalization and PCA basis only",
        "target_pm25_used_in_similarity":False,
        "target_aware_macro_rmse_change_vs_global":float(target.macro_rmse-global_result.macro_rmse),
        "target_aware_macro_rmse_change_vs_macro":float(target.macro_rmse-macro_result.macro_rmse),
        "target_aware_mae_epoch_error_change_vs_global":float(target.mean_absolute_epoch_error-global_result.mean_absolute_epoch_error),
        "target_aware_improves_macro_rmse_vs_global":bool(target.macro_rmse<global_result.macro_rmse),
        "target_aware_closer_to_station_best_than_global":bool(target.mean_absolute_epoch_error<global_result.mean_absolute_epoch_error),
        "hypothesis_supported_on_12_station_loso":bool(
            target.macro_rmse<global_result.macro_rmse
            and target.mean_absolute_epoch_error<global_result.mean_absolute_epoch_error
        ),
        "important_limitations":[
            "Only 12 validation pseudo-targets provide unseen-station epoch labels.",
            "Exact pooled R2 for heterogeneous selected epochs cannot be reconstructed from station metrics alone; comparisons report macro and sample-weighted station R2.",
            "Current trainer stores only the best checkpoint, so this analysis validates selection retrospectively from epoch-wise metrics; deployment requires saving candidate epoch checkpoints.",
        ],
    }
    (output/"target_aware_epoch_conclusion.json").write_text(json.dumps(conclusion,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output_dir":str(output),**conclusion},ensure_ascii=False,indent=2))
    print("\nPerformance summary:")
    print(performance.to_string(index=False))


if __name__=="__main__":
    main()
