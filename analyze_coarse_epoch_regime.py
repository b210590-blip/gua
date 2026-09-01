from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from config import CFG
from data_pipeline import load_static


REGIMES=("early","middle","late")
EPOCHS_BY_REGIME={
    "early":np.arange(1,6,dtype=int),
    "middle":np.arange(6,11,dtype=int),
    "late":np.arange(11,16,dtype=int),
}


def epoch_regime(epoch:int) -> str:
    if 1<=epoch<=5: return "early"
    if 6<=epoch<=10: return "middle"
    if 11<=epoch<=15: return "late"
    raise ValueError(f"epoch超出1..15: {epoch}")


def load_history(root:Path) -> pd.DataFrame:
    files=sorted(root.glob("fold_*/validation_station_metrics_all_epochs.csv"))
    if len(files)!=6: raise FileNotFoundError(f"應有6個fold station history，實際{len(files)}")
    history=pd.concat([pd.read_csv(path,encoding="utf-8-sig") for path in files],ignore_index=True)
    required={"station_index","siteid","sitename","epoch","n","rmse","mae","r2","bias"}
    missing=required-set(history.columns)
    if missing: raise RuntimeError(f"station history缺少欄位: {sorted(missing)}")
    history["station_index"]=history.station_index.astype(int)
    history["epoch"]=history.epoch.astype(int)
    if history.station_index.nunique()!=72: raise RuntimeError("history不是72個OOF stations")
    counts=history.groupby("station_index").epoch.nunique()
    if not (counts==CFG.max_epochs).all(): raise RuntimeError("不是每站都有完整15個epoch")
    return history


def oracle_table(history:pd.DataFrame) -> pd.DataFrame:
    oracle=(history.sort_values(["station_index","rmse","epoch"])
            .groupby("station_index",as_index=False).first())
    oracle=oracle[["station_index","siteid","sitename","epoch","rmse","r2"]].rename(columns={
        "epoch":"true_best_epoch","rmse":"oracle_rmse","r2":"oracle_r2",
    })
    oracle["true_regime"]=oracle.true_best_epoch.map(epoch_regime)
    return oracle


def make_models(seed:int=20260902):
    # Hyperparameters are fixed before LOSO; no held-out station is used to
    # choose depth, leaf size, or feature subset.
    tree=DecisionTreeClassifier(
        max_depth=3,min_samples_leaf=5,class_weight="balanced",random_state=seed,
    )
    forest=ExtraTreesClassifier(
        n_estimators=400,max_depth=5,min_samples_leaf=3,max_features="sqrt",
        class_weight="balanced",random_state=seed,n_jobs=-1,
    )
    return {
        "cart_depth3_leaf5":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",tree)]),
        "extra_trees_depth5_leaf3":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",forest)]),
    }


def choose_epoch_from_training_curves(history:pd.DataFrame,train_stations:np.ndarray,regime:str) -> int:
    allowed=EPOCHS_BY_REGIME[regime]
    score=(history.loc[
        history.station_index.isin(train_stations)&history.epoch.isin(allowed)
    ].groupby("epoch").rmse.mean())
    return int(score.idxmin())


def select_station_metric(history:pd.DataFrame,station:int,epoch:int) -> dict:
    row=history.loc[(history.station_index==station)&(history.epoch==epoch)]
    if len(row)!=1: raise RuntimeError(f"station={station}, epoch={epoch}不是唯一一列")
    return row.iloc[0].to_dict()


def bootstrap_stability(model,x_train,y_train,x_target,bootstraps:int,seed:int):
    if bootstraps<=0: return {regime:float("nan") for regime in REGIMES}
    rng=np.random.default_rng(seed); counts={regime:0 for regime in REGIMES}; n=len(y_train)
    for b in range(bootstraps):
        # Stratified station bootstrap keeps all three regimes represented.
        sampled=[]
        for regime in REGIMES:
            positions=np.flatnonzero(y_train==regime)
            sampled.extend(rng.choice(positions,size=len(positions),replace=True).tolist())
        estimator=clone(model)
        # Bootstrap is only a stability diagnostic. Fewer trees are enough and
        # avoid repeating the full 400-tree forest 7,200 times.
        if isinstance(estimator.named_steps["model"],ExtraTreesClassifier):
            estimator.set_params(model__n_estimators=50,model__n_jobs=1)
        estimator.fit(x_train[np.asarray(sampled)],y_train[np.asarray(sampled)])
        counts[str(estimator.predict(x_target)[0])]+=1
    return {regime:counts[regime]/bootstraps for regime in REGIMES}


def truth_sufficient_statistics(root:Path):
    result={}
    for fold in range(6):
        path=root/f"fold_{fold:02d}"/"epoch_predictions"/"epoch_001.npz"
        data=np.load(path); station=np.asarray(data["station_index"],dtype=int)
        truth=np.asarray(data["y_true"],dtype="float64")
        for idx in np.unique(station):
            values=truth[station==idx]
            result[int(idx)]={"n_truth":len(values),"sum_y":float(values.sum()),"sum_y2":float(np.square(values).sum())}
    return result


def summarize_selected(rows:pd.DataFrame,truth_stats:dict,name:str) -> dict:
    n=rows.n.to_numpy("float64"); rmse=rows.rmse.to_numpy("float64")
    total_n=float(n.sum()); sse=float(np.sum(n*np.square(rmse)))
    expected_n=sum(truth_stats[int(i)]["n_truth"] for i in rows.station_index)
    if int(total_n)!=int(expected_n):
        raise RuntimeError(f"{name}: metric n={int(total_n)} 與truth timestamps={expected_n}不一致")
    sum_y=sum(truth_stats[int(i)]["sum_y"] for i in rows.station_index)
    sum_y2=sum(truth_stats[int(i)]["sum_y2"] for i in rows.station_index)
    sst=float(sum_y2-sum_y*sum_y/total_n)
    return {
        "method":name,"stations":len(rows),"regime_accuracy":float(rows.regime_correct.mean()),
        "mean_absolute_epoch_error":float(rows.epoch_error_abs.mean()),
        "macro_rmse":float(rows.rmse.mean()),"macro_mae":float(rows.mae.mean()),
        "macro_r2":float(rows.r2.mean()),"macro_bias":float(rows.bias.mean()),
        "pooled_rmse":float(np.sqrt(sse/total_n)),
        "pooled_mae":float(np.sum(n*rows.mae)/total_n),
        "pooled_bias":float(np.sum(n*rows.bias)/total_n),
        "pooled_r2_exact":float(1.0-sse/sst),
        "mean_rmse_regret_vs_oracle":float(rows.rmse_regret_vs_oracle.mean()),
        "median_rmse_regret_vs_oracle":float(rows.rmse_regret_vs_oracle.median()),
    }


def main():
    root=Path(os.environ.get("DL_TCN_CROSSFIT_ROOT",str(CFG.formal_output_root/"crossfit_target_conditioned_snapshots")))
    output=Path(os.environ.get("DL_TCN_COARSE_OUTPUT",str(root/"coarse_epoch_regime_analysis")))
    output.mkdir(parents=True,exist_ok=True)
    bootstraps=int(os.environ.get("DL_TCN_COARSE_BOOTSTRAPS","100"))
    history=load_history(root); oracle=oracle_table(history)
    static,_,static_cols=load_static()
    station_ids=oracle.station_index.to_numpy(int)
    x=static.loc[station_ids,static_cols].apply(pd.to_numeric,errors="coerce").to_numpy("float64")
    y=oracle.true_regime.to_numpy(str)
    models=make_models(); all_predictions=[]; all_selected_metrics=[]
    print(f"72-station LOSO coarse classification | bootstraps={bootstraps}",flush=True)

    for model_name,base_model in models.items():
        model_rows=[]; selected_rows=[]
        for position,station in enumerate(station_ids):
            train=np.arange(len(station_ids))!=position
            estimator=clone(base_model); estimator.fit(x[train],y[train])
            predicted=str(estimator.predict(x[[position]])[0])
            proba={regime:0.0 for regime in REGIMES}
            if hasattr(estimator,"predict_proba"):
                values=estimator.predict_proba(x[[position]])[0]
                for label,value in zip(estimator.classes_,values): proba[str(label)]=float(value)
            stability=bootstrap_stability(
                base_model,x[train],y[train],x[[position]],bootstraps,20260902+position*101,
            )
            chosen_epoch=choose_epoch_from_training_curves(history,station_ids[train],predicted)
            metric=select_station_metric(history,int(station),chosen_epoch)
            oracle_row=oracle.loc[oracle.station_index==station].iloc[0]
            common={
                "model":model_name,"station_index":int(station),
                "siteid":str(oracle_row.siteid),"sitename":str(oracle_row.sitename),
                "true_best_epoch":int(oracle_row.true_best_epoch),"true_regime":str(oracle_row.true_regime),
                "predicted_regime":predicted,"regime_correct":predicted==oracle_row.true_regime,
                "selected_epoch":chosen_epoch,"epoch_error_abs":abs(chosen_epoch-int(oracle_row.true_best_epoch)),
                "oracle_rmse":float(oracle_row.oracle_rmse),"rmse_regret_vs_oracle":float(metric["rmse"]-oracle_row.oracle_rmse),
                **{f"prob_{key}":value for key,value in proba.items()},
                **{f"bootstrap_{key}":value for key,value in stability.items()},
                "bootstrap_predicted_regime_stability":float(stability[predicted]),
            }
            model_rows.append(common)
            selected_rows.append({**common,**{key:metric[key] for key in ("n","mae","rmse","r2","bias")}})
            if (position+1)%12==0:
                print(f"  {model_name}: {position+1}/72 stations",flush=True)
        pred_frame=pd.DataFrame(model_rows); selected_frame=pd.DataFrame(selected_rows)
        all_predictions.append(pred_frame); all_selected_metrics.append(selected_frame)

    predictions=pd.concat(all_predictions,ignore_index=True)
    selected_metrics=pd.concat(all_selected_metrics,ignore_index=True)
    truth_stats=truth_sufficient_statistics(root)
    summaries=[]; confusion_rows=[]
    for model_name in models:
        rows=selected_metrics[selected_metrics.model==model_name].copy()
        summary=summarize_selected(rows,truth_stats,model_name)
        summary.update({
            "balanced_accuracy":balanced_accuracy_score(rows.true_regime,rows.predicted_regime),
            "macro_f1":f1_score(rows.true_regime,rows.predicted_regime,labels=list(REGIMES),average="macro"),
        })
        summaries.append(summary)
        cm=confusion_matrix(rows.true_regime,rows.predicted_regime,labels=list(REGIMES))
        for i,true in enumerate(REGIMES):
            for j,pred in enumerate(REGIMES):
                confusion_rows.append({"model":model_name,"true_regime":true,"predicted_regime":pred,"stations":int(cm[i,j])})

    # Global hard LOSO and station oracle are comparison bounds.
    baseline_rows=[]; oracle_rows=[]
    for position,station in enumerate(station_ids):
        train_ids=np.delete(station_ids,position)
        global_epoch=int(history[history.station_index.isin(train_ids)].groupby("epoch").rmse.mean().idxmin())
        oracle_epoch=int(oracle.loc[oracle.station_index==station,"true_best_epoch"].iloc[0])
        true_regime=str(oracle.loc[oracle.station_index==station,"true_regime"].iloc[0])
        for name,epoch,target in (("global_macro_hard_epoch_loso",global_epoch,baseline_rows),("station_oracle_hard_epoch",oracle_epoch,oracle_rows)):
            metric=select_station_metric(history,int(station),epoch); oracle_rmse=float(oracle.loc[oracle.station_index==station,"oracle_rmse"].iloc[0])
            target.append({
                "model":name,"station_index":int(station),"n":metric["n"],"mae":metric["mae"],"rmse":metric["rmse"],
                "r2":metric["r2"],"bias":metric["bias"],"regime_correct":epoch_regime(epoch)==true_regime,
                "epoch_error_abs":abs(epoch-oracle_epoch),"rmse_regret_vs_oracle":metric["rmse"]-oracle_rmse,
            })
    summaries.append(summarize_selected(pd.DataFrame(baseline_rows),truth_stats,"global_macro_hard_epoch_loso"))
    summaries.append(summarize_selected(pd.DataFrame(oracle_rows),truth_stats,"station_oracle_hard_epoch"))

    # Fit interpretable final models on all 72 development labels. These rules
    # are for description/deployment to a truly unseen station, not OOF scoring.
    importance_rows=[]
    for name,model in models.items():
        model.fit(x,y); fitted=model.named_steps["model"]
        for feature,value in zip(static_cols,fitted.feature_importances_):
            importance_rows.append({"model":name,"feature":feature,"importance":float(value)})
        if name.startswith("cart"):
            rules=export_text(fitted,feature_names=list(static_cols))
            (output/"cart_rules.txt").write_text(rules,encoding="utf-8")

    summary_frame=pd.DataFrame(summaries)
    predictions.to_csv(output/"station_loso_regime_predictions.csv",index=False,encoding="utf-8-sig")
    selected_metrics.to_csv(output/"station_loso_selected_epoch_metrics.csv",index=False,encoding="utf-8-sig")
    summary_frame.to_csv(output/"coarse_regime_performance_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(confusion_rows).to_csv(output/"coarse_regime_confusion_matrix.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(importance_rows).sort_values(["model","importance"],ascending=[True,False]).to_csv(output/"coarse_regime_feature_importance.csv",index=False,encoding="utf-8-sig")
    protocol={
        "labels":{"early":"epoch 1-5","middle":"epoch 6-10","late":"epoch 11-15"},
        "features":"raw static 49; no PCA","evaluation":"station-label LOSO; fixed model hyperparameters",
        "bootstraps":bootstraps,"tcn_retraining":False,
        "limitation":"Development-set cross-fitted analysis, not fully nested independent outer evaluation; final clean external target remains the station excluded before all six TCN folds.",
    }
    (output/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nPerformance summary:",flush=True); print(summary_frame.to_string(index=False),flush=True)
    print(f"\nOutput: {output}",flush=True)


if __name__=="__main__":
    main()
