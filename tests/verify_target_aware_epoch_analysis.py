import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

import analyze_target_aware_epoch as analysis
from config import CFG
from data_pipeline import choose_split, load_static
from train_formal import resolve_target


def main_test():
    static,clusters,static_cols=load_static()
    outer=resolve_target(static,CFG.target_site)
    train_idx,val_idx=choose_split(clusters,outer)
    sx=static[static_cols].apply(pd.to_numeric,errors="coerce").to_numpy("float32")
    median=np.nanmedian(sx[train_idx],axis=0)
    train=np.where(np.isfinite(sx[train_idx]),sx[train_idx],median)
    mean=train.mean(axis=0); std=train.std(axis=0); std[std<1e-6]=1.0
    checkpoint={
        "train_indices":train_idx,"validation_indices":val_idx,"outer_index_excluded":outer,
        "static_columns":static_cols,
        "scaler":{"static_mean":mean,"static_std":std,"static_median":median},
    }
    station_rows=[]
    epochs=[1,2,3,4]
    for position,station in enumerate(val_idx):
        best=1+(position%4)
        for epoch in epochs:
            rmse=5.0+0.1*position+0.4*(epoch-best)**2
            station_rows.append({
                "epoch":epoch,"station_index":int(station),"siteid":str(static.loc[station,"siteid"]),
                "sitename":str(static.loc[station,"sitename"]),"n":100+position,
                "mae":rmse*0.8,"rmse":rmse,"r2":0.8-rmse/100,"bias":0.1*(epoch-best),
            })
    training_history=pd.DataFrame({
        "epoch":epochs,
        "validation_rmse":[6.0,5.5,5.7,6.2],
        "validation_macro_station_rmse":[6.1,5.8,5.6,5.9],
    })
    with tempfile.TemporaryDirectory() as temporary:
        source=Path(temporary)/"source"; output=Path(temporary)/"output"
        source.mkdir()
        pd.DataFrame(station_rows).to_csv(source/"validation_station_metrics_all_epochs.csv",index=False,encoding="utf-8-sig")
        training_history.to_csv(source/"training_history.csv",index=False,encoding="utf-8-sig")
        torch.save(checkpoint,source/"best_checkpoint.pt")
        os.environ["DL_TCN_ANALYSIS_DIR"]=str(source)
        os.environ["DL_TCN_ANALYSIS_OUTPUT"]=str(output)
        os.environ["DL_TCN_SIMILARITY"]="rbf_euclidean"
        # The local minimal test runtime need not install plotting packages;
        # Colab runs the real matplotlib functions through requirements-colab.
        analysis.save_heatmap=lambda similarity,validation_indices,static,output: (output/"station_similarity_heatmap.png").write_bytes(b"smoke")
        analysis.save_pca=lambda x,train_indices,validation_indices,clusters,static,true_best,output: (output/"station_static_pca.png").write_bytes(b"smoke")
        analysis.main()
        expected=(
            "station_epoch_matrix.csv","station_similarity_matrix.csv","station_similarity_long.csv",
            "station_similarity_heatmap.png","station_static_pca.png","epoch_selection_by_station.csv",
            "selected_epoch_station_performance.csv","epoch_selection_performance_summary.csv",
            "primary_performance_comparison.csv","nested_loso_inner_method_scores.csv",
            "station_best_epoch_plateau_diagnostics.csv","similarity_epoch_gap_diagnostics.csv",
            "target_aware_epoch_conclusion.json",
        )
        missing=[name for name in expected if not (output/name).is_file()]
        if missing: raise RuntimeError(f"analysis outputs missing: {missing}")
        comparison=pd.read_csv(output/"epoch_selection_by_station.csv")
        performance=pd.read_csv(output/"epoch_selection_performance_summary.csv")
        required_methods={"global_validation_rmse","macro_station_rmse","target_aware_static_similarity_loso","target_aware_nested_loso","station_oracle"}
        if len(comparison)!=12 or not required_methods.issubset(set(performance.method)):
            raise RuntimeError("analysis output rows/methods錯誤")
        print({"analysis_smoke":True,"stations":len(comparison),"epochs":len(epochs),"outputs":len(expected)})


if __name__=="__main__":
    main_test()
