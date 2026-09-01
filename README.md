# Unseen-station PM2.5 TCN + cross-attention

這是已通過protocol review與pipeline等價測試的Colab版本。固定設定為60 training stations／12 validation stations、24小時history、11個dynamic channels、49個static features、shared causal TCN與target-conditioned cross-attention。`train_formal.py`只做60/12模型選擇；`evaluate_outer.py`在模型選擇鎖定後，讀取最佳checkpoint並對1個outer target做一次正式test。此階段不做72站refit，也不搜尋超參數。

## Colab快速執行

先在Colab選擇GPU runtime，然後依序執行：

```python
%cd /content
!unzip -q /content/dl_tcn_colab_data.zip -d /content
!git clone YOUR_GITHUB_REPOSITORY_URL dl_tcn_cross_attention
%cd /content/dl_tcn_cross_attention
!pip install -q -r requirements-colab.txt
!python verify_colab_setup.py
```

資料ZIP解壓後必須形成：

```text
/content/dl_tcn_data/
  AQX_P_15_Resource/       # 72 CSV
  resources/
    station_static_features_49.csv
    station_static_clusters.csv
```

正式訓練前可選擇跑少量新舊pipeline等價測試：

```python
!python tests/verify_pipeline_equivalence.py
```

開始正式training：

```python
!python train_formal.py
```

目前鎖定config：learning rate `5e-4`、weight decay `1e-4`、dropout `0.20`、gradient clip `5.0`、station-balanced MSE、max epochs `15`、patience `5`。station-balanced權重依60個training stations各自可用樣本數計算，使每站對期望loss的貢獻相同。Colab L4會自動使用batch size `512`、4個輕量index workers、BF16 AMP、TF32與fused AdamW；其他CUDA GPU會依VRAM保守選擇batch。資料、split、特徵與模型架構不變。

CUDA正式訓練會在每次run開始時，一次性預計算9個raw channels的mask/標準化值，以及所有target-donor的WIND_ALONG/WIND_CROSS；每個epoch只擷取24小時窗口，不重算三角函數，也不預展開全年sample tensor。啟動時會印出預計算秒數與table VRAM。

如L4仍發生OOM，不需改程式，可在正式訓練前覆寫：

```python
%env DL_TCN_BATCH_SIZE=256
```

其他常用訓練設定也可在啟動程式前覆寫：

```python
%env DL_TCN_LEARNING_RATE=5e-4
%env DL_TCN_WEIGHT_DECAY=1e-4
%env DL_TCN_MAX_EPOCHS=15
%env DL_TCN_PATIENCE=5
```

## 建議將輸出寫入Google Drive

Colab的`/content`會在runtime結束後消失。若要保存checkpoint與history，先掛載Drive並指定輸出根目錄：

```python
from google.colab import drive
drive.mount('/content/drive')
%env DL_TCN_OUTPUT_ROOT=/content/drive/MyDrive/dl_tcn_runs
!python train_formal.py
```

AQ memory-map cache預設留在`/content/dl_tcn_work`，避免將頻繁cache I/O寫到Drive。可用環境變數覆寫：

- `DL_TCN_DATA_ROOT`
- `DL_TCN_WORK_ROOT`
- `DL_TCN_OUTPUT_ROOT`

## 正式輸出

輸出資料夾為`DL_TCN_CA_v1_formal_lr5e-4_station_balanced_output`，包含：

- `best_checkpoint.pt`
- `training_history.csv`
- `training_summary.json`
- `validation_predictions.csv`
- `best_validation_station_metrics.csv`
- `validation_station_metrics_all_epochs.csv`
- `pretraining_sanity.json`

## Outer test（60/12/1，不refit）

模型選擇完成後，將`DL_TCN_SELECTION_CHECKPOINT`指向鎖定實驗的`best_checkpoint.pt`，再執行：

```bash
python evaluate_outer.py
```

程式會核對checkpoint的Reduced cluster-aware 60/12 split與train-only scaler。Outer target完全不參與training、validation、scaler或epoch選擇；outer推論只使用60個training stations，12個validation stations不會混入donor pool。輸出資料夾`DL_TCN_CA_v1_outer_60donors_output`包含逐時預測、整體指標、coverage分母與protocol audit。72站refit是另一個後續階段，本程式不會執行。

## Target-aware checkpoint selection離線分析

`analyze_target_aware_epoch.py`只讀取既有`training_history.csv`、`validation_station_metrics_all_epochs.csv`、`best_checkpoint.pt`與49項static features，不修改模型也不重新訓練。它使用checkpoint保存的60站train-only static scaler，對12個validation pseudo-target做leave-one-station-out static similarity加權epoch預測。每個target自己的best epoch不會進入其預測，PM2.5與future observation也不會進入similarity。

程式輸出station×epoch矩陣、similarity矩陣、PCA／heatmap、global／macro／target-aware／oracle逐站表現，以及明確列出12站樣本量與目前只保存best checkpoint的限制。
