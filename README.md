# Unseen-station PM2.5 TCN + cross-attention

這是已通過protocol review與pipeline等價測試的Colab版本。固定設定為60 training stations／12 validation stations、24小時history、11個dynamic channels、49個static features、shared causal TCN與target-conditioned cross-attention。程式不建立outer test、不做72站refit，也不搜尋超參數。

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

目前正式config：learning rate `3e-4`、max epochs `15`、patience `7`。Colab L4會自動使用batch size `512`、4個輕量index workers、BF16 AMP、TF32與fused AdamW；其他CUDA GPU會依VRAM保守選擇batch。這些調整不改資料、split、特徵、模型架構或loss。

如L4仍發生OOM，不需改程式，可在正式訓練前覆寫：

```python
%env DL_TCN_BATCH_SIZE=256
```

其他常用訓練設定也可在啟動程式前覆寫：

```python
%env DL_TCN_LEARNING_RATE=3e-4
%env DL_TCN_WEIGHT_DECAY=1e-4
%env DL_TCN_MAX_EPOCHS=15
%env DL_TCN_PATIENCE=7
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

輸出資料夾為`DL_TCN_CA_v1_formal_lr3e-4_output`，包含：

- `best_checkpoint.pt`
- `training_history.csv`
- `training_summary.json`
- `validation_predictions.csv`
- `best_validation_station_metrics.csv`
- `pretraining_sanity.json`
