from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
IS_COLAB = Path("/content").is_dir()
DEFAULT_DATA_ROOT = Path(os.environ.get(
    "DL_TCN_DATA_ROOT", "/content/dl_tcn_data" if IS_COLAB else str(SCRIPT_DIR / "dl_tcn_data")
))
DEFAULT_WORK_ROOT = Path(os.environ.get(
    "DL_TCN_WORK_ROOT", "/content/dl_tcn_work" if IS_COLAB else str(SCRIPT_DIR / "dl_tcn_work")
))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get(
    "DL_TCN_OUTPUT_ROOT", "/content" if IS_COLAB else str(SCRIPT_DIR)
))

@dataclass
class Config:
    data_root: Path = DEFAULT_DATA_ROOT
    aq_data_dir: Path = DEFAULT_DATA_ROOT / "AQX_P_15_Resource"
    target_site: str = "桃園"
    script_dir: Path = SCRIPT_DIR
    resource_dir: Path = DEFAULT_DATA_ROOT / "resources"
    static_path: Path = resource_dir / "station_static_features_49.csv"
    cluster_path: Path = resource_dir / "station_static_clusters.csv"
    output_dir: Path = DEFAULT_WORK_ROOT
    formal_output_root: Path = DEFAULT_OUTPUT_ROOT

    train_start: str = "2024-07-01 00:00:00"
    train_end: str = "2025-06-30 23:00:00"
    test_start: str = "2025-07-01 00:00:00"
    test_end: str = "2026-06-30 23:00:00"
    validation_stations: int = 12
    seed: int = 42

    history_hours: int = 24
    raw_dynamic_items: tuple[str, ...] = (
        "SO2", "CO", "O3", "PM10", "NO", "NO2", "AMB_TEMP", "PM2.5", "RH",
    )
    derived_dynamic_items: tuple[str, ...] = ("WIND_ALONG", "WIND_CROSS")
    aq_cube_items: tuple[str, ...] = (
        "SO2", "CO", "O3", "PM10", "NO", "NO2",
        "WIND_SPEED", "WIND_DIREC", "AMB_TEMP", "PM2.5", "RH",
    )

    tcn_hidden: int = 32
    tcn_kernel_size: int = 3
    tcn_dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.10
    attention_dim: int = 32
    attention_heads: int = 2
    final_hidden: int = 32

    batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    smoke_epochs: int = 2
    smoke_max_train_batches: int = 20
    smoke_max_val_batches: int = 10
    use_amp: bool = True
    amp_init_scale: float = 128.0
    amp_growth_interval: int = 1_000_000
    cache_schema_version: str = "masked_tcn_ca_v2_exact_pair_wind_scaler"

    # Formal 60/12 supervised training (no outer test/refit/search).
    max_epochs: int = 15
    early_stopping_patience: int = 7
    formal_output_dirname: str = "DL_TCN_CA_v1_formal_lr3e-4_output"
    progress_every_batches: int = 1000
    formal_num_workers: int = 0

    @property
    def dynamic_items(self) -> tuple[str, ...]:
        return self.raw_dynamic_items + self.derived_dynamic_items

    @property
    def n_dynamic_channels(self) -> int:
        return len(self.dynamic_items)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = Config()
