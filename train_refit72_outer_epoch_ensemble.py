from __future__ import annotations

import json
import os
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
    standardize_static,
)
from evaluate_full72_oof_outer_ensemble import (
    fit_epoch_weights,
    load_all_oof,
    select_method_by_sixfold_oof,
)
from model import TCNTargetCrossAttention
from sanity_check import audit_dataset, smoke_forward
from train_formal import (
    DeviceFeatureBuilder,
    amp_dtype_for,
    make_grad_scaler,
    make_index_loader,
    make_loader,
    make_optimizer,
    make_station_sample_weights,
    make_vectorized_loader,
    regression_metrics,
    resolve_target,
    seed_all,
    train_epoch,
    validate_epoch,
)


def main() -> None:
    seed_all(CFG.seed)
    runtime = apply_runtime_profile(CFG)
    device = CFG.device
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = CFG.enable_tf32
        torch.backends.cudnn.allow_tf32 = CFG.enable_tf32
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    crossfit_root = Path(os.environ.get(
        'DL_TCN_CROSSFIT_ROOT',
        str(CFG.formal_output_root / 'crossfit_target_conditioned_snapshots'),
    ))
    oof_base, oof_matrix = load_all_oof(crossfit_root)
    method, method_rows = select_method_by_sixfold_oof(oof_base, oof_matrix)
    oof_stations = np.unique(oof_base.station_index.to_numpy(int))
    epoch_weights = fit_epoch_weights(
        oof_base, oof_matrix, oof_stations,
        method['family'], method['parameter'],
    )
    lock = {
        'protocol': '72-station OOF locks epoch ensemble before 72-station refit and outer truth',
        'selection_used_outer_truth': False,
        'selected_method': method,
        'candidate_scores': sorted(method_rows, key=lambda row: row['sixfold_oof_macro_rmse']),
        'epoch_weights': [
            {'epoch': epoch, 'weight': float(weight)}
            for epoch, weight in enumerate(epoch_weights, start=1) if weight > 1e-8
        ],
    }
    print('\nLOCKED EPOCH ENSEMBLE BEFORE REFIT/OUTER TRUTH')
    print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)

    static, _, static_cols = load_static()
    outer = resolve_target(static, CFG.target_site)
    known = np.asarray([idx for idx in range(len(static)) if idx != outer], dtype=int)
    if len(known) != 72:
        raise RuntimeError(f'outer排除後不是72站: {len(known)}')
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)
    scaler = fit_train_only_scaler(cube, timestamps, static, static_cols, known)
    static_scaled = standardize_static(static, static_cols, scaler)
    train_ds = ColdStartStationDataset(
        known, known, CFG.train_start, CFG.train_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    outer_ds = ColdStartStationDataset(
        [outer], known, CFG.test_start, CFG.test_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    if audit_dataset(train_ds, max_samples=64)['sampled_donor_counts'] != [71]:
        raise RuntimeError('72-refit training donors不是71')
    if audit_dataset(outer_ds, max_samples=64)['sampled_donor_counts'] != [72]:
        raise RuntimeError('72-refit outer donors不是72')

    feature_builder = None
    if device.type == 'cuda':
        max_time = int(max(train_ds.row_times.max(), outer_ds.row_times.max()))
        feature_builder = DeviceFeatureBuilder(
            known, cube, max_time, timestamps, static, static_scaled,
            distance, scaler, outer, device,
        )
        train_loader = make_index_loader(train_ds, True)
        outer_loader = make_index_loader(outer_ds, False)
    else:
        train_loader = make_vectorized_loader(
            train_ds, known, cube, timestamps, static, static_scaled,
            distance, scaler, True,
        )
        outer_loader = make_vectorized_loader(
            outer_ds, known, cube, timestamps, static, static_scaled,
            distance, scaler, False,
        )

    base_model = TCNTargetCrossAttention(static_dim=len(static_cols)).to(device)
    smoke = smoke_forward(
        base_model, make_loader(train_ds, False, CFG.smoke_batch_size), device
    )
    base_model.zero_grad(set_to_none=True)
    optimizer, fused = make_optimizer(base_model, device)
    model = base_model
    if CFG.compile_mode != 'off':
        if not hasattr(torch, 'compile'):
            raise RuntimeError('目前PyTorch沒有torch.compile，請設DL_TCN_COMPILE_MODE=off')
        model = torch.compile(
            base_model, mode=CFG.compile_mode, fullgraph=False, dynamic=True
        )
    amp_dtype = amp_dtype_for(device)
    grad_scaler = make_grad_scaler(amp_dtype == torch.float16)
    station_weights = make_station_sample_weights(train_ds, len(static), device)

    print(json.dumps({
        'runtime_profile': runtime,
        'train_samples': len(train_ds),
        'outer_truth_timestamps': len(outer_ds),
        'training_targets': 72,
        'training_donors_per_sample': 71,
        'outer_donors': 72,
        'model_parameters': sum(p.numel() for p in base_model.parameters()),
        'smoke': smoke,
        'fused_adamw': fused,
        'files_written': 0,
    }, ensure_ascii=False, indent=2), flush=True)

    outer_base = None
    outer_columns = []
    history = []
    started = time.perf_counter()
    peak_vram = 0.0
    for epoch in range(1, 16):
        epoch_started = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        train_loss = train_epoch(
            model, train_loader, optimizer, grad_scaler, device, epoch,
            feature_builder, station_weights,
        )
        # Outer predictions are collected for the already-locked snapshot
        # ensemble. Outer metrics are deliberately not printed or selected here.
        _, _, current = validate_epoch(
            model, outer_loader, device, static, timestamps, epoch, feature_builder
        )
        if outer_base is None:
            outer_base = current[['station_index', 'timestamp', 'y_true']].copy()
        else:
            if not np.array_equal(
                pd.to_datetime(outer_base.timestamp).astype('int64').to_numpy(),
                pd.to_datetime(current.timestamp).astype('int64').to_numpy(),
            ):
                raise RuntimeError('72-refit outer snapshot timestamps不一致')
            if not np.allclose(outer_base.y_true, current.y_true, rtol=0, atol=1e-6):
                raise RuntimeError('72-refit outer snapshot truth不一致')
        outer_columns.append(current.y_pred.to_numpy(float))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            peak_vram = max(
                peak_vram,
                torch.cuda.max_memory_allocated(device) / 1024**2,
            )
        row = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'runtime_seconds': time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    assert outer_base is not None
    outer_matrix = np.stack(outer_columns, axis=1)
    deployed = outer_matrix @ epoch_weights
    result = {
        'target_station_index': int(outer),
        'siteid': str(static.loc[outer, 'siteid']),
        'sitename': str(static.loc[outer, 'sitename']),
        'selection_used_outer_truth': False,
        'training_targets': 72,
        'training_donors_per_sample': 71,
        'outer_donors': 72,
        'epochs': 15,
        'metrics': regression_metrics(
            outer_base.y_true.to_numpy(float), deployed
        ),
        'training_runtime_seconds': time.perf_counter() - started,
        'peak_vram_mb': peak_vram,
        'files_written': 0,
    }
    print('\nREFIT-72 OUTER EPOCH-ENSEMBLE RESULT')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
