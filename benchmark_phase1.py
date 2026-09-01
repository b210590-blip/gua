import json,subprocess,threading,time
import numpy as np
import torch
from config import CFG
from data_pipeline import *
from model import TCNTargetCrossAttention
from train_formal import DeviceFeatureBuilder,make_grad_scaler,make_index_loader,resolve_target


def main():
    if CFG.batch_size!=64: raise RuntimeError("Phase 1 benchmark固定batch size 64")
    static,clusters,cols=load_static(); outer=resolve_target(static,CFG.target_site); train_idx,_=choose_split(clusters,outer)
    cube,timestamps=build_or_load_hourly_cube(static); distance=haversine_matrix(static.longitude,static.latitude)
    scaler=fit_train_only_scaler(cube,timestamps,static,cols,train_idx); ss=standardize_static(static,cols,scaler)
    dataset=ColdStartStationDataset(train_idx,train_idx,CFG.train_start,CFG.train_end,cube,timestamps,static,ss,distance,scaler)
    builder=DeviceFeatureBuilder(train_idx,cube,int(dataset.row_times.max()),timestamps,static,ss,distance,scaler,outer,CFG.device)
    loader=make_index_loader(dataset,True); model=TCNTargetCrossAttention(len(cols)).to(CFG.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CFG.learning_rate,weight_decay=CFG.weight_decay); grad_scaler=make_grad_scaler(True)
    samples=[]; stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                text=subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader,nounits"],text=True,creationflags=subprocess.CREATE_NO_WINDOW)
                util,memory=text.strip().splitlines()[0].split(","); samples.append((float(util),float(memory)))
            except Exception: pass
            stop.wait(0.25)
    thread=threading.Thread(target=monitor,daemon=True); thread.start(); torch.cuda.reset_peak_memory_stats()
    measured_batches=500; warmup_batches=50; measured_samples=0; started=None
    for batch_number,index_batch in enumerate(loader,start=1):
        batch=builder(index_batch); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",enabled=True):
            pred,_=model(batch["values"],batch["mask"],batch["donor_static"],batch["geometry"],batch["donor_padding_mask"],batch["target_static"],batch["time_features"])
            loss=torch.nn.functional.mse_loss(pred,batch["label"])
        grad_scaler.scale(loss).backward(); grad_scaler.unscale_(optimizer)
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CFG.gradient_clip_norm)
        if not torch.isfinite(grad_norm): raise RuntimeError("benchmark gradient非finite")
        grad_scaler.step(optimizer); grad_scaler.update()
        if batch_number==warmup_batches: torch.cuda.synchronize(); started=time.perf_counter()
        elif batch_number>warmup_batches: measured_samples+=int(batch["label"].numel())
        if batch_number>=warmup_batches+measured_batches: break
    torch.cuda.synchronize(); elapsed=time.perf_counter()-started; stop.set(); thread.join(timeout=2)
    rate=measured_samples/elapsed; util=[x[0] for x in samples]; nvidia_memory=[x[1] for x in samples]
    result={"batch_size":CFG.batch_size,"measured_batches":measured_batches,"measured_samples":measured_samples,
            "seconds":elapsed,"samples_per_second":rate,"estimated_train_epoch_minutes":len(dataset)/rate/60,
            "torch_peak_vram_mb":torch.cuda.max_memory_allocated()/1024**2,
            "gpu_utilization_mean_percent":float(np.mean(util)),"gpu_utilization_p95_percent":float(np.percentile(util,95)),
            "nvidia_memory_used_peak_mb":float(np.max(nvidia_memory)),"gpu_samples":len(samples)}
    (CFG.script_dir/"phase1_benchmark.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
