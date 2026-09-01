from pathlib import Path
import torch
from config import CFG

required=(CFG.static_path,CFG.cluster_path)
missing=[str(path) for path in required if not path.is_file()]
aq_files=sorted(CFG.aq_data_dir.glob("*.csv")) if CFG.aq_data_dir.is_dir() else []
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"data_root={CFG.data_root}")
print(f"aq_csv_count={len(aq_files)}")
print(f"static_path={CFG.static_path}")
print(f"cluster_path={CFG.cluster_path}")
print(f"formal_output_root={CFG.formal_output_root}")
if missing: raise FileNotFoundError(f"缺少資料檔: {missing}")
if len(aq_files)!=72: raise RuntimeError(f"AQ CSV應為72份，實際{len(aq_files)}")
if not torch.cuda.is_available(): raise RuntimeError("Colab沒有啟用GPU runtime")
print("COLAB SETUP PASSED")
