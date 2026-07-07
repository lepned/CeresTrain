"""Dump real 'squares' inputs (FP16) + ground-truth WDL labels to .npy so a
minimal Windows TRT-python script can run the value-correctness check without
TPGDataset/torch/zstandard on Windows.

Usage (WSL): CERES_AUX_FEATURES_PER_SQUARE=4 python3 wsl_dump_for_win.py <tpg_dir> <out_dir> [num_batches] [batch]
"""
import os, sys
import numpy as np
sys.path.insert(0, '/mnt/c/Users/lepne/source/repos/CeresTrain/src/CeresTrainPy')
from tpg_dataset import TPGDataset

tpg = sys.argv[1]
out = sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 30
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 64

ds = TPGDataset(tpg, BATCH, 0.0, 0, 1, 0, 1, 0, False)
sq_list, lab_list = [], []
for _ in range(NB):
    it = ds[0][0]
    s = it['squares'].numpy().astype(np.float16)
    lab = it['wdl_deblundered'].numpy().astype(np.float32)
    if s.shape[0] != BATCH:
        continue
    sq_list.append(s)
    lab_list.append(lab)

sq = np.concatenate(sq_list, 0)
lab = np.concatenate(lab_list, 0)
os.makedirs(out, exist_ok=True)
np.save(os.path.join(out, 'win_squares.npy'), sq)
np.save(os.path.join(out, 'win_wdl.npy'), lab)
print('saved squares', sq.shape, sq.dtype, '-> win_squares.npy')
print('saved wdl    ', lab.shape, lab.dtype, '-> win_wdl.npy')
