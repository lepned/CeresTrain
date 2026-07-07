"""Fate heatmap dumper — visual validation of the K-ply survival aux head.

Loads a survival-trained checkpoint (CPU), runs REAL TPG records through it
(never hand-encoded FENs — known standalone-input trap), and prints, per
position: the pieces, the head's predicted fate per square, and the actual
sidecar labels. Boards are shown in side-to-move perspective (mover at bottom).

Symbols in fate grids: '.' empty · '+' survives horizon · digit d = captured
at ply d. Predicted grid uses the argmax class; '?' marks occupied squares
where prediction != label.

Usage (WSL venv python, from anywhere):
  python3 survival_heatmap.py <CONFIG_ID> <CKPT_POS> <corpus_dir> [num_samples] [min_captures]
  e.g. python3 survival_heatmap.py kvxcmb20M 20000256 /mnt/d/kovax_lc0_4cells_tpg_v2_surv 8 4
"""
import glob, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'CeresTrainPy'))

# Env must be set BEFORE importing ceres_net/config (module-level reads).
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
os.environ.setdefault('CERES_TPG_SQUARE_BYTES', '137')
os.environ.setdefault('CERES_SURVIVAL_TARGET_WEIGHT', '0.3')
os.environ.setdefault('CERES_SURVIVAL_HORIZON', '8')

import numpy as np
import torch
import zstandard as zstd

torch.set_num_threads(2)  # stay out of the way of any running training dataloader

from config import Configuration
from ceres_net import CeresNet

OUTPUTS_DIR = '/mnt/c/Dev/Chess/CeresTrain'
V2 = 9378
PREFIX = 610
K = int(os.environ['CERES_SURVIVAL_HORIZON'])

CONFIG_ID = sys.argv[1] if len(sys.argv) > 1 else 'kvxcmb20M'
CKPT_POS = sys.argv[2] if len(sys.argv) > 2 else '20000256'
CORPUS = sys.argv[3] if len(sys.argv) > 3 else '/mnt/d/kovax_lc0_4cells_tpg_v2_surv'
NUM_SAMPLES = int(sys.argv[4]) if len(sys.argv) > 4 else 8
MIN_CAPTURES = int(sys.argv[5]) if len(sys.argv) > 5 else 4


def build_model():
    config = Configuration('.', os.path.join(OUTPUTS_DIR, 'configs', CONFIG_ID))
    model = CeresNet(None, config,
                     policy_loss_weight=config.Opt_LossPolicyMultiplier,
                     value_loss_weight=config.Opt_LossValueMultiplier,
                     moves_left_loss_weight=config.Opt_LossMLHMultiplier,
                     unc_loss_weight=config.Opt_LossUNCMultiplier,
                     value2_loss_weight=config.Opt_LossValue2Multiplier,
                     q_deviation_loss_weight=config.Opt_LossQDeviationMultiplier,
                     value_diff_loss_weight=config.Opt_LossValueDMultiplier,
                     value2_diff_loss_weight=config.Opt_LossValue2DMultiplier,
                     action_loss_weight=config.Opt_LossActionMultiplier,
                     uncertainty_policy_weight=config.Opt_LossUncertaintyPolicyMultiplier,
                     action_uncertainty_loss_weight=config.Opt_LossActionUncertaintyMultiplier,
                     q_ratio=config.Data_FractionQ)
    ckpt_path = os.path.join(OUTPUTS_DIR, 'nets', f'ckpt_lepdev_{CONFIG_ID}_{CKPT_POS}')
    loaded = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(loaded['model'], strict=True)
    model = model.float()
    model.train()  # survival stash is gated on self.training (export safety); no BN/dropout in arch
    print(f'loaded {ckpt_path}')
    return model


def read_records(corpus, want, min_caps):
    """Stream shard+sidecar; return list of (squares_bytes[64,137], labels[64]) with >= min_caps captures."""
    shards = sorted(s for s in glob.glob(os.path.join(corpus, '*.zst')) if not s.endswith('.tgt.zst'))
    out = []
    ds, dt = zstd.ZstdDecompressor(), zstd.ZstdDecompressor()
    with open(shards[0], 'rb') as fs, open(shards[0][:-4] + '.tgt.zst', 'rb') as ft:
        rs, rt = ds.stream_reader(fs), dt.stream_reader(ft)
        assert rt.read(16)[:4] == b'TPGT'
        while len(out) < want:
            raw = rs.read(4096 * V2)
            lab = rt.read(4096 * 64)
            if not raw:
                break
            n = len(raw) // V2
            recs = np.frombuffer(raw[:n * V2], dtype=np.uint8).reshape(n, V2)
            labels = np.frombuffer(lab[:n * 64], dtype=np.uint8).reshape(n, 64)
            caps = ((labels >= 1) & (labels <= K)).sum(axis=1)
            for i in np.where(caps >= min_caps)[0]:
                out.append((recs[i, PREFIX:].reshape(64, 137).copy(), labels[i].copy()))
                if len(out) >= want:
                    break
    return out


PIECES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']


def piece_char(onehot13):
    idx = int(np.argmax(onehot13))
    return PIECES[idx] if onehot13[idx] == 100 else '?'


def fate_char(v):
    return '.' if v == 0 else ('+' if v == K + 1 else str(int(v)))


def grid(rows):
    return [''.join(r) for r in rows]


def main():
    model = build_model()
    samples = read_records(CORPUS, NUM_SAMPLES, MIN_CAPTURES)
    print(f'{len(samples)} tactically busy positions (>= {MIN_CAPTURES} captures within {K} plies)\n')

    agree_occ = total_occ = agree_cap = total_cap = 0
    for si, (sq_bytes, labels) in enumerate(samples):
        squares = torch.tensor(sq_bytes.astype(np.float32) / 100.0).unsqueeze(0)  # [1, 64, 137]
        with torch.no_grad():
            model(squares, None)
            logits = model._last_survival_out[0]        # [64, K+2]
            model._last_survival_out = None
        pred = logits.argmax(dim=1).numpy()             # [64]
        p_cap = torch.softmax(logits.float(), dim=1)[:, 1:K + 1].sum(dim=1).numpy()

        piece_rows, pred_rows, lab_rows = [], [], []
        for rank in range(7, -1, -1):                   # stm perspective: mover at bottom
            prow, drow, lrow = [], [], []
            for file in range(8):
                s = rank * 8 + file
                pc = piece_char(sq_bytes[s, :13])
                lab = labels[s]
                pd = fate_char(pred[s]) if pc != '.' else '.'
                if pc != '.' and pred[s] != lab:
                    pd = '?' if fate_char(pred[s]) == fate_char(lab) else pd.lower() if pd.isalpha() else pd
                prow.append(pc); drow.append(pd); lrow.append(fate_char(lab))
            piece_rows.append(prow); pred_rows.append(drow); lab_rows.append(lrow)

        occ = sq_bytes[:, 0] != 100
        agree_occ += int((pred[occ] == labels[occ]).sum()); total_occ += int(occ.sum())
        cap_mask = occ & (labels >= 1) & (labels <= K)
        agree_cap += int((pred[cap_mask] == labels[cap_mask]).sum()); total_cap += int(cap_mask.sum())

        print(f'--- sample {si}  (side to move at bottom; UPPER = side to move) ---')
        print(f'{"pieces":10s}  {"predicted":10s}  {"actual":10s}')
        for a, b, c in zip(grid(piece_rows), grid(pred_rows), grid(lab_rows)):
            print(f'{a:10s}  {b:10s}  {c:10s}')
        hot = np.argsort(-p_cap * occ)[:3]
        descr = ', '.join(f'{piece_char(sq_bytes[s, :13])}@slot{s} P(cap)={p_cap[s]:.0%} (label {fate_char(labels[s])})' for s in hot)
        print(f'most-endangered per net: {descr}\n')

    print(f'agreement on occupied squares : {100 * agree_occ / max(total_occ, 1):.1f}%  ({total_occ} squares)')
    print(f'agreement on CAPTURED squares : {100 * agree_cap / max(total_cap, 1):.1f}%  ({total_cap} squares, exact-ply match)')


if __name__ == '__main__':
    main()
