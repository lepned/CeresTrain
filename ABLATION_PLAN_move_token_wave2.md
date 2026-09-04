# Move-token wave 2 — ablation plan (2026-09-04)

Source: the 4-agent ideation of 2026-09-04 (memory `project_ideation_round_2026_09_04`,
SHARED_NOTES 20:40 09-04). Everything below is implemented and unit-tested
(`src/CeresTrainPy/test_move_tokens.py`, `test_export_folds.py`: ALL OK). Nothing here has a
training read yet except a 3M smoke of the value-order head (loss falls 14.3 -> 9.8, top-1
0.21 -> 0.42 on 3M positions).

## Chassis and protocol

- Chassis: `srv_256_fs_PX3` = 256x10 trunk + move-token decoder dm256 x 4 blocks x 8 heads
  (= the prod decoder shape), M=128, 100M positions, LR 8e-4, decay at 0.8, BSF 4096,
  TorchSeed 777 / ShuffleSeed 20260828, EMA 100/10. **New control = `srv_256_fs_PX3pl05`**
  (PX3 + PolicyPL 0.05/K5, the prod recipe). The control must be trained on the SAME machine,
  world size and corpus as the arms (pairing). Every arm is a one-key diff from the control
  (verified programmatically when the configs were written).
- Read: RAW net at 100M (EMA is irrelevant at 100M), `eval_net.py puzzle`, rg2500 + rg2700,
  4000 puzzles, policy / pTop3 / value; plus the 95-100M val/pol loss from the log. A paired
  4000-puzzle read resolves ~5-10 Elo when all rows move the same way; single-row 5 Elo is not a
  result.
- TB tags to watch per arm: `mt_vord_loss`, `mt_vord_top1` (value-order), `mt_opp_count_*`,
  `mt_opp_trunc_rate` (must be 0; raise MoveTokenOppMax if not), `mt_qk_max_self/cross`
  (decoder logit monitor, NEW for every arm incl. control), `grad_2.0_norm/*move_tokens*`.
- Launch on the server: `bash ~/scan/launch_ea_wave.sh --pairs <IDs>` after copying the
  configs with `cp -f` (the launcher does NOT overwrite stale copies). Data path in
  `*_ceres_data.json` is `/mnt/lepned/T80_v2`; adjust for another machine.

## Arms (priority order)

| # | config | knob(s) | hypothesis / evidence | expected read |
|---|--------|---------|-----------------------|---------------|
| 0 | `srv_256_fs_PX3pl05` | control | prod recipe on the 256 chassis | baseline for all rows |
| 1 | `srv_256_fs_PX3dmu` | `MuonAdamWScope: all-non-trunk+decoder` | decoder body trains under AdamW at the Muon peak rate today (576-A/B trap); Muon on the decoder | both heads via better-optimized decoder; watch `mt_qk_max_*` (no clip in the decoder) |
| 2 | `srv_256_fs_PX3dlr` | `LearningRateDecoderRatio: 0.33` | same mismatch corrected without changing optimizer class (safer) | as above |
| 3 | `srv_256_fs_PX3dmulr` | both of the above | | |
| 4 | `srv_256_fs_PX3vo` | `LossMoveTokenValueOrderMultiplier: 0.5, TopK 5` | PXpl 1.0 gave value +15/+32/+28/+10 but broke policy CE; the order loss now lives on a training-only token scalar, so policy CE stays intact | value +10..+25 if the PXpl gain was an ordering effect; policy 0 |
| 5 | `srv_256_fs_PX3vo005` | same at 0.05 | dose response (PXpl05 kept 2/3 of the PL-1.0 value gain) | |
| 6 | `srv_256_fs_PX3ork` | `MoveTokenOppMax: 80` | opponent's candidate replies as extra keys/values of the own-move self-attention: one-ply threat structure in move space (the campaign's only working thesis is 'compute in move space') | policy/pTop3 on rg2700 first; +24 decoder kernels (~-2 % EPS) |
| 7 | `srv_256_fs_PX3orkp` | + `MoveTokenOppPool: true` | also pool opp tokens into the value inject | value |
| 8 | `srv_256_fs_PX3wb` | `MoveTokenWriteBack: true` | squares query the move tokens; zero-init projection added to `flow` before the head front-end = a per-square value pathway (X4: the pooled inject IS the value mechanism; this replaces the global bag with a spatial signal) | value +10..+25, policy 0; ~-1 % EPS |

Suggested pair order on 2-GPU pairs: (0,1) (0 must run first or in parallel), then (2,4),
(6,8), (3,5), (7). Stack winners into one arm afterwards.

## Memory / speed notes

- ORK: attention maps grow from [B,h,M,M] to [B,h,M,M+80]; ~+1 GB at 1024/rank on the 640 form.
  `MoveTokenMax: 96` (training-time cap; exact, max observed candidates 72) frees ~2 GB if needed.
- Write-back: +~0.5 M params on 640, one extra attention block at serving (~-1 % EPS).
- Value-order head: 0 serving cost (never exported).
- Export folds (NetDef `ExportFolds: none|mt|ffn|all`, `export_folds.py` via `save_model.py`):
  exact rewrites. Measured on the 700M prod net in EngineBattle: `mt` (decoder attention-scale +
  pre-norm scale folds) **+6-8 % EPS** (1.08x / 1.06x both orders) -> `"ExportFolds": "mt"` is
  set in every wave-2 config (and worth adopting for prod exports); `ffn` (SwiGLU gate|up
  fusion) 0.96-1.00x and `all` 0.91-0.96x -> A/B tools only. NOTE: with `ExportFolds` set,
  every export of the run is named `<pos>fldmt.onnx` (also the `ema` files) — eval/def tooling
  must glob that name, and a paired control must carry the same setting (it does).

## Not in this wave (decided)

FP8/FP4 (closed 09-04), bigger decoder (dm384x6 null), decoder decorations (rich features,
pools, value query, post-move: all null), replay/reweighting (three designs falsified),
QK-clip tau changes (guard inactive), export M < 80.
