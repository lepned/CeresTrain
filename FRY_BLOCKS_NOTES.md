# Fry building blocks in CeresTrain (branch `fry-blocks`, worktree `C:\Dev\Chess\CeresTrain_fry`), 2026-11-26

Local work only: uncommitted changes on top of 8a7d72e. Nothing here goes to a public repository (the parity test
loads GPL-3.0 weights by path; no weights and none of the author's code are in this tree).

## Summary

- **Implemented, each behind its own config key, all off by default:** square features; grouped-query attention with
  a per-layer query/key width; the 70-token input; Fry's policy head on our 1858 moves, his value-head body, the
  layer-0 input skip, his optimizer partition and weight-decay rule. Plus three small keys that an exact description
  of his net needed: `NormEps`, `FFNNoBias`, `AttnQKVBias`.
- **Proven: our trainer reproduces Fry.** `parity_768_28_fry117` (our `CeresNet`, 768 x 28, 70 tokens) with the
  released checkpoint copied onto it (all 113,227,484 parameters consumed, none left over on either side of the Fry
  part) against the author's ONNX on the 450 verification positions (214 Black to move), float32 on CPU:

  | oracle | policy logits, 12,795 legal moves: max / mean abs diff | same top move | value (expected score): max / mean | 128 value logits: max |
  |---|---|---|---|---|
  | the ONNX as released (FP16 trunk) vs the checkpoint weights in our net | 0.176 / 0.0013 | 450 / 450 | 0.00048 / 0.000056 | 0.104 |
  | the same graph with FP16 tensors and casts turned FP32 (its FP16-rounded weights, unrounded activations) | 0.085 / 0.0011 | 450 / 450 | 0.00054 / 0.000044 | 0.060 |
  | as the line above, and our net given the file's own FP16-rounded weights, norm epsilon and logit divisor | **0.00014 / 0.000004** | 450 / 450 | **< 0.000005** | **0.00014** |

  The third line is the proof that the forward pass is the same function (embedding output and layer-0 input are
  bit-identical; over the 28 layer inputs the worst mean difference is 1e-6 of the rms). The first two lines are
  what FP16 costs in his file, not a difference in our code: see "Where the FP16 differences come from".
- **Regression:** with all new keys off, four existing configs (384 flat-BT6 base arm, 256 plain Fry-style arm, 384
  NBT arm, 768 x 16 arm) build bit-identical nets and give bit-identical eval outputs, training loss, gradients and
  weight-decay partition against a pristine copy of 8a7d72e. Eight existing test files that touch the changed code pass.
- **Smoke:** the real `train.py` on CPU, a 64 x 4 net with every new key on, 16,384 real T80 positions: total loss
  1.73 -> 1.34 (last run; 1.71 -> 1.27 in an earlier one), checkpoint and ONNX written, the ONNX runs in
  onnxruntime and agrees with the checkpoint net.
- **Not verified (needs the GPU):** TensorRT build and FP16 behaviour of an export with the new integer operators,
  training speed, bf16 autocast backward, torch.compile on CUDA, DDP. Listed at the end.
- **Things in the review that turned out wrong or incomplete** are listed at the end (the record byte numbers; no
  colour in our record; the ONNX's norm gains and logit divisor are FP16-rounded).

## Config keys

Net (`*_ceres_net.json`):

| key | default | meaning |
|---|---|---|
| `SquareFeatures` | false | per square the sum of 23 rows of one zero-init [1735, ModelDim] table, added to the embedding |
| `FryInput` | false | 70 tokens (64 squares + colour, 4 castling, e.p.), token + position embedding instead of the Linear embedding and its norm |
| `AttnKVHeads` | 0 | number of key/value heads shared by the `NumHeads` query heads (0 = NumHeads) |
| `AttnQKDims` | 0 | per-head query/key width: an int, or a list with one int per layer (0 = ModelDim / NumHeads) |
| `AttnValueDim` | 0 | per-head value width (0 = ModelDim / NumHeads) |
| `AttnQKVBias` | false | bias on the Q / K / V projections (only with the three keys above) |
| `PolicyHeadFry` | false | from-square x to-square bilinear head + promotion term, on our 1858 moves |
| `PolicyHeadFryKingCastling` | false | e1h1 / e1a1 read the king's two-square pairs (only for loading his weights) |
| `ValueHeadFry` | false | the HL-Gauss head's body: per token ModelDim -> 16, GELU, flatten, -> hidden, GELU, -> buckets |
| `ValueHeadFryHidden` | 0 | hidden width of that head (0 = ModelDim) |
| `ValueHLGaussServeDraw` | "wdl" | "none": served D := 0 (his head alone); needs `LossValueMultiplier` 0 |
| `InputSkipLayer0` | false | with `NBTInputSkip` on a plain trunk: layer 0 gets the two scalars too |
| `NormEps` | null (= 1e-6) | epsilon of the trunk norms and of the grouped-query QK-norm (his: 1e-5) |
| `FFNNoBias` | false | no bias on the FFN's first and last Linear (his SwiGLU has none; ours has them) |

Opt (`*_ceres_opt.json`):

| key | default | meaning |
|---|---|---|
| `MuonAdamWScope: "fry"` | "all-non-trunk" | Muon on every 2-D tensor inside the trunk layers, template mixers included |
| `WeightDecayScheme: "fry"` | "default" | decay everything except QK-norm gains and input-skip scalars, in both optimizers |
| `WeightDecayFromSchedule` | null | c: `WeightDecay := 1 / (c * LearningRateBase * steps)`, steps = NumTrainingPositions / BatchSizeBackwardPass (his c: 0.17) |

Existing keys used for the rest of his recipe: `MuonMomentum` 0.95, `Beta1` / `Beta2` 0.9 / 0.999, `HLGaussWeight`
0.5, `HLGaussBuckets` 128, `AttnTemplates` 8, `NBTInputSkip`, `UseQKNorm`, `FFNMultiplier` 992 / 768, `FractionQ` 0.8.

## Files

New (src/CeresTrainPy): `square_features.py`, `fry_input.py`, `fry_heads.py`; test side: `fry_testlib.py`
(python-chess reference encoders, the 1968-move table), `fry_parity.py` (the Fry config and the checkpoint map),
`test_fry_blocks.py`, `test_fry_parity.py`, `test_fry_regression.py`, `test_fry_record_real.py`, `test_fry_smoke.py`,
`test_fry_configs.py`. Configs: `configs/fry_blocks/` (+ `make_fry_configs.py`, which wrote them).

Changed: `config.py` (keys + validation, one block at the end of `__init__`), `ceres_net.py` (embedding, layer
factory, trunk loop, trunk-end norm, heads), `dot_product_attention.py` (grouped-query projections, `qk_div`),
`encoder_layer.py` (pass-through, template padding for metadata tokens), `mlp2_layer.py` (optional no-bias),
`nbt_layer.py` (`_raw_logits` with grouped projections), `bt6_native_terms.py` (optional q/k width), `train.py`
(Muon scope 'fry', weight-decay scheme, name rules), `wd_partition.py` (the square-feature table).

## Facts about our record that the blocks rely on (checked, not assumed)

Record = 137 bytes per square (`TPGSquareRecord`), floats after / 100:
0..103 eight boards x 13 one-hot (0 empty, 1..6 our P N B R Q K, 7..12 theirs); 104..111 repetition flags;
112..115 CanOO, CanOOO, OpponentCanOO, OpponentCanOOO; 116 rule-50; 117 unused; 118 en passant, set on the square of
the pawn that just made a double step (32 + file); 119 / 120 blunder counters; 121..128 rank one-hot; 129..136 file
one-hot.

- **Token t is square t** (rank * 8 + file, a1 = 0, frame of the side to move, ranks mirrored for Black, files not):
  true on 4,000 records of a C#-written TPG shard (`D:\T80_v2_sub`) and on 20,000 positions through the
  DirectFromV6 loader (`test_fry_record_real.py`), both colours. (The C# writer's `squareRecords[63 - i]` branch for
  Black would reverse the files in the token order; no training record takes it: converted lc0 positions are stored
  as the side to move sees them.)
- **The piece order of bytes 0..12 is Fry's token id** (0..12), so the piece codes are a plain read.
- **No colour.** The record is side-to-move relative and carries no colour; `tokens_from_record` writes the constant
  14 ("White"). A net trained on our records never sees the other id. For Fry's own weights this matters only when
  his net is fed from our record: measured on the 214 Black-to-move verification positions, value changes by mean
  0.0021 / max 0.0127 expected score, policy logits by up to 1.4, the top move is the same in 213.
- **En passant.** The record flags every double step (5.6 % of T80 positions); a capture is possible in 0.12 %.
  Fry wants the token only when a capture is LEGAL; the record path sets it when one of our pawns stands beside the
  pushed pawn (pseudo-legal). On 19,787 real positions the two never differed (24 of 24); they differ only when the
  capture would expose our king (the pinned-pawn test position).
- **T80 contains Chess960 games** (213 of 20,000 sampled positions had castling rights with king or rooks off their
  standard squares). The castling tokens then carry the right without the standard rook square; nothing breaks, but
  Fry's own data was standard chess.
- **Castling in the policy:** our targets use king-takes-rook (e1h1 / e1a1), confirmed on 1,486 real positions with
  a legal castling move; Fry's weights use the king's two-square move.

## Block 1: square features (`SquareFeatures`)

`square_features.py`. Rows = slot base + local index; 23 slots per square, 1735 rows:
offsets to our king / their king / our queen (lowest square if several) / their queen (4 x 226 rows, 225 = absent);
the 8 neighbours (8 x 14, 13 = off the board); first piece and distance on 8 rays (8 x 85); knight-attack counts
(9); nearest pawn up / down the file (2 x 15). Directions (file, rank): (0,+1) (0,-1) (+1,0) (-1,0) (+1,+1) (-1,+1)
(+1,-1) (-1,-1).

- Computed inside forward from the 64 piece codes (record bytes 0..12, or the first 64 Fry tokens): integer gathers
  on constant index tables, comparisons, a 7-step unrolled scan for "first piece on the ray". No ArgMax, CumSum or
  loop over squares. Training / eval use `embedding_bag(sum)` (no [B, 64, 23, D] tensor); the export path is
  Gather + ReduceSum.
- Added to the embedding after its norm (Linear embedding, BT6-form embedding or FryInput); with FryInput the six
  metadata tokens get none. Zero-initialised table: the net is its feature-free self at step 0 (bit-identical, tested).
- Optimizer class by default: no weight decay, AdamW in every Muon scope (lookup table). `WeightDecayScheme: "fry"`
  decays it, as he does.
- Verified: module rows == a per-square reference written from the description on 300 game positions (155 Black to
  move) and on 1,979 real T80 positions; with Fry's table the embedding output equals the ONNX's (bit-identical in
  the exact mode, 0.035 max on rms 2.46 against the FP16 file); 1499 of the 1735 rows occur in 300 positions.
- Unsupported: `PriorStateDim`, `PieceTokens`, `RegisterTokens` (refused).

## Block 2: grouped-query attention with its own q/k width (`AttnKVHeads`, `AttnQKDims`, `AttnValueDim`, `AttnQKVBias`)

In `DotProductAttention` (`project_gqa`): separate `q_proj` (heads x dqk), `k_proj` (kv heads x dqk), `v_proj` (kv
heads x dv) instead of the fused `qkv`; per-head RMSNorm of q and k over the dqk channels with one gain vector
for all heads (`qLN` / `kLN`), keys and values repeated so query head h reads key/value head h // (heads / kv heads);
logits / sqrt(dqk); `W_h` reads heads x dv. Everything after the projections is the existing score path, so the
per-head logit bias (attention templates, smolgen) is added as before.

- Three matrices instead of one fused: each is its own matrix for Muon (his partition orthogonalises Q, K and V
  separately); one GEMM more per layer than a fused form in training.
- Works on: the plain encoder trunk; with attention templates; with the flat BT6 trunk of the 384 base arm (term
  reads on the layer's q / k -- the q- and k-side relation tables `Pq` / `Pk` take the layer's own q/k width --, edge
  sites via the logit readback, decoder with edge input); with FryInput.
- Refused in config (not modelled): NBT trunks, BT6Layer inner layers (BT6Hybrid without BT6NativeLayers),
  `BT6FusedAttention` (the Triton kernel), `BT6EdgeValues`, `EGTEdgeStream`, RPE, RoPE, non-linear / diff attention,
  `AttentionMultiplier` != 1, vis-edge gates, graph-route heads, `MuonPerHeadAttention`, `GatedAttentionOutput`, LoRA;
  and it needs `UseQKNorm` (QK-clip is inert with QK-norm, so it needed no grouped form).
- NOT updated: the BT6 exports for lc0ex / the TensorRT plugin (`bt6_terms_export.py`, `bt6_trt_plugin_export.py`)
  read the fused `attention.qkv`; with grouped attention they fail on `qkv is None`. The ordinary ONNX export works.
- Verified: == a direct einsum reference (max diff 3e-8); with kv heads = heads and q/k = value width = the plain
  width it equals our fused attention on the same weights (1e-7), also on the flat BT6 trunk with live reader tables
  (0.0); gradients on every parameter; and the Fry parity above (24 on 12, widths 8 / 16 / 24, value 32, with bias).

## Block 3: the 70-token input (`FryInput`)

`fry_input.py`: `tokens_from_record` builds the ids inside forward; `FryInput` = `nn.Embedding(81, D)` + learned
position embedding [70, D]; no embedding norm. The six metadata tokens are appended after the 64 squares and run
through every layer (attention and FFN). After the trunk-end norm the heads read the 64 square states; `ValueHeadFry`
reads all 70.

Where the trunk and heads assume 64 tokens, and what happens with 70:

| place | with FryInput |
|---|---|
| attention templates (64 x 64 tables, mixer on the 64 squares) | supported: bias zero-padded to 70 x 70 (his form) |
| QK-norm, grouped-query attention, FFN, norms, input skip (the 70-token embedding is the skip source) | supported (token-count free) |
| shared head front-end (`headPremap` -> 64 x C flatten), MLP heads, value pool / private value channels | supported: read the 64 square states |
| move-token decoder (policy from the 64 square states + record) | supported, WITHOUT `MoveTokenEdgeInput` |
| depth probes | supported: the depth states are sliced to 64 squares (tested) |
| decoder trunk-layer mix (`MoveTokenTrunkMix`), mid-depth decoder | read the same sliced states; not refused, NOT tested |
| `PolicyHeadFry`, `ValueHeadFry` | supported (the value head takes 70 tokens) |
| BT6 hybrid (relations, pair stream, edge state / sites, its embedding and policy head) | refused |
| EGT edge stream; smolgen in all forms (64-token generator, basis / static banks); RPE / RoPE | refused |
| NBT trunk (bottleneck blocks, shared smolgen, hyper-connections, MoD, carry) | refused |
| piece tokens, register tokens, DenseFormer, LoopCount > 1, post-norm | refused |
| piece-relation / ray / vis-edge biases, graph-route heads, king-distance channels, spectral PE, dual plane, tactical codebook, refiner, FFN global, TSB, SoftMoE, PriorStateDim, LoRA | refused |
| env-gated: phase FiLM, GTAB, RPE-from-embedding | assert in forward |
| value depth attention / policy depth attention (env / config) | not refused, NOT tested |

- With FryInput the net sees the current board, castling rights and e.p. only: no history, repetition flags or
  rule-50 counter (as Fry). `InputHistoryKeep` is then irrelevant.
- Verified: tokens from the record == the python-chess board encoder on 300 game positions and 19,787 real T80
  positions in every column except colour (and non-legal e.p., see above); 70 tokens through the trunk; template bias
  exactly 0 on metadata rows / columns; a castling right changes the value; with the decoder and depth probes the loss
  path runs; the Fry parity.

## Block 4: heads, input skip, optimizer

**(a) `PolicyHeadFry`** (`fry_heads.FryPolicyHead`): F = y Wf + bf, T = y Wt + bt, L = F T^T / sqrt(D) in float32,
gathered to our 1858 moves; + (y[to] Wp + bp)[piece] for the 66 explicit q / r / b promotions, and for the 22 plain
rank-7 -> rank-8 moves with |file step| <= 1 the knight term when one of OUR PAWNS is on the from square (our layout
has no knight-promotion index; the plain move is the knight promotion). Promotion projection order N, B, R, Q. With
that every legal move's logit equals his 1968-move formula (tested on 3,492 legal moves incl. 28 promotions, 10
castling moves; and on 12,795 moves in the parity). No legality mask (ours neither: the loss masks). Refused with the
decoder, the BT6 policy head, dual plane, refiner, policy depth attention, LoRA.
Reminder from our own ladder: a bilinear policy head was catastrophic on puzzles in the 5M puzzle-only test; this
head is of that family.

**(b) `ValueHeadFry`** (`fry_heads.FryValueHead`) replaces the body of our HL-Gauss head (`hlg_head`), reading the
token states after the trunk-end norm instead of the shared head front-end. Our buckets (equal width over q in
[-1, 1], sigma 0.75 bucket widths) are his over expected score bucket for bucket, so his 128 logits load unchanged.
- with `ValueHLGaussServe` (default draw "wdl"): the served value is our existing construction, Q from the HL-Gauss
  head, D from the WDL head, exported as log-WDL: Ceres and the user's engine get W/D/L as before.
- `ValueHLGaussServeDraw: "none"`: D := 0, W = (1 + Q) / 2 (his head alone; the WDL head is built but neither
  trained nor served; `LossValueMultiplier` must be 0).
- without `ValueHLGaussServe`: training-only HL-Gauss head with his body.
- Decoder / pool value injects do not reach this head.

**(c) Input skip.** `NBTInputSkip` on the plain trunk already is his form (r = a * x + b * x0, the mixed r replaces
the stream, scalars not decayed) for layers >= 1; its parameter vectors have an element for layer 0 that is unused.
`InputSkipLayer0` uses it: layer 0 reads (a0 + b0) * x0. Identical at init. Also acts on the flat BT6 trunk.
His learned values (layer 0: a 2.94, b 1.94) show the layer-0 scalar is in practice an embedding gain.

**(d) Optimizer.** How ours assigns parameters today: `train.py` picks the Muon set by `MuonAdamWScope` (name-based
predicates; default 'all-non-trunk' = 2-D tensors in `transformer_layer` minus embeddings, template mixers, BT6
reader tables, LoRA), everything else goes to the AdamW group inside our `Muon` class; `wd_partition.py` defines
decay / no_decay (norm gains, biases, embeddings, raw tables and gates = no decay), which Muon honours only with
`MuonHonorNoDecay`.
- `MuonAdamWScope: "fry"`: every 2-D tensor in the trunk layers (Q, K, V, attention output, three FFN matrices, the
  template mixers); AdamW for the rest. Plain trunk only, refused with BT6Hybrid.
- `WeightDecayScheme: "fry"`: group weight decay on everything except `.qLN.` / `.kLN.` gains and the input-skip
  scalars (so norm gains, biases, embeddings, templates and the square-feature table ARE decayed), in both
  branches. Exclusive with `MuonHonorNoDecay`.
- `WeightDecayFromSchedule: 0.17`: wd = 1 / (0.17 * lr * steps); at lr 1.5e-3, 6.0B positions, batch 4104 it gives
  0.00268 (his table). Refused when it would exceed 0.5 (short runs). NOTE: for a 300M arm at 9e-4 and batch 4608
  the rule gives 0.100; it keeps lr * wd * steps constant (5.9), so short arms get a strong per-step decay.
- Our Muon vs his (torch.optim.Muon, match_rms_adamw): same update direction (the momentum buffers differ by a
  constant factor, which the orthogonalisation removes), same shape factor 0.2 * sqrt(max(rows, cols)), same
  decoupled decay 1 - lr * wd. His AdamW betas are torch's defaults (0.9, 0.999): `Beta1` / `Beta2`.
- Verified: partition kinds listed and asserted in `test_fry_blocks.py` (f); the trainer's own log in the smoke
  ("Muon partition scope=fry: 32 muon / 67 adamw", "WeightDecayScheme 'fry': decay on 89 ... no decay on 10").

## Where the FP16 differences come from (first two lines of the parity table)

Found while making the third line exact:
- the file's norm gains are FP32 tensors holding FP16-rounded values (all norms, the trunk-end norm included): up to
  3.6e-4 off the checkpoint;
- the attention logit divisor is an FP16 constant: 2.828125 for sqrt(8), 4.8984375 for sqrt(24) (1.1e-4 off; 4 is exact);
- trunk weights, biases, skip scalars, embeddings and tables are FP16 (stream writers after the 1 / 16 scaling);
- the 768-wide norms use epsilon 1e-5 on the 1/16 stream = 2.56e-3 in training units (training: 1e-5);
- activations: single stream channels reach |x| = 6,140 in training units at the input of layer 7 (rms 49); one FP16
  step of the 1/16-scaled stream is 4 there. Single elements differ by up to 7 between the FP16 file and float32
  while the mean difference stays 6e-4 of the rms.
Heads are FP32 in the file and equal the checkpoint.

## Tests (all CPU; run from `src/CeresTrainPy` with `CUDA_VISIBLE_DEVICES=""`)

| command | what |
|---|---|
| `python test_fry_blocks.py` | unit tests (a)-(h): each block, refusals, optimizer classes, all together, bf16 forward, ONNX, torch.compile |
| `FRY_ORACLE=fp16\|fp32\|exact python test_fry_parity.py [n]` | Fry in our trainer vs the ONNX, stage by stage (embedding, 28 layer inputs, end norm, heads), then the record path |
| `python test_fry_regression.py all <pristine 8a7d72e src/CeresTrainPy> <config dir> <ids...>` | keys off = bit-identical (pristine copy: `git archive 8a7d72e src/CeresTrainPy \| tar -x -C <dir>`) |
| `python test_fry_record_real.py [tar] [n]` | record path on real T80 positions (legal-move mask, tokens, feature rows) |
| `python test_fry_smoke.py <dir of loose v6 .gz chunks> [work dir]` | real train.py on CPU + its ONNX export + onnxruntime |
| `python test_fry_configs.py` | the example configs build and take a loss / backward step |

Output of the last full run of all six (after the last code change): `scratch_fry/final_verification.txt`; the
existing test files that were run (8, all passing): `scratch_fry/existing_tests.txt`. `scratch_fry/t80_small` = 400 games extracted from
`D:\TAR\training-run1-test80-20240427-1117.tar` for the smoke.

## Smoke and export

`dev_64_4_fryall_t80_smoke` (64 x 4, 4 heads on 2, q/k 4 / 8 / 12 / 12, 70 tokens, square features, 4 templates, skip
with layer 0, Fry heads, served WDL, Muon scope 'fry', decay scheme 'fry', wd 0.01), DirectFromV6 on 400 T80 games,
16,384 positions at batch 256, CPU: total loss 1.7313 -> 1.3370 between the first and last logged step of the last
run (the runs differ a little: time-based logging, loader shuffle); `ckpt_*`, `*.onnx` written; the ONNX (opset 18,
FP16, input `squares` [B, 64, 137] as today) gives served WDL summing to 1 and matches the checkpoint net in
PyTorch (policy logits on legal moves 1.8e-3, log-WDL 1e-3: FP16 file vs float32).

What the export contains that TensorRT has to accept (NOT built with TensorRT here):
- integer work for the piece codes, tokens and feature rows (counts of the 64 x 4 smoke net): `Cast` to INT64 (30),
  `Round` (2), `Gather` (20, on INT64 indices: the ray / knight lookups, the table lookup [B * 64 * 23], the token
  embedding, the policy gathers), `Equal` (22) / `Greater` (8) / `Less` (4) / `And` (27) / `Or` (8) / `Not` (12) /
  `Where` (9) on integers, `ReduceMin` (4: the king / lowest queen squares), INT64 constant index tables (ray
  table [3584], knight table [512]);
- `Expand` for the grouped keys / values; `Pad` for the template bias on the metadata tokens; `Erf` (GELU) in the
  value head; no ArgMax, CumSum, Loop, ScatterND, Einsum or Tile (Fry's own file has ArgMax, Einsum, Tile, ScatterND
  and built in TensorRT 11.3 per the review).
- the square-feature sum is 23 table rows added in the stream's dtype; the stream itself is ours (FP32 norm islands
  as for every net); no 1/16 stream scaling is applied by our exporter.

## Example configs (`configs/fry_blocks/`, not launched)

Base arm: `prod_384_12_fry_hlg_tpl8qkbt6Ant135notripei_t80t91_300M` (26,659,867 parameters).

| config | what | parameters |
|---|---|---|
| `parity_768_28_fry117` | Fry itself (for `test_fry_parity.py`; `fry_parity.FRY_NET` / `FRY_OPT` are the same dictionaries) | 115,920,656 = his 113,227,484 + 2,693,172 in our always-built head front-end, WDL and uncertainty heads |
| `prod_384_12_fry_hlg_tpl8qkbt6sqfAnt135notripei_t80t91_300M` | base + `SquareFeatures` | +666,240 (the table) |
| `prod_384_12_fry_hlg_tpl8qkbt6gqaAnt135notripei_t80t91_300M` | base + 12 query heads on 6 key/value heads, q/k 8, 8, 16, 16, 16, 24 x 7, value 32, no bias | -2,944,432 |
| `prod_384_12_fryfaithful_t80t91_300M` | 384 x 12 with all blocks and his stage-3 style recipe | 13,867,496 |

The faithful arm: FryInput, square features, 8 templates, QK-norm, 12 heads on 6, q/k 0.25 / 0.5 / 0.75 of the value
width by depth, FFN 1.29x without biases, input skip in every layer, NormEps 1e-5, his policy head, his value head
as the served Q; policy 1 + HL-Gauss 0.5, no auxiliary losses; FractionQ 0.8; Muon 0.95 scope 'fry', AdamW betas
0.9 / 0.999, decay scheme 'fry', wd from the rule (0.100), one learning rate (no heads ratio), clip 1.0.
Still different from his stage 3: (1) a WDL head at 0.1 for the served draw probability (he has none; set
`ValueHLGaussServeDraw` "none" and `LossValueMultiplier` 0 for his exact form); (2) the colour token is constant;
(3) e.p. token pseudo-legal; (4) our value-target blend partner is the deblundered result; (5) T80 + T91 with our
position sampling, including Chess960 games; (6) peak LR 9e-4 without decay, 3 % warm-up, batch 4608 (his: 5e-4
re-warmed, held, decayed over the last 56 %, 500 warm-up steps, batch 4104); (7) from scratch (his stage 3 starts
from 12B positions of Stockfish-labelled training); (8) bf16 autocast instead of a bf16 compute copy; (9) our
always-built head front-end / uncertainty heads exist untrained; (10) no mirror augmentation (his use is unknown).

## Needs the GPU or a decision

1. TensorRT: build an export with SquareFeatures / FryInput (INT64 casts and gathers) and check FP16 outputs against
   onnxruntime; whether the Ceres TensorRT wrapper's FP32 norm marker still catches every norm with 70 tokens.
2. Training speed of each block (embedding_bag on GPU; three projection GEMMs instead of one; 70 tokens).
3. bf16 autocast backward, torch.compile on CUDA (the CPU compile test passes), DDP with the new parameters
   (static graph, `MuonDistributed`, `DDPManualGradSync`).
4. The weight-decay rule on short arms: 0.100 at 300M. Keep the rule, or his 0.0026?
5. The served value of a Fry-headed net: WDL head for D (as configured) or D = 0.
6. Grouped attention for the lc0ex / TensorRT-plugin BT6 exports (not done; they read the fused qkv).
7. Resume of an existing net with a block added (`CheckpointResumeAllowNewParameters`): `resume_new_params.py` does
   not know `square_features.table`; SquareFeatures is function-preserving at init, so it could be added there.
8. Colour for record-fed serving of Fry's own weights: our record cannot supply it (see above).

## Review document: wrong or incomplete, found while implementing

1. Section 5 says Fry's tokens come from our record at "castling 104-107, colour 108". Those are lc0 plane numbers.
   In our record 104..111 are repetition flags, castling is 112..115, e.p. is byte 118 on the pushed pawn's square,
   and there is NO colour.
2. Section 1.6: "heads and all norm gains FP32". The gains are FP32 tensors with FP16-rounded values; and the logit
   divisor sqrt(dqk) is an FP16 constant. Not mentioned: stream channels up to |x| 6,140 in training units.
3. Section 5, FFN "SAME form": ours has biases on the first and last Linear, his has none (`FFNNoBias`).
4. Sections 1.3 and 3.3 give two different promotion-piece orders without saying they are different things: the id
   order is N, R, B, Q; the projection's output order is N, B, R, Q (both confirmed against the checkpoint buffers).
5. Section 6 step 2a counts colour among what forward can derive; it cannot (item 1).
6. Not in the review: T80 has Chess960 games; our e.p. byte marks every double step.
