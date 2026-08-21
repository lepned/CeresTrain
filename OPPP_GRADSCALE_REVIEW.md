# Code review: a70d06e..8db0c5d (GradScale + OppPolicy aux + multi-dir DirectFromV6) — 2026-08-21

No effort level was given, so I reused **max** — the level you typed last time; typing a level (for example `/code-review high`) changes it.

Review of a70d06e..8db0c5d (18e5b84, 4fe2f6e, f4a9e1d, 8db0c5d), code files only; recipe .md read for intent (it enables all three features together on the 4xA100 run: DirectFromV6 over `t91_v7_op1;cv2`, `DualPlanePolicyGradScale: 0.25`, `LossOppPolicyMultiplier: 0.1`). Ten finder angles ran; the Phase-3 sweep was skipped per quota constraint, so coverage is the 10-angle pass plus direct verification reads.

## Verified clean

- **OppPlayedIndex parsing vs the v7 struct**: field order in `V6ChunkDataset`'s `V7_DTYPE` matches the authoritative C# struct (`C:\Users\lepne\source\repos\Ceres\src\Ceres.Chess\LC0\Positions\Position\Training\EncodedTrainingPositionExtraV7.cs`) field-for-field; offsets corroborated by `V7_EXTRAS_SIDECAR_SPEC.md` (+8368/+8372). `NO_MOVE_INDEX = 0xFFFF` is correctly folded to -1 by the `< 1858` mask (u2 is cast to int32 first, no wraparound); target frame (opponent's stm-relative 1858 space) matches the record convention; head is training-only.
- **Grad-scale core semantics**: with scale=1.0, absent, or eval/export, the branch is skipped and `_dpt` aliases `_dp_tokens` — byte-identical to committed behavior. Autograd factor is exactly `a`. Consumer split is right: policy-decode reads (`dp_pol_p`, `dpv_a`, `dpv_b`) scaled; value-family reads (`dpva_k/dpva_v`, `dp_surv_head`) untouched; `_dpt` is always defined where read; the attr is always set when forward can reach it.
- **torch.compile(dynamic=False)**: the stash write and grad-scale branch gate on constant attrs plus `self.training` (specializes; same contract as `opt_head`); `compute_loss` runs eager on the raw module, so the masked CE and `.any()` are outside the compiled graph.
- **Export**: `save_model` calls `eval()` before trace/ONNX export, so the training-gated head and stash never enter exported graphs; `recover_export` rebuilds from the stored config (keys match) and its `strict=False` load is benign in both mismatch directions; the FP16 Cast reconciliation is untouched; EMA dual-export unaffected.
- **Resume (non-LoRA)**: `'oppp_head.'` in `_AUX_HEAD_PREFIXES` handles both resume directions; Muon partitioning and the family-LR lists treat the head exactly like `opt_head`.
- **Data plumbing**: `filter_tensor` slices the int64 [B,1] tensor correctly; `batch` is a dict at all five `compute_loss` call sites; `_move_batch_to_device` carries the new key to GPU; every `v7x` consumer is length-agnostic (no stale 3-tuple unpacks); yield arity matches `batch[14]` on both paths.
- **Gradnorm two-pass stash lifecycle**: the no-clear-in-gradnorm-mode is the deliberate pairing (the following real pass on the same batch consumes the stash); no stale-stash leak into validation.
- **Back-compat**: single-dir TPG configs and non-dp configs are byte-identical; both new config keys default inert.
- **All 15 previous review fixes intact** (checked item-by-item against `V6_DATASET_REVIEW.md`): EP/Move50/history/policy-floor parity, prev-record suboptimality, `_RUN_SHUFFLE_SEED` rank-consistent shuffle (multi-root union is built deterministically before the seeded shuffle; DDP seed refusal intact), fd-LRU, isal catch, version pinning, NaN guards, atomic chunkindex cache, fail-fast, block conversion, knob rejection.
- The per-step `bool(_ovalid.any())` host sync is immaterial (the loss path already syncs via `.item()` calls). No CLAUDE.md violations in code or commit messages.

Refuted along the way: the "0-as-sentinel mislabel" concern (0 is a genuine move index; sentinel is 0xFFFF) and the stale-stash-into-validation concern (deliberate design).

Latent notes below the cap (all unreachable today): the oppp block lacks the `value_out is not None` 4-board gate its siblings have (v6 asserts BoardsPerBatch=1, TPG never carries the key); `SECONDARY_WEIGHT_OVERRIDES` has hlg/opt/soft policy-family follows but no oppp entry — the same historical gap class that later needed one-by-one fixes; mirror-aug would corrupt the geometry-dependent index if v7x ever flowed through it; the `hasattr(batch, 'get')` guard is dead code.

## Findings

```json
[
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\train.py",
    "line": 1293,
    "summary": "The new stash-only oppp_head is missing from the DDP aux-head guard (which lists dp_surv/placement/survival/stvalue/depth-probes), so the exact planned server run crashes at step 1 with the raw reducer error instead of the guard's actionable message.",
    "failure_scenario": "4xA100 torchrun, LossOppPolicyMultiplier=0.1 (the committed recipe), DirectFromV6 forces BOARDS_PER_BATCH=1 -> train.py:820 defaults static_graph=0, find_unused_parameters=True. forward() returns only the 11 head outputs; _last_oppp_out is a stash, so DDP's output traversal pre-marks oppp_head params 'ready as unused', then the backward through compute_loss's oppp CE delivers real grads and every rank dies with 'Expected to mark a variable ready only once' on the first step — the mechanism train.py:1276-1292's own comment documents. Fix: add opp_policy_weight to the guard condition (opt_head shares the omission pre-existing; the diff repeats it for the head it adds)."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\ceres_net.py",
    "line": 1929,
    "summary": "The oppp loss emits no zero-weighted participation term when 'opp_played_idx' is absent or all targets are -1 (oppp_loss stays int 0), unlike survival/dp_surv/stvalue which emit 0.0*out.float().sum() — so even the documented CERES_DDP_STATIC_GRAPH=1 workaround crashes mid-run.",
    "failure_scenario": "DDP with static_graph=1 and a mixed recipe (DirectFromV6 v7 primary + TPG secondary via TrainingFilesDirectory2 — TPG batches never carry the key since .v7x sidecars are 3-tuples): the first secondary batch shrinks the used-parameter set and DDP aborts with 'Your training graph has changed in this iteration'. Same trigger on a (rare) all--1 batch. Fix: participation term in both the _ot-is-None and no-valid branches, per the in-file template at ~2117/2165 — or the sync-free form F.cross_entropy(..., ignore_index=-1, reduction='sum')/count.clamp(min=1), which keeps the head in the graph unconditionally."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\ceres_net.py",
    "line": 774,
    "summary": "float(getattr(config, 'NetDef_DualPlanePolicyGradScale', 1.0) or 1.0) — the 'or 1.0' swallows a configured 0.0, the full-detach endpoint of the very value-oscillation experiment this knob was built for.",
    "failure_scenario": "\"DualPlanePolicyGradScale\": 0.0 in the NetDef JSON: 0.0 is falsy, so the expression yields 1.0; the != 1.0 gate also suppresses the startup banner, and the arm silently trains as an exact copy of the control — a burned A/B cell recorded as 'full detach changes nothing'. The forward at 1659 handles a=0.0 correctly; drop the 'or 1.0' (config.py:513 already defaults 1.0)."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\train.py",
    "line": 1349,
    "summary": "No startup preflight that the source can supply 'opp_played_idx' — survival hard-raises on wrong source, stvalue and prov-weights check 7 in primary_dataset._diag_versions, but LossOppPolicyMultiplier>0 gets no equivalent check.",
    "failure_scenario": "Weight>0 with a TPG SourceType, or a DirectFromV6 corpus whose version pins to 6: the head is built and announced, forward pays a full [B,1858] Head every step, oppp_loss stays int 0 forever, and the isinstance-int gate means opp_policy_loss is never even logged — a silent zero-supervision whole run, the exact failure class the neighboring preflights state verbatim they exist to prevent. Mirror of the stvalue check is the template."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\v6_dataset.py",
    "line": 329,
    "summary": "Multi-root comment promises 'pinning applies across the union', but version pinning is first-decoded-chunk-wins per forked worker, and nothing enforces the documented same-version constraint that _startup_diagnosis already computes the data for.",
    "failure_scenario": "TrainingFilesDirectory='v7root;v6root' under DDP: workers/ranks pin different versions depending on which chunk their shard's shuffle serves first, each silently discarding a different root (reported only in an epoch-end diag line, and an epoch here is weeks). Batches then carry v7x keys on some ranks and not others — rank-divergent used-parameter sets that hang/crash NCCL under static_graph with opp/stvalue/prov-weights, and ~half the corpus silently dropped either way. _startup_diagnosis already collects versions across the union; assert len(versions)==1 there."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\v6_dataset.py",
    "line": 180,
    "summary": "Multi-root entry union does no duplicate/overlap detection: a repeated root, or one root nested inside another, double-counts every affected chunk via the recursive glob.",
    "failure_scenario": "'/data/lc0;/data/lc0/2026-07' (nested) or a copy-paste-duplicated root: chunks under the overlap appear twice in entries, giving 2x sampling weight after the flat shuffle — the 'volume-proportional mixing' contract silently becomes oversampling, with only an inflated startup chunk count as a clue. A realpath-keyed dedup (or duplicate-count assert) at build time catches it."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\v6_dataset.py",
    "line": 471,
    "summary": "PLAUSIBLE: sentinel mapping only folds values >=1858 to -1; a v7 corpus whose writer zero-filled the ExtraV7 tail would yield 100% 'valid' class-0 targets, and _startup_diagnosis samples versions/provenance but not opp-index population.",
    "failure_scenario": "Enabling the head on any v7 corpus other than the two measured Kovax ones (99.1% populated), e.g. an older run where the 40-byte tail exists but OppPlayedIndex was never populated: every record trains a hard one-hot CE toward move index 0 at full weight through the shared fS_policy front-end, with a plausible-looking loss curve and nothing in the logs to catch it. Cheap hardening: report the populated fraction in _startup_diagnosis and warn/raise near 0%."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\train.py",
    "line": 126,
    "summary": "The rewritten validation accepts ';' lists for EVERY SourceType (only V6ChunkDataset splits), and a separator-only/empty TrainingFilesDirectory now yields zero loop iterations — both directions lose the old loud config-time exit.",
    "failure_scenario": "TPG source with 'D:/a;D:/b': per-part validation passes, then TPGDataset receives the raw string and dies in os.listdir('D:/a;D:/b') (tpg_dataset.py:345) deep in dataset init, right after startup asserted both dirs exist — before the diff this failed immediately with the clear 'does not exist' message. TrainingFilesDirectory=';' or '' skips the loop body entirely and fails late in dataset construction. Fix: reject multi-part strings unless SourceType==DirectFromV6, and error when the parts list is empty."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\train.py",
    "line": 1584,
    "summary": "'oppp_head.' was added to _AUX_HEAD_PREFIXES (non-LoRA resume) but not to the LoRA-mode remap's pass-list, whose else clause indexes the checkpoint by every non-excepted model key.",
    "failure_scenario": "LoRA fine-tune with LossOppPolicyMultiplier>0 resumed from a base checkpoint predating the head (the standard LoRA-from-orig flow): the remap loop reaches 'oppp_head.*', matches none of lora_/tactical_/tsb/value_premap/vis_edge/graph_route/_pool_inject, and new_state_dict[name] = loaded['model'][name] raises KeyError at startup. Same latent gap exists for opt_head/hlg_head/sp_head (pre-existing); the diff repeats it for the head it introduces."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\v6_dataset.py",
    "line": 186,
    "summary": "num_files_to_skip is applied to the concatenated pre-shuffle entry list, so under multi-root every skipped chunk comes from root1's sorted entries first — skip >= len(root1) silently deletes root1 from the run.",
    "failure_scenario": "NumTPGFilesToSkip=40000 with 'rootA;rootB' where rootA holds 30000 chunks: all of rootA plus rootB's first 10000 sorted chunks are dropped before the shuffle; startup still prints a plausible total and nothing indicates the skip consumed an entire root (single-root semantics were 'skip N of this corpus'). Either apply the skip per-root or document/assert the interaction."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\ceres_net.py",
    "line": 775,
    "summary": "The grad-scale banner prints 'enabled' whenever scale != 1.0, but forward only applies _dpt inside the dp_policy_decode branch — no assert ties DualPlanePolicyGradScale to DualPlanePolicyDecode.",
    "failure_scenario": "Config sets DualPlanePolicyGradScale=0.25 but omits DualPlanePolicyDecode (e.g. cloned from a non-decode baseline): init prints 'DUAL-PLANE POLICY-GRAD SCALE enabled' yet the run is bit-identical to control — a misleading log that silently invalidates the arm, contrary to the repo's loud-assert convention for config combos (compare the victim-decode assert at line 816)."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\ceres_net.py",
    "line": 1661,
    "summary": "'_dp_tokens * a + _dp_tokens.detach() * (1-a)' is forward-identity only up to floating-point rounding for fractional a (about 1-2 ulp under bf16), contradicting the 'forward-IDENTITY' comment; a bit-exact form exists.",
    "failure_scenario": "With a=0.25 under bf16 autocast, training forward activations differ from the unscaled net by rounding noise, so any step-0 parity or paired-seed byte-comparison methodology (the repo's standard A/B hygiene) reports spurious diffs for the grad-scale arm. 'x.detach() + (x - x.detach()) * a' computes x + 0*a = x exactly in any dtype with the same gradient a; eval/export paths are unaffected either way (scale=1/disabled verified byte-identical)."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\train.py",
    "line": 883,
    "summary": "count_zst_files is a write-only dead variable (its only other occurrence is a separate local in tpg_dataset.py's __main__ harness), and the diff extended it into a full os.listdir of every ';' root — guaranteed 0 for DirectFromV6 roots (.gz/.tar, never .zst); the validation loop's 'not os.listdir()' also materializes complete listings just to test emptiness.",
    "failure_scenario": "Loose-chunk v6 roots can hold ~10^6 directory entries; each rank lists every root twice at startup (once for the dead count, once for the emptiness test) — seconds locally, tens of seconds to minutes on network mounts, for a variable with zero consumers. Delete lines 883-884 and use next(os.scandir(d), None) is None for the emptiness check."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\train.py",
    "line": 884,
    "summary": "The ';'-root parse '[d.strip() for d in str(X).split(';') if d.strip()]' is hand-rolled in three places (train.py:126, train.py:883-884, v6_dataset.py:180) with no shared helper.",
    "failure_scenario": "The parse IS the definition of which roots a config names; three copies drift independently (separator, quoting, trailing-';' handling), so startup validation can approve a different directory set than the loader enumerates — precisely the silent mismatch the validation loop exists to prevent. One split_roots() helper used by all three (and by any future consumer) closes it."
  },
  {
    "file": "C:\\Users\\lepne\\source\\repos\\CeresTrain\\src\\CeresTrainPy\\tpg_dataset.py",
    "line": 813,
    "summary": "The v7x sidecar payload grew positionally from 3 to 4 elements discriminated by len(), with two producers (sidecar 3-tuple at tpg_dataset.py:575-577, v6 4-tuple at v6_dataset.py:473-476) and no named layout contract.",
    "failure_scenario": "The v6 dtype already carries next_played_idx — the obvious next target. If a 5th/reordered field is appended by one producer in a different order than the consumer assumes, v7x[3] silently becomes the wrong array: torch.tensor(..., dtype=torch.int64) happily casts float cens/prov data, so the head trains on garbage with plausible loss curves and no crash. A namedtuple or shared index constants make producers and consumer agree by name; positional len() checks cannot catch a reordering."
  }
]
```

The two highest-severity findings (DDP guard omission and missing participation term) both fire on the exact configuration the committed recipe prescribes for the next fresh-1B server run; fixing them means adding `opp_policy_weight` to the train.py:1293 guard and giving the oppp block the same participation-term (or `ignore_index`) treatment the survival/stvalue blocks already have.
