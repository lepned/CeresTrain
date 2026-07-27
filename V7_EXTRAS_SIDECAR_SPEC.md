# V7-extras sidecar format (`<shard>.v7x.zst`)

Emitted by `gen-tpg --v7-extras` when converting **V7 training data** (8396-byte
records = stock lc0 V6 + 40-byte rescorer tail, e.g. from `lc0-rescorer-v7`).
Carries per-position rescorer fields that have no slot in the TPG record:

- **censored q_st** — short-term value target: backward EMA of `root_q`
  (weight 5/6 on the carry, sign-alternating, STM-relative) whose carry
  re-initializes to `best_q` at every record where a deblunder trigger fired,
  so the average never blends across a detected blunder.
- **censored d_st** — same censored walk over `root_d`/`best_d`, clamped >= 0.
- **z-provenance** — origin of the record's `result_*` fields:
  `0` original game result | `1` syzygy rescore | `2` deblunder (noise) |
  `3` deblunder (unintended cross-ply) | `4` op1 8-man relabel.

## File layout

One `.v7x.zst` per output set, zstd-compressed, alongside `<shard>_setN.zst`
(and `.tgt.zst` survival sidecars if enabled). Decompressed stream:

```
16-byte header:
  [0..3]  magic "TPGX"
  [4]     format version = 1
  [5]     bytes per row = 9
  [6..15] reserved (zero)

then one 9-byte row per TPG record, in EXACTLY main-shard record order:
  [0..3]  censored q_st   float32 LE   (STM-relative, matches TPG conventions)
  [4..7]  censored d_st   float32 LE
  [8]     z-provenance    uint8
```

Row `i` of the sidecar corresponds to TPG record `i` of the same set — rows are
buffered in lockstep with the main writer buffers and flushed at the same append
point (identical mechanism and guarantees as the survival `.tgt.zst` sidecars;
see SURVIVAL_TARGET_SPEC.md, including the shutdown/flush integrity handling).

Python read:

```python
import zstandard, numpy as np
raw = zstandard.ZstdDecompressor().decompress(open(fn, 'rb').read(), max_output_size=1 << 34)
assert raw[:4] == b"TPGX" and raw[4] == 1 and raw[5] == 9
rows = np.frombuffer(raw[16:], dtype=[('cens_q', '<f4'), ('cens_d', '<f4'), ('prov', 'u1')])
```

## Constraints

- Requires V7 source data: a V6 game encountered with `--v7-extras` is a hard
  failure (no silent zero-fill).
- Same sidecar constraints as survival: `NumRelatedPositionsPerBlock == 1`,
  no annotation evaluator/postprocessor (record omission would desync rows),
  Zstandard output only.
- Composes freely with `--survival-horizon K` (independent files).
- The scalar values pass through untransformed: they are STM-relative in the
  source record, which is the TPG convention; no mirroring/remap applies.

## Provenance of the format

Corpus fields defined by the `k2hybrid` V7 rescorer (Kovax, 2026-07); byte
offsets in the source record: censored q_st +8368, censored d_st +8372,
z-provenance +8364 (stored as float in the file, converted to u8 here).
