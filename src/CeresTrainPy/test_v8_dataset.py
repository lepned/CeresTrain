"""V8 (cv4) record support in v6_dataset.py — dtype, dispatch, and real records.

  python3 test_v8_dataset.py                       # synthetic only
  python3 test_v8_dataset.py <tar-or-dir> [cell]   # + real cv4 records

The real-data mode is the one that matters: the offsets in V8_DTYPE come from a
written spec, and a spec can be misread. Decoding actual games and checking the
corpus invariants (README_cv4_v8_record_format.txt, section 7) is what proves the
layout, because every invariant spans fields the layout would have to shift.
"""
import os, sys, gzip, tarfile, io
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
from v6_dataset import (V6ChunkDataset, V6_DTYPE, V7_DTYPE, V8_DTYPE, _BY_VERSION,
                        _HAS_V7_TAIL, V6_RECORD_BYTES, V7_RECORD_BYTES, V8_RECORD_BYTES,
                        V8_SLOTS)


def synthetic():
  # Byte accounting: v8 = v7 + 2,640, and the v6/v7 prefix must stay untouched so
  # every existing decode path keeps working on a v8 record.
  assert V8_DTYPE.itemsize == V8_RECORD_BYTES == 11036
  assert V8_RECORD_BYTES - V7_RECORD_BYTES == 2640
  assert V8_DTYPE.descr[:len(V7_DTYPE.descr)] == V7_DTYPE.descr
  for name in V7_DTYPE.names:
    assert V8_DTYPE.fields[name][1] == V7_DTYPE.fields[name][1], name

  # Spot-check the absolute offsets the spec gives, not just the running total:
  # a pair of compensating size errors would keep itemsize right and the data wrong.
  for name, want in (('v8_layout_version', 8396), ('n_legal', 8402), ('block_flags', 8406),
                     ('slot_idx', 8412), ('slot_prior', 8668), ('slot_q', 8924),
                     ('slot_d', 9180), ('slot_m', 9436), ('slot_n_raw', 9692),
                     ('slot_n_deforced', 9948), ('slot_reply_idx', 10204),
                     ('slot_reply_q', 10460), ('slot_reply_n', 10716),
                     ('recipe', 10972), ('reserved_v8', 11000)):
    got = V8_DTYPE.fields[name][1]
    assert got == want, f'{name}: offset {got}, spec says {want}'

  assert set(_BY_VERSION) == {6, 7, 8}
  # Regression: the same-version guard must be derived from _BY_VERSION. A literal
  # {6, 7} silently admitted a mixed v7+v8 union (workers pin divergently, ~half the
  # corpus dropped, rank-divergent batch keys under DDP).
  for mixed in ({7, 8}, {6, 8}, {6, 7}):
    assert len(mixed & set(_BY_VERSION)) > 1, mixed
  assert _HAS_V7_TAIL == {7, 8} and _HAS_V7_TAIL <= set(_BY_VERSION)
  for v, (rb, dt) in _BY_VERSION.items():
    assert rb == dt.itemsize, (v, rb, dt.itemsize)

  # Round-trip: write known values through the dtype and read them back.
  rec = np.zeros(1, dtype=V8_DTYPE)
  rec['version'] = 8
  rec['v8_layout_version'] = 2
  rec['recipe_id'] = 4
  rec['n_legal'] = 20
  rec['n_stored'] = 20
  rec['slot_idx'][0, :3] = [100, 200, 300]
  rec['slot_q'][0, :3] = [32767, 0, -32767]
  rec['slot_n_raw'][0, :3] = [500, 200, 99]
  back = np.frombuffer(rec.tobytes(), dtype=V8_DTYPE)
  assert back['version'][0] == 8 and back['v8_layout_version'][0] == 2
  assert list(back['slot_idx'][0, :3]) == [100, 200, 300]
  assert abs(back['slot_q'][0, 0] / 32767.0 - 1.0) < 1e-4
  print(f'  synthetic OK: v8 dtype {V8_DTYPE.itemsize} B, {len(V8_DTYPE.names)} fields, '
        f'{V8_SLOTS} slots, spec offsets match, round-trip exact')


def _games_from(path, limit):
  """Yield (name, gunzipped bytes) from a .tar of .gz members, or a dir of .gz."""
  if os.path.isdir(path):
    n = 0
    for root, _d, files in os.walk(path):
      for f in sorted(files):
        if not f.endswith('.gz'):
          continue
        yield f, gzip.open(os.path.join(root, f)).read()
        n += 1
        if n >= limit:
          return
    return
  with tarfile.open(path) as tf:
    n = 0
    for m in tf:
      if not (m.isfile() and m.name.endswith('.gz')):
        continue
      yield m.name, gzip.open(io.BytesIO(tf.extractfile(m).read())).read()
      n += 1
      if n >= limit:
        return


def real(path, cell_kind=None, limit=200):
  ngames = nrec = 0
  for name, raw in _games_from(path, limit):
    assert len(raw) % V8_RECORD_BYTES == 0, (name, len(raw))
    recs = np.frombuffer(raw, dtype=V8_DTYPE, count=len(raw) // V8_RECORD_BYTES)
    assert (recs['version'] == 8).all(), name
    assert (recs['v8_layout_version'] == 2).all(), name
    assert (recs['recipe_id'] == 4).all(), name

    probs = recs['probs']
    legal = probs >= 0
    assert np.allclose(np.where(legal, probs, 0).sum(1), 1.0, atol=1e-3), name
    assert (recs['n_legal'] == legal.sum(1)).all(), name

    for r in recs:
      ns = int(r['n_stored'])
      idx, nraw = r['slot_idx'], r['slot_n_raw']
      assert (idx[ns:] == 0xFFFF).all(), (name, 'slots past n_stored not sentinel')
      assert (idx[:ns] < 1858).all(), (name, 'slot index out of 1858 space')
      assert (np.diff(nraw[:ns].astype(np.int64)) <= 0).all(), (name, 'n_raw not descending')
      assert int(nraw[:ns].sum()) == int(r['visits']) - 1, (name, 'visit sum')
      # provenance bit0 <=> class 1, per README section 4
      assert bool(int(r['provenance_flags']) & 1) == (round(float(r['z_provenance'])) == 1), name

    for f in ('us_oo', 'us_ooo', 'them_oo', 'them_ooo'):
      assert np.isin(recs[f], (0, 1)).all(), (name, f)

    # Signedness. The i16 sentinel is -32768; read as u16 it would be +32768, so these
    # assertions fail loudly if slot_q / slot_d / slot_reply_q are ever typed unsigned.
    for r in recs:
      ns = int(r['n_stored'])
      for f in ('slot_q', 'slot_d', 'slot_reply_q'):
        assert (r[f][ns:] == -32768).all(), (name, f, 'sentinel not signed')
      live = r['slot_n_raw'][:ns] > 0
      if live.any():
        assert (np.abs(r['slot_q'][:ns][live]) <= 32767).all(), (name, 'q out of range')
        d = r['slot_d'][:ns][live]
        assert ((d >= 0) & (d <= 32767)).all(), (name, 'd out of range')

    # Anchors in regions no other assertion reaches: a 4-byte shift at the recipe block
    # would read ml_slope (0.007) instead of pst (1.45).
    assert (recs['plies_until_progress'] == 0xFFFF).all(), name
    assert np.allclose(recs['recipe'][:, 0], 1.45, atol=1e-6), name
    assert (recs['reserved_v8'] == 0).all(), name
    ngames += 1
    nrec += len(recs)

  assert ngames > 0, f'no .gz games found under {path}'
  print(f'  real OK: {ngames} games, {nrec} records from {os.path.basename(path)} — '
        f'stride/sum(n_raw)==visits-1/n_legal/sentinels/provenance all hold')

  # The stride trap (README section 2): a v7 reader must NOT quietly appear to work.
  # Record 0 decodes fine either way; the proof is that record 1 does not.
  raw = next(iter(_games_from(path, 1)))[1]
  if len(raw) >= 2 * V8_RECORD_BYTES:
    bad = np.frombuffer(raw[:2 * V7_RECORD_BYTES], dtype=V7_DTYPE, count=2)
    assert bad['version'][0] == 8, 'record 0 should still look like a header'
    assert bad['version'][1] != 8, 'v7 stride on v8 data must desynchronise at record 1'
    print('  stride trap OK: a v7-stride read desynchronises at record 1 as documented')

  if cell_kind == 'std':
    # Section 5a anchor: record 0 of a std game is the start position.
    raw = next(iter(_games_from(path, 1)))[1]
    r = np.frombuffer(raw, dtype=V8_DTYPE, count=1)[0]
    planes = np.unpackbits(r['planes'].view(np.uint8)).reshape(104, 64)
    assert list(np.nonzero(planes[0])[0]) == list(range(8, 16)), 'our pawns not on rank 2'
    assert list(np.nonzero(planes[5])[0]) == [4], 'our king not on e1'
    assert list(np.nonzero(planes[11])[0]) == [60], 'their king not on e8'
    print('  startpos anchor OK: pawns 8..15, our king 4, their king 60')


def _bare():
  """A V6ChunkDataset with only the attributes _decode_chunk touches.

  Constructing the real thing needs a corpus root, a DataLoader context and a shuffle
  seed; the decode path needs seven fields. Filters are set to pass-through so the
  comparison measures decoding, not sampling.
  """
  d = object.__new__(V6ChunkDataset)
  d._version = None
  d._skipped_other_version = 0
  d._skipped_formats = 0
  d._ragged_chunks = 0
  d._zfiltered = 0
  d.max_resultq_delta = 0.0      # off: no z-integrity filtering
  d.skip_count = 1               # keep every record
  return d


def loader(path):
  """Decode a real v8 chunk through the loader, and again as v7 over the same bytes.

  The v6/v7 prefix of a v8 record is byte-identical, so every output array must match.
  This is the check that would catch a prefix regression -- the dtype assertions above
  only prove the LAYOUT, not that the decode path produces the same training tuple.
  """
  raw = next(iter(_games_from(path, 1)))[1]
  nrec = min(len(raw) // V8_RECORD_BYTES, 64)
  assert nrec >= 2, 'need at least 2 records to be meaningful'
  v8_bytes = raw[:nrec * V8_RECORD_BYTES]

  # Same records restrided to v7: keep the first 8,396 bytes and rewrite the version word.
  parts = []
  for i in range(nrec):
    rec = bytearray(v8_bytes[i * V8_RECORD_BYTES:i * V8_RECORD_BYTES + V7_RECORD_BYTES])
    rec[0:4] = (7).to_bytes(4, 'little')
    parts.append(bytes(rec))
  v7_bytes = b''.join(parts)

  def decode(data):
    d = _bare()
    recs = d._decode_chunk(data)
    assert recs is not None and len(recs) == nrec, (len(recs) if recs is not None else None)
    return d, d._records_to_arrays(recs)

  d8, out8 = decode(v8_bytes)
  d7, out7 = decode(v7_bytes)
  assert d8._version == 8 and d7._version == 7
  assert d8._ragged_chunks == 0 and d7._ragged_chunks == 0

  def flat(out):
    fields = []
    for a in out[:13]:
      fields.append(('arr', a))
    v7x = out[13]
    if v7x is not None:
      fields += [(k, getattr(v7x, k)) for k in v7x._fields]
    return fields

  f8, f7 = flat(out8), flat(out7)
  assert len(f8) == len(f7)
  ncmp = 0
  for (k8, a8), (k7, a7) in zip(f8, f7):
    assert k8 == k7
    if a8 is None or a7 is None:
      assert a8 is None and a7 is None, k8
      continue
    assert np.array_equal(np.nan_to_num(a8), np.nan_to_num(a7)), f'{k8} differs between v8 and v7 decode'
    ncmp += 1
  print(f'  loader OK: {nrec} records decoded as v8 and as v7-over-the-same-bytes, '
        f'{ncmp} output arrays identical; version pinned 8 vs 7, 0 ragged')

  # A chunk whose length is not a multiple of the stride must be refused, not truncated.
  d = _bare()
  assert d._decode_chunk(v8_bytes[:-17]) is None and d._ragged_chunks == 1
  print('  ragged tripwire OK: a short-tailed chunk is refused and counted')


def child_table(path):
  """The v8 child table as the trainer will see it, checked against the raw records.

  The masking rule is the part worth testing: a slot is usable only inside n_stored AND
  visited (n_raw > 0). q carries the -32768 sentinel wherever n_raw == 0, so a consumer
  that trusted n_stored alone would train on -1.0 values for moves the search never
  looked at.
  """
  raw = next(iter(_games_from(path, 1)))[1]
  nrec = min(len(raw) // V8_RECORD_BYTES, 64)
  d = _bare()
  recs = d._decode_chunk(raw[:nrec * V8_RECORD_BYTES])
  out = d._records_to_arrays(recs)
  v8x = out[14]
  assert v8x is not None, 'v8 records must produce a child table'
  ci, cq, cn = v8x.child_idx, v8x.child_q, v8x.child_n
  assert ci.shape == cq.shape == cn.shape == (len(recs), V8_SLOTS), (ci.shape, len(recs))

  live = ci >= 0
  # The mask must agree exactly with the raw record, computed independently here.
  ns = recs['n_stored'].astype(np.int64)[:, None]
  k = np.arange(V8_SLOTS, dtype=np.int64)[None, :]
  want = (k < ns) & (recs['slot_n_raw'].astype(np.int64) > 0) & (recs['slot_idx'].astype(np.int64) < 1858)
  assert np.array_equal(live, want), 'mask disagrees with the raw record'

  assert (ci[live] < 1858).all() and (ci[~live] == -1).all()
  assert (np.abs(cq[live]) <= 1.0).all(), 'q outside [-1, 1]'
  assert (cq[~live] == 0).all(), 'masked q must be zeroed, not left at the sentinel'
  assert (cn[live] > 0).all() and (cn[~live] == 0).all()
  # Dequantisation: q must match the raw i16 divided by 32767 exactly where live.
  assert np.allclose(cq[live], recs['slot_q'].astype(np.float32)[live] / 32767.0, atol=0)
  # Visit accounting survives the masking: zeroing unvisited slots removes nothing.
  assert (cn.sum(1) == recs['visits'].astype(np.int64) - 1).all(), 'visit sum broken by masking'

  # SIGN AND FRAME. Slots are sorted by n_raw descending, so slot 0 IS the most-visited
  # child -- which is exactly what best_q is defined as. They must agree. A flipped sign
  # or a frame error would pass every assertion above and silently train a ranking head
  # backwards, so this is the one check that pins the semantics rather than the layout.
  s0 = live[:, 0]
  if s0.any():
    dq = np.abs(cq[s0, 0] - np.nan_to_num(recs['best_q'])[s0])
    assert dq.max() < 2e-4, f'slot 0 q vs best_q: max |diff| {dq.max():.2e} (sign/frame?)'
    # And the negated version must NOT match, or the test would pass under a sign flip.
    dq_flip = np.abs(-cq[s0, 0] - np.nan_to_num(recs['best_q'])[s0])
    assert dq_flip.max() > 1e-3, 'negated q also matches — check cannot detect a sign flip'
    print(f'    sign/frame OK: slot 0 q == best_q to {dq.max():.1e} on {int(s0.sum())} records')

  nlive = int(live.sum())
  print(f'  child table OK: {len(recs)} records, {nlive:,} live children '
        f'({nlive / len(recs):.1f} per position, max {int(live.sum(1).max())}), '
        f'q in [{cq[live].min():+.3f}, {cq[live].max():+.3f}], mask matches raw, '
        f'sum(n_raw) preserved')


if __name__ == '__main__':
  synthetic()
  if len(sys.argv) >= 2:
    real(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else None)
    loader(sys.argv[1])
    child_table(sys.argv[1])
  print('ALL OK')
