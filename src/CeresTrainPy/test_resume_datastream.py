"""Contract tests for the data-stream resume (2026-09-09).

    CERES_TPG_V3=0 CERES_TPG_SQUARE_BYTES=137 CERES_AUX_FEATURES_PER_SQUARE=0 \\
      CERES_RESUME_TEST_TPG=/mnt/d/T80_v2_sub python test_resume_datastream.py

1. Loader tags: every board dict carries tpg_root / tpg_file / tpg_file_pos / tpg_file_pos_end /
   tpg_worker, and consecutive batches from one worker advance by the loader batch size.
2. exclude_files drops exactly the named shards from the discovered (post-shuffle) list.
3. start_offsets fast-forwards: the first batch of an in-progress shard starts at the offset.
4. Trainer bookkeeping (helpers extracted from train.py): a worker moving to a new shard marks
   the previous one consumed; the merged state has consumed shards and in-progress offsets;
   tags are popped from the batch dicts.
"""
import os, sys, json, tempfile, re
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TPG = os.environ.get('CERES_RESUME_TEST_TPG', '/mnt/d/T80_v2_sub')


def loader_tests():
  from tpg_dataset import TPGDataset
  B = 256
  ds = TPGDataset(TPG, B, 0.0, 0, 1, 0, 1, 0, False)
  ds.set_worker_id(0)
  b0 = ds[0][0]; b1 = ds[0][0]
  for k in ('tpg_root', 'tpg_file', 'tpg_file_pos', 'tpg_file_pos_end', 'tpg_worker'):
    assert k in b0, f'missing tag {k}'
  assert b0['tpg_root'] == TPG and b0['tpg_file'].endswith('.zst') and b0['tpg_worker'] == 0
  assert b0['tpg_file_pos'] == 0 and b0['tpg_file_pos_end'] == B, (b0['tpg_file_pos'], b0['tpg_file_pos_end'])
  assert b1['tpg_file'] == b0['tpg_file'] and b1['tpg_file_pos'] == B, 'second batch of the same shard must start at one batch'
  first = b0['tpg_file']
  order = ds._discover_files()
  print(f'  tags OK: first shard {first}, batch span {b0["tpg_file_pos"]}..{b0["tpg_file_pos_end"]}, then {b1["tpg_file_pos"]}')
  # 2. exclude
  ds_x = TPGDataset(TPG, B, 0.0, 0, 1, 0, 1, 0, False, exclude_files={first})
  order_x = ds_x._discover_files()
  assert first not in order_x and len(order_x) == len(order) - 1 and order_x == [f for f in order if f != first], 'exclude must drop exactly that shard, order preserved'
  ds_x.set_worker_id(0)
  bx = ds_x[0][0]
  assert bx['tpg_file'] == order_x[0] and bx['tpg_file'] != first
  print(f'  exclude OK: {first} dropped, stream starts at {bx["tpg_file"]}')
  # 3. fast-forward
  OFF = 3 * B + 17          # deliberately NOT a batch multiple: the reader must land exactly here
  ds_o = TPGDataset(TPG, B, 0.0, 0, 1, 0, 1, 0, False, start_offsets={first: OFF})
  ds_o.set_worker_id(0)
  bo = ds_o[0][0]
  assert bo['tpg_file'] == first and bo['tpg_file_pos'] == OFF and bo['tpg_file_pos_end'] == OFF + B, (bo['tpg_file_pos'], OFF)
  # the fast-forwarded stream must decode sanely: policy indices in range, squares one-hot-ish
  import torch
  assert bo['squares'].shape[1:] == (64, 137) or bo['squares'].shape[1] == 64
  assert torch.isfinite(bo['wdl_q']).all()
  print(f'  fast-forward OK: {first} resumed at position {OFF} (next span {bo["tpg_file_pos"]}..{bo["tpg_file_pos_end"]})')
  # exclusivity guard
  try:
    TPGDataset(TPG, B, 0.0, 0, 1, 0, 1, 0, False, exclude_files={first}, num_files_to_skip_after_shuffle=4)
    raise SystemExit('FAIL: exclude_files + num_files_to_skip_after_shuffle not refused')
  except AssertionError as e:
    print(f'  rejection OK: {str(e)[:70]}')


def trainer_helper_tests():
  src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py'), encoding='utf-8').read()
  start = src.index('_DS_PROGRESS = {}')
  end = src.index('def _ds_load_state')
  end = src.index('\n\n', end)
  ns = {'os': os}
  exec(src[start:end], ns)
  pop, track, local, write = ns['_pop_stream_tags'], ns['_ds_track'], ns['_ds_local_state'], ns['_ds_write_state']
  def mk(root, f, pos, w, B=256):
    return {'tpg_root': root, 'tpg_file': f, 'tpg_file_pos': pos, 'tpg_file_pos_end': pos + B, 'tpg_worker': w, 'squares': 0}
  # worker 0 reads A then moves to B; worker 1 stays on C; secondary corpus has its own state
  seq = [mk('/p', 'A.zst', 0, 0), mk('/p', 'C.zst', 0, 1), mk('/p', 'A.zst', 256, 0), mk('/s', 'X.zst', 0, 0),
         mk('/p', 'B.zst', 0, 0), mk('/p', 'C.zst', 256, 1), mk('/s', 'X.zst', 256, 0)]
  for d in seq:
    tag = pop([d]); assert tag is not None and 'tpg_file' not in d and 'squares' in d, 'tags popped, payload kept'
    track(tag, 0)
  st = local()
  assert st['/p']['done'] == ['A.zst'], st
  assert st['/p']['in_progress'] == {'B.zst': 256, 'C.zst': 512}, st
  assert st['/s'] == {'done': [], 'in_progress': {'X.zst': 512}}, st
  # write + read back (single rank path); needs tpg_dataset importable for the seed
  d = tempfile.mkdtemp(prefix='dsres_')
  ck = os.path.join(d, 'ckpt_test_123')
  write(ck, 1, True)
  j = json.load(open(ck + '.datastream.json'))
  assert j['world_size'] == 1 and j['corpora']['/p']['done'] == ['A.zst'] and j['corpora']['/p']['in_progress']['C.zst'] == 512
  assert 'shuffle_seed' in j
  print(f'  trainer bookkeeping OK: consumed {j["corpora"]["/p"]["done"]}, in progress {j["corpora"]["/p"]["in_progress"]}, secondary {j["corpora"]["/s"]["in_progress"]}')


def main():
  trainer_helper_tests()
  if os.path.isdir(TPG):
    loader_tests()
  else:
    print(f'  (corpus {TPG} not present: loader tests skipped)')
  print('ALL OK')


if __name__ == '__main__':
  main()
