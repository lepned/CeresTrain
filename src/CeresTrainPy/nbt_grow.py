# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

"""NBT trunk grow on resume (config TrunkGrowInsertAfter, 2026-09-29).

Renumbers a checkpoint's trunk blocks (transformer_layer.N.*) into a deeper NBT trunk. The
inserted blocks keep their fresh init; an NBT block's up projection is zero-init
(nbt_layer.py), so each inserted block is an exact identity and the grown net computes the
same function as the checkpoint at the switch. Nothing else in the net is keyed on the block
index at run time (layerNum only drives LoRA ranges, the DIFF++ lambda plan and print-once
logs, none of which a served NBT trunk uses).
"""

from typing import Dict, List, Tuple

TRUNK_PREFIX = 'transformer_layer.'


def grow_trunk_state_dict(sd: Dict, n_new: int, insert_after: List[int]) -> Tuple[Dict, Dict[int, int], List[int]]:
  """Returns (renumbered state dict, checkpoint block -> new index, indices of the fresh blocks).

  insert_after lists CHECKPOINT block indices; a fresh block goes right after each (-1 = before
  block 0, a repeated index inserts several)."""
  ckpt_blocks = sorted({int(k[len(TRUNK_PREFIX):].split('.')[0]) for k in sd if k.startswith(TRUNK_PREFIX)})
  n_old = n_new - len(insert_after)
  if ckpt_blocks != list(range(n_old)):
    raise ValueError(f'TrunkGrowInsertAfter: checkpoint has {len(ckpt_blocks)} trunk blocks, expected {n_old} '
                     f'= NumLayers {n_new} - {len(insert_after)} inserted')
  old_to_new, fresh_idx, pos = {}, [], 0
  for j in [-1] + list(range(n_old)):
    if j >= 0:
      old_to_new[j] = pos
      pos += 1
    for _ in range(insert_after.count(j)):
      fresh_idx.append(pos)
      pos += 1
  assert pos == n_new, (pos, n_new)
  out = {}
  for k, v in sd.items():
    if k.startswith(TRUNK_PREFIX):
      b, rest = k[len(TRUNK_PREFIX):].split('.', 1)
      k = f'{TRUNK_PREFIX}{old_to_new[int(b)]}.{rest}'
    out[k] = v
  return out, old_to_new, fresh_idx
