# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Warm-start growth of the move-token decoder (2026-09-16): resuming a checkpoint that
has FEWER decoder blocks than the config (e.g. MoveTokenLayers 4 -> 6).

The existing blocks load by name (blocks.0..3); the new ones (blocks.4, 5) arrive through
the resume aux-key path as fresh keys. A fresh pre-norm block is NOT a no-op: its three
output projections (self-attn `proj`, cross-attn `xproj`, FFN `ffn_out`) are randomly
initialised, so it would inject noise into a trained policy at the switch. Zeroing those
three projections makes each new block an exact identity at step 0 (x + 0 for every
branch), so the grown net is function-identical to the checkpoint. The projections
still receive gradient from step 0 (their inputs are non-zero), so the blocks wake up
like every other zero-init mechanism here. `pm_proj` (post-move) is zero-init already.

Only whole blocks whose parameters are ALL fresh are touched; a partially-fresh block
means a shape/name mismatch and is refused.
"""
import torch


_OUT_PROJ = ('proj', 'xproj', 'ffn_out')
# Per-block parameters that are an exact step-0 no-op when they arrive fresh at a resume (so an EXISTING block may
# gain them without becoming 'partially fresh'): the post-move branch (pm_proj zero-init) and the relational bias (zero-init).
NOOP_FRESH = ('ln_pm', 'pm_q', 'pm_kv', 'pm_proj', 'pm_dk', 'pm_dv', 'rel_w')


def zero_init_fresh_decoder_blocks(model, fresh_keys):
  """fresh_keys: iterable of state-dict keys that were NOT in the checkpoint.
  Returns the sorted list of decoder block indices that were zero-initialised."""
  dec = getattr(model, 'move_tokens', None)
  if dec is None:
    return []
  fresh = set(fresh_keys)
  new_blocks = []
  for i, blk in enumerate(dec.blocks):
    prefix = f'move_tokens.blocks.{i}.'
    keys = [prefix + n for n, _ in blk.named_parameters()]
    fresh_here = [k for k in keys if k in fresh]
    if not fresh_here:
      continue
    if len(fresh_here) != len(keys):
      # An EXISTING block whose only fresh keys are the post-move branch (MoveTokenPostMove switched on at the
      # same resume): that branch is an exact no-op by itself (pm_proj is zero-init), so nothing to zero here.
      if all(k[len(prefix):].split('.')[0] in NOOP_FRESH for k in fresh_here):
        continue
      raise RuntimeError(f'decoder grow: block {i} is partially fresh ({len(fresh_here)}/{len(keys)} keys, '
                         f'e.g. {fresh_here[0]}) — name/shape mismatch, not a new block; refusing')
    new_blocks.append(i)
  if not new_blocks:
    return []
  if len(new_blocks) == len(dec.blocks):
    # No block came from the checkpoint: this is the documented warm start of a whole decoder from a
    # base net (random init by construction), NOT growth — leave it on that path.
    return []
  with torch.no_grad():
    for i in new_blocks:
      blk = dec.blocks[i]
      for name in _OUT_PROJ:
        lin = getattr(blk, name)
        if not isinstance(lin, torch.nn.Linear):
          raise RuntimeError(f'decoder grow: blocks.{i}.{name} is {type(lin).__name__}, not a plain Linear')
        lin.weight.zero_()
        if lin.bias is not None:
          lin.bias.zero_()
  return new_blocks
