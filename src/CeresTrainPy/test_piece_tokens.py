# Piece tokens (piece_tokens.py) and NBT Mixture-of-Depths (nbt_layer.py, NBTMoDCapacity).
#   python3 test_piece_tokens.py                       block-level tests
#   python3 test_piece_tokens.py <CFG_DIR> <CFG_ID>    + real CeresNets built from an NBT + shared-smolgen config
import sys

import torch

from encoder_layer import EncoderLayer
from nbt_layer import NestedBottleneckLayer, SharedSmolgen, gather_pair_bias, gather_tokens
from piece_tokens import select_piece_tokens, PieceTokenDecompress


def _make(hidden, i, heads=4):
  return EncoderLayer('T', 64, 64, 4, hidden, 2 * hidden, True, 100, False, heads,
                      ffn_activation_type='SwiGLU', norm_type='RMSNorm', layerNum=i, pre_norm=True)


def _block(seed=0, mod_capacity=0, smolgen=True):
  torch.manual_seed(seed)
  sm = None
  if smolgen:
    prep = torch.nn.Linear(128, 64 * 64)
    sm = SharedSmolgen(128, 4, 32, 256, 128, prep, 'Swish', 'RMSNorm', 1e-6)
  blk = NestedBottleneckLayer(128, 64, lambda j: _make(64, j), 2, 'RMSNorm', 1e-6, proj_activation='Swish',
                              shared_smolgen=sm, mod_capacity=mod_capacity)
  with torch.no_grad():
    blk.up.weight.normal_(0, 0.05)
  return blk.eval()


def _squares(B, seed=0):
  # [B, 64, 137]: current board (0..12) and previous board (13..25) one-hot with 0 = empty, rest noise.
  g = torch.Generator().manual_seed(seed)
  sq = torch.rand(B, 64, 137, generator=g)
  for b0 in (0, 13):
    sq[:, :, b0:b0 + 13] = 0
    kind = torch.randint(1, 13, (B, 64), generator=g)
    kind[torch.rand(B, 64, generator=g) < 0.6] = 0
    sq[:, :, b0:b0 + 13].scatter_(2, kind.unsqueeze(-1), 1.0)
  # keep the current board at <= 32 pieces
  occ = 1 - sq[:, :, 0]
  for b in range(B):
    extra = occ[b].nonzero().flatten()[32:]
    sq[b, extra, 0:13] = 0
    sq[b, extra, 0] = 1
  return sq


def test_gather_pair_bias():
  bias = torch.randn(2, 3, 64, 64)
  idx = torch.stack([torch.randperm(64)[:10] for _ in range(2)])
  ref = torch.stack([bias[b][:, idx[b]][:, :, idx[b]] for b in range(2)])
  assert torch.equal(gather_pair_bias(bias, idx), ref)


def test_select_piece_tokens():
  sq = _squares(6)
  idx = select_piece_tokens(sq, 32)
  assert idx.shape == (6, 32)
  for b in range(6):
    sel = set(idx[b].tolist())
    assert len(sel) == 32, 'indices must be distinct'
    now = set((1 - sq[b, :, 0]).nonzero().flatten().tolist())
    prev = set((1 - sq[b, :, 13]).nonzero().flatten().tolist()) - now
    assert now <= sel, 'every piece must get a token'
    n_prev = min(len(prev), 32 - len(now))
    assert len(prev & sel) == n_prev, 'free slots go to previous-board pieces first'
  # fp16 input selects the same squares
  assert torch.equal(select_piece_tokens(sq.half(), 32).sort(1).values, idx.sort(1).values)


def test_block_token_equivariance():
  # A block on all 64 squares in permuted order (token_idx = the permutation) == the plain block, permuted:
  # inner attention and the scattered/gathered shared smolgen are permutation-equivariant.
  blk = _block()
  x = torch.randn(3, 64, 128)
  perm = torch.stack([torch.randperm(64) for _ in range(3)])
  with torch.no_grad():
    ref = gather_tokens(blk(x), perm)
    out = blk(gather_tokens(x, perm), token_idx=perm)
  assert torch.allclose(out, ref, atol=1e-5), float((out - ref).abs().max())
  # K = 32 tokens runs and the smolgen bias it uses is the 64-square bias of the scattered state, gathered.
  idx = perm[:, :32]
  with torch.no_grad():
    y = blk(gather_tokens(x, idx), token_idx=idx)
  assert y.shape == (3, 32, 128) and torch.isfinite(y).all()


def test_mod_block():
  blk = _block(mod_capacity=16)
  x = torch.randn(3, 64, 128)
  with torch.no_grad():
    y = blk(x)
    h0 = blk.norm_in(x)
    score = blk.mod_router(h0).squeeze(-1)
    idx = torch.topk(score, 16, dim=1).indices
  changed = (y != x).any(-1)
  routed = torch.zeros(3, 64, dtype=torch.bool).scatter_(1, idx, True)
  assert torch.equal(changed, routed), 'exactly the routed squares change, the rest is identity'
  # routed squares == the unrouted block on those squares (smolgen from all 64, gathered), times the gate
  ref_blk = _block(mod_capacity=0)
  ref_blk.load_state_dict({k: v for k, v in blk.state_dict().items() if not k.startswith('mod_router')})
  with torch.no_grad():
    sb = gather_pair_bias(ref_blk.shared_smolgen(h0), idx)
    h = ref_blk.down(torch.nn.functional.silu(gather_tokens(h0, idx)))
    for l in ref_blk.inner:
      h = l(h, piece_relation_bias=sb)
    out = ref_blk.up(torch.nn.functional.silu(ref_blk.norm_out(h)))
    gate = 2 * torch.sigmoid(torch.gather(score, 1, idx)).unsqueeze(-1)
    ref = gather_tokens(x, idx) + gate * out
  assert torch.allclose(gather_tokens(y, idx), ref, atol=1e-5)
  # the router trains through the gate
  blk.train(); blk(x).square().sum().backward()
  assert float(blk.mod_router.weight.grad.abs().sum()) > 0


def test_decompress_init():
  dec = PieceTokenDecompress(128, 64, 4, 'RMSNorm', 1e-6)
  flow64, tokens = torch.randn(2, 64, 128), torch.randn(2, 32, 128)
  idx = torch.stack([torch.randperm(64)[:32] for _ in range(2)])
  out = dec(flow64, tokens, idx)
  mask = torch.zeros(2, 64, dtype=torch.bool).scatter_(1, idx, True)
  assert torch.equal(gather_tokens(out, idx), tokens), "token squares take their trunk output"
  assert torch.equal(out[~mask], flow64[~mask]), 'zero-init: non-token squares keep their post-embedding state'
  with torch.no_grad():
    dec.out.weight.normal_(0, 0.05)
  out = dec(flow64, tokens, idx)
  out.square().sum().backward()
  assert float(dec.q.weight.grad.abs().sum()) > 0 and float(dec.kv.weight.grad.abs().sum()) > 0


def _real(cfg_dir, cfg_id):
  import json, os, shutil
  from config_bootstrap import bootstrap_env_from_config
  bootstrap_env_from_config(os.path.dirname(cfg_dir), cfg_id, quiet=True, configs_dir=cfg_dir)   # input width etc.
  from config import Configuration
  from ceres_net import CeresNet
  tag = 'zz_piece_tokens'
  base = json.load(open(f'{cfg_dir}/{cfg_id}_ceres_net.json'))
  # small shape, same feature set (NBT /3 + shared smolgen + move tokens)
  base.update(ModelDim=384, NumLayers=4, NumHeads=4, NBTInnerHeads=4, NBTInnerLayers=2, NBTWidthDivisor=3, EGTEdgeStream=False)
  def build(**over):
    net = dict(base); net.update(over)
    json.dump(net, open(f'{cfg_dir}/{tag}_ceres_net.json', 'w'), indent=1)
    cfg = Configuration(cfg_dir, tag)
    torch.manual_seed(0)
    return CeresNet(None, cfg, policy_loss_weight=1, value_loss_weight=1, moves_left_loss_weight=0, unc_loss_weight=0,
                    value2_loss_weight=0, q_deviation_loss_weight=0, value_diff_loss_weight=0, value2_diff_loss_weight=0,
                    action_loss_weight=0, uncertainty_policy_weight=0, action_uncertainty_loss_weight=0, q_ratio=1).eval()
  try:
    for p in ('data', 'exec', 'monitoring', 'opt'):
      shutil.copy(f'{cfg_dir}/{cfg_id}_ceres_{p}.json', f'{cfg_dir}/{tag}_ceres_{p}.json')
    sq = _squares(3)
    with torch.no_grad():
      base_net = build()
      ref = base_net(sq, None)
      # Conversion: a trained 64-square net's weights load into a piece-token net (only pt_decompress is new); with
      # PieceTokens 64 the function is identical (permutation equivariance + zero-init decompression), also hybrid.
      for over in (dict(PieceTokens=64), dict(PieceTokens=64, PieceTokensFromBlock=2),
                   dict(PieceTokens=64, PieceTokensFromBlock=2, PieceTokenSmolgenFill='stale'),
                   dict(PieceTokens=64, PieceTokensToBlock=2), dict(PieceTokens=64, PieceTokensFromBlock=1, PieceTokensToBlock=3)):
        net = build(**over)
        missing, unexpected = net.load_state_dict(base_net.state_dict(), strict=False)
        assert not unexpected and missing and all(k.startswith('pt_decompress.') for k in missing), (missing, unexpected)
        out = net(sq, None)
        for a, r in zip(out, ref):
          if isinstance(r, torch.Tensor):
            assert torch.allclose(a, r, atol=1e-4, rtol=1e-4), (over, float((a - r).abs().max()))
        print(f'  conversion OK: {over} with the base weights == base net')
      for name, over in (('PieceTokens 32', dict(PieceTokens=32)), ('MoD 32 all blocks', dict(NBTMoDCapacity=32)),
                         ('MoD 32 blocks 2,3', dict(NBTMoDCapacity=32, NBTMoDBlocks=[2, 3])),
                         ('PieceTokens 32 from block 2', dict(PieceTokens=32, PieceTokensFromBlock=2)),
                         ('PieceTokens 32 stale smolgen fill', dict(PieceTokens=32, PieceTokenSmolgenFill='stale')),
                         ('PieceTokens 32 to block 2 (tail on 64 squares)', dict(PieceTokens=32, PieceTokensToBlock=2)),
                         ('PieceTokens 32 blocks [1, 3) + registers 4 + value read',
                          dict(PieceTokens=32, PieceTokensFromBlock=1, PieceTokensToBlock=3, RegisterTokens=4, RegisterTokensToValue=True)),
                         ('Registers 8', dict(RegisterTokens=8)),
                         ('Registers 8 + value read', dict(RegisterTokens=8, RegisterTokensToValue=True)),
                         ('PieceTokens 32 from block 2 + registers 4 + value read',
                          dict(PieceTokens=32, PieceTokensFromBlock=2, RegisterTokens=4, RegisterTokensToValue=True))):
        net = build(**over)
        out = net(sq, None)
        assert len(out) == len(ref)
        for a, r in zip(out, ref):
          if isinstance(r, torch.Tensor):
            assert a.shape == r.shape and torch.isfinite(a).all(), name
        n = sum(p.numel() for p in net.parameters())
        print(f'  real net OK: {name}, {n:,} params, outputs match the base net in shape')
      # registers are built in a forked RNG: every other parameter initializes exactly as in the base net
      reg_net = build(RegisterTokens=8, RegisterTokensToValue=True)
      bsd = base_net.state_dict()
      extra = [k for k in reg_net.state_dict() if k not in bsd]
      assert sorted(extra) == ['register_tokens', 'register_value_inject.weight'], extra
      for k, v in bsd.items():
        assert torch.equal(v, reg_net.state_dict()[k]), f'init differs with registers: {k}'
      print('  registers: other parameters init bit-identical to the base net')
    # gradients reach the registers through attention (training mode, depth probes on if the config has them); the
    # zero-init NBT up projections make that gradient exactly 0 at step 0, so give them (and the value read) weights first
    with torch.no_grad():
      for n, p in reg_net.named_parameters():
        if n.endswith('.up.weight') or n.startswith('register_value_inject.'):
          p.normal_(0, 0.02)
    reg_net.train()
    out = reg_net(sq, None)
    sum(o.float().square().mean() for o in out if isinstance(o, torch.Tensor) and o.is_floating_point()).backward()
    g = reg_net.register_tokens.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, 'no gradient into register_tokens'
    print(f'  registers: gradient into register_tokens |g|={float(g.abs().mean()):.2e}')
    # tail hybrid: step-0 input of the first 64-square block = the K token states at their squares, the switch state
    # elsewhere; gradients reach the decompression and the tail blocks
    th = build(PieceTokens=32, PieceTokensToBlock=2)
    seen = {}
    th.transformer_layer[1].register_forward_hook(lambda m, a, o: seen.__setitem__('tok', o))
    th.transformer_layer[2].register_forward_pre_hook(lambda m, a: seen.__setitem__('in', a[0]))
    with torch.no_grad():
      th(sq, None)
    from piece_tokens import select_piece_tokens
    from nbt_layer import gather_tokens
    idx = select_piece_tokens(sq, 32)
    assert seen['in'].shape[1] == 64 and torch.equal(gather_tokens(seen['in'], idx), seen['tok']), 'tail input at tokens != token states'
    with torch.no_grad():
      for n, p in th.named_parameters():
        if n.endswith('.up.weight') or n == 'pt_decompress.out.weight':
          p.normal_(0, 0.02)
    th.train()
    out = th(sq, None)
    sum(o.float().square().mean() for o in out if isinstance(o, torch.Tensor) and o.is_floating_point()).backward()
    for n in ('pt_decompress.q.weight', 'transformer_layer.3.down.weight'):
      g = dict(th.named_parameters())[n].grad
      assert g is not None and float(g.abs().sum()) > 0, f'no gradient into {n}'
    print('  tail hybrid: step-0 scatter semantics + gradients into decompression and tail blocks OK')
    for over in (dict(PieceTokens=16), dict(PieceTokens=32, NBTMoDCapacity=32), dict(NBTMoDCapacity=64),
                 dict(PieceTokenDecompressDim=640), dict(NBTMoDBlocks=[1]), dict(PieceTokensFromBlock=1), dict(PieceTokenSmolgenFill='stale'), dict(PieceTokens=32, PieceTokensFromBlock=4), dict(NBTMoDCapacity=32, NBTSharedSmolgen=False),
                 dict(RegisterTokensToValue=True), dict(RegisterTokens=8, NBTMoDCapacity=32), dict(RegisterTokens=33),
                 dict(RegisterTokens=8, NBTSharedSmolgen=False),
                 dict(PieceTokensToBlock=2), dict(PieceTokens=32, PieceTokensToBlock=4),
                 dict(PieceTokens=32, PieceTokensFromBlock=2, PieceTokensToBlock=2), dict(PieceTokens=32, PieceTokensToBlock=5)):
      refused = False
      try:
        build(**over)
      except (ValueError, AssertionError):
        refused = True
      assert refused, f'must be refused: {over}'
    print('  guards OK')
  finally:
    for p in ('data', 'exec', 'monitoring', 'opt', 'net'):
      fn = f'{cfg_dir}/{tag}_ceres_{p}.json'
      if os.path.exists(fn):
        os.remove(fn)


if __name__ == '__main__':
  for name, fn in list(globals().items()):
    if name.startswith('test_') and callable(fn):
      fn()
      print(f'{name} OK')
  if len(sys.argv) == 3:
    _real(sys.argv[1], sys.argv[2])
