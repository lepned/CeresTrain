"""BT6FusedAttention (bt6_fused_attn.py) against the reference path of BT6NativeTerms.attend, on the GPU.

  (a) one term layer, float32: output, H max and the gradients of Q, K, V, bias, e, Z0 and every reader parameter,
      with live tables (C1, C2, Wm, Wg, bg, t_edge, t_node != defaults), with and without a bias (templates)
  (b) the same under bf16 autocast (the training numerics): fused vs reference-bf16 is compared with the size of
      reference-bf16 vs reference-float32 (same numerics class)
  (c) full nets, fused on vs off, same weights: the flat Fry + BT6 smoke config and an NBT native config -- loss and
      all gradients; torch.compile of the fused net
usage (from src/CeresTrainPy, WSL): python3 test_bt6_fused_attn.py"""

import copy
import io
import os
import sys

os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import torch

from bt6_net import BT6Config, RelationContext
from bt6_native_terms import BT6NativeTerms

DEV = 'cuda'


def rel(a, b):
  return float((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-12))


def make_case(B=6, H=8, hd=32, bias=True, seed=0):
  g = torch.Generator().manual_seed(seed)
  cfg = BT6Config(d_model=H * hd, heads=H, pair_channels=16, edge_channels=16)
  assert cfg.head_dim == hd
  terms = BT6NativeTerms(cfg)
  with torch.no_grad():
    for name, p in terms.named_parameters():
      p.copy_(torch.randn(p.shape, generator=g) * (0.3 if name not in ('t_node',) else 0.2) + (1.0 if name == 't_node' else 0.0))
  terms = terms.to(DEV)
  E0 = (torch.rand(B, 17, 64, 64, generator=g) < 0.15).float()
  E = torch.cat((E0, E0.transpose(-1, -2)), dim=1).to(DEV)
  Z0 = (torch.randn(B, 16, 64, 64, generator=g) * 0.5).to(DEV).requires_grad_(True)
  e = (torch.randn(B, 16, 64, 64, generator=g) * 0.5).to(DEV).requires_grad_(True)
  Q = torch.randn(B, H, 64, hd, generator=g).to(DEV).requires_grad_(True)
  K = torch.randn(B, H, 64, hd, generator=g).to(DEV).requires_grad_(True)
  V = torch.randn(B, H, 64, hd, generator=g).to(DEV).requires_grad_(True)
  b = (torch.randn(1, H, 64, 64, generator=g) * 0.5).to(DEV).requires_grad_(True) if bias else None
  W = torch.randn(B, H, 64, hd, generator=g).to(DEV)
  return terms, E, Z0, e, Q, K, V, b, W


def run(terms, E, Z0, e, Q, K, V, b, W, fused, autocast):
  terms.fused = fused
  for t in (Z0, e, Q, K, V) + ((b,) if b is not None else ()):
    t.grad = None
  for p in terms.parameters():
    p.grad = None
  ctx = RelationContext(E, None, Z0)
  with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
    pd = torch.bfloat16 if autocast else torch.float32
    rs = ctx.read_state(e, pd)
    Qc, Kc, Vc = (Q.to(pd), K.to(pd), V.to(pd))
    out, H = terms.attend(Qc, Kc, Vc, b, ctx, e, rs)
    loss = (out.float() * W).sum()
  loss.backward()
  hmax = H.float().amax(dim=(2, 3)) if H.shape[-1] == 64 else H.float()[..., 0, 0]
  grads = {'Q': Q.grad, 'K': K.grad, 'V': V.grad, 'e': e.grad, 'Z0': Z0.grad}
  if b is not None:
    grads['bias'] = b.grad
  grads.update({n: p.grad for n, p in terms.named_parameters()})
  return out.detach().float(), hmax.detach(), {k: v.detach().clone() for k, v in grads.items()}


def test_layer(bias):
  case = make_case(bias=bias)
  o_r, h_r, g_r = run(*case, fused=False, autocast=False)
  o_f, h_f, g_f = run(*case, fused=True, autocast=False)
  worst = max([rel(o_f, o_r), rel(h_f, h_r)] + [rel(g_f[k], g_r[k]) for k in g_r])
  for k in g_r:
    assert g_f[k] is not None, k
    assert rel(g_f[k], g_r[k]) < 2e-4, (k, rel(g_f[k], g_r[k]))
  assert rel(o_f, o_r) < 2e-5 and rel(h_f, h_r) < 2e-5, (rel(o_f, o_r), rel(h_f, h_r))
  print(f'(a) float32, bias {bias}: out {rel(o_f, o_r):.1e}, H max {rel(h_f, h_r):.1e}, worst grad '
        f'{max(rel(g_f[k], g_r[k]) for k in g_r):.1e} ({max(g_r, key=lambda k: rel(g_f[k], g_r[k]))}), {len(g_r)} gradients')
  # (b) bf16 autocast
  o_b, h_b, g_b = run(*case, fused=False, autocast=True)
  o_fb, h_fb, g_fb = run(*case, fused=True, autocast=True)
  ref_err = {k: rel(g_b[k], g_r[k]) for k in g_r}
  fus_err = {k: rel(g_fb[k], g_r[k]) for k in g_r}
  print(f'(b) bf16 autocast, bias {bias}: out vs float32 reference: reference-bf16 {rel(o_b, o_r):.1e}, fused-bf16 '
        f'{rel(o_fb, o_r):.1e}; worst grad vs float32: reference-bf16 {max(ref_err.values()):.1e}, fused-bf16 '
        f'{max(fus_err.values()):.1e}')
  assert rel(o_fb, o_r) <= 2 * rel(o_b, o_r) + 1e-3
  for k in g_r:
    assert fus_err[k] <= 2 * ref_err[k] + 2e-2, (k, fus_err[k], ref_err[k])


def build_net(cfg_dir, ident, fused):
  from config import Configuration
  from ceres_net import CeresNet
  import json
  cfg = Configuration(cfg_dir, ident)
  cfg.NetDef_BT6FusedAttention = fused
  out, sys.stdout = sys.stdout, io.StringIO()
  try:
    net = CeresNet(None, cfg, policy_loss_weight=cfg.Opt_LossPolicyMultiplier, value_loss_weight=cfg.Opt_LossValueMultiplier,
                   moves_left_loss_weight=cfg.Opt_LossMLHMultiplier, unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                   value2_loss_weight=cfg.Opt_LossValue2Multiplier, q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                   value_diff_loss_weight=cfg.Opt_LossValueDMultiplier, value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                   action_loss_weight=cfg.Opt_LossActionMultiplier, uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                   action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier, q_ratio=cfg.Data_FractionQ)
  finally:
    sys.stdout = out
  return net


def test_net(cfg_dir, ident, label, B=16, compile_check=False):
  from test_bt6_hybrid import batch_for
  from losses import LossCalculator
  import numpy as np
  torch.manual_seed(1)
  ref = build_net(cfg_dir, ident, False).to(DEV).train()
  with torch.no_grad():
    for n, p in ref.named_parameters():
      if '.bt6t.' in n and not n.endswith('.We'):
        p.add_(torch.randn_like(p) * 0.2)
  fused = copy.deepcopy(ref)
  for m in fused.modules():
    if hasattr(m, 'fused') and type(m).__name__ == 'BT6NativeTerms':
      m.fused = True
  n_fused = sum(1 for m in fused.modules() if type(m).__name__ == 'BT6NativeTerms')
  raw = np.load(os.environ.get('CERES_TEST_RECORDS', '/mnt/c/Users/lepne/AppData/Local/Temp/claude/fed277be149b42da/'
                               'f7bf9136-da7e-47e9-97f1-56906f6b19dd/scratchpad/eval_set13.npz'))['squares_byte'].reshape(-1, 64, 137)
  squares = (torch.from_numpy(raw[:B]).float() / 100.0).to(DEV)
  batch = {k: v.to(DEV) for k, v in batch_for(B).items()}
  lc = LossCalculator(torch.nn.Linear(4, 4))

  def loss_of(net, model=None, autocast=False):
    model = net if model is None else model
    for p in net.parameters():
      p.grad = None
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
      outs = model(squares, None)
      (po, vo, mo, uo, v2, qdl, qdu, up, ao, _) = outs[:10]
      loss = net.compute_loss(lc, batch, po, vo, mo, uo, v2, qdl, qdu, up, None, None, None, ao, None, 0, 0, 0, False)
    loss.backward()
    return float(loss), {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in net.named_parameters()}

  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  l_r, g_r = loss_of(ref)
  l_f, g_f = loss_of(fused)
  worst, wname = 0.0, ''
  for n, g in g_r.items():
    if g is None:
      assert g_f[n] is None, n
      continue
    r = float((g_f[n] - g).abs().max() / (g.abs().max() + 1e-12))
    if r > worst:
      worst, wname = r, n
  assert abs(l_f - l_r) <= 1e-5 * max(1.0, abs(l_r)), (l_f, l_r)
  assert worst < 2e-3, (worst, wname)
  print(f'(c) {label}: {n_fused} fused term layers; float32 loss {l_r:.6f} vs {l_f:.6f}, worst gradient rel {worst:.1e} ({wname})')
  if compile_check:
    torch.backends.cuda.matmul.allow_tf32 = True
    l_e, _ = loss_of(fused, autocast=True)
    l_c, _ = loss_of(fused, model=torch.compile(fused, dynamic=False), autocast=True)
    l_c2, _ = loss_of(fused, model=torch.compile(fused, dynamic=False), autocast=True)
    assert abs(l_c - l_e) <= 2e-3 * max(1.0, abs(l_e)), (l_c, l_e)
    print(f'(c) {label}: torch.compile (bf16) loss {l_c:.5f} == eager {l_e:.5f}')


if __name__ == '__main__':
  test_layer(bias=True)
  test_layer(bias=False)
  test_net('/mnt/c/Dev/Chess/CeresTrain/configs', 'smoke_256_12_fry_tpl8qkbt6_1M', 'flat Fry + BT6 smoke', compile_check=True)
  test_net('/mnt/c/Dev/Chess/lc0ex_work/terms/configs_notripei', 'prod_384_12_nbt3d3_tpl8Ant135notripei_t80t91_300M', 'NBT 384 tpl8')
  print('test_bt6_fused_attn: ALL OK')
