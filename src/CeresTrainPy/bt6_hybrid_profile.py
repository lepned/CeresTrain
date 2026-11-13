"""Timing / memory profile of one training step of a BT6-hybrid net on ONE GPU (manual synchronised timers).

    python bt6_hybrid_profile.py --records R.npz [--batch 512] [--checkpoint 0|1] [--compile MODE] [--steps 6]
                                 [--shape 1536,768,24,10,3] [--sites 0,2,4,6,8] [--embedding bt6] [--no-components]
                                 [--net KEY=VALUE ...]

`R.npz` holds `squares` uint8 [N, 64, 137] (real records; the bytes are read unsigned / 100). The step is what
train.py does per forward batch: bf16 autocast forward, the real compute_loss, backward, and a plain AdamW step as
a stand-in for the optimiser (reported separately). Reported: ms per phase (median over the timed steps), positions
per second, peak memory, and -- unless --no-components -- the time inside the forward per component, measured with
synchronising module hooks in a separate pass (hooks cost time themselves, so shares are what matters there).
"""

import argparse
import os
import sys
import time

os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sync():
  torch.cuda.synchronize()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--records', required=True)
  parser.add_argument('--batch', type=int, default=512)
  parser.add_argument('--checkpoint', type=int, default=1)
  parser.add_argument('--compile', default='')
  parser.add_argument('--steps', type=int, default=6)
  parser.add_argument('--warmup', type=int, default=2)
  parser.add_argument('--shape', default='1536,768,24,10,3')
  parser.add_argument('--sites', default='0,2,4,6,8')
  parser.add_argument('--embedding', default='bt6')
  parser.add_argument('--no-components', action='store_true')
  parser.add_argument('--net', nargs='*', default=[])
  parser.add_argument('--soft', type=float, default=1.0)
  arguments = parser.parse_args()

  from test_bt6_hybrid import NET_TARGET, batch_for, build, run_loss
  model_dim, inner, heads, blocks, k = (int(v) for v in arguments.shape.split(','))
  over = {**NET_TARGET, 'ModelDim': model_dim, 'NumLayers': blocks, 'NumHeads': heads, 'NBTInnerLayers': k,
          'NBTInnerHeads': heads, 'NBTInnerWidth': inner, 'BT6EdgeSites': [int(s) for s in arguments.sites.split(',') if s],
          'BT6Embedding': arguments.embedding, 'BT6CheckpointBlocks': bool(arguments.checkpoint)}
  for item in arguments.net:
    key, value = item.split('=')
    over[key] = {'True': True, 'False': False}.get(value, value if not value.lstrip('-').replace('.', '').isdigit()
                                                   else (float(value) if '.' in value else int(value)))
  torch.backends.cuda.matmul.allow_tf32 = True
  net = build(over, {'SoftPolicyWeight': arguments.soft, 'LossMLHMultiplier': 0, 'LossValue2Multiplier': 0.04,
                     'LossUNCMultiplier': 0.005}).cuda().train()
  with np.load(arguments.records) as archive:
    raw = archive['squares']
  index = np.arange(arguments.batch) % len(raw)
  squares = (torch.from_numpy(raw[index]).float() / 100.0).cuda()
  batch = {key: value.cuda() for key, value in batch_for(arguments.batch).items()}
  parameters = sum(p.numel() for p in net.parameters())
  print(f'net {arguments.shape} sites {arguments.sites} embedding {arguments.embedding}: {parameters:,} parameters; batch '
        f'{arguments.batch}; checkpoint {bool(arguments.checkpoint)}; compile {arguments.compile or "off"}; '
        f'{torch.cuda.get_device_name(0)}; extra net keys {arguments.net}', flush=True)
  model = torch.compile(net, mode=arguments.compile, dynamic=False) if arguments.compile else net
  optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)

  def step(timed):
    optimizer.zero_grad(set_to_none=True)
    sync()
    t0 = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
      outs = model(squares, None)
      (policy_out, value_out, mlh_out, unc_out, value2_out, qdl, qdu, unc_pol, action_out, _) = outs[:10]
      from losses import LossCalculator
      loss = net.compute_loss(LossCalculator(torch.nn.Linear(4, 4)), batch, policy_out, value_out, mlh_out, unc_out,
                              value2_out, qdl, qdu, unc_pol, None, None, None, action_out, None, 0, 0, 0, False)
    sync()
    t1 = time.perf_counter()
    loss.backward()
    sync()
    t2 = time.perf_counter()
    optimizer.step()
    sync()
    t3 = time.perf_counter()
    if timed is not None:
      timed.append((t1 - t0, t2 - t1, t3 - t2, float(loss)))

  compile_started = time.perf_counter()
  for _ in range(arguments.warmup):
    step(None)
  sync()
  warm = time.perf_counter() - compile_started
  torch.cuda.reset_peak_memory_stats()
  timed = []
  for _ in range(arguments.steps):
    step(timed)
  forward, backward, optimise = (float(np.median([t[i] for t in timed])) * 1e3 for i in range(3))
  total = forward + backward + optimise
  peak = torch.cuda.max_memory_allocated() / 2**30
  reserved = torch.cuda.max_memory_reserved() / 2**30
  print(f'STEP batch {arguments.batch}: forward+loss {forward:.0f} ms, backward {backward:.0f} ms, optimiser(AdamW) {optimise:.0f} ms, '
        f'total {total:.0f} ms = {arguments.batch / total * 1e3:.0f} positions/s on one GPU (without optimiser '
        f'{arguments.batch / (forward + backward) * 1e3:.0f}); peak allocated {peak:.1f} GiB, reserved {reserved:.1f} GiB; '
        f'warm-up {warm:.0f} s; loss {timed[0][3]:.3f} -> {timed[-1][3]:.3f}', flush=True)
  if arguments.no_components or arguments.compile:
    return

  # ---- component times: synchronising hooks, forward under grad and the checkpoint recompute separated
  times: dict = {}
  starts: dict = {}

  def attach(module, label):
    def pre(_m, _i):
      sync()
      starts[id(_m)] = time.perf_counter()

    def post(_m, _i, _o):
      sync()
      key = (label, 'grad' if torch.is_grad_enabled() else 'no-grad')
      times[key] = times.get(key, 0.0) + time.perf_counter() - starts[id(_m)]
    return [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]

  hooks = []
  hybrid = net.bt6h
  hooks += attach(hybrid.relations, 'BT6Relations (E from the record)')
  hooks += attach(hybrid.prologue, 'prologue (S, e0)')
  if hybrid.embed is not None:
    hooks += attach(hybrid.embed, 'embedding')
  for block in net.transformer_layer:
    for layer in block.inner:
      hooks += attach(layer, 'inner layers (30)')
      for name in ('ln1', 'ln2'):
        hooks += attach(getattr(layer, name), '  inner: LayerNorm x2')
      for name in ('q', 'k', 'v', 'g'):
        hooks += attach(getattr(layer, name), '  inner: Q K V G projections')
      hooks += attach(layer.out, '  inner: out projection')
      hooks += attach(layer.ffn, '  inner: GLU FFN')
    hooks += attach(block.norm_in, 'block: norm_in / norm_out')
    hooks += attach(block.norm_out, 'block: norm_in / norm_out')
    hooks += attach(block.down, 'block: down / up projections')
    hooks += attach(block.up, 'block: down / up projections')
  for site in hybrid.sites.values():
    hooks += attach(site, 'edge sites (5)')
  hooks += attach(hybrid.policy.embedding, 'policy head: embedding + Q + K')
  hooks += attach(hybrid.policy.query, 'policy head: embedding + Q + K')
  hooks += attach(hybrid.policy.key, 'policy head: embedding + Q + K')
  hooks += attach(net.headPremap, 'value front end + heads')
  hooks += attach(net.headSharedLinear, 'value front end + heads')
  hooks += attach(net.value_head, 'value front end + heads')
  repeats = 3
  whole = []
  for _ in range(repeats):
    step(whole)
  for hook in hooks:
    hook.remove()
  forward_hooked = float(np.mean([t[0] for t in whole])) * 1e3
  print(f'COMPONENTS (hooked forward {forward_hooked:.0f} ms incl. hook synchronisation; un-hooked {forward:.0f} ms). '
        f'"no-grad" = the first pass of a checkpointed block, "grad" = with autograd (the recompute inside backward when '
        f'checkpointing is on, otherwise the forward itself)')
  labels = sorted({key[0] for key in times}, key=lambda s: (s.startswith('  '), s))
  ordered = [l for l in labels if not l.startswith('  ')]
  for label in ordered:
    row = ''.join(f'{mode:>8s} {times.get((label, mode), 0.0) / repeats * 1e3:8.1f} ms' for mode in ('no-grad', 'grad'))
    print(f'  {label:40s}{row}')
    if label.startswith('inner layers'):
      accounted = {'no-grad': 0.0, 'grad': 0.0}
      for sub in [l for l in labels if l.startswith('  ')]:
        row = ''.join(f'{mode:>8s} {times.get((sub, mode), 0.0) / repeats * 1e3:8.1f} ms' for mode in ('no-grad', 'grad'))
        for mode in accounted:
          accounted[mode] += times.get((sub, mode), 0.0)
        print(f'  {sub:40s}{row}')
      rest = ''.join(f'{mode:>8s} {(times.get((label, mode), 0.0) - accounted[mode]) / repeats * 1e3:8.1f} ms' for mode in ('no-grad', 'grad'))
      print(f'  {"  inner: attention core (pairwise part)":40s}{rest}')


if __name__ == '__main__':
  main()
