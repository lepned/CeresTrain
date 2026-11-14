"""QK-clip in the NBT-BT6 hybrid's inner layers (BT6Layer.qk_clip_monitor / apply_qk_clip, armed by train.py when
QKClipTau > 0). Small hybrid net with terms in the first inner layer, shared smolgen live in the others.

  (a) the training stash equals the per-head max of the logits that enter the softmax: term layer H = L (1 + door),
      plain layer q.k / sqrt(hd) + smolgen bias; max-accumulated over two micro-batches; nothing stashed in eval
  (b) apply_qk_clip: per head gamma = min(1, tau / max); the q/k-dependent part of the logits scales by exactly
      gamma (term layer: q.k + q.Pq E + k.Pk E; C1, read and door stay), the rest of the layer is untouched; heads
      under tau unchanged; stash reset; returns the number of heads clipped
  (c) a training step with the clip armed stays finite

  python test_bt6_qk_clip.py
"""
import torch

from test_bt6_hybrid import batch_for, build, randomise, records, run_loss
import bt6_net

BASE = {'NumLayers': 2, 'NBTInnerLayers': 3, 'BT6EdgeSites': [0], 'BT6TermsInnerLayers': [0],
        'BT6SharedSmolgen': True, 'BT6SharedSmolgenDim': 48, 'BT6SmolgenPerSquareDim': 8, 'BT6SmolgenHeadDim': 16}


def layers(net):
  return [(n, m) for n, m in net.named_modules() if isinstance(m, bt6_net.BT6Layer)]


def capture(net, sq):
  """Forward (no grad) recording per BT6 layer: term layer L and door (the attention_core inputs), plain layer
  the independent reference logits q.k / sqrt(hd) + bias from the layer's input."""
  rec, cur, handles = {}, [None], []
  orig = bt6_net.attention_core

  def core(L, door, *a, **k):
    rec[cur[0]] = ('term', L.detach().float().clone(), door.detach().float().clone())
    return orig(L, door, *a, **k)

  def pre(mod, args, name):
    cur[0] = name
    if not mod.terms:
      x, bias = args[0], (args[8] if len(args) > 8 else None)
      hd, heads = mod.cfg.head_dim, mod.cfg.heads
      n = mod.ln1(x)
      q, k = (m(n).view(x.shape[0], 64, heads, hd).transpose(1, 2) for m in (mod.q, mod.k))
      H = q @ k.transpose(-1, -2) / hd ** 0.5
      rec[name] = ('plain', H.float(), None if bias is None else bias.detach().float().clone())

  for name, mod in layers(net):
    handles.append(mod.register_forward_pre_hook(lambda m, a, n=name: pre(m, a, n)))
  bt6_net.attention_core = core
  try:
    with torch.no_grad():
      net(sq, None)
  finally:
    bt6_net.attention_core = orig
    for h in handles:
      h.remove()
  return rec


def logits(entry):
  kind, a, b = entry
  if kind == 'term':
    return a * (1 + b)
  return a if b is None else a + b


def test_qk_clip():
  net = build(BASE)
  randomise(net)
  with torch.no_grad():                                   # a live smolgen bias
    net.smolgenPrepLayer.weight.normal_(0, 0.05), net.smolgenPrepLayer.bias.normal_(0, 0.05)
  sq = records(8)
  L = layers(net)
  assert sum(m.terms for _, m in L) == 2 and sum(not m.terms for _, m in L) == 4
  # ---- (a) stash
  for _, m in L:
    m.qk_clip_monitor = True
  net.eval()
  with torch.no_grad():
    net(sq, None)
  assert all(m._last_max_logit is None for _, m in L), 'eval must not stash'
  net.train()
  rec_a, rec_b = capture(net, sq[:4]), capture(net, sq[4:])
  for _, m in L:
    m._last_max_logit = None
  with torch.no_grad():
    net(sq[:4], None)
    net(sq[4:], None)
  worst = 0.0
  for name, m in L:
    want = torch.maximum(logits(rec_a[name]).amax(dim=(0, 2, 3)), logits(rec_b[name]).amax(dim=(0, 2, 3)))
    got = m._last_max_logit
    assert got is not None and got.shape == (m.cfg.heads,), name
    worst = max(worst, float((got - want).abs().max() / want.abs().max()))
  assert worst < 1e-5, worst
  # ---- (b) clip exactness, one layer at a time (a clip changes every later layer's input, so only the clipped
  #      layer's own logits are compared; all earlier layers and the layer's inputs are unchanged)
  rec0 = capture(net, sq)
  saved = {k: v.clone() for k, v in net.state_dict().items()}
  touched = ('.q.weight', '.q.bias', '.k.weight', '.k.bias', '.Pq', '.Pk')
  for _, m in L:
    m._last_max_logit = None
  with torch.no_grad():
    net(sq, None)
  maxes = {name: m._last_max_logit.clone() for name, m in L}
  for _, m in L:
    m._last_max_logit = None
  worst, clipped_total = 0.0, 0
  for name, m in L:
    # static part of a term layer's L: the same forward with this layer's q / k / Pq / Pk zero
    if m.terms:
      with torch.no_grad():
        for p in (m.q.weight, m.q.bias, m.k.weight, m.k.bias, m.Pq, m.Pk):
          p.zero_()
      static = capture(net, sq)[name][1]
      net.load_state_dict(saved)
    tau = float(maxes[name].median())                       # some heads of this layer clip, some do not
    m._last_max_logit = maxes[name].clone()
    clipped = m.apply_qk_clip(tau)
    expected = int((maxes[name] > tau * 1.0001).sum())
    assert clipped == expected and 0 < clipped < m.cfg.heads, (name, clipped, expected)
    assert m._last_max_logit is None
    clipped_total += clipped
    after = net.state_dict()
    for k, v in saved.items():
      if not (k.startswith(name + '.') and k.endswith(touched)):
        assert torch.equal(after[k], v), f'{k} changed by the clip of {name}'
    rec1 = capture(net, sq)
    gamma = (tau / maxes[name]).clamp(max=1.0).view(1, -1, 1, 1)
    if m.terms:
      assert torch.allclose(rec1[name][2], rec0[name][2], atol=0, rtol=0), 'door changed'
      want = gamma * (rec0[name][1] - static) + static
    else:
      want = gamma * rec0[name][1]                          # q.k part only (the smolgen bias is added outside)
    worst = max(worst, float((rec1[name][1] - want).abs().max() / want.abs().max()))
    net.load_state_dict(saved)
  assert worst < 1e-5, worst
  tau = float(torch.cat(list(maxes.values())).median())
  clipped = clipped_total
  # ---- (c) one armed training step stays finite
  net.train()
  opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
  for _ in range(2):
    opt.zero_grad()
    loss = run_loss(net, batch_for(8), sq)
    loss.backward()
    opt.step()
    n = sum(m.apply_qk_clip(tau) for _, m in L)
    assert torch.isfinite(loss)
  print(f'test_qk_clip: OK  (stash == max logit; layer by layer {clipped} of {sum(m.cfg.heads for _, m in L)} heads '
        f'clipped, q/k part scaled by gamma to rel {worst:.1e}, static part / door / other params unchanged; '
        f'armed training steps finite)')


if __name__ == '__main__':
  test_qk_clip()
