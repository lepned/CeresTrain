"""Self-consistency tests for bt6_net.py on small random nets (CPU, seconds; no carrier, no GPU).

Run from src/CeresTrainPy:

    python test_bt6_net.py

1. Fold exactness: a randomly initialised BT6Net with EVERY parameter made non-trivial (the zero-init ones too)
   computes the same outputs as `forward_served` on its own `export_served()` tensors (float32 tables; the FP16
   dense weights are compared after loading the exported values back, so only the FP32 folds are under test).
2. Round trip: export_served -> load_served into a second module -> export_served reproduces every tensor; the
   second module's forward equals the first's (t_edge / We are recovered as 1 / (t_edge * We): same function).
3. Carrier file: write_carrier / read_carrier round trip of names, shapes, dtypes and bytes, with and without a
   template Net.
4. Init contract (spec A3 / A4): at initialisation the edge read, door, gate, relation terms, output gate and the
   edge sites are neutral -- H equals q.k / sqrt(hd), the gates are exactly 1 and the sites leave rms(e) unchanged.
5. Other sizes build and run (no triplet, no reverse edge, no error head, capped gate, "ag" contraction).
6. Gradients reach every parameter (one backward of a dummy loss).
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bt6_carrier import read_carrier, write_carrier  # noqa: E402
from bt6_net import BT6Config, BT6Net, count_parameters, forward_served, served_to_torch  # noqa: E402
from bt6_planes import encode_fens, expand_planes  # noqa: E402

FENS = [
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R b KQ - 3 8",
  "5r1k/ppp1q2p/8/3BnQ2/4p1n1/2N3K1/PPP2PP1/R4R2 w - - 1 21",
  "8/P6k/8/8/8/8/7p/K7 b - - 12 60",
  "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
]
SMALL = dict(d_model=64, layers=4, heads=4, ffn_hidden=48, embedding_dense=24, embedding_ffn_hidden=80,
             edge_sites=(1, 2), site_hidden=32, policy_width=64, value_width=8, value_hidden=16, moves_width=4,
             moves_hidden=16)


def planes():
  masks, values = encode_fens(FENS)
  return torch.from_numpy(expand_planes(masks, values))


def randomise(model: BT6Net, seed: int = 1) -> None:
  """Give every parameter (including the zero-initialised ones) a non-trivial value."""
  generator = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, parameter in model.named_parameters():
      noise = torch.randn(parameter.shape, generator=generator)
      if name.endswith("t_node"):
        parameter.copy_(0.5 + noise.abs())
      elif name.endswith("mult_gate"):
        parameter.copy_(1.0 + 0.5 * noise)           # some gates negative: the export applies relu
      elif parameter.ndim == 1 and ("ln" in name or "norm" in name) and name.endswith("weight"):
        parameter.copy_(1.0 + 0.1 * noise)
      elif any(key in name for key in ("C1", "Pk", "Pq", "C2", "We", "Wm", "Wg", "bg", "t_edge", "readback", "triplet",
                                       "ffn_in", "ffn_rev", "ffn_out", "mix", ".D", ".T", "Pin", "Toff")):
        parameter.copy_(0.2 * noise)
      else:
        parameter.add_(0.05 * noise)


def max_diff(a: dict, b: dict) -> float:
  return max(float((a[key].double() - b[key].double()).abs().max()) for key in ("policy", "wdl", "mlh", "err") if key in a)


def with_fp16_weights(model: BT6Net) -> BT6Net:
  """Round the module's dense weights to what the carrier stores (FP16), so module and served forward share values."""
  clone = BT6Net(model.config)
  clone.load_served(model.export_served())
  return clone


def test_folds_and_round_trip() -> None:
  torch.manual_seed(0)
  config = BT6Config(**SMALL)
  model = BT6Net(config).eval()
  randomise(model)
  x = planes()
  exported = model.export_served()
  assert BT6Config.from_served(exported) == config, "from_served must recover the sizes"
  rounded = with_fp16_weights(model).eval()            # same function up to the FP16 rounding of the dense weights
  with torch.no_grad():
    out_module = rounded(x)
    out_served = forward_served(served_to_torch(rounded.export_served()), x, config)
  diff = max_diff(out_module, out_served)
  scale = float(out_module["policy"].abs().max())
  print(f"1. folds: module (FP16-rounded weights) vs forward_served on its export: max abs diff {diff:.3e} (policy scale {scale:.2f})")
  assert diff < 1e-4 * max(1.0, scale), diff  # what remains: the policy pair folds are served in FP16
  # the folds alone, without the FP16 rounding of the dense weights: served forward on float64-exact FP32 tables
  # generation 1 = export of the float32 module; generation 2 = export after loading generation 1. The only tensors
  # that may move are the policy pair folds: generation 1 folds the float32 u, generation 2 the FP16-rounded u
  # that the carrier carries (one FP16 unit). From generation 2 on the export is a fixed point.
  again = rounded.export_served()
  third = with_fp16_weights(rounded).export_served()
  worst, worst_name, fixed = 0.0, "", 0.0
  for name, array in exported.items():
    assert again[name].dtype == array.dtype and again[name].shape == array.shape, name
    difference = float(np.abs(again[name].astype(np.float64) - array.astype(np.float64)).max())
    if difference > worst:
      worst, worst_name = difference, name
    fixed = max(fixed, float(np.abs(third[name].astype(np.float64) - again[name].astype(np.float64)).max()))
  print(f"2. round trip export -> load -> export: {len(exported)} tensors, max abs diff {worst:.3e} (at {worst_name}); "
        f"second -> third generation {fixed:.3e}")
  assert worst < 1e-4 and (worst < 5e-6 or worst_name.startswith("/policy/pair/")), (worst, worst_name)
  assert fixed < 5e-6, fixed
  with torch.no_grad():
    drift = max_diff(rounded(x), with_fp16_weights(rounded).eval()(x))
  print(f"2. second-generation module vs first: max abs output diff {drift:.3e}")
  assert drift < 1e-3


def test_carrier_file() -> None:
  torch.manual_seed(0)
  model = BT6Net(BT6Config(**SMALL))
  randomise(model)
  exported = model.export_served()
  with tempfile.TemporaryDirectory() as folder:
    first = os.path.join(folder, "a.pb.gz")
    write_carrier(first, exported)
    tensors, net = read_carrier(first)
    assert list(tensors) == list(exported)
    for name, array in exported.items():
      assert tensors[name].dtype == array.dtype and tensors[name].shape == array.shape and np.array_equal(tensors[name], array), name
    second = os.path.join(folder, "b.pb.gz")
    write_carrier(second, exported, template_net=net)
    tensors2, net2 = read_carrier(second)
    assert net2 == net and all(np.array_equal(tensors2[name], exported[name]) for name in exported)
  print(f"3. carrier file round trip: {len(exported)} initializers, names / shapes / dtypes / bytes identical")


def test_init_contract() -> None:
  torch.manual_seed(0)
  config = BT6Config(**SMALL)
  model = BT6Net(config).eval()
  x = planes()
  with torch.no_grad():
    E = model.relations(x[:, 0:12] > 0.5, x[:, 13:25] > 0.5).float()
    S, e = model.prologue(E)
    h = model.embedding(x)
    layer = model.layers[0]
    out, H = layer(h, E, S, e, model.prologue)
    y = layer.ln1(h)
    q = layer.q(y).view(len(x), 64, config.heads, config.head_dim)
    k = layer.k(y).view(len(x), 64, config.heads, config.head_dim)
    plain = torch.einsum("bihd,bjhd->bhij", q, k) / config.head_dim ** 0.5
    assert float((H - plain).abs().max()) < 1e-5, "at init H must be the plain scaled dot product"
    attended = torch.einsum("bhij,bjhd->bihd", torch.softmax(plain, -1), layer.v(y).view(len(x), 64, config.heads, config.head_dim))
    reference = h + layer.out(attended.reshape(len(x), 64, -1))
    reference = reference + layer.ffn(layer.ln2(reference))
    assert float((out - reference).abs().max()) < 1e-4, "at init the gates must be exactly 1"
    site = model.sites[str(config.edge_sites[0])]
    assert float((site(e, H) - e).abs().max()) < 1e-4, "at init a site leaves the (already rms-normed) state unchanged"
  zero = [name for name, p in model.named_parameters() if float(p.detach().abs().max()) == 0.0]
  print(f"4. init contract holds; {len(zero)} zero-initialised parameter tensors, e.g. {sorted(set(n.split('.')[-1] for n in zero))}")


def test_other_sizes_and_gradients() -> None:
  torch.manual_seed(0)
  x = planes()
  variants = [dict(), dict(triplet=False), dict(rev_edge=False), dict(error_head=False), dict(gate_cap=True),
              dict(triplet_contraction="ag"), dict(edge_sites=()), dict(triplet_heads=2, triplet_dots=8),
              dict(pair_channels=8, edge_channels=8, triplet_heads=2, triplet_dots=4)]
  for extra in variants:
    config = BT6Config(**{**SMALL, **extra})
    model = BT6Net(config).eval()
    randomise(model)
    rounded = with_fp16_weights(model).eval()
    with torch.no_grad():
      diff = max_diff(rounded(x), forward_served(served_to_torch(model.export_served()), x, config))
    assert diff < 5e-3, (extra, diff)
  print(f"5. {len(variants)} size / option variants: module == forward_served on the export")
  model = BT6Net(BT6Config(**SMALL))
  randomise(model)
  out = model(x)
  (out["policy"].pow(2).mean() + out["wdl"].pow(2).mean() + out["mlh"].mean() + out["err"].mean()).backward()
  missing = [name for name, p in model.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
  # the last layer's edge site does not exist; a site after the LAST layer would be unread, and nothing here is
  assert not missing, f"parameters without gradient: {missing}"
  print(f"6. gradients reach all {len(list(model.parameters()))} parameter tensors ({count_parameters(model):,} parameters)")
  default = BT6Net(BT6Config())
  print(f"   default (BT6 flagship) config: {count_parameters(default):,} parameters "
        f"(prologue {count_parameters(default.prologue):,}, embedding {count_parameters(default.embedding):,}, "
        f"layer {count_parameters(default.layers[0]):,}, site {count_parameters(default.sites['3']):,}, "
        f"policy {count_parameters(default.policy):,}, value {count_parameters(default.value):,}, "
        f"error {count_parameters(default.value_error):,}, moves {count_parameters(default.moves):,})")


if __name__ == "__main__":
  test_folds_and_round_trip()
  test_carrier_file()
  test_init_contract()
  test_other_sizes_and_gradients()
  print("test_bt6_net: OK")
