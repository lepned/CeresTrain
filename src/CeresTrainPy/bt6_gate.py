"""Gate of the PyTorch BT6 forward (bt6_net.py) against the real compiled BT6 program.

    python bt6_gate.py --carrier C.pb.gz --reference reference.npz [--count 1024] [--legal legal.npz]
                       [--write-carrier OUT.pb.gz] [--save-outputs OUT.npz]

`reference.npz` holds the inputs (`masks` u64 [N, 112], `values` f32 [N, 112]) and the outputs of the real FP16
program for them (`policy` [N, 1858] logits, `wdl` [N, 3] logits, `mlh`, `err`), written by the lc0ex runner.

What is run, all on CPU in float32:
  (i)   `forward_served`: the forward straight from the carrier's served tensors;
  (ii)  `BT6Net` after `load_served()` (the fold inversion), in trained-parameter form;
  (iii) the round trip carrier -> load_served -> export_served, compared tensor by tensor with the carrier, and
        optionally written as a new carrier (`--write-carrier`) for the real program to run.
Reported per output: max and mean absolute error, correlation with the reference, policy top-1 agreement (over all
1858 entries and, with `--legal`, over the legal moves only), and the softmaxed W-L.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bt6_carrier import read_carrier, write_carrier  # noqa: E402
from bt6_net import BT6Config, BT6Net, count_parameters, forward_served, served_to_torch  # noqa: E402
from bt6_planes import expand_planes  # noqa: E402
from bt6_relations import BT6Relations  # noqa: E402


def _is_float_weight(name: str) -> bool:
  """The carrier's FP16 / FP32 tensors of the FP16 program; the int8 artifact's extras are skipped."""
  return not ("/i8/" in name or "/quant/" in name or name.endswith("_d1"))


def softmax_wl(logits: np.ndarray) -> np.ndarray:
  shifted = logits.astype(np.float64) - logits.max(-1, keepdims=True)
  p = np.exp(shifted)
  p /= p.sum(-1, keepdims=True)
  return p[:, 0] - p[:, 2]


def _line(name: str, got: np.ndarray, want: np.ndarray) -> str:
  got, want = got.astype(np.float64), want.astype(np.float64)
  error = np.abs(got - want)
  correlation = np.corrcoef(got.reshape(-1), want.reshape(-1))[0, 1] if want.std() > 0 else float("nan")
  bias = (got - want).mean()
  return (f"  {name:34s} max abs err {error.max():.5f}  mean abs err {error.mean():.6f}  mean signed err {bias:+.6f}  "
          f"corr {correlation:.8f}  (reference mean |x| {np.abs(want).mean():.4f}, std {want.std():.4f})")


def report(label: str, got: dict, reference: dict, legal: np.ndarray | None) -> None:
  count = len(got["policy"])
  print(f"{label}: {count} positions")
  policy, want = got["policy"], reference["policy"][:count]
  print(_line("policy logits (1858 entries)", policy, want))
  top = (policy.argmax(1) == want.argmax(1)).mean()
  print(f"  {'policy top-1 (1858 entries)':34s} agreement {100 * top:.2f} %  ({int(round(top * count))}/{count})")
  if legal is not None:
    mask = legal[:count]
    print(_line("policy logits (legal moves)", policy[mask], want[mask]))
    masked_got, masked_want = np.where(mask, policy, -np.inf), np.where(mask, want, -np.inf)
    has = mask.any(1)
    top_legal = (masked_got.argmax(1) == masked_want.argmax(1))[has].mean()
    # policy distribution over the legal moves: KL(reference || ours) and the largest probability error
    def distribution(x):
      x = x - x.max(1, keepdims=True)
      p = np.exp(x)
      return p / p.sum(1, keepdims=True)
    p_want, p_got = distribution(masked_want[has].astype(np.float64)), distribution(masked_got[has].astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
      kl = np.where(p_want > 0, p_want * (np.log(p_want) - np.log(np.maximum(p_got, 1e-300))), 0.0).sum(1)
    print(f"  {'policy top-1 (legal moves)':34s} agreement {100 * top_legal:.2f} %  ({int(top_legal * has.sum() + 0.5)}/{int(has.sum())}); "
          f"KL(ref || ours) mean {kl.mean():.3e} max {kl.max():.3e}; max |dp| {np.abs(p_want - p_got).max():.5f}")
  print(_line("WDL logits", got["wdl"], reference["wdl"][:count]))
  print(_line("W-L (softmax)", softmax_wl(got["wdl"]), softmax_wl(reference["wdl"][:count])))
  print(_line("moves left", got["mlh"], reference["mlh"][:count]))
  if "err" in got and "err" in reference:
    print(_line("value error", got["err"], reference["err"][:count]))


def run_forward(function, planes: np.ndarray, batch: int) -> dict:
  outputs: dict = {}
  for start in range(0, len(planes), batch):
    chunk = torch.from_numpy(planes[start:start + batch])
    with torch.no_grad():
      result = function(chunk)
    for name in ("policy", "wdl", "mlh", "err"):
      if name in result:
        outputs.setdefault(name, []).append(result[name].float().cpu().numpy())
  return {name: np.concatenate(parts) for name, parts in outputs.items()}


def family(name: str) -> str:
  return re.sub(r"/encoder\d+/", "/encoder*/", name)


def round_trip(tensors: dict, exported: dict) -> bool:
  """Compare the exported tensors with the carrier's, per tensor family."""
  rows: dict = {}
  wrong_form, missing = [], [name for name in tensors if name not in exported]
  for name, array in exported.items():
    if name not in tensors:
      wrong_form.append(f"{name}: not in the carrier")
      continue
    original = tensors[name]
    if original.dtype != array.dtype or original.shape != array.shape:
      wrong_form.append(f"{name}: {array.dtype} {array.shape} vs carrier {original.dtype} {original.shape}")
      continue
    difference = np.abs(array.astype(np.float64) - original.astype(np.float64))
    scale = np.abs(original.astype(np.float64)).max()
    row = rows.setdefault(family(name), [0, 0.0, 0.0, 0, 0, str(array.dtype)])
    row[0] += 1
    row[1] = max(row[1], float(difference.max()))
    row[2] = max(row[2], float(difference.max() / scale) if scale > 0 else 0.0)
    row[3] += int((array != original).sum())
    row[4] += array.size
  print(f"round trip carrier -> load_served -> export_served: {len(exported)} tensors exported, "
        f"{len(tensors)} float tensors in the carrier")
  print(f"  {'tensor family':46s} {'dtype':8s} {'tensors':>7s} {'max abs diff':>13s} {'max diff / max |x|':>19s} {'elements differing':>22s}")
  for name, (count, absolute, relative, differing, size, dtype) in rows.items():
    print(f"  {name:46s} {dtype:8s} {count:7d} {absolute:13.3e} {relative:19.3e} {differing:12d} / {size}")
  print(f"  name / shape / dtype mismatches: {wrong_form if wrong_form else 'none'}")
  print(f"  carrier float tensors not exported: {missing if missing else 'none'}")
  return not wrong_form


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--carrier", required=True)
  parser.add_argument("--reference", required=True)
  parser.add_argument("--count", type=int, default=1024)
  parser.add_argument("--batch", type=int, default=64)
  parser.add_argument("--legal", default=None, help=".npz with `legal` bool [N, 1858] in the reference's order")
  parser.add_argument("--skip", default="", help="comma list of: served, module, roundtrip")
  parser.add_argument("--write-carrier", default=None)
  parser.add_argument("--save-outputs", default=None)
  parser.add_argument("--gate-cap", action="store_true")
  parser.add_argument("--contraction", default="path")
  parser.add_argument("--round16", action="store_true",
                      help="also run forward_served with FP16 rounding of the stored activations (noise study)")
  parser.add_argument("--accumulate16", action="store_true",
                      help="with --round16: also emulate the FP16 accumulators of the program's GEMMs")
  arguments = parser.parse_args()
  skip = set(filter(None, arguments.skip.split(",")))
  torch.manual_seed(0)

  with np.load(arguments.reference) as archive:
    reference = {key: archive[key] for key in archive.files}
  count = min(arguments.count, len(reference["masks"]))
  planes = expand_planes(reference["masks"][:count], reference["values"][:count])
  legal = None
  if arguments.legal:
    with np.load(arguments.legal) as archive:
      legal = archive["legal"]
  started = time.perf_counter()
  tensors, net = read_carrier(arguments.carrier, keep=_is_float_weight)
  config = BT6Config.from_served(tensors, gate_cap=arguments.gate_cap, triplet_contraction=arguments.contraction)
  print(f"carrier: {len(tensors)} float tensors read in {time.perf_counter() - started:.0f} s; config {config}")
  print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads, float32; {count} positions, batch {arguments.batch}")
  saved: dict = {}

  if "served" not in skip:
    served = served_to_torch(tensors)
    relations = BT6Relations()
    started = time.perf_counter()
    got = run_forward(lambda chunk: forward_served(served, chunk, config, relations), planes, arguments.batch)
    print(f"(i) forward from served tensors: {time.perf_counter() - started:.0f} s")
    report("(i) forward_served vs the real program", got, reference, legal)
    saved.update({f"served_{name}": array for name, array in got.items()})
    if arguments.round16:
      got16 = run_forward(lambda chunk: forward_served(served, chunk, config, relations, round16=True,
                                                       accumulate16=arguments.accumulate16), planes, arguments.batch)
      report("(i-16) forward_served with FP16-rounded activations vs the real program", got16, reference, legal)
      report("(i-16') forward_served FP16-rounded vs forward_served FP32 (the size of FP16 noise)", got16, got, legal)
      saved.update({f"round16_{name}": array for name, array in got16.items()})
    del served

  if not {"module", "roundtrip"} <= skip or arguments.write_carrier:
    model = BT6Net(config).eval()
    checks = model.load_served(tensors)
    print(f"(ii) BT6Net: {count_parameters(model):,} parameters; load_served consistency checks (max abs deviation):")
    for name, value in checks.items():
      print(f"  {name:66s} {value:.3e}")
    if "module" not in skip:
      started = time.perf_counter()
      got_module = run_forward(model, planes, arguments.batch)
      print(f"(ii) module forward after load_served: {time.perf_counter() - started:.0f} s")
      report("(ii) BT6Net after load_served vs the real program", got_module, reference, legal)
      if "served" not in skip:
        report("(ii') BT6Net after load_served vs (i) forward_served", got_module, got, legal)
      saved.update({f"module_{name}": array for name, array in got_module.items()})
    exported = model.export_served()
    if "roundtrip" not in skip:
      round_trip(tensors, exported)
    if arguments.write_carrier:
      started = time.perf_counter()
      write_carrier(arguments.write_carrier, exported, template_net=net)
      print(f"wrote {arguments.write_carrier}: {len(exported)} initializers, "
            f"{os.path.getsize(arguments.write_carrier) / 2**20:.0f} MB, {time.perf_counter() - started:.0f} s")
  if arguments.save_outputs:
    np.savez(arguments.save_outputs, **saved)
  return 0


if __name__ == "__main__":
  sys.exit(main())
