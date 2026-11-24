"""One command: a trained 768x16 flat BT6 net (served m80 ONNX + .ts / checkpoint) -> the round-4 TensorRT plugin ONNX
(bt6_trt_plugin_r4 library), FP16 or INT8, optionally gated against PyTorch. Runs the existing bt6_trt_plugin_* scripts
unchanged, in the same order and with the same arguments as the hand-made 200M / 400M nets (trt_bt6_plugin/REPORT.md).

USAGE (Windows, from the TensorRT 11.3 venv that has onnx + numpy + tensorrt + polygraphy):
    C:/Dev/Chess/lc0ex_work/terms/trt113_venv/Scripts/python.exe make_plugin_onnx.py \
        <served_m80.onnx> <net.ts | checkpoint.pt> <out_dir> (--fp16 | --int8) [--calib DIR] [--gate] [--keep]

  --fp16    params -> bt6_trt_plugin_onnx -> onnx_r2 --site -> merge --coef --ffn -> dec_r4   => <stem>_bt6r4.onnx
  --int8    params -> bt6_trt_plugin_onnx -> onnx_r2 --site, and in parallel the plain INT8 QDQ ONNX of the served net:
            qdq_export.py (percentile 99.9999, 8 calibration batches, shape allowlist 768x2304,768x768,768x1536,1536x768,
            ONNX Runtime calibration on the CPU) -> qdq_to_fp16.py (fp32 islands); then port_qdq --exclude 1536x768 ->
            merge --coef -> glu_r3 -> i8out_r3 -> dec_r4                                          => <stem>_bt6r4_int8.onnx
  --calib DIR      TPG shard folder for the INT8 calibration, Windows or WSL path (default D:/t91_skip1_v2_surv)
  --run-config DIR / --run-id ID   training config the calibration reader takes the record layout from
                   (defaults: lc0ex_work/terms/configs_notripei, prod_768_16_fry_hlg_tpl8qkbt6Ant135notripei_t80t91_4B)
  --gate    three-set gate with the plugin library loaded (TensorRT strongly typed), against PyTorch FP32 references made
            here from the .ts (WSL, CUDA: the .ts has device-bound ops; uses the GPU):
              [64]  work/squares_real_b64.bin   [x4] the same 64 as batch 256   [256] eval_set13 0..255
            limits FP16: W-L mean <= 0.0003 on all sets, policy KL mean <= 3e-6 on [256]
                   INT8: W-L mean <= 0.0025 on all sets, top-1 >= 63/64 and >= 253/256 ([x4], [256]), KL <= 3e-4 on [256]
            prints PASS / FAIL, exit code 3 on FAIL. The GPU must be free (no EngineBattle / Mycel / Ceres running).
  --keep    keep the intermediates in <out_dir>/work (default: deleted after a successful run)
  --plugin DLL    library for the gate (default C:/Dev/Chess/trt_bt6_plugin/build_r4/bt6_trt_plugin_r4.dll)

Writes <out_dir>/<stem>_bt6r4[_int8].onnx, <out_dir>/params.npz and <out_dir>/make_plugin_onnx.log (every step with its
command line, duration and the step's last output lines; gate lines). Any failing step stops the run with exit code 2.
Steps needing torch / onnxruntime run in WSL (/home/lep/cerestrain-env, qdq_to_fp16 in /home/lep/modelopt-env); the
graph steps run with this interpreter. --_ref is internal (the reference maker, run by WSL python).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WSL_PY = "/home/lep/cerestrain-env/bin/python"
WSL_MO_PY = "/home/lep/modelopt-env/bin/python"
QDQ_SCRIPTS = r"C:\Users\lepne\source\repos\CeresTrain\scripts"
PLUGIN_ROOT = r"C:\Dev\Chess\trt_bt6_plugin"
DEF_PLUGIN = PLUGIN_ROOT + r"\build_r4\bt6_trt_plugin_r4.dll"
DEF_CALIB = r"D:\t91_skip1_v2_surv"
DEF_RUN_CONFIG = r"C:\Dev\Chess\lc0ex_work\terms\configs_notripei"
DEF_RUN_ID = "prod_768_16_fry_hlg_tpl8qkbt6Ant135notripei_t80t91_4B"
SHAPES = "768x2304,768x768,768x1536,1536x768"
SQ64 = PLUGIN_ROOT + r"\work\squares_real_b64.bin"
POS256 = PLUGIN_ROOT + r"\work\r3\ref_h768_200M_lc0ex.npz"   # carries eval_set13 0..255 (squares_byte + signedness)
GATE = PLUGIN_ROOT + r"\tools\gate_r3.py"
TRT_BIN = r"C:\Dev\TensorRT-11.3.0.99\bin"


def wsl_path(p: str) -> str:
  p = os.path.abspath(p).replace("\\", "/")
  m = re.match(r"^([A-Za-z]):/(.*)$", p)
  return f"/mnt/{m.group(1).lower()}/{m.group(2)}" if m else p


class Run:
  def __init__(self, log_path):
    self.log = open(log_path, "w", encoding="utf-8")

  def say(self, s):
    print(s, flush=True); self.log.write(s + "\n"); self.log.flush()

  def step(self, name, cmd, wsl=False, cwd=None, tail=4):
    """Runs one step; stops the whole run with exit code 2 when it fails."""
    if wsl:
      inner = (f"cd {wsl_path(cwd)} && " if cwd else "") + " ".join(cmd)
      full = ["wsl", "-e", "bash", "-c", inner]
      shown = "wsl: " + inner
    else:
      full = cmd; shown = subprocess.list2cmdline(cmd)
    self.say(f"== {time.strftime('%H:%M:%S')} {name}\n   $ {shown}")
    env = dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    p = subprocess.run(full, cwd=None if wsl else cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    out = [l for l in (p.stdout + p.stderr).splitlines() if l.strip() and not re.search(r"Warning|warn\(|vulnerable", l)]
    for l in out[-tail:]:
      self.say("   | " + l[:400])
    self.say(f"   ({time.time() - t0:.0f} s, exit {p.returncode})")
    if p.returncode != 0:
      self.say(f"FAILED: {name}")
      for l in out[-40:]:
        self.log.write("   ! " + l + "\n")
      sys.exit(2)
    return out


def make_ref(ts, positions, out):
  """Internal (WSL torch): PyTorch FP32 reference from the .ts. positions = squares .bin (FP16 [N,64,137]) or an npz
  with squares_byte (signed / 100 unless 'unsigned'); FP16-rounded inputs exactly as gate_r3.py feeds TensorRT."""
  import numpy as np
  import torch
  if positions.endswith(".npz"):
    P = dict(np.load(positions))
    b8 = P["squares_byte"][:256]
    uns = bool(P["unsigned"]) if "unsigned" in P else False
    x = ((b8 if uns else b8.view(np.int8)).astype(np.float32) / 100.0).astype(np.float16)
    extra = dict(squares_byte=b8, unsigned=np.array(uns))
  else:
    x = np.fromfile(positions, dtype=np.float16); extra = {}
  x = x.astype(np.float32).reshape(-1, 64, 137)
  m = torch.jit.load(ts, map_location="cuda").float().eval()
  ys = []
  with torch.no_grad():
    for i in range(0, x.shape[0], 32):
      xb = torch.from_numpy(x[i:i + 32]).cuda()
      o = [t.float().cpu().numpy() for t in m(xb, torch.zeros(xb.shape[0], 1, device="cuda"))]
      ys.append((o[0], o[1], o[4]))
  pol, val, val2 = (np.concatenate([y[j] for y in ys]) for j in range(3))
  np.savez(out, policy=pol, value=val, value2=val2, **extra)
  print(f"reference {pol.shape[0]} positions -> {out}")


def parse_gate(line):
  g = {}
  m = re.search(r"value W-L mean ([\d.]+) max ([\d.]+);\s+value2 W-L mean ([\d.]+)", line)
  if m:
    g["wl"] = max(float(m.group(1)), float(m.group(3))); g["wlmax"] = float(m.group(2))
  m = re.search(r"top-1 (\d+)/(\d+)", line)
  if m:
    g["top"], g["n"] = int(m.group(1)), int(m.group(2))
  m = re.search(r"KL mean ([\d.e+-]+)", line)
  if m:
    g["kl"] = float(m.group(1))
  g["finite"] = "finite True" in line
  return g


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("onnx", nargs="?"); ap.add_argument("net", nargs="?"); ap.add_argument("out_dir", nargs="?")
  mode = ap.add_mutually_exclusive_group()
  mode.add_argument("--fp16", action="store_true"); mode.add_argument("--int8", action="store_true")
  ap.add_argument("--calib", default=DEF_CALIB); ap.add_argument("--run-config", default=DEF_RUN_CONFIG)
  ap.add_argument("--run-id", default=DEF_RUN_ID)
  ap.add_argument("--gate", action="store_true"); ap.add_argument("--keep", action="store_true")
  ap.add_argument("--plugin", default=DEF_PLUGIN)
  ap.add_argument("--_ref", nargs=3, metavar=("TS", "POSITIONS", "OUT"), help=argparse.SUPPRESS)
  a = ap.parse_args(argv)
  if a._ref:
    make_ref(*a._ref); return 0
  if not (a.onnx and a.net and a.out_dir) or not (a.fp16 or a.int8):
    ap.error("need <served_m80.onnx> <net.ts|ckpt> <out_dir> and --fp16 or --int8")
  for f in (a.onnx, a.net):
    if not os.path.isfile(f):
      ap.error(f"missing input {f}")
  if a.gate and not a.net.endswith(".ts"):
    ap.error("--gate makes its references from the .ts; give the .ts as <net>")

  out = os.path.abspath(a.out_dir); wk = os.path.join(out, "work"); os.makedirs(wk, exist_ok=True)
  stem = os.path.splitext(os.path.basename(a.onnx))[0]
  final = os.path.join(out, f"{stem}_bt6r4{'_int8' if a.int8 else ''}.onnx")
  R = Run(os.path.join(out, "make_plugin_onnx.log"))
  R.say(f"make_plugin_onnx {'INT8' if a.int8 else 'FP16'}  {time.strftime('%Y-%m-%d %H:%M')}\n  onnx {a.onnx}\n  net  {a.net}\n"
        f"  out  {final}")
  py = sys.executable
  S = lambda n: os.path.join(HERE, n)
  W = lambda n: os.path.join(wk, n)
  params = os.path.join(out, "params.npz")

  R.step("params (bt6_trt_plugin_params.py, WSL torch)",
         [WSL_PY, wsl_path(S("bt6_trt_plugin_params.py")), wsl_path(a.net), wsl_path(params)], wsl=True, cwd=HERE)
  R.step("round 1 (bt6_trt_plugin_onnx.py)", [py, S("bt6_trt_plugin_onnx.py"), a.onnx, params, W("r1_bt6p.onnx")], cwd=HERE)
  R.step("round 2 (bt6_trt_plugin_onnx_r2.py --site)",
         [py, S("bt6_trt_plugin_onnx_r2.py"), W("r1_bt6p.onnx"), params, W("r2_site.onnx"), "--site"], cwd=HERE)
  if a.fp16:
    R.step("merge --coef --ffn", [py, S("bt6_trt_plugin_merge.py"), W("r2_site.onnx"), W("r2_sitem.onnx"), "--coef",
                                  "--ffn"], cwd=HERE)
    R.step("decoder plugins (bt6_trt_plugin_dec_r4.py)", [py, S("bt6_trt_plugin_dec_r4.py"), W("r2_sitem.onnx"), final],
           cwd=HERE)
  else:
    q = W(os.path.basename(a.onnx)); shutil.copyfile(a.onnx, q)
    qn = os.path.basename(a.onnx); qs = os.path.splitext(qn)[0]
    calib = a.calib if a.calib.startswith("/") else wsl_path(a.calib)
    R.step("qdq_export (WSL, ONNX Runtime calibration on the CPU, percentile 99.9999, shape allowlist)",
           [WSL_PY, wsl_path(os.path.join(QDQ_SCRIPTS, "qdq_export.py")), qn, calib, "--run_config",
            wsl_path(a.run_config), "--run_id", a.run_id, "--precision", "int8", "--method", "percentile", "--percentile",
            "99.9999", "--calib_batches", "8", "--no_verify", "--out", f"{qs}.qdq.fp16io.onnx", "--include_shapes",
            SHAPES], wsl=True, cwd=wk, tail=6)
    R.step("qdq_to_fp16 (WSL modelopt-env, fp32 islands)",
           [WSL_MO_PY, wsl_path(os.path.join(QDQ_SCRIPTS, "qdq_to_fp16.py")), f"{qs}.qdq.fp16io.onnx",
            f"{qs}.int8f16isl.onnx"], wsl=True, cwd=wk, tail=3)
    R.step("port_qdq --exclude 1536x768", [py, S("bt6_trt_plugin_port_qdq.py"), W("r2_site.onnx"),
                                           W(f"{qs}.int8f16isl.onnx"), W("r2_site_int8nd.onnx"), "--exclude", "1536x768"],
           cwd=HERE)
    R.step("merge --coef", [py, S("bt6_trt_plugin_merge.py"), W("r2_site_int8nd.onnx"), W("i8nd_coefonly.onnx"), "--coef"],
           cwd=HERE)
    R.step("BT6GluQuant (bt6_trt_plugin_glu_r3.py)", [py, S("bt6_trt_plugin_glu_r3.py"), W("i8nd_coefonly.onnx"),
                                                      W("i8g_glu.onnx")], cwd=HERE)
    R.step("INT8 attention output (bt6_trt_plugin_i8out_r3.py)", [py, S("bt6_trt_plugin_i8out_r3.py"), W("i8g_glu.onnx"),
                                                                  W("r3_int8.onnx")], cwd=HERE)
    R.step("decoder plugins (bt6_trt_plugin_dec_r4.py)", [py, S("bt6_trt_plugin_dec_r4.py"), W("r3_int8.onnx"), final],
           cwd=HERE)
  R.say(f"== final {final} ({os.path.getsize(final) / 1e6:.1f} MB)")

  rc = 0
  if a.gate:
    ref64, ref256 = W("ref64.npz"), W("ref256.npz")
    me = wsl_path(os.path.abspath(__file__))
    R.step("reference [64] (PyTorch .ts, CUDA)", [WSL_PY, me, "--_ref", wsl_path(a.net), wsl_path(SQ64), wsl_path(ref64)],
           wsl=True, cwd=HERE)
    R.step("reference [256] (PyTorch .ts, CUDA)", [WSL_PY, me, "--_ref", wsl_path(a.net), wsl_path(POS256),
                                                   wsl_path(ref256)], wsl=True, cwd=HERE)
    os.environ["PATH"] = TRT_BIN + os.pathsep + os.environ.get("PATH", "")
    sets = {"64": [SQ64, "64", ref64], "x4": [SQ64, "64", ref64, "--tile", "4"], "256": [ref256, "256", ref256, "--batch", "64"]}
    res = {}
    for k, args in sets.items():
      o = R.step(f"gate [{k}]", [py, GATE, final, a.plugin] + args, cwd=HERE, tail=1)
      res[k] = parse_gate(o[-1])
    lim_wl = 0.0025 if a.int8 else 0.0003
    lim_kl = 3e-4 if a.int8 else 3e-6
    fails = []
    for k, g in res.items():
      if not g.get("finite") or "wl" not in g:
        fails.append(f"[{k}] unparsed / not finite"); continue
      if g["wl"] > lim_wl:
        fails.append(f"[{k}] W-L mean {g['wl']} > {lim_wl}")
      if a.int8 and g["top"] < (63 if g["n"] == 64 else 253):
        fails.append(f"[{k}] top-1 {g['top']}/{g['n']}")
    if res["256"].get("kl", 1) > lim_kl:
      fails.append(f"[256] KL mean {res['256'].get('kl')} > {lim_kl}")
    R.say(f"== GATE {'INT8' if a.int8 else 'FP16'}: {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    rc = 0 if not fails else 3
  if not a.keep and rc == 0:
    shutil.rmtree(wk, ignore_errors=True)
    R.say("== intermediates deleted (--keep keeps them)")
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
