"""The example configs of the Fry blocks (configs/fry_blocks) build and train a step on CPU.

  python test_fry_configs.py [config dir] [base config dir]

For each config: Configuration validates, CeresNet builds, one forward + loss + backward on 4 positions is finite;
prints the parameter count and what differs from the 384 base arm (parameter names added / removed).
"""
import io
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
sys.dont_write_bytecode = True

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CONFIG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'configs', 'fry_blocks')
BASE_DIR = sys.argv[2] if len(sys.argv) > 2 else '/mnt/c/Dev/Chess/lc0ex_work/terms/configs_notripei'
BASE_ID = 'prod_384_12_fry_hlg_tpl8qkbt6Ant135notripei_t80t91_300M'
IDS = ['prod_384_12_fry_hlg_tpl8qkbt6sqfAnt135notripei_t80t91_300M',
       'prod_384_12_fry_hlg_tpl8qkbt6gqaAnt135notripei_t80t91_300M',
       'prod_384_12_fryfaithful_t80t91_300M',
       'parity_768_28_fry117']


def build(config_dir, cid):
  from config import Configuration
  from ceres_net import CeresNet
  sink, stdout = io.StringIO(), sys.stdout
  sys.stdout = sink
  try:
    cfg = Configuration(config_dir, cid)
    torch.manual_seed(777)
    net = CeresNet(None, cfg,
                   policy_loss_weight=cfg.Opt_LossPolicyMultiplier, value_loss_weight=cfg.Opt_LossValueMultiplier,
                   moves_left_loss_weight=cfg.Opt_LossMLHMultiplier, unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                   value2_loss_weight=cfg.Opt_LossValue2Multiplier, q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                   value_diff_loss_weight=cfg.Opt_LossValueDMultiplier, value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                   action_loss_weight=cfg.Opt_LossActionMultiplier, uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                   action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier, q_ratio=cfg.Data_FractionQ)
  finally:
    sys.stdout = stdout
  return cfg, net


def kinds(net):
  import re
  return {re.sub(r'\.\d+\.', '.N.', n): 0 for n, _ in net.named_parameters()}


def main():
  import test_bt6_hybrid as H
  torch.set_num_threads(8)
  _, base = build(BASE_DIR, BASE_ID)
  n_base = sum(p.numel() for p in base.parameters())
  base_kinds = set(kinds(base))
  print(f'base arm {BASE_ID}: {n_base:,} parameters')
  sq, batch = H.records(4), H.batch_for(4)
  for cid in IDS:
    cfg, net = build(CONFIG_DIR, cid)
    n = sum(p.numel() for p in net.parameters())
    net.train()
    loss = H.run_loss(net, batch, sq)
    loss.backward()
    assert torch.isfinite(loss), cid
    none = sorted({n.split('.')[0] for n, p in net.named_parameters() if p.grad is None})
    k = set(kinds(net))
    added, removed = sorted(k - base_kinds), sorted(base_kinds - k)
    line = f'{cid}: builds, loss {float(loss.detach()):.4f} finite; {n:,} parameters ({n - n_base:+,} vs the base arm); weight decay {cfg.Opt_WeightDecay:.5g}'
    if len(added) + len(removed) <= 12:
      line += f'; parameter kinds added {added}, removed {removed}'
    else:
      line += f'; {len(added)} parameter kinds added, {len(removed)} removed'
    line += f'; parameters without a gradient (grad None): {none if none else "none"}'
    print(line)
  print('test_fry_configs: OK')


if __name__ == '__main__':
  main()
