# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Weight-decay partition (decay / no_decay) of a CeresNet's parameters.

Factored out of train.py (2026-09-02) so a unit test can run the SAME loop the
optimizer build runs. Motivation: the edge-aux readout (`dp_eaux_w/b`, raw
nn.Parameters, boelge 13) reached the bench as `ea4fe1a` and died before step 0
with "parameters were not fully partitioned" — the smoke test built the net and
computed the loss but never touched this partition. Any new raw nn.Parameter
(not owned by an nn.Linear / norm / embedding) needs an explicit branch here,
and `assert_partition_complete` is what the tests call to prove it has one.

Based on the minGPT recipe (https://github.com/karpathy/minGPT).
NB: under Muon the no_decay set is INERT (muon.py applies the group weight decay
to everything it owns); the partition still has to be complete or the assert fires.
"""

import torch

from rms_norm import RMSNorm
from derf_norm import DerfNorm
from dyt_norm import DyTNorm
from soft_moe_batched_dual import SoftMoEBatchedDual
from multi_expert import MultiExpertLayer

WHITELIST_WEIGHT_MODULES = (torch.nn.Linear, SoftMoEBatchedDual, MultiExpertLayer)
BLACKLIST_WEIGHT_MODULES = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm, DerfNorm, DyTNorm)


def partition_weight_decay(model):
  """Returns (decay, no_decay): two sets of full parameter names covering EVERY
  parameter of `model` exactly once. Raises AssertionError otherwise."""
  decay = set()
  no_decay = set()

  for mn, m in model.named_modules():
      for pn, p in m.named_parameters():
          fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
          if pn.endswith('bias'):
              no_decay.add(fpn)
          elif "rpe" in fpn:
              decay.add(fpn)
          elif "lora" in fpn:
              no_decay.add(fpn)
          elif fpn.endswith('softmin_log_tau') or fpn.endswith('softmax_log_tau') \
                or fpn.endswith('head_logit_temp'):
              # (Flyttet HIT 2026-08-28 — bugfunn: grenen laa ETTER catch-all-en
              # og var doed, stikk i strid med sin egen 2026-08-20#6-kommentar.)
              no_decay.add(fpn)
          elif "transformer_layer" in fpn:
              decay.add(fpn)
          elif "rpe_factor" in fpn:
              pass
          elif "alphas" in fpn: # for Denseformer
              decay.add(fpn)
          elif "vda_query" in fpn: # depth-attention pseudo-query (bare 1-D vector, bias-like)
              no_decay.add(fpn)
          elif "rc_btype" in fpn or "rc_u" in fpn or "rc_v" in fpn or "rc_w" in fpn: # ray-context bare vectors (bias-like)
              no_decay.add(fpn)
          elif "rc_W" in fpn: # ray-context projections (plain Linear weights)
              decay.add(fpn)
          elif "dual_plane" in fpn and "log_tau" in fpn:
              # P-plane soft-min temperatures: bias-like 1-D log params.
              no_decay.add(fpn)
          elif "dp_eaux_" in fpn:
              # Edge-aux readout (boelge 13): raw nn.Parameters [T,C] + [T], NOT an
              # nn.Linear (fixed-key init, no global RNG draw), so no catch-all
              # sees them. no_decay, per the bench's fix (09-02): (1) it is a
              # training-only PROBE readout, and WD shrinking its logit scale would
              # damp the `_sep` decodability metric in both arms — the very
              # measurement the arm exists for; (2) follows the raw-table
              # convention (cbk_*, smol_*_bank). Hits B and C identically.
              # NB inert under Muon anyway (group wd) — completeness is the point.
              no_decay.add(fpn)
          elif "smol_basis_bank" in fpn or "smol_static_bank" in fpn:
              # Smolbasis/smbstatic-tabellbankene: raa logit-tabeller, ikke
              # projeksjonsvekter — embedding-konvensjonen (no decay). NB
              # no_decay er INERT under Muon (muon.py bruker gruppe-wd paa alt);
              # Muon-ortogonaliserings-unntaket haandteres separat i
              # _use_muon_final_only (review-funn 3/4 2026-09-01).
              no_decay.add(fpn)
          elif "cbk_keys" in fpn or "cbk_vals" in fpn:
              # Tactical-codebook motif tables: embedding-like raw matrices
              # (row = motif), not projection weights — follow the embedding
              # convention (no decay; also keeps them out of Muon's
              # orthogonalization, which targets true weight matrices).
              no_decay.add(fpn)
          elif ".mem_" in fpn:
              decay.add(fpn)
          elif "mlp.linear" in fpn:
              decay.add(fpn)
          elif "qkv" in fpn:
              decay.add(fpn)
          elif "embedding" in fpn:
              no_decay.add(fpn)
          elif isinstance(m, BLACKLIST_WEIGHT_MODULES):
              no_decay.add(fpn)
          elif isinstance(m, WHITELIST_WEIGHT_MODULES):
              decay.add(fpn)

  assert_partition_complete(model, decay, no_decay)
  return decay, no_decay


def assert_partition_complete(model, decay, no_decay):
  param_dict = {pn: p for pn, p in model.named_parameters()}
  inter_params = decay & no_decay
  union_params = decay | no_decay
  assert len(inter_params) == 0, "parameters %s appear in both decay/no_decay sets" % (str(inter_params), )
  assert len(param_dict.keys() - union_params) == 0, "parameters %s were not fully partitioned into decay/no_decay sets" \
                                              % (str(param_dict.keys() - union_params), )
