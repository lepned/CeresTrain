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
NB: under Muon the no_decay set is INERT unless Opt_MuonHonorNoDecay is on (muon.py
applies the group weight decay to everything it owns; HonorNoDecay passes per-param
wd scale 0 for this set); the partition still has to be complete or the assert fires.

2026-09-11 fix: norm-module parameters (RMSNorm.scale, LayerNorm.weight, DyT/Derf
alpha/gamma/beta) are resolved FIRST, by ownership, wherever they live. Before this
the `"transformer_layer" in fpn` catch-all sent every trunk norm gain to `decay`
(the isinstance(BLACKLIST) rule sat below it and never saw them), so HonorNoDecay
freed biases/embeddings/decoder norms but kept decaying the trunk gains — the
asymmetric half-measure diagnosed on the 8B run's seg2 (09-10/11).
NB this is NOT Muon-only: the AdamW-family optimizers build their two param groups
from these sets, so from this fix on they also stop decaying trunk norm gains
(the minGPT convention the BLACKLIST always intended). Consequences: (1) post-fix
AdamW runs differ from pre-fix ones in that respect — train.py logs the count;
(2) resuming a PRE-fix AdamW-family checkpoint changes the per-group sizes, which
train.py's resume path treats as a group mismatch => optimizer state starts fresh
(it says so). Muon is unaffected (single group; wd scales fixed at construction).
"""

import torch

from rms_norm import NORM_MODULE_TYPES
from soft_moe_batched_dual import SoftMoEBatchedDual
from multi_expert import MultiExpertLayer

WHITELIST_WEIGHT_MODULES = (torch.nn.Linear, SoftMoEBatchedDual, MultiExpertLayer)
# Embedding only: norm types are claimed by ownership (norm_owned_param_names)
# before any name rule, so listing them here would be dead code.
BLACKLIST_WEIGHT_MODULES = (torch.nn.Embedding,)


def norm_owned_param_names(model):
  """Full names of every parameter owned DIRECTLY by a norm module
  (NORM_MODULE_TYPES; recurse=False so a container that merely contains a norm
  does not match). These are gains/offsets, never weight matrices."""
  names = set()
  for mn, m in model.named_modules():
      if isinstance(m, NORM_MODULE_TYPES):
          for pn, _ in m.named_parameters(recurse=False):
              names.add('%s.%s' % (mn, pn) if mn else pn)
  return names


def partition_weight_decay(model):
  """Returns (decay, no_decay): two sets of full parameter names covering EVERY
  parameter of `model` exactly once. Raises AssertionError otherwise."""
  decay = set()
  no_decay = set()

  # Norm gains are resolved by OWNERSHIP before any name-based rule, so trunk
  # norms ("transformer_layer.N.ln*") are not swept into `decay` by the catch-all
  # below. Keep this the first branch: anything inserted above it re-opens the bug.
  norm_owned = norm_owned_param_names(model)

  for mn, m in model.named_modules():
      for pn, p in m.named_parameters():
          fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
          if pn.endswith('bias') or fpn in norm_owned:
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
          elif "move_tokens." in fpn and (fpn.endswith(".pm_dk") or fpn.endswith(".pm_dv") or fpn.endswith("vq_block.vq")
                                           or fpn.endswith(".opp_side")):
              # Move-token post-move deltas (per-piece key/value edits) and the learned value
              # query token (2026-09-03): raw nn.Parameters, embedding-like -> no_decay.
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
          elif fpn.endswith("mt_ev_dir"):
              # 2026-09-08 expected-value readout: zero-init 3-vector mapping the decoder's
              # expected value onto the WDL logits. A raw direction/gain, not a weight matrix
              # -> no_decay (inert under Muon anyway; keeps it out of orthogonalization).
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
  # Owner-based invariant (2026-09-11): every param directly owned by a norm or
  # embedding module must be in no_decay, wherever it lives in the tree. This is
  # the check the tests inherit; it is independent of the rule order above.
  owned_nodecay = set(norm_owned_param_names(model))
  for mn, m in model.named_modules():
      if isinstance(m, BLACKLIST_WEIGHT_MODULES):
          for pn, _ in m.named_parameters(recurse=False):
              owned_nodecay.add('%s.%s' % (mn, pn) if mn else pn)
  leaked = owned_nodecay & decay
  assert not leaked, "norm/embedding-owned parameters landed in the decay set: %s" % (sorted(leaked),)
