# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

# NOTE: this code derived from: https://github.com/Rocketknight1/minimal_lczero.

import os

import torch
from torch import nn
from torch.nn import functional as F

# Survival-loss variants (SURVIVAL_TARGET_SPEC.md; pure loss-side — sidecar labels and the
# K+2-logit head shape are unchanged, so checkpoints stay comparable across modes):
#   CERES_SURVIVAL_LOSS_BUCKETS: comma-separated upper bounds of capture-distance buckets,
#     e.g. "2,4,8" -> buckets [1-2],[3-4],[5-8],[survives]. Bucket logits are formed by
#     logsumexp-pooling the exact-ply logits, then CE is applied at bucket granularity —
#     exact capture TIMING depends on move-order choices the position does not determine
#     (measured: 13% exact-ply vs ~high bucket agreement), so bucketing concentrates the
#     gradient on the learnable distinction. Empty = off (exact-ply CE, legacy).
#   CERES_SURVIVAL_CAPTURE_WEIGHT: CE class weight applied to all CAPTURE classes/buckets
#     (the "survives" class keeps weight 1). Counters the ~10:1 survive:capture imbalance.
#     Default 1.0 = off.
# NOTE: two further shaping modes (all-threshold ORDINAL loss; piece-value square weighting)
# were implemented, A/B-tested at 20M on 2026-07-07 (t91s1k4i20M), found Pareto-NEGATIVE
# (value −11..−52 all bands, policy −9..−14), and REMOVED. See SURVIVAL_TARGET_SPEC.md §6.1.
_SURV_BUCKETS_ENV = os.environ.get('CERES_SURVIVAL_LOSS_BUCKETS', '').strip()
SURVIVAL_BUCKET_BOUNDS = [int(x) for x in _SURV_BUCKETS_ENV.split(',') if x.strip()] if _SURV_BUCKETS_ENV else None
SURVIVAL_CAPTURE_WEIGHT = float(os.environ.get('CERES_SURVIVAL_CAPTURE_WEIGHT', '1') or 1)
if SURVIVAL_BUCKET_BOUNDS is not None:
  print(f'[losses] survival loss: BUCKET mode, capture-distance bucket bounds {SURVIVAL_BUCKET_BOUNDS} (+survives)')
if SURVIVAL_CAPTURE_WEIGHT != 1.0:
  print(f'[losses] survival loss: capture-class weight {SURVIVAL_CAPTURE_WEIGHT}')

# Short-term value head (V7_EXTRAS_SIDECAR_SPEC.md): optional per-record loss weighting
# by z-provenance code (0=orig result, 1=syzygy, 2=deblunder-noise, 3=deblunder-unint,
# 4=op1-8man). CERES_STVALUE_PROV_WEIGHTS="w0,w1,w2,w3,w4"; unset = uniform.
_STVALUE_PROV_ENV = os.environ.get('CERES_STVALUE_PROV_WEIGHTS', '').strip()
STVALUE_PROV_WEIGHTS = [float(x) for x in _STVALUE_PROV_ENV.split(',')] if _STVALUE_PROV_ENV else None
if STVALUE_PROV_WEIGHTS is not None:
  assert len(STVALUE_PROV_WEIGHTS) == 5, f'CERES_STVALUE_PROV_WEIGHTS needs 5 values, got {STVALUE_PROV_WEIGHTS}'
  print(f'[losses] stvalue loss: per-provenance weights {STVALUE_PROV_WEIGHTS}')

# Per-provenance weighting of the MAIN value/value2 losses (CERES_VALUE_PROV_WEIGHTS,
# same 5-value format/codes as above). Weights records by how trustworthy their result
# label is: e.g. "0.5,3,1,1,5" learns hardest from syzygy-proven (1) and op1-relabeled
# (4) positions, whose labels are ground truth the net may genuinely disagree with.
# Adds NO parameters (pure loss reweighting) so it composes with LoRA fine-tuning.
# Requires v7x sidecars (batch['z_provenance']); silently inactive per-batch when the
# key is absent, with a loud startup check in train.py against whole-run no-ops.
_VALUE_PROV_ENV = os.environ.get('CERES_VALUE_PROV_WEIGHTS', '').strip()
VALUE_PROV_WEIGHTS = [float(x) for x in _VALUE_PROV_ENV.split(',')] if _VALUE_PROV_ENV else None
# Scope: 'both' (default) applies provenance weighting to value1 AND value2;
# 'value2' restricts it to value2 only — the right setting when value1 trains
# toward Q (FractionQ=1), since provenance describes the z labels, not Q.
VALUE_PROV_SCOPE = (os.environ.get('CERES_VALUE_PROV_SCOPE', 'both') or 'both').strip().lower()
if VALUE_PROV_WEIGHTS is not None:
  assert len(VALUE_PROV_WEIGHTS) == 5, f'CERES_VALUE_PROV_WEIGHTS needs 5 values, got {VALUE_PROV_WEIGHTS}'
  assert VALUE_PROV_SCOPE in ('both', 'value2'), f"CERES_VALUE_PROV_SCOPE must be 'both' or 'value2', got {VALUE_PROV_SCOPE!r}"
  print(f'[losses] value loss provenance weights {VALUE_PROV_WEIGHTS} (scope: {VALUE_PROV_SCOPE})')

# CERES_VALUE_FOCAL_GAMMA: hardness-weighted value1 loss (default 0 = off).
# Late in training the batch-mean value KL sits at the teacher-noise floor:
# most samples contribute pure label-noise gradient while a minority carry the
# remaining real signal. Reweight each sample by its own (detached) KL^gamma,
# normalized to keep the loss scale comparable: gradient budget moves from
# noise-fitting the solved samples to the genuinely-wrong ones. value1 only —
# value2 targets are one-hot outcomes where "hard" = mislabeled (blunders),
# exactly the samples NOT to upweight.
VALUE_FOCAL_GAMMA = float(os.environ.get('CERES_VALUE_FOCAL_GAMMA', '0') or 0)
if VALUE_FOCAL_GAMMA > 0:
  assert VALUE_PROV_WEIGHTS is None or VALUE_PROV_SCOPE == 'value2', \
    'CERES_VALUE_FOCAL_GAMMA and value1-scope provenance weighting are mutually exclusive'
  print(f'[losses] value1 FOCAL hardness weighting enabled: gamma={VALUE_FOCAL_GAMMA}')

# CERES_POLICY_ONLYMOVE_LAMBDA: only-move criticality weighting of the policy
# CE (tactics toolbox T1.3, default 0 = off). Tactics are only-move chains,
# but plain mean CE prices a forced mate-in-3 exactly like a quiet
# 12-candidate position. Weight each sample by the sharpness of its own
# POLICY TARGET: w_i = 1 + lambda * gap_i, gap_i = top1 - top2 of the target
# distribution (in [0,1], detached — it is pure label data). Batch-normalized
# (sum w*ce / sum w) so the loss SCALE — and thus the effective policy LR —
# is unchanged: the mechanism only REDISTRIBUTES gradient toward forced
# positions. Pure loss-side: no params, serving graph untouched.
POLICY_ONLYMOVE_LAMBDA = float(os.environ.get('CERES_POLICY_ONLYMOVE_LAMBDA', '0') or 0)
if POLICY_ONLYMOVE_LAMBDA > 0:
  print(f'[losses] policy ONLY-MOVE sharpness weighting enabled: lambda={POLICY_ONLYMOVE_LAMBDA}')

# CERES_POLICY_SIBLING_MARGIN_WEIGHT (+ CERES_POLICY_SIBLING_MARGIN, nats):
# sibling-margin policy term (tactics toolbox T1.4, default 0 = off). CE only
# asks for probability mass on the target; tactics additionally require the
# forced move to DOMINATE its best-scoring WRONG sibling. Adds a hinge in
# log-prob space (scale-free): relu(margin + logp_bestwrong - logp_target),
# weighted per-sample by the target's own sharpness gap (top1-top2, detached)
# so ambiguous multi-candidate labels contribute ~nothing — the term reprices
# only genuinely forced positions. Added to the RETURNED loss only; the
# logged/TRAIN policy loss stays pure CE for cross-run comparability. Watch
# gate KLD for over-sharpening (pre-registered risk).
POLICY_SIBLING_MARGIN_WEIGHT = float(os.environ.get('CERES_POLICY_SIBLING_MARGIN_WEIGHT', '0') or 0)
POLICY_SIBLING_MARGIN = float(os.environ.get('CERES_POLICY_SIBLING_MARGIN', '1.0') or 1.0)
if POLICY_SIBLING_MARGIN_WEIGHT > 0:
  print(f'[losses] policy SIBLING-MARGIN enabled: weight={POLICY_SIBLING_MARGIN_WEIGHT}, '
        f'margin={POLICY_SIBLING_MARGIN} nats, gap-weighted')



class LossCalculator():
  """Class to compute and keep track of losses on various training target heads.
   """

  def __init__(self, model : nn.Module):
    super().__init__()

    self.MASK_POLICY_VALUE = -6E4 # for illegal moves (stay within range of float16)

    # Keep running statistics (counts/totals) in between calls to reset_counters.
    self.reset_counters()
    self.ce_loss = nn.CrossEntropyLoss()
    self.model = model


  def reset_counters(self):
    self.PENDING_COUNT = 0
    self.PENDING_VALUE_LOSS = 0
    self.PENDING_POLICY_LOSS = 0
    self.PENDING_PLACEMENT_VALUE_LOSS = 0
    self.PENDING_SURVIVAL_LOSS = 0
    self.PENDING_SURVIVAL_ACC = 0
    self.PENDING_STVALUE_LOSS = 0
    self.PENDING_VALUE_ACC = 0
    self.PENDING_POLICY_ACC = 0
    self.PENDING_MLH_LOSS = 0
    self.PENDING_UNC_LOSS = 0
    self.PENDING_VALUE2_LOSS = 0
    self.PENDING_Q_DEVIATION_LOWER_LOSS = 0
    self.PENDING_Q_DEVIATION_UPPER_LOSS = 0
    self.PENDING_UNCERTAINTY_POLICY_LOSS = 0
    self.PENDING_VALUE_DIFF_LOSS = 0
    self.PENDING_VALUE2_DIFF_LOSS = 0
    self.PENDING_ACTION_LOSS = 0
    self.PENDING_ACTION_UNCERTAINTY_LOSS = 0
    
  @property
  def LAST_VALUE_LOSS(self):
    return self.PENDING_VALUE_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_VALUE2_LOSS(self):
    return self.PENDING_VALUE2_LOSS / self.PENDING_COUNT

  @property
  def LAST_PLACEMENT_VALUE_LOSS(self):
    return self.PENDING_PLACEMENT_VALUE_LOSS / self.PENDING_COUNT

  @property
  def LAST_SURVIVAL_LOSS(self):
    return self.PENDING_SURVIVAL_LOSS / self.PENDING_COUNT

  @property
  def LAST_SURVIVAL_ACC(self):
    return self.PENDING_SURVIVAL_ACC / self.PENDING_COUNT

  @property
  def LAST_STVALUE_LOSS(self):
    return self.PENDING_STVALUE_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_VALUE_DIFF_LOSS(self):
    return self.PENDING_VALUE_DIFF_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_VALUE2_DIFF_LOSS(self):
    return self.PENDING_VALUE2_DIFF_LOSS / self.PENDING_COUNT

  @property
  def LAST_POLICY_LOSS(self):
    return self.PENDING_POLICY_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_VALUE_ACC(self):
    return self.PENDING_VALUE_ACC / self.PENDING_COUNT
  
  @property
  def LAST_POLICY_ACC(self):
    return self.PENDING_POLICY_ACC / self.PENDING_COUNT

  @property
  def LAST_MLH_LOSS(self):
    return self.PENDING_MLH_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_UNC_LOSS(self):
    return self.PENDING_UNC_LOSS / self.PENDING_COUNT

  @property
  def LAST_Q_DEVIATION_LOWER_LOSS(self):
    return self.PENDING_Q_DEVIATION_LOWER_LOSS / self.PENDING_COUNT

  @property
  def LAST_Q_DEVIATION_UPPER_LOSS(self):
    return self.PENDING_Q_DEVIATION_UPPER_LOSS / self.PENDING_COUNT

  @property
  def LAST_ACTION_LOSS(self):
    return self.PENDING_ACTION_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_UNCERTAINTY_POLICY_LOSS(self):
    return self.PENDING_UNCERTAINTY_POLICY_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_ACTION_UNCERTAINTY_LOSS(self):
    return self.PENDING_ACTION_UNCERTAINTY_LOSS / self.PENDING_COUNT


  # calculates and returns the gradient norm of the loss
  # warning: this zeros the other gradients of the model
  def calc_loss_grad_norm(self, loss_name : str, loss : torch.Tensor, loss_wt : float):
    self.model.zero_grad()
    loss.backward(retain_graph = True)
    norm = sum((p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None)) ** 0.5
    self.model.zero_grad()
    # GRADNORM: prefix keeps these lines grep-able alongside TRAIN:/SURV: in run logs.
    print('GRADNORM:', loss_name, ',', round(norm, 5), ',', round(norm * loss_wt, 5), flush=True)
    return norm
  

  def calc_accuracy(self, target: torch.Tensor, output: torch.Tensor, apply_masking : bool) -> float:
    if apply_masking:
      legalMoves = target.greater(0)
      illegalMaskValue = torch.zeros_like(output).add_(self.MASK_POLICY_VALUE)
      output = torch.where(legalMoves, output, illegalMaskValue)
    
    max_scores, max_idx_class = target.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    max_scores_out, max_idx_class_out = output.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    n = target.size(0)
    acc = (max_idx_class == max_idx_class_out).sum().item() / n
    return 100 * acc


  def value_metrics(self, target: torch.Tensor, output: torch.Tensor) -> dict:
    """Base-rate-aware value diagnostics (added 2026-08-22).

    Plain argmax accuracy has two defects that make cross-run reads unsound:

    1. It is dominated by the corpus draw rate. An "always draw" predictor
       scores exactly the draw fraction, so a corpus with 4 pp more draws looks
       4 pp better at identical skill. Our own T80/T91 gap is ~5 pp, so every
       value_acc comparison across those two corpora has been contaminated.
    2. It is blind to SHARPNESS. Predicting 0.34/0.33/0.33 and 0.9/0.05/0.05
       score the same. This project's documented value pathology is that the
       head is COMPRESSED rather than weak — so the headline metric cannot see
       the actual problem.

    Everything here is base-rate normalised and/or sharpness-sensitive, so the
    numbers stay comparable when the corpus changes.
    """
    with torch.no_grad():
      t = target.float()
      p = torch.softmax(output.float(), dim=-1)
      tc = t.argmax(dim=1)
      pc = p.argmax(dim=1)
      n = max(t.shape[0], 1)

      # Reference: accuracy of always predicting the majority class.
      base_rate = torch.bincount(tc, minlength=3).float().max() / n
      acc = (pc == tc).float().mean()
      # 0 = no better than always-majority, 1 = perfect. Base-rate free.
      # A single-class batch makes the denominator 0; the clamp keeps it finite
      # but would still emit a huge value that wrecks the TB axis, so report 0
      # (undefined skill) instead. torch.where keeps this sync-free.
      _headroom = 1.0 - base_rate
      skill = torch.where(_headroom > 1e-4,
                          (acc - base_rate) / _headroom.clamp(min=1e-6),
                          torch.zeros_like(acc))

      # Balanced accuracy = mean per-class recall; an always-draw predictor
      # scores 1/3 regardless of how drawish the corpus is. Vectorised to
      # avoid per-class host syncs.
      correct = (pc == tc).float()
      ones = torch.ones_like(correct)
      cls_total = torch.zeros(3, device=t.device, dtype=correct.dtype).index_add_(0, tc, ones)
      cls_hit = torch.zeros(3, device=t.device, dtype=correct.dtype).index_add_(0, tc, correct)
      recall = cls_hit / cls_total.clamp(min=1.0)
      bal_acc = recall.sum() / (cls_total > 0).float().sum().clamp(min=1.0)

      # Brier skill score against the batch climatology (the mean target).
      # Proper scoring rule: rewards honest probabilities, not a lucky argmax,
      # and the climatology reference divides out the base rate.
      brier = ((p - t) ** 2).sum(dim=1).mean()
      clim = t.mean(dim=0, keepdim=True)
      brier_base = ((clim - t) ** 2).sum(dim=1).mean()
      # Degenerate batch (all targets identical): climatology is perfect, the
      # reference Brier is ~0, and the ratio explodes. Skill against a perfect
      # reference is undefined — report 0 rather than a nonsense magnitude.
      bss = torch.where(brier_base > 1e-6,
                        1.0 - brier / brier_base.clamp(min=1e-9),
                        torch.zeros_like(brier))

      # Dispersion of the predicted scalar q — measures compression directly.
      # unbiased=False: this is the spread of THIS batch, not an estimate of a
      # wider population, and the Bessel correction would make n==1 produce NaN.
      q_spread = (p[:, 0] - p[:, 2]).std(unbiased=False)

    return {'value_base_rate': 100.0 * base_rate,
            'value_skill': skill,
            'value_bal_acc': 100.0 * bal_acc,
            'value_bss': bss,
            'value_q_spread': q_spread}


  def entropy(self, probabilities : torch.Tensor):
    # entropy is same as cross entropy with itself
    clipped_probabilities = torch.clamp(probabilities + 1e-6, min=1e-6)
    return torch.nn.functional.cross_entropy(torch.log(clipped_probabilities),clipped_probabilities)


  def policy_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    legalMoves = target.greater(0)
    illegalMaskValue = torch.zeros_like(output).add_(self.MASK_POLICY_VALUE)
    output = torch.where(legalMoves, output, illegalMaskValue)

    entropy = self.entropy(target) if subtract_entropy else 0.0
    if POLICY_ONLYMOVE_LAMBDA > 0:
      # Only-move weighting (see module header). fp32 CE mirrors the value
      # path; entropy subtraction stays unweighted (informational only — the
      # target entropy carries no gradient wrt the output).
      _ce_i = F.cross_entropy(output.float(), target.float(), reduction='none')
      _t2 = target.float().topk(2, dim=1).values
      _w_i = 1.0 + POLICY_ONLYMOVE_LAMBDA * (_t2[:, 0] - _t2[:, 1]).detach()
      loss = (_w_i * _ce_i).sum() / _w_i.sum().clamp_min(1e-6) - entropy
      # Pure-CE logging invariant (review finding #10, mirroring the
      # sibling-margin block): the LOGGED number stays the unweighted mean CE
      # for cross-run comparability; only the RETURNED loss is reweighted.
      _log_loss = _ce_i.mean() - entropy
    else:
      loss = self.ce_loss.forward(output, target) - entropy
      _log_loss = loss

    self.PENDING_POLICY_LOSS += _log_loss.item() if not calc_grad_norm_mode else 0
    if POLICY_SIBLING_MARGIN_WEIGHT > 0:
      # Sibling-margin hinge (see module header). Computed on the MASKED
      # logits, so illegal moves (at MASK_POLICY_VALUE) can never be the
      # best-wrong sibling. Added after the PENDING update: logged policy
      # loss stays pure CE.
      _logp = F.log_softmax(output.float(), dim=1)
      _tgt_idx = target.argmax(dim=1, keepdim=True)
      _lp_t = _logp.gather(1, _tgt_idx).squeeze(1)
      _lp_w = _logp.scatter(1, _tgt_idx, float('-inf')).max(dim=1).values
      _t2 = target.float().topk(2, dim=1).values
      _gap = (_t2[:, 0] - _t2[:, 1]).detach()
      _hinge = F.relu(POLICY_SIBLING_MARGIN + _lp_w - _lp_t)
      loss = loss + POLICY_SIBLING_MARGIN_WEIGHT * (_gap * _hinge).mean()
    self.PENDING_POLICY_ACC += self.calc_accuracy(target, output, True) if not calc_grad_norm_mode else 0
    self.PENDING_COUNT += 1 if not calc_grad_norm_mode else 0 # increment only for policy, not other losses

#   cos = nn.CosineSimilarity(dim=1, eps=1e-6) # cosine similarity and correlation metrics are related
#   pearson = cos(target - target.mean(dim=1, keepdim=True), output - output.mean(dim=1, keepdim=True))
#   print ('policy ', loss.item(), ' ', (sum(pearson) / len(pearson)).item(), '  acc ', self.LAST_POLICY_ACC)
#   return 100 * torch.nn.functional.mse_loss(output, target)

    return self.calc_loss_grad_norm('policy', loss, loss_wt) if calc_grad_norm_mode else loss


  def _prov_weighted_ce(self, target: torch.Tensor, output: torch.Tensor, provenance: torch.Tensor):
    """Cross entropy with optional per-record z-provenance weighting
    (CERES_VALUE_PROV_WEIGHTS). Falls back to plain mean CE when weighting is
    off or the batch carries no provenance (v7x-less shard in auto mode)."""
    # fp32 cast on the logits (2026-08-09 comparative audit vs lc0, row 8):
    # under BFloat16Pure the value logits would otherwise enter the CE already
    # quantized to bf16 (~±0.004), non-negligible at the late-run 0.01-nat KL
    # scale. lc0 guards this with an explicit fp32 cast in the value head.
    output = output.float()
    target = target.float()
    if VALUE_PROV_WEIGHTS is not None and provenance is not None:
      prov_wt = torch.tensor(VALUE_PROV_WEIGHTS, device=output.device, dtype=torch.float32)
      rec_wt = prov_wt[provenance.reshape(-1).long()]
      per_rec = F.cross_entropy(output, target, reduction='none')
      return (per_rec * rec_wt).sum() / rec_wt.sum().clamp_min(1e-6)
    return self.ce_loss.forward(output, target)


  def value_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float, provenance: torch.Tensor = None):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    # Entropy subtraction stays unweighted under provenance weighting (informational
    # comparability of the logged number; the gradient comes from the CE term only).
    entropy = self.entropy(target) if subtract_entropy else 0.0
    if VALUE_FOCAL_GAMMA > 0:
      # Hardness-weighted per-sample KL (see module header). Per-sample entropy
      # is subtracted BEFORE weighting so the hardness measure is the true KL,
      # not inflated by high-entropy (drawish) targets.
      _ce_i = F.cross_entropy(output.float(), target.float(), reduction='none')
      _t = torch.clamp(target.float() + 1e-6, min=1e-6)
      _h_i = -(_t * torch.log(_t)).sum(dim=-1)
      _kl_i = _ce_i - _h_i
      _w_i = (_kl_i.detach().clamp_min(0) + 1e-3) ** VALUE_FOCAL_GAMMA
      loss = (_w_i * _kl_i).sum() / _w_i.sum().clamp_min(1e-6)
    else:
      _prov_v1 = provenance if VALUE_PROV_SCOPE == 'both' else None
      loss = self._prov_weighted_ce(target, output, _prov_v1) - entropy
    # Guarded like the other heads: the grad-norm diagnostic pass must not double-count stats.
    self.PENDING_VALUE_LOSS += loss.item() if not calc_grad_norm_mode else 0
    self.PENDING_VALUE_ACC += self.calc_accuracy(target, output, False) if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('value', loss, loss_wt) if calc_grad_norm_mode else loss


  def value2_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float, provenance: torch.Tensor = None):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    entropy = self.entropy(target) if subtract_entropy else 0.0
    loss = self._prov_weighted_ce(target, output, provenance) - entropy
    self.PENDING_VALUE2_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('value2', loss, loss_wt) if calc_grad_norm_mode else loss


  def placement_value_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float):
    """Auxiliary placement value head (additive per-square WDL decomposition).
    Same CE-minus-entropy form as value_loss/value2_loss so the logged number is
    directly comparable to those heads against the identical target."""
    if calc_grad_norm_mode:
      self.model.zero_grad()

    entropy = self.entropy(target) if subtract_entropy else 0.0
    loss = self.ce_loss.forward(output, target) - entropy
    self.PENDING_PLACEMENT_VALUE_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('placement_value', loss, loss_wt) if calc_grad_norm_mode else loss


  def _survival_tables(self, num_classes, device):
    """Build (once, cached) the class->bucket map and CE class-weight vectors for the
    configured survival-loss mode. num_classes = K+2 (class 0 = empty, 1..K = ply, K+1 = survives)."""
    cached = getattr(self, '_surv_tables', None)
    if cached is not None and cached[0] == num_classes and cached[1] == str(device):
      return cached[2], cached[3]

    K = num_classes - 2
    if SURVIVAL_BUCKET_BOUNDS is not None:
      bounds = SURVIVAL_BUCKET_BOUNDS
      assert bounds == sorted(bounds) and bounds[-1] == K, \
        f'CERES_SURVIVAL_LOSS_BUCKETS must be ascending and end at K={K}, got {bounds}'
      class_to_bucket = torch.zeros(num_classes, dtype=torch.long, device=device)
      class_to_bucket[0] = -1                         # empty class: excluded from pooling
      for c in range(1, K + 1):
        class_to_bucket[c] = next(i for i, b in enumerate(bounds) if c <= b)
      num_buckets = len(bounds) + 1
      class_to_bucket[K + 1] = num_buckets - 1        # survives = last bucket
      weights = torch.ones(num_buckets, device=device)
      weights[:num_buckets - 1] = SURVIVAL_CAPTURE_WEIGHT
    else:
      class_to_bucket = None
      weights = torch.ones(num_classes, device=device)
      weights[1:num_classes - 1] = SURVIVAL_CAPTURE_WEIGHT  # capture classes; empty+survives stay 1

    self._surv_tables = (num_classes, str(device), class_to_bucket, weights)
    return class_to_bucket, weights


  def survival_loss(self, target: torch.Tensor, output: torch.Tensor, calc_grad_norm_mode : bool, loss_wt : float):
    """K-ply survival aux head (SURVIVAL_TARGET_SPEC.md): per-square fate classification.
    target: [B, 64] uint8 (0 = empty square, masked out; 1..K = captured at ply d; K+1 = survives).
    output: [B, 64, C] logits with C = K+2 (class 0 exists but never appears under the mask).
    Modes (env; see module header): exact-ply CE (default), ordinal-bucket CE via
    logsumexp-pooled logits, optional capture-class weighting. Reported ACC matches the mode."""
    if calc_grad_norm_mode:
      self.model.zero_grad()

    mask = target > 0
    target_masked = target[mask].long()
    output_masked = output[mask].float()
    class_to_bucket, weights = self._survival_tables(output.shape[-1], output_masked.device)

    if class_to_bucket is not None:
      # Ordinal buckets: pool exact-ply logits into bucket logits (logsumexp = probability
      # sum in log space), grade at bucket granularity. Class 0 (empty) never appears under
      # the mask and its logit is EXCLUDED from pooling (it used to leak into bucket 0 as
      # noise mass; fixed 2026-07-07 — survival CE values shift negligibly vs older logs).
      num_buckets = int(weights.shape[0])
      bucket_logits = output_masked.new_full((output_masked.shape[0], num_buckets), float('-inf'))
      for b in range(num_buckets):
        cols = (class_to_bucket == b).nonzero(as_tuple=True)[0]
        bucket_logits[:, b] = torch.logsumexp(output_masked[:, cols], dim=1)
      target_graded = class_to_bucket[target_masked]
      loss = F.cross_entropy(bucket_logits, target_graded, weight=weights)
      pred_graded = bucket_logits.argmax(dim=1)
    else:
      loss = F.cross_entropy(output_masked, target_masked, weight=weights)
      target_graded = target_masked
      pred_graded = output_masked.argmax(dim=1)

    if not calc_grad_norm_mode:
      self.PENDING_SURVIVAL_LOSS += loss.item()
      self.PENDING_SURVIVAL_ACC += 100.0 * (pred_graded == target_graded).float().mean().item()
    return self.calc_loss_grad_norm('survival', loss, loss_wt) if calc_grad_norm_mode else loss


  def stvalue_loss(self, cens_q: torch.Tensor, cens_d: torch.Tensor, provenance: torch.Tensor,
                   output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float):
    """Short-term value aux head vs the blunder-censored q_st/d_st sidecar values
    (V7_EXTRAS_SIDECAR_SPEC.md). Target WDL is built from the STM-relative pair as
    w=(1-d+q)/2, l=(1-d-q)/2 (clamped >= 0 and renormalized against float drift).
    Same CE-minus-entropy form as value_loss so the logged number is comparable.
    Optional per-record weighting by z-provenance (CERES_STVALUE_PROV_WEIGHTS);
    the subtracted entropy stays unweighted (informational comparability only)."""
    if calc_grad_norm_mode:
      self.model.zero_grad()

    q = cens_q.reshape(-1).float()
    d = cens_d.reshape(-1).float()
    w = (1.0 - d + q) * 0.5
    l = (1.0 - d - q) * 0.5
    target = torch.stack((w, d, l), dim=1).clamp_min(0)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)

    entropy = self.entropy(target) if subtract_entropy else 0.0
    if STVALUE_PROV_WEIGHTS is not None and provenance is not None:
      prov_wt = torch.tensor(STVALUE_PROV_WEIGHTS, device=output.device, dtype=torch.float32)
      rec_wt = prov_wt[provenance.reshape(-1).long()]
      per_rec = F.cross_entropy(output, target, reduction='none')
      loss = (per_rec * rec_wt).sum() / rec_wt.sum().clamp_min(1e-6) - entropy
    else:
      loss = self.ce_loss.forward(output, target) - entropy
    self.PENDING_STVALUE_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('stvalue', loss, loss_wt) if calc_grad_norm_mode else loss


  def value_diff_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    target_softmax = F.softmax(target.detach(), dim=-1)  # maalsiden skal ikke motta gradient (bugfunn 2026-08-28)
    entropy = self.entropy(target_softmax) if subtract_entropy else 0.0
    loss = self.ce_loss.forward(output, target_softmax) - entropy

    self.PENDING_VALUE_DIFF_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('value_diff', loss, loss_wt) if calc_grad_norm_mode else loss


  def value2_diff_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    target_softmax = F.softmax(target.detach(), dim=-1)  # maalsiden skal ikke motta gradient (bugfunn 2026-08-28)
    entropy = self.entropy(target_softmax) if subtract_entropy else 0.0
    loss = self.ce_loss(output, target_softmax) - entropy
   
    self.PENDING_VALUE2_DIFF_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('value2_diff', loss, loss_wt) if calc_grad_norm_mode else loss


  def action_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    target_softmax = F.softmax(target, dim=-1)
    entropy = self.entropy(target_softmax) if subtract_entropy else 0.0
    loss = self.ce_loss(output, target_softmax) - entropy
  
    self.PENDING_ACTION_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('action', loss, loss_wt) if calc_grad_norm_mode else loss


  def moves_left_loss(self, target: torch.Tensor, output: torch.Tensor, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    # Scale the loss to similar range as other losses.
    self.POST_SCALE = 5.0
    loss = self.POST_SCALE * F.huber_loss(output, target, reduction="mean", delta=0.5)
    self.PENDING_MLH_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('moves_left', loss, loss_wt) if calc_grad_norm_mode else loss


  def unc_loss(self, target: torch.Tensor, output: torch.Tensor, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    # Scale the loss to similar range as other losses.
    self.POST_SCALE = 150.0
    loss = self.POST_SCALE * F.huber_loss(output, target, reduction="mean", delta=0.5)
    self.PENDING_UNC_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('uncertainty', loss, loss_wt) if calc_grad_norm_mode else loss


  def q_deviation_lower_loss(self, target: torch.Tensor, output: torch.Tensor, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    self.POST_SCALE = 10.0
    loss = self.POST_SCALE * nn.MSELoss().forward(output, target)
    self.PENDING_Q_DEVIATION_LOWER_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('qdev_lower', loss, loss_wt) if calc_grad_norm_mode else loss


  def q_deviation_upper_loss(self, target: torch.Tensor, output: torch.Tensor, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    self.POST_SCALE = 10.0
    loss = self.POST_SCALE * nn.MSELoss().forward(output, target)
    self.PENDING_Q_DEVIATION_UPPER_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('qdev_upper', loss, loss_wt) if calc_grad_norm_mode else loss


  def uncertainty_policy_loss(self, target: torch.Tensor, output: torch.Tensor, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    self.POST_SCALE = 10.0
    loss = self.POST_SCALE * nn.MSELoss().forward(output, target)
    self.PENDING_UNCERTAINTY_POLICY_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('policy_unc', loss, loss_wt) if calc_grad_norm_mode else loss


  def action_unc_loss(self, target: torch.Tensor, output: torch.Tensor, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    # Scale the loss to similar range as other losses.
    self.POST_SCALE = 150.0
    loss = self.POST_SCALE * F.huber_loss(output, target, reduction="mean", delta=0.5)
    self.PENDING_ACTION_UNCERTAINTY_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('action_uncertainty', loss, loss_wt) if calc_grad_norm_mode else loss
    