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
    loss = self.ce_loss.forward(output, target) - entropy
       
    self.PENDING_POLICY_LOSS += loss.item() if not calc_grad_norm_mode else 0
    self.PENDING_POLICY_ACC += self.calc_accuracy(target, output, True) if not calc_grad_norm_mode else 0
    self.PENDING_COUNT += 1 if not calc_grad_norm_mode else 0 # increment only for policy, not other losses

#   cos = nn.CosineSimilarity(dim=1, eps=1e-6) # cosine similarity and correlation metrics are related
#   pearson = cos(target - target.mean(dim=1, keepdim=True), output - output.mean(dim=1, keepdim=True))
#   print ('policy ', loss.item(), ' ', (sum(pearson) / len(pearson)).item(), '  acc ', self.LAST_POLICY_ACC)
#   return 100 * torch.nn.functional.mse_loss(output, target)

    return self.calc_loss_grad_norm('policy', loss, loss_wt) if calc_grad_norm_mode else loss


  def value_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    entropy = self.entropy(target) if subtract_entropy else 0.0
    loss = self.ce_loss.forward(output, target) - entropy
    # Guarded like the other heads: the grad-norm diagnostic pass must not double-count stats.
    self.PENDING_VALUE_LOSS += loss.item() if not calc_grad_norm_mode else 0
    self.PENDING_VALUE_ACC += self.calc_accuracy(target, output, False) if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('value', loss, loss_wt) if calc_grad_norm_mode else loss


  def value2_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    entropy = self.entropy(target) if subtract_entropy else 0.0
    loss = self.ce_loss.forward(output, target) - entropy
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
      # the mask; its logit is pooled into bucket 0 but contributes only as (tiny) noise mass.
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


  def value_diff_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    target_softmax = F.softmax(target, dim=-1)
    entropy = self.entropy(target_softmax) if subtract_entropy else 0.0
    loss = self.ce_loss.forward(output, target_softmax) - entropy

    self.PENDING_VALUE_DIFF_LOSS += loss.item() if not calc_grad_norm_mode else 0
    return self.calc_loss_grad_norm('value_diff', loss, loss_wt) if calc_grad_norm_mode else loss


  def value2_diff_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool, calc_grad_norm_mode : bool, loss_wt : float):
    if calc_grad_norm_mode:
      self.model.zero_grad()

    target_softmax = F.softmax(target, dim=-1)
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
    