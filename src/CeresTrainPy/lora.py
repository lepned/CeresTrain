# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice


import struct
import re

import math
from typing import Tuple, NamedTuple

import torch
import torch.nn as nn

# PiSSA (Principal Singular values & Singular vectors Adaptation, arXiv:2404.02948).
# Replaces vanilla LoRA's Kaiming/zero init with an SVD-based init that captures
# the top-r principal components of the base weight in lora_A @ lora_B, and
# subtracts that rank-r approximation from the base weight so the model output
# at init equals the original. Both heads have non-zero gradient flow from step 1.
# Set to True to enable; default False keeps vanilla LoRA behavior.
LORA_USE_PISSA = False


class LoRALinear(nn.Module):
  """
  A PyTorch module for applying Low-Rank Adaptation (LoRA) to a linear layer.
    
  LoRA introduces trainable low-rank matrices (A and B) to fine-tune large
  pre-trained models efficiently. This reduces the number of trainable 
  parameters. This implementation replaces a standard linear layer 
  with a LoRA-enabled linear layer.

  Parameters:
  - original_layer: The original nn.Linear layer to be adapted with LoRA.
  - rank_divisor: Determines the rank of the low-rank decomposition. The rank is 
    calculated as the in_features divided by the rank_divisor.
  - enable_lora: If True, enables LoRA by adding the low-rank matrices A and B.

  Reference:
  Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, 
  Weizhu Chen. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
  https://arxiv.org/abs/2106.09685
  """
  def __init__(self, original_layer, rank_divisor: int, enable_lora: bool = False):
    super(LoRALinear, self).__init__()
    self.original_layer = original_layer  # Original linear layer
    self.enable_lora = enable_lora

    MIN_RANK = 4
    self.rank = max(MIN_RANK, original_layer.in_features // rank_divisor)
    self.rank = min(original_layer.out_features, self.rank)  # Ensure rank is not greater than the output size

    if enable_lora:
      # LoRA trainable parameters
      # Note that the specific names "lora_A", "lora_B" and "lora_alpha" are
      # referenced elsewhere (e.g. in train.py)
      self.lora_A = nn.Parameter(torch.zeros((original_layer.out_features, self.rank)))
      self.lora_B = nn.Parameter(torch.zeros((self.rank, original_layer.in_features)))
      nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
      nn.init.zeros_(self.lora_B)

      # Trainable alpha parameter. Initialized to sqrt(rank) so the rsLoRA
      # scaling factor (alpha/sqrt(rank)) starts at 1.0 — i.e., the adapter's
      # effective contribution is at the same scale as a direct base-weight
      # update. Prior default was 0.1, which gave scaling ≈ 0.016 for rank=40
      # and meant LoRA adapters trained at ~1/60th the intended effective LR.
      # Identity at init still holds because lora_B is zeros.
      self.lora_alpha = nn.Parameter(torch.tensor(math.sqrt(float(self.rank))))
    else:
      self.lora_A = None
      self.lora_B = None
      self.lora_alpha = None

  def forward(self, x):
    if self.enable_lora:
      # Compute the LoRA update and scale it by alpha/sqrt(r)
      # (see "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA" by Kalajdzievski)
      scaling = self.lora_alpha / math.sqrt(self.rank)
      lora_update = (self.lora_A @ self.lora_B)
      return self.original_layer(x) + scaling * (x @ lora_update.T)
    else:
      return self.original_layer(x)

  def apply_pissa(self):
    """Re-initialize lora_A, lora_B, lora_alpha using PiSSA (SVD of base weight).

    Math: SVD W = U Σ Vᵀ. Take top-r components U_r, S_r, V_rᵀ.
      A = U_r * sqrt(S_r),  B = sqrt(S_r) * V_rᵀ
      lora_alpha = sqrt(rank)  ⇒  scaling factor (alpha/sqrt(rank)) = 1
      base_weight ← W − U_r·S_r·V_rᵀ    (subtract rank-r approximation)
    At init: base_modified + 1·A·B = (W − top_r) + top_r = W   (identity).
    During training, A and B move along the top-r directions.

    MUST be called AFTER the base checkpoint has been loaded.
    """
    if not self.enable_lora:
      return
    with torch.no_grad():
      W = self.original_layer.weight.detach().to(torch.float32)
      U, S, Vh = torch.linalg.svd(W, full_matrices=False)
      r = self.rank
      U_r = U[:, :r]
      S_r = S[:r]
      Vh_r = Vh[:r, :]
      sqrt_S = torch.sqrt(S_r)
      A_init = U_r * sqrt_S.unsqueeze(0)            # (out, r)
      B_init = sqrt_S.unsqueeze(1) * Vh_r           # (r, in)
      top_r_approx = (U_r * S_r.unsqueeze(0)) @ Vh_r
      dtype = self.original_layer.weight.dtype
      self.lora_A.data.copy_(A_init.to(dtype))
      self.lora_B.data.copy_(B_init.to(dtype))
      # Set alpha = sqrt(rank) so the rsLoRA scaling alpha/sqrt(rank) = 1.
      self.lora_alpha.data.fill_(math.sqrt(float(r)))
      self.original_layer.weight.data.sub_(top_r_approx.to(dtype))


def apply_pissa_to_model(model):
  """Walk the model and apply PiSSA init to every enabled LoRALinear layer.

  Skips layers where rank >= min(out_features, in_features). On those, PiSSA's
  top-r SVD captures the full weight matrix and zeros the base, leaving the
  layer 100% adapter-driven through a rank-saturated subspace — gradient
  dynamics through that adapter don't track direct base-weight updates well.
  Skipped layers keep vanilla LoRA init (lora_B=0), so they start as identity
  and train normally. Empirically observed on small output-dim heads (e.g.
  value_head.fcFinal 640->...->3) where this was the cause of v47's value
  head regression.

  Should be called after the base checkpoint has been loaded into the model.
  """
  n = 0
  skipped = []
  for m in model.modules():
    if isinstance(m, LoRALinear) and m.enable_lora:
      out_dim = m.original_layer.out_features
      in_dim = m.original_layer.in_features
      if m.rank >= min(out_dim, in_dim):
        skipped.append((out_dim, in_dim, m.rank))
        # rsLoRA-consistent scaling: vanilla init defaults alpha=0.1, giving
        # scaling = 0.1/sqrt(r). PiSSA-initialized layers get alpha=sqrt(r) →
        # scaling = 1.0. Without this fix the skipped layers train at ~17x
        # smaller effective LR than the rest of the model — empirically
        # observed in v48/v49 to leave value_head.fcFinal essentially
        # untrained while upstream layers reoriented under PiSSA, breaking
        # the value head's calibration. lora_B=0 is preserved, so identity
        # at init still holds; only the post-init learning rate changes.
        m.lora_alpha.data.fill_(math.sqrt(float(m.rank)))
        continue
      m.apply_pissa()
      n += 1
  print(f"PiSSA: re-initialized {n} LoRALinear layers using SVD of base weights.")
  if skipped:
    print(f"PiSSA: skipped {len(skipped)} rank-saturated layers (kept vanilla init, alpha=sqrt(r)): {skipped}")



def serialize_lora_to_binary(filename : str, lora_matrices):
  """
  Serializes LoRA updates to a binary file for integration with ONNX.

  This method writes LoRA parameters (A, B, and alpha matrices) to a binary
  file in a format that can be procesed by other code (e.g. written in C#)
  to fold these LoRA updates into the base pretrained model (e.g. in ONNX). 

  Each layer's parameters are written with its name, dimensions, and data 
  in a compact binary representation.
  """
  with open(filename, "wb") as f:
    # Write the number of layers
    f.write(struct.pack("I", len(lora_matrices)))

    for layer_name, matrices in lora_matrices.items():
      A, B, alpha = matrices["A"], matrices["B"], matrices["alpha"]

      # Write layer name length and name
      encoded_name = layer_name.encode("utf-8")
      f.write(struct.pack("I", len(encoded_name)))
      f.write(encoded_name)

      # Write alpha as a single float
      f.write(struct.pack("f", alpha[0]))

      # Write dimensions of matrix A and data
      rows_A, cols_A = len(A), len(A[0])
      f.write(struct.pack("II", rows_A, cols_A))
      for row in A:
        f.write(struct.pack(f"{len(row)}f", *row))

      # Write dimensions of matrix B and data
      rows_B, cols_B = len(B), len(B[0])
      f.write(struct.pack("II", rows_B, cols_B))
      for row in B:
        f.write(struct.pack(f"{len(row)}f", *row))

  print(f"Serialized LoRA matrices to {filename} count {len(lora_matrices.items())}")



def collect_and_save_lora_parameters(model, output_file : str):
  """
  Collects LoRA parameters from the model and saves them to a binary file.

  Parameters:
    model: The model containing LoRA parameters.
    output_file: The file path to save the serialized LoRA parameters.
  """
  lora_matrices = {}

  # Iterate over all parameters in the model
  for name, param in model.named_parameters():
    # Match lora_A_* to extract the layer name
    if "lora_A" in name:
      A = param.detach().cpu().numpy()  # Convert to NumPy array

      # Extract the base layer name by removing the ".lora_A" suffix.
      layer_name = re.sub(r"\.lora_A$", "", name)

      # Then find corresponding lora_B and lora_alpha parameters.
      lora_B_name = f"{layer_name}.lora_B"
      lora_alpha_name = f"{layer_name}.lora_alpha"
           
      if lora_B_name in dict(model.named_parameters()) and lora_alpha_name in dict(model.named_parameters()):
        B = dict(model.named_parameters())[lora_B_name].detach().cpu().numpy()
        alpha = dict(model.named_parameters())[lora_alpha_name].detach().cpu().numpy()

        # Ensure alpha is wrapped in a list for compatibility with serialize_lora_to_binary
        alpha = [float(alpha)] if alpha.size == 1 else alpha.tolist()

        # Add to the dictionary
        lora_matrices[layer_name] = {
            "alpha": alpha,
            "A": A.tolist(),
            "B": B.tolist(),
        }
      else:
        raise ValueError(f"Missing lora_B or lora_alpha for layer {layer_name}")

  # Serialize the collected LoRA parameters
  serialize_lora_to_binary(output_file, lora_matrices)
# print(f"LoRA parameters saved to {output_file}")
