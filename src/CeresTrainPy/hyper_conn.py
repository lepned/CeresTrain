"""Manifold-constrained hyper-connections on the NBT outer residual (2026-11-08, NetDef NBTHyperConn; NBT trunks only).

Hyper-Connections (Zhu et al., ICLR 2025, arXiv 2409.19606) widen the residual stream into n parallel streams of width D;
mHC (DeepSeek, arXiv 2512.24880) constrains the stream-mixing matrix to be doubly stochastic (Sinkhorn), so its products
across depth stay bounded. Per NBT block l:

    read    h_in = sum_i alpha_i H_i                          (H_pre, [n])          -> the block sees one D-wide input
    write   H'   = H_res @ H  +  beta (x) branch_l(h_in)       (H_res [n,n], H_post = beta [n])

and the trunk end (and every collected depth state) reads the mean of the streams. Static coefficients by default;
NBTHyperConnDynamic 'prepost' adds input-dependent alpha/beta (HC's dynamic form: tanh(RMSNorm(H) W) * s, W zero-init).
H_res stays static (an in-graph Sinkhorn per token would be ~4 x iters tiny kernels per block): it depends on parameters
only, and save_model bakes it into an [n, n] buffer on the export copy (bake_res_matrix_), so the graph carries a constant
whatever the exporter folds.

The stream math runs in the stream dtype (bf16 under autocast; matmul accumulates in fp32): an fp32 upcast of the n*D
stream would be saved for backward in every block (review 11-08: ~54 GB at 1920/3, B=1024). The write is computed as
H + (H_res - I) H + beta (x) branch: at init the rows of H are identical and the rows of H_res - I sum to ~0, so the
correction is far below half an ulp of H and the plain residual is reproduced bit for bit.

Step-0 identity for ANY trunk weights: the n streams start as copies of the embedding; block l reads stream l % n
(one-hot alpha), writes its branch to every stream (beta = 1), and a doubly-stochastic H_res maps identical rows to
themselves (rows sum to 1), so all streams stay equal to the plain residual and the mean readout returns it (exact for
n a power of 2). The one-hot read is what breaks the symmetry from step 1 (stream i is read only by blocks l % n == i, so
d loss / d beta_i differs across i); a uniform alpha would keep the streams identical forever.
"""
import torch
import torch.nn as nn

from rms_norm import make_norm

RES_INIT_LOGIT = 5.0   # H_res = Sinkhorn(5 I): ~0.98 on the diagonal at init


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
  """[.., n, n] logits -> doubly-stochastic matrix (fp32); rows normalized last, so rows sum to 1 exactly."""
  K = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
  for _ in range(iters):
    K = K / K.sum(dim=-2, keepdim=True)
    K = K / K.sum(dim=-1, keepdim=True)
  return K


class HyperConnection(nn.Module):
  def __init__(self, n: int, model_dim: int, block_index: int, dynamic: str, sinkhorn_iters: int,
               norm_type: str, eps: float = 1e-6, dyn_scale_init: float = 0.01):
    super().__init__()
    assert dynamic in ('none', 'prepost'), dynamic
    self.n, self.d, self.iters = n, model_dim, sinkhorn_iters
    self.dynamic = dynamic == 'prepost'
    self.alpha = nn.Parameter(nn.functional.one_hot(torch.tensor(block_index % n), n).float())   # H_pre
    self.beta = nn.Parameter(torch.ones(n))                                                          # H_post
    self.res_logits = nn.Parameter(RES_INIT_LOGIT * torch.eye(n))                                    # H_res logits
    if self.dynamic:
      self.dyn_norm = make_norm(norm_type, n * model_dim, eps=eps)
      self.dyn_w = nn.Parameter(torch.zeros(n * model_dim, 2 * n))
      self.dyn_scale = nn.Parameter(torch.full((2,), dyn_scale_init))

  def res_matrix(self) -> torch.Tensor:
    baked = getattr(self, 'res_baked', None)
    return baked if baked is not None else sinkhorn(self.res_logits.float(), self.iters)

  @torch.no_grad()
  def bake_res_matrix_(self) -> bool:
    """Export copy only: H_res as a constant buffer (no Sinkhorn subgraph in the exported graph)."""
    self.register_buffer('res_baked', sinkhorn(self.res_logits.float(), self.iters).clone(), persistent=False)
    return True

  def coeffs(self, x: torch.Tensor):
    """x [B, T, n*D] -> (alpha, beta), each [n] (static) or [B, T, n] (dynamic), fp32."""
    if not self.dynamic:
      return self.alpha, self.beta
    d = torch.tanh(self.dyn_norm(x) @ self.dyn_w.to(x.dtype))
    s = self.dyn_scale.to(d.dtype)
    return self.alpha.to(d.dtype) + s[0] * d[..., :self.n], self.beta.to(d.dtype) + s[1] * d[..., self.n:]

  def read(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """x [B, T, n*D] -> [B, T, D]: the combination of the streams the block reads."""
    B, T, _ = x.shape
    xv = x.reshape(B, T, self.n, self.d)
    a = (alpha.reshape(1, 1, 1, self.n) if alpha.dim() == 1 else alpha.unsqueeze(-2)).to(x.dtype)
    return torch.matmul(a, xv).reshape(B, T, self.d)

  def write(self, x: torch.Tensor, branch: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """x [B, T, n*D], branch [B, T, D] -> H + (H_res - I) H + beta (x) branch, [B, T, n*D], in the stream dtype."""
    B, T, _ = x.shape
    xv = x.reshape(B, T, self.n, self.d)
    b = (beta.reshape(1, 1, self.n, 1) if beta.dim() == 1 else beta.unsqueeze(-1)).to(x.dtype)
    delta = (self.res_matrix() - torch.eye(self.n, device=x.device)).to(x.dtype)
    out = xv + torch.matmul(delta, xv) + b * branch.to(x.dtype).unsqueeze(-2)
    return out.reshape(B, T, self.n * self.d)


class HyperConnReadout(nn.Module):
  """Trunk end / depth states: a learned combination of the streams, init = the mean (exact for n a power of 2)."""
  def __init__(self, n: int, model_dim: int):
    super().__init__()
    self.n, self.d = n, model_dim
    self.alpha = nn.Parameter(torch.full((n,), 1.0 / n))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, _ = x.shape
    xv = x.reshape(B, T, self.n, self.d)
    return torch.matmul(self.alpha.to(x.dtype).reshape(1, 1, 1, self.n), xv).reshape(B, T, self.d)
