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

The stream math runs in the stream dtype (bf16 under autocast; the unrolled mixes are computed in fp32 inside the fused kernels): an fp32 upcast of the n*D
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
  """[.., n, n] logits -> doubly-stochastic matrix (fp32); rows normalized last, so rows sum to 1 exactly.

  The row/column sums are unrolled over n (slices added one by one) instead of .sum(dim): every iteration is then
  pointwise and torch.compile fuses the whole loop into a few kernels. With .sum it emitted a reduction kernel per
  normalization, ~40 forward + ~100 backward tiny kernels per block per step (11-08 profile: +2355 kernels/step at
  384x12, n=4, ~10 ms of launch overhead)."""
  K = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
  for _ in range(iters):
    K = K / _unrolled_sum(K, -2)
    K = K / _unrolled_sum(K, -1)
  return K


def _unrolled_sum(K: torch.Tensor, dim: int) -> torch.Tensor:
  """K.sum(dim, keepdim=True) as a chain of slice adds (n <= 8: pointwise for torch.compile)."""
  s = K.narrow(dim, 0, 1)
  for j in range(1, K.shape[dim]):
    s = s + K.narrow(dim, j, 1)
  return s


def res_matrices(hcs) -> list:
  """H_res of every block, one batched Sinkhorn over the stacked logits (one fused chain instead of one per block);
  modules with a baked constant (export copy) return it."""
  if any(getattr(hc, 'res_baked', None) is not None for hc in hcs):
    return [hc.res_matrix() for hc in hcs]
  return list(sinkhorn(torch.stack([hc.res_logits.float() for hc in hcs]), hcs[0].iters).unbind(0))


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
    return _combine(x.reshape(B, T, self.n, self.d), _coeff_list(alpha, self.n, x.dtype))

  def write(self, x: torch.Tensor, branch: torch.Tensor, beta: torch.Tensor, res: torch.Tensor = None) -> torch.Tensor:
    """x [B, T, n*D], branch [B, T, D] -> H + (H_res - I) H + beta (x) branch, [B, T, n*D], in the stream dtype.
    res: this block's H_res from res_matrices() (batched over the blocks); computed here when None."""
    B, T, _ = x.shape
    xv = x.reshape(B, T, self.n, self.d)
    b = _coeff_list(beta, self.n, x.dtype)
    delta = ((self.res_matrix() if res is None else res) - torch.eye(self.n, device=x.device)).to(x.dtype)
    br = branch.to(x.dtype)
    rows = [xv[..., i, :] + _combine(xv, [delta[i, j] for j in range(self.n)]) + b[i] * br for i in range(self.n)]
    return torch.stack(rows, dim=-2).reshape(B, T, self.n * self.d)


# The stream mixes are unrolled over n (<= 8) as scalar-times-slice sums instead of torch.matmul: a matmul of
# [.., 1, n] or [n, n] against [B, T, n, D] runs as B*T tiny batched GEMMs/GEMVs that torch.compile cannot fuse
# (11-08 profile at 384x12: ~26 ms of the ~40 ms per-step overhead of n=2 static); the unrolled form is pointwise and
# fuses into the neighbouring kernels (computed in fp32 inside the fused kernel, one rounding at the store).
def _coeff_list(w: torch.Tensor, n: int, dtype):
  """[n] (static) or [B, T, n] (dynamic) -> n coefficients, scalar or [B, T, 1], in dtype."""
  w = w.to(dtype)
  return [w[i] for i in range(n)] if w.dim() == 1 else [w[..., i:i + 1] for i in range(n)]


def _combine(xv: torch.Tensor, coeffs) -> torch.Tensor:
  """xv [B, T, n, D], n coefficients -> sum_i c_i xv[..., i, :], [B, T, D]."""
  out = coeffs[0] * xv[..., 0, :]
  for i in range(1, len(coeffs)):
    out = out + coeffs[i] * xv[..., i, :]
  return out


class HyperConnReadout(nn.Module):
  """Trunk end / depth states: a learned combination of the streams, init = the mean (exact for n a power of 2)."""
  def __init__(self, n: int, model_dim: int):
    super().__init__()
    self.n, self.d = n, model_dim
    self.alpha = nn.Parameter(torch.full((n,), 1.0 / n))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, _ = x.shape
    return _combine(x.reshape(B, T, self.n, self.d), _coeff_list(self.alpha, self.n, x.dtype))
