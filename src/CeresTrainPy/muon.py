import os
import math

import torch
import torch.distributed as dist
from torch import Tensor

# Muon optimizer invented by Keller Jordan (https://github.com/KellerJordan/Muon)

# Enhanced implementation from Moonlight AI (https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py)

# NOTE: See bottom of this file for approximate changes needed to train.py

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile
def zeropower_via_newtonschulz5_batched(G, steps):
    """
    Batched variant of zeropower_via_newtonschulz5 for a stack of equally-shaped
    blocks [nb, r, c] (per-head Muon). Each block is normalized and orthogonalized
    independently — identical math to running the 2-D version per block, but one
    batched matmul chain instead of nb kernel launches.
    """
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_lr=None,
        head_split_specs=None,
        lr_ratios=None,
    ):
        # adamw_lr: separate learning rate for the internal-AdamW group (heads, embeddings,
        # norms, biases). The docstring always advertised it but it was never implemented -
        # both groups silently shared `lr`, i.e. one knob for two optimizers with different
        # natural scales. Stored as a RATIO so the external LR scheduler (which rescales
        # group['lr']) keeps the two rates proportional through warmup/decay.
        # None (default) = legacy behavior, adamw group uses `lr` unchanged.
        adamw_lr_ratio = (adamw_lr / lr) if adamw_lr is not None else 1.0

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_lr_ratio=adamw_lr_ratio,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

        # Per-head Muon (Kimi K3-style): head_split_specs maps param -> (axis, nb).
        # The gradient of such a param is split into nb equal blocks along `axis`
        # (0 = row blocks, 1 = column blocks) and each block is orthogonalized
        # INDEPENDENTLY, so heads stop sharing one global orthogonalization.
        # Momentum stays full-matrix; only the NS post-processing is blockwise.
        # LR scaling uses the BLOCK shape, so blocks get the spectral scaling
        # appropriate to their own dimensions.
        self.head_split_specs = dict(head_split_specs) if head_split_specs else {}
        for p, spec in self.head_split_specs.items():
            assert self.state[p].get("use_muon", False), "head_split only valid for Muon params"
            axis, nb = spec
            assert p.ndim == 2 and axis in (0, 1) and p.shape[axis] % nb == 0, \
                f"bad head_split {spec} for param shape {tuple(p.shape)}"
            self.state[p]["head_split"] = (int(axis), int(nb))

        # Per-PARAM LR ratios (split-LR program 2026-08-20): {param: ratio},
        # applied multiplicatively on top of the group lr in BOTH branches —
        # so a "head family" or "couplings family" can run at e.g. base/3
        # regardless of whether its members landed in the Muon or the AdamW
        # partition (the old adamw_lr knob could only throttle the whole
        # internal-AdamW group, which also contains embeddings/norms/taus).
        # Kept OUT of self.state (head_split_specs precedent) so ratios are
        # construction-time config, never resurrected from a stale checkpoint.
        # Ratios are constants => the external LambdaLR scheduler keeps every
        # family proportional through warmup/decay automatically.
        self._lr_ratios = dict(lr_ratios) if lr_ratios else {}
        for p, r in self._lr_ratios.items():
            assert r > 0, f"lr ratio must be positive, got {r}"

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                lr_p = lr * self._lr_ratios.get(p, 1.0)
                head_split = state.get("head_split")
                if head_split is not None:
                    # Per-head Muon: orthogonalize each head's block independently.
                    axis, nb = head_split
                    rows, cols = g.shape
                    if axis == 0:
                        G3 = g.view(nb, rows // nb, cols)
                    else:
                        G3 = g.view(rows, nb, cols // nb).transpose(0, 1)
                    U3 = zeropower_via_newtonschulz5_batched(G3, steps=group["ns_steps"])
                    if axis == 0:
                        u = U3.reshape(rows, cols)
                        block_shape = (rows // nb, cols)
                    else:
                        u = U3.transpose(0, 1).reshape(rows, cols)
                        block_shape = (rows, cols // nb)
                    adjusted_lr = self.adjust_lr_for_muon(lr_p, block_shape)
                else:
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                    # scale update
                    adjusted_lr = self.adjust_lr_for_muon(lr_p, p.shape)

                # apply weight decay (family-scaled lr => decay stays
                # proportional to the actual step size, matching AdamW branch)
                p.data.mul_(1 - lr_p * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            # Split-LR: heads/embeddings/norms/biases run at their own (typically much
            # lower) rate; ratio-based so schedules stay proportional. 1.0 = legacy.
            lr = group['lr'] * group.get("adamw_lr_ratio", 1.0)
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                lr_p = lr * self._lr_ratios.get(p, 1.0)
                p.data.mul_(1 - lr_p * weight_decay)
                p.data.add_(g, alpha=-lr_p / scale)

        return loss
