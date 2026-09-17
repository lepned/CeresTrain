"""The minimax readout: per-move value and value-after-reply folded into the policy logit.

This is the architectural arm, so the tests that matter are (a) it is EXACTLY the old net
at initialisation, and (b) the two alphas actually carry gradient, i.e. the readout can
earn its way in rather than sitting inert.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
from move_tokens import MoveTokenDecoder, move_token_minimax_loss

S_DIM, DM, M = 64, 32, 12


def _decoder(minimax):
    torch.manual_seed(7)
    return MoveTokenDecoder(s_dim=S_DIM, norm_type='RMSNorm', dm=DM, layers=2, heads=4,
                            ffn_mult=2, max_tokens=M, minimax=minimax)


def identity_at_init():
    """alpha is zero-init, so the readout must not move a single logit at step 0."""
    torch.manual_seed(11)
    B = 3
    squares = torch.randn(B, 64, 13)
    flow = torch.randn(B, 64, S_DIM)

    a, b = _decoder(False), _decoder(True)
    # Copy every shared parameter so the ONLY difference is the readout.
    sd_a = a.state_dict()
    missing = [k for k in b.state_dict() if k not in sd_a]
    assert set(missing) == {'mm_v.weight', 'mm_r.weight', 'mm_alpha_v', 'mm_alpha_r'}, missing
    b.load_state_dict(sd_a, strict=False)

    a.eval(); b.eval()
    with torch.no_grad():
        pa = a(squares, flow)
        pb = b(squares, flow)
    ncmp = 0
    for x, y in zip(pa if isinstance(pa, tuple) else (pa,), pb if isinstance(pb, tuple) else (pb,)):
        if not (torch.is_tensor(x) and torch.is_tensor(y)):
            continue
        if x.dtype == torch.bool:
            assert torch.equal(x, y), 'a boolean output differs at init'
        else:
            d = (x - y).abs().max().item()
            assert d == 0.0, f'readout changed the output at init by {d:.3e}'
        ncmp += 1
    assert ncmp > 0, 'nothing was compared'
    print(f'  identity at init OK: {ncmp} outputs bit-identical with zero-init alpha')


def alphas_learn():
    """The readout is useless if the mixing weights cannot move off zero."""
    d = _decoder(True)
    d.train()
    torch.manual_seed(12)
    squares = torch.randn(2, 64, 13)
    flow = torch.randn(2, 64, S_DIM)
    out = d(squares, flow)
    pol = out[0] if isinstance(out, tuple) else out
    pol.sum().backward()
    for name in ('mm_alpha_v', 'mm_alpha_r'):
        g = getattr(d, name).grad
        assert g is not None and torch.isfinite(g) and g.abs() > 0, f'{name} got no gradient ({g})'
    # The value heads get NOTHING from the policy path at init, by construction:
    # d(alpha * v)/dv = alpha = 0. They are trained by the supervision loss instead, and
    # alpha separately learns how much the logit should listen. That decoupling is the
    # point -- the heads can become accurate before they are allowed to influence ranking.
    assert d.mm_v.weight.grad is None or d.mm_v.weight.grad.abs().sum() == 0,         'value head got policy gradient at alpha=0 — the readout is not warm-started'
    print(f'  alphas learn OK: d(loss)/d(alpha_v)={d.mm_alpha_v.grad.item():+.4f}, '
          f'alpha_r={d.mm_alpha_r.grad.item():+.4f}; value heads correctly get NO policy '
          f'gradient while alpha is 0 (supervision trains them)')


def supervision():
    """The loss must weight by visits and must not train the reply head on absent replies."""
    B, MM, SL = 4, 6, 10
    v = torch.zeros(B, MM, requires_grad=True)
    r = torch.zeros(B, MM, requires_grad=True)
    sel = torch.arange(MM).unsqueeze(0).expand(B, MM).contiguous()
    valid = torch.ones(B, MM, dtype=torch.bool)
    mvp = torch.arange(1858) % 4096

    ci = torch.full((B, SL), -1, dtype=torch.int64)
    cq = torch.zeros(B, SL)
    cn = torch.zeros(B, SL, dtype=torch.int64)
    crq = torch.full((B, SL), -2.0)
    for k in range(MM):
        ci[:, k] = k
        cn[:, k] = 100 - 15 * k
        cq[:, k] = 0.6 - 0.2 * k
    crq[:, 0] = -0.4                       # only the first child has a recorded reply

    loss, diag = move_token_minimax_loss(v, r, sel, valid, mvp, (ci, cq, cn, crq))
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    # Reply head: gradient only where a reply exists.
    assert r.grad[:, 0].abs().max() > 0, 'reply head got no gradient where a reply exists'
    assert r.grad[:, 1:].abs().max() == 0, 'reply head trained on an ABSENT reply (-2 sentinel)'
    # Value head: every visited child contributes, most-visited most strongly.
    gv = v.grad.abs()[0]
    assert (gv[:MM] > 0).all()
    assert gv[0] > gv[MM - 1], f'visit weighting not applied ({gv[0]:.4f} vs {gv[MM-1]:.4f})'
    print(f'  supervision OK: reply head trains only on recorded replies '
          f'({int(diag["mt_mm_rows_r"].item() * B)}/{B} rows), value grad on n=100 is '
          f'{(gv[0] / gv[MM - 1]).item():.1f}x that on n=25')


if __name__ == '__main__':
    identity_at_init()
    alphas_learn()
    supervision()
    print('ALL OK')
