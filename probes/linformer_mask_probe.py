"""Numerical probe: can Linformer masking be done on the n x k score matrix?"""

import itertools
import math

import torch

torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)

DT = torch.float64
n, k, d = 16, 4, 8
SQ = math.sqrt(d)

m = torch.tensor([1.0] * 11 + [0.0] * 5, dtype=DT)
pad = m == 0
val = m == 1

Q0 = torch.randn(n, d, dtype=DT)
K0 = torch.randn(n, d, dtype=DT)
V0 = torch.randn(n, d, dtype=DT)
E0 = torch.randn(k, n, dtype=DT) / math.sqrt(n)
F0 = torch.randn(k, n, dtype=DT) / math.sqrt(n)

# large perturbation applied to padded rows
PERT_K = 1e3 * torch.randn(n, d, dtype=DT)
PERT_V = 1e3 * torch.randn(n, d, dtype=DT)


def leaves():
    return (
        Q0.clone().requires_grad_(True),
        K0.clone().requires_grad_(True),
        V0.clone().requires_grad_(True),
        E0.clone().requires_grad_(True),
        F0.clone().requires_grad_(True),
    )


def perturb(K, V):
    Kp = torch.where(pad[:, None], PERT_K, K)
    Vp = torch.where(pad[:, None], PERT_V, V)
    return Kp, Vp


def linformer(Q, K, V, E, F, mask=None, colmask=None, prezero=True):
    """mask: per-token 0/1 vector applied pre-projection when prezero."""
    if prezero and mask is not None:
        K = K * mask[:, None]
        V = V * mask[:, None]
    Kp, Vp = E @ K, F @ V
    S = Q @ Kp.T / SQ
    if colmask is not None:
        S = S.masked_fill(colmask[None, :], float("-inf"))
    P = torch.softmax(S, dim=-1)
    return P @ Vp, P, Kp, Vp


def std_attn(Q, K, V, keymask=None, causal=False):
    S = Q @ K.T / SQ
    if keymask is not None:
        S = S.masked_fill(~keymask[None, :], float("-inf"))
    if causal:
        S = S.masked_fill(
            torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf")
        )
    return torch.softmax(S, dim=-1) @ V


def gn(t):
    return float(t.norm()) if t is not None else float("nan")


def hdr(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# ---------------------------------------------------------------- T1
hdr("T1  pre-projection zeroing (Km = K*m, Vm = V*m)")
Q, K, V, E, F = leaves()
out1, _, _, _ = linformer(Q, K, V, E, F, mask=m)
Kp_, Vp_ = perturb(K.detach(), V.detach())
out1p, _, _, _ = linformer(Q.detach(), Kp_, Vp_, E.detach(), F.detach(), mask=m)
print(
    f"(a) max|out - out_perturbed|                = {float((out1 - out1p).abs().max()):.6e}"
)
print(f"    control max|out| (same code path)      = {float(out1.abs().max()):.6e}")
print(
    f"    perturbation magnitude max|dK_pad|     = {float((PERT_K[pad] - K0[pad]).abs().max()):.6e}"
)

out1.sum().backward()
print(
    f"(b) ||grad K[pad]||   = {gn(K.grad[pad]):.6e}   control ||grad K[val]||   = {gn(K.grad[val]):.6e}"
)
print(
    f"    ||grad V[pad]||   = {gn(V.grad[pad]):.6e}   control ||grad V[val]||   = {gn(V.grad[val]):.6e}"
)
print(
    f"    ||grad E[:,pad]|| = {gn(E.grad[:, pad]):.6e}   control ||grad E[:,val]|| = {gn(E.grad[:, val]):.6e}"
)
print(
    f"    ||grad F[:,pad]|| = {gn(F.grad[:, pad]):.6e}   control ||grad F[:,val]|| = {gn(F.grad[:, val]):.6e}"
)
print(f"    ||grad Q||        = {gn(Q.grad):.6e}  (control, whole tensor)")
T1_KG, T1_VG = K.grad.clone(), V.grad.clone()

# ---------------------------------------------------------------- T2
hdr("T2  masking columns of the n x k score matrix, NO pre-projection zeroing")
subsets = [s for r in range(1, k) for s in itertools.combinations(range(k), r)]
print(f"non-empty proper subsets of {k} columns: {len(subsets)}")
print(
    f"{'subset':>14} {'max|out-out_pert|':>20} {'||grad K[pad]||':>18} {'||grad K[val]||':>18}"
)
rows = []
for s in subsets:
    cm = torch.zeros(k, dtype=torch.bool)
    cm[list(s)] = True
    Q, K, V, E, F = leaves()
    o, _, _, _ = linformer(Q, K, V, E, F, colmask=cm, prezero=False)
    Kp_, Vp_ = perturb(K.detach(), V.detach())
    op, _, _, _ = linformer(
        Q.detach(), Kp_, Vp_, E.detach(), F.detach(), colmask=cm, prezero=False
    )
    o.sum().backward()
    dmax = float((o - op).abs().max())
    rows.append(dmax)
    print(f"{s!s:>14} {dmax:>20.6e} {gn(K.grad[pad]):>18.6e} {gn(K.grad[val]):>18.6e}")
print(f"min over subsets of max|out-out_pert|      = {min(rows):.6e}")
print(f"(last-column-only subset (3,) is row {subsets.index((3,))})")

# ---------------------------------------------------------------- T3
hdr("T3  projected-up picture: Atilde = Pbar F, zero padded COLUMNS of Atilde")
Q, K, V, E, F = leaves()
_, Pbar, _, _ = linformer(Q, K, V, E, F, prezero=False)
At = Pbar @ F  # n x n, never formed in practice
lhs = (At * m[None, :]) @ V
rhs = Pbar @ (F @ (V * m[:, None]))
print(
    f"max|(Atilde*m) V  -  Pbar (F (V*m))|       = {float((lhs - rhs).abs().max()):.6e}"
)
print(f"control max|lhs|                           = {float(lhs.abs().max()):.6e}")
print(
    f"control max|(Atilde*m) V - Atilde V|       = {float((lhs - At @ V).abs().max()):.6e}"
)
rs_mask = (At * m[None, :]).sum(-1)
rs_full = At.sum(-1)
print(
    f"row sums of Atilde*m : min={float(rs_mask.min()):.6f} max={float(rs_mask.max()):.6f}"
)
print(f"  first 5: {[round(float(x), 6) for x in rs_mask[:5]]}")
print(
    f"row sums of Atilde   : min={float(rs_full.min()):.6f} max={float(rs_full.max()):.6f}"
)
print(f"  first 5: {[round(float(x), 6) for x in rs_full[:5]]}")
print(
    f"min entry of Atilde*m = {float((At * m[None, :]).min()):.6f}   min entry Atilde = {float(At.min()):.6f}"
)
print(
    f"count of negative entries in Atilde*m = {int((At * m[None, :] < 0).sum())} / {n * n}"
)

# ---------------------------------------------------------------- T4
hdr("T4  masked Linformer (T1) vs standard key-masked attention")
so = std_attn(Q0, K0, V0, keymask=val)
print(
    f"max|std_masked - linformer_masked|         = {float((so - out1.detach()).abs().max()):.6e}"
)
su = std_attn(Q0, K0, V0)
lu, _, _, _ = linformer(Q0, K0, V0, E0, F0, prezero=False)
print(
    f"max|std_unmasked - linformer_unmasked|     = {float((su - lu).abs().max()):.6e}  (scale ref)"
)
print(
    f"max|std_masked|={float(so.abs().max()):.6e}  max|linformer_masked|={float(out1.abs().max()):.6e}"
)

# ---------------------------------------------------------------- T5
hdr("T5  gradient surgery via backward hook")
Q, K, V, E, F = leaves()
K.register_hook(lambda g: g * m[:, None])
V.register_hook(lambda g: g * m[:, None])
o, _, _, _ = linformer(Q, K, V, E, F, mask=m)  # forward still pre-zeroed
o.sum().backward()
print(
    f"max|grad K (hook+prezero) - grad K (T1)|   = {float((K.grad - T1_KG).abs().max()):.6e}"
)
print(
    f"max|grad V (hook+prezero) - grad V (T1)|   = {float((V.grad - T1_VG).abs().max()):.6e}"
)
print(f"control ||grad K (T1)[val]||               = {gn(T1_KG[val]):.6e}")

Q, K, V, E, F = leaves()
K.register_hook(lambda g: g * m[:, None])
V.register_hook(lambda g: g * m[:, None])
o2, _, _, _ = linformer(Q, K, V, E, F, prezero=False)  # forward NOT zeroed
Kp_, Vp_ = perturb(K.detach(), V.detach())
o2p, _, _, _ = linformer(Q.detach(), Kp_, Vp_, E.detach(), F.detach(), prezero=False)
o2.sum().backward()
print(
    f"hook-only: max|out - out_perturbed|        = {float((o2 - o2p).abs().max()):.6e}"
)
print(f"hook-only: ||grad K[pad]|| (zeroed by hook)= {gn(K.grad[pad]):.6e}")
print(f"hook-only: ||grad K[val]|| (control)       = {gn(K.grad[val]):.6e}")

# ---------------------------------------------------------------- T6
hdr("T6  scale dependence of ||K'_j|| with number of valid tokens")


def kprime_norm(mask, renorm):
    Em = E0 * mask[None, :]
    if renorm:
        Em = Em / Em.sum(-1, keepdim=True)
    Kp_ = Em @ (K0 * mask[:, None])
    return float(Kp_.norm(dim=-1).mean()), Kp_


for nv in (16, 11, 3):
    mm = torch.tensor([1.0] * nv + [0.0] * (n - nv), dtype=DT)
    a, _ = kprime_norm(mm, False)
    b, _ = kprime_norm(mm, True)
    print(
        f"valid={nv:>2}  mean_j ||K'_j|| plain = {a:.6f}   row-renormalized E = {b:.6f}"
    )
mm11 = torch.tensor([1.0] * 11 + [0.0] * 5, dtype=DT)
mm16 = torch.ones(n, dtype=DT)
mm3 = torch.tensor([1.0] * 3 + [0.0] * 13, dtype=DT)
p16, _ = kprime_norm(mm16, False)
p11, _ = kprime_norm(mm11, False)
p3, _ = kprime_norm(mm3, False)
r16, _ = kprime_norm(mm16, True)
r11, _ = kprime_norm(mm11, True)
r3, _ = kprime_norm(mm3, True)
print(f"plain ratios   11/16 = {p11 / p16:.6f}   3/16 = {p3 / p16:.6f}")
print(f"renorm ratios  11/16 = {r11 / r16:.6f}   3/16 = {r3 / r16:.6f}")

# ---------------------------------------------------------------- T7
hdr("T7  causality: query t=5 must not see original position 9")
t, fut = 5, 9
Kc = K0.clone()
Vc = V0.clone()
Kc[fut] = 1e3 * torch.randn(d, dtype=DT)
Vc[fut] = 1e3 * torch.randn(d, dtype=DT)

base, _, _, _ = linformer(Q0, K0, V0, E0, F0, prezero=False)
pert, _, _, _ = linformer(Q0, Kc, Vc, E0, F0, prezero=False)
print(
    f"plain Linformer: max|out[{t}] change|       = {float((base[t] - pert[t]).abs().max()):.6e}"
)
mins = []
print(f"{'subset':>14} {'max|out[5] change|':>22}")
for s in subsets:
    cm = torch.zeros(k, dtype=torch.bool)
    cm[list(s)] = True
    b, _, _, _ = linformer(Q0, K0, V0, E0, F0, colmask=cm, prezero=False)
    p, _, _, _ = linformer(Q0, Kc, Vc, E0, F0, colmask=cm, prezero=False)
    c = float((b[t] - p[t]).abs().max())
    mins.append(c)
    print(f"{s!s:>14} {c:>22.6e}")
print(f"min over subsets of max|out[{t}] change|     = {min(mins):.6e}")
sb = std_attn(Q0, K0, V0, causal=True)
sp = std_attn(Q0, Kc, Vc, causal=True)
print(
    f"standard causal: max|out[{t}] change|       = {float((sb[t] - sp[t]).abs().max()):.6e}  (control)"
)
print(
    f"standard causal: max|out[{fut}] change| (control, should be >0) = {float((sb[fut] - sp[fut]).abs().max()):.6e}"
)

# ---------------------------------------------------------------- T8
hdr("T8  NaN hazard")
Sfull = Q0 @ (E0 @ K0).T / SQ
Srow = Sfull.clone()
Srow[0] = float("-inf")
outn = torch.softmax(Srow, dim=-1) @ (F0 @ V0)
print(
    f"full -inf row: isnan count in out          = {int(torch.isnan(outn).sum())} / {outn.numel()}"
)
print(
    f"full -inf row: isnan count in row 0        = {int(torch.isnan(outn[0]).sum())} / {d}"
)
print(
    f"control (unmasked) isnan count             = {int(torch.isnan(torch.softmax(Sfull, -1) @ (F0 @ V0)).sum())}"
)
m0 = torch.zeros(n, dtype=DT)
o0, P0, Kp0, Vp0 = linformer(Q0, K0, V0, E0, F0, mask=m0)
print(
    f"all-padded pre-projection: max|K'| = {float(Kp0.abs().max()):.6e}  max|V'| = {float(Vp0.abs().max()):.6e}"
)
print(
    f"all-padded pre-projection: softmax row 0 = {[round(float(x), 6) for x in P0[0]]}"
)
print(
    f"all-padded pre-projection: max|out| = {float(o0.abs().max()):.6e}  isnan count = {int(torch.isnan(o0).sum())}"
)
print(f"control: max|out| with m all ones = {float(lu.abs().max()):.6e}")
