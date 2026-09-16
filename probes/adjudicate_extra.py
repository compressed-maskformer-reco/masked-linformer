"""Adjudicator re-measurement: torch ports of the math report's unreproducible numpy claims,
plus the gaps none of the three reports measured."""

import itertools
import math

import torch

torch.manual_seed(0)
DT = torch.float64
SM = lambda x: torch.softmax(x, -1)
H = lambda P: float(-(P * (P + 1e-300).log()).sum(-1).mean())


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ---------------------------------------------------------------- A  C1 dense leak (was lin2.py)
hdr(
    "A  C1: best achievable |Atilde[:,pad]| over ALL column subsets, dense E,F, 200 draws"
)
n, k, d = 12, 4, 6
leaks = []
for s in range(200):
    g = torch.Generator().manual_seed(s)
    r = lambda *sh, g=g: torch.randn(*sh, generator=g, dtype=DT)
    Q, K, E, F = r(n, d), r(n, d), r(k, n), r(k, n)
    m = torch.ones(n, dtype=DT)
    m[8:] = 0
    S0 = Q @ (E @ K).T / math.sqrt(d)
    best = float("inf")
    for rr in range(k):  # proper subsets only (full set -> NaN)
        for M in itertools.combinations(range(k), rr):
            S = S0.clone()
            S[:, list(M)] = float("-inf")
            best = min(best, float((SM(S) @ F)[:, m == 0].abs().max()))
    leaks.append(best)
lk = torch.tensor(leaks)
print(
    f"dense E,F : min={lk.min():.3f} median={lk.median():.3f} max={lk.max():.3f}  (0 would mean maskable)"
)
print(
    f"control   : unmasked leak median={lk.median():.3f} -> nonzero everywhere, instrument live"
)

# block-structured E=F (mean pool, w=3): the refutation case
Fb = torch.zeros(k, n, dtype=DT)
w = n // k
for j in range(k):
    Fb[j, j * w : (j + 1) * w] = 1.0 / w
g = torch.Generator().manual_seed(7)
Q, K = (
    torch.randn(n, d, generator=g, dtype=DT),
    torch.randn(n, d, generator=g, dtype=DT),
)
S0 = Q @ (Fb @ K).T / math.sqrt(d)
for L, kill in ((6, [2, 3]), (9, [3]), (7, [2, 3]), (7, [3])):
    S = S0.clone()
    S[:, kill] = float("-inf")
    A = SM(S) @ Fb
    mm = torch.zeros(n, dtype=DT)
    mm[:L] = 1
    print(
        f"block E,F L={L} kill={kill}: leak onto pads={float(A[:, mm == 0].abs().max()):.3f}  "
        f"weight lost from VALID cols={float((1 - A[:, mm == 1].sum(1)).max()):.3f}"
    )

# ---------------------------------------------------------------- C  C6 causal-safe columns
hdr("C  C6: dense F -> projected keys with support inside the prefix [0..t]")
g = torch.Generator().manual_seed(3)
F = torch.randn(k, n, generator=g, dtype=DT)
safe = [int(((F[:, t + 1 :].abs() < 1e-12).all(1)).sum()) for t in range(n - 1)]
Fpre = torch.zeros(k, n, dtype=DT)  # positive control: prefix-supported F
for j in range(k):
    Fpre[j, : (j + 1) * (n // k)] = 1.0
safe_pre = [int(((Fpre[:, t + 1 :].abs() < 1e-12).all(1)).sum()) for t in range(n - 1)]
print(f"dense F  #causal-safe cols per query t: {safe}")
print(
    f"prefix F #causal-safe cols per query t: {safe_pre}   (positive control, instrument live)"
)

# ---------------------------------------------------------------- D  absolute-position sensitivity
hdr("D  same content at a different offset (left- vs right-aligned padding)")
n, k, d, L = 12, 4, 6, 8
g = torch.Generator().manual_seed(11)
Kv, Vv, Qv = (torch.randn(L, d, generator=g, dtype=DT) for _ in range(3))
E = torch.randn(k, n, generator=g, dtype=DT).abs()
F = torch.randn(k, n, generator=g, dtype=DT).abs()


def run(off):
    Z = lambda: torch.zeros(n, d, dtype=DT)
    Q, K, V = Z(), Z(), Z()
    Q[off : off + L], K[off : off + L], V[off : off + L] = Qv, Kv, Vv
    P = SM(Q @ (E @ K).T / math.sqrt(d))
    return (P @ (F @ V))[off : off + L]


a, b = run(0), run(4)
print(
    f"max|out(offset 0) - out(offset 4)| = {float((a - b).abs().max()):.3f}   "
    f"scale max|out| = {float(a.abs().max()):.3f}"
)
print(f"control max|out(0) - out(0)| = {float((run(0) - run(0)).abs().max()):.3e}")

# ---------------------------------------------------------------- E/F/G/H/I/J  torch setup
n, k, d = 16, 4, 8
SQ = math.sqrt(d)
g = torch.Generator().manual_seed(0)
R = lambda *sh: torch.randn(*sh, generator=g, dtype=DT)
Q0, K0, V0 = R(n, d), R(n, d), R(n, d)
E0, F0 = R(k, n) / math.sqrt(n), R(k, n) / math.sqrt(n)
m = torch.tensor([1.0] * 11 + [0.0] * 5, dtype=DT)
pad = m == 0
PK, PV = 1e3 * R(n, d), 1e3 * R(n, d)


def lin(Q, K, V, E, F, mk=None, mv=None):
    if mk is not None:
        K = K * mk[:, None]
    if mv is not None:
        V = V * mv[:, None]
    return SM(Q @ (E @ K).T / SQ) @ (F @ V)


hdr("E  which of K / V must be masked pre-projection?")
Kp_, Vp_ = torch.where(pad[:, None], PK, K0), torch.where(pad[:, None], PV, V0)
for name, mk, mv in (
    ("neither", None, None),
    ("V only", None, m),
    ("K only", m, None),
    ("both", m, m),
):
    o = lin(Q0, K0, V0, E0, F0, mk, mv)
    op = lin(Q0, Kp_, Vp_, E0, F0, mk, mv)
    print(f"{name:>8}: max|out - out_perturbed| = {float((o - op).abs().max()):.6e}")

hdr("F  pad rows holding NaN/inf: k*mask vs torch.where, forward AND backward")
Kbad = K0.clone()
Kbad[pad] = float("nan")
Kbad[12] = float("inf")
for name, fn in (
    ("K * m      ", lambda K: K * m[:, None]),
    (
        "where(m,K,0)",
        lambda K: torch.where(m[:, None].bool(), K, torch.zeros((), dtype=DT)),
    ),
):
    K = Kbad.clone().requires_grad_(True)
    out = SM(Q0 @ (E0 @ fn(K)).T / SQ) @ (F0 @ (V0 * m[:, None]))
    out.sum().backward()
    print(
        f"{name}: nan in out = {int(out.isnan().sum())}/{out.numel()}   "
        f"nan in grad K = {int(K.grad.isnan().sum())}/{K.grad.numel()}   "
        f"||grad K[valid]|| = {float(K.grad[~pad].norm()):.3e}"
    )

hdr("G  fairseq E[:, :tgt_len] slicing  ==  zero-masking the trailing pad?")
L = 11
sl = SM(Q0[:L] @ (E0[:, :L] @ K0[:L]).T / SQ) @ (F0[:, :L] @ V0[:L])
mk = lin(Q0, K0, V0, E0, F0, m, m)[:L]
print(
    f"max|sliced - masked| = {float((sl - mk).abs().max()):.3e}   control max|sliced| = {float(sl.abs().max()):.3f}"
)
print(
    f"control (slice vs UNmasked full) = {float((sl - lin(Q0, K0, V0, E0, F0)[:L]).abs().max()):.3f}"
)

hdr("H  batched: is grad E[:, i] zero for a position padded in ONLY SOME batch rows?")
B = 3
Qb, Kb, Vb = R(B, n, d), R(B, n, d), R(B, n, d)
mb = torch.ones(B, n, dtype=DT)
mb[0, 8:] = 0
mb[1, 11:] = 0  # row 2 full length
E = E0.clone().requires_grad_(True)
F = F0.clone().requires_grad_(True)
Kb_, Vb_ = Kb * mb[..., None], Vb * mb[..., None]
Pb = SM(torch.einsum("bnd,bkd->bnk", Qb, torch.einsum("kn,bnd->bkd", E, Kb_)) / SQ)
outb = torch.einsum("bnk,bkd->bnd", Pb, torch.einsum("kn,bnd->bkd", F, Vb_))
outb.sum().backward()
never = (mb == 0).all(0)
print(
    f"positions padded in ALL rows: {int(never.sum())}  ||grad E[:, never]|| = {float(E.grad[:, never].norm()):.3e}"
)
print(
    f"positions padded in SOME rows: {int(((mb == 0).any(0) & ~never).sum())}  "
    f"||grad E[:, some]|| = {float(E.grad[:, (mb == 0).any(0) & ~never].norm()):.3e}  (NONZERO expected)"
)

hdr(
    "I  output scale drift with valid count under pre-projection masking, and a row-sum renorm"
)
for L in (16, 11, 6, 3):
    mm = torch.zeros(n, dtype=DT)
    mm[:L] = 1
    P = SM(Q0 @ (E0 @ (K0 * mm[:, None])).T / SQ)
    rows = P @ (F0 @ mm)  # = row sums of Atilde*m
    out = P @ (F0 @ (V0 * mm[:, None]))
    print(
        f"L={L:>2}: mean|out|={float(out.abs().mean()):.4f}  Atilde row-sum mean={float(rows.mean()):+.4f} "
        f"min|row-sum|={float(rows.abs().min()):.4f}  Pbar entropy={H(P):.3f} (max {math.log(k):.3f})"
    )
rs = SM(R(500, d) @ (E0 @ (K0 * m[:, None])).T / SQ) @ (F0 @ m)
print(
    f"renorm hazard: over 500 random query rows, min|Atilde row-sum| = {float(rs.abs().min()):.5f}, "
    f"sign changes present = {bool((rs.max() > 0) and (rs.min() < 0))}  -> out/rowsum is unsafe"
)
mm = torch.zeros(n, dtype=DT)
mm[:6] = 1
Pv = SM(Q0 @ (E0 @ (K0 * mm[:, None])).T / SQ)
print(
    f"block-kill cost check: with block E,F (L=7,kill=[2,3]) valid col 6 receives "
    f"Atilde[:,6] weight = {float((SM((Q0[:, :6] @ (Fb @ K)[:, :6].T if False else S0.clone()).index_fill(1, torch.tensor([2, 3]), float('-inf'))) @ Fb)[:, 6].abs().max()):.3f} (0 => a VALID token is unattendable)"
)

hdr("J  multi-head + attention dropout on Pbar: still invariant to pad content?")
Hn = 4
dh = d // Hn
Qh, Kh, Vh = R(Hn, n, dh), R(Hn, n, dh), R(Hn, n, dh)
Eh, Fh = R(Hn, k, n) / math.sqrt(n), R(Hn, k, n) / math.sqrt(n)
Khp = torch.where(pad[None, :, None], 1e3 * R(Hn, n, dh), Kh)
Vhp = torch.where(pad[None, :, None], 1e3 * R(Hn, n, dh), Vh)


def mh(K, V, drop_seed=None):
    Kp = torch.einsum("hkn,hnd->hkd", Eh, K * m[None, :, None])
    Vp = torch.einsum("hkn,hnd->hkd", Fh, V * m[None, :, None])
    P = SM(torch.einsum("hnd,hkd->hnk", Qh, Kp) / math.sqrt(dh))
    if drop_seed is not None:
        gg = torch.Generator().manual_seed(drop_seed)
        P = P * (torch.rand(P.shape, generator=gg, dtype=DT) > 0.3) / 0.7
    return torch.einsum("hnk,hkd->hnd", P, Vp)


print(
    f"no dropout : max|out - out_perturbed| = {float((mh(Kh, Vh) - mh(Khp, Vhp)).abs().max()):.3e}"
    f"   control max|out| = {float(mh(Kh, Vh).abs().max()):.3f}"
)
print(
    f"dropout 0.3: max|out - out_perturbed| = {float((mh(Kh, Vh, 1) - mh(Khp, Vhp, 1)).abs().max()):.3e}"
    f"   control max|out| = {float(mh(Kh, Vh, 1).abs().max()):.3f}"
)

hdr("K  softmax temperature vs valid count: plain vs E-row renormalization (entropy)")
n2, k2, d2 = 64, 8, 32
g2 = torch.Generator().manual_seed(5)
E2 = torch.randn(k2, n2, generator=g2, dtype=DT).abs()
Q2 = torch.randn(n2, d2, generator=g2, dtype=DT)
mu = torch.randn(d2, generator=g2, dtype=DT)
print(
    f"  L   H_plain  H_renorm   (max = {math.log(k2):.3f})   mean||K'|| plain / renorm"
)
for L in (8, 16, 32, 64):
    mm = torch.zeros(n2, dtype=DT)
    mm[:L] = 1
    K2 = mu + torch.randn(n2, d2, generator=g2, dtype=DT)
    Kp = E2 @ (mm[:, None] * K2)
    D = (E2 @ mm).clamp_min(1e-12)
    Pp, Pn = SM(Q2 @ Kp.T / math.sqrt(d2)), SM(Q2 @ (Kp / D[:, None]).T / math.sqrt(d2))
    print(
        f"{L:>3} {H(Pp):>9.3f} {H(Pn):>9.3f}          "
        f"{float(Kp.norm(dim=1).mean()):>8.3f} / {float((Kp / D[:, None]).norm(dim=1).mean()):.3f}"
    )

hdr("L  fp16: -1e9 additive mask instead of -inf")
S = (Q0 @ (E0 @ K0).T / SQ).half()
S2 = S.clone()
S2[:, 2] = S2[:, 2] - 1e9
print(
    f"fp16 S - 1e9 -> {float(S2[0, 2])} (inf-saturated), softmax col2 max = {float(SM(S2.float())[:, 2].max()):.3e}, "
    f"nan = {int(SM(S2.float()).isnan().sum())}"
)
S3 = S.clone().float()
S3[:, 2] = -1e9
print(
    f"fp32 hard -1e9: softmax col2 max = {float(SM(S3)[:, 2].max()):.3e} (nonzero residual grad path)"
)
