import torch

torch.manual_seed(0)
n, k, d = 256, 32, 64
E = torch.empty(k, n, dtype=torch.float64)
torch.nn.init.xavier_uniform_(E)  # fairseq-style init, signed


def proj(E, K, m, mode):
    Em = E * m[None, :]
    if mode == "none":
        s = torch.ones(k, 1, dtype=E.dtype)
    if mode == "count":
        s = torch.full((k, 1), n / m.sum(), dtype=E.dtype)
    if mode == "sqrtc":
        s = torch.full((k, 1), (n / m.sum()).sqrt(), dtype=E.dtype)
    if mode == "l1":
        s = E.abs().sum(1, keepdim=True) / Em.abs().sum(1, keepdim=True).clamp_min(
            1e-12
        )
    if mode == "l2":
        s = E.norm(dim=1, keepdim=True) / Em.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if mode == "signed":
        s = E.sum(1, keepdim=True) / Em.sum(1, keepdim=True)
    return (s * Em) @ K


for corr in (0.0, 0.9):  # token correlation: iid content vs strongly shared component
    print(
        f"\n== token correlation {corr}: mean ||K'_j|| by n_valid (target = full-length value)"
    )
    base = torch.randn(1, d, dtype=torch.float64)
    K = corr * base + (1 - corr**2) ** 0.5 * torch.randn(n, d, dtype=torch.float64)
    full = proj(E, K, torch.ones(n, dtype=torch.float64), "none").norm(dim=1).mean()
    print(f"{'mode':>7} " + " ".join(f"nv={nv:>3}" for nv in (256, 128, 32, 8)))
    for mode in ("none", "count", "sqrtc", "l1", "l2", "signed"):
        row = []
        for nv in (256, 128, 32, 8):
            m = torch.zeros(n, dtype=torch.float64)
            m[:nv] = 1
            v = proj(E, K, m, mode).norm(dim=1).mean() / full
            row.append(f"{v:6.2f}")
        print(f"{mode:>7} " + " ".join(row))
m = torch.zeros(n, dtype=torch.float64)
m[:8] = 1
den = (E * m[None, :]).sum(1)
print(
    f"\nsigned denominators at nv=8: min|sum|={den.abs().min():.2e}, max|sum|={den.abs().max():.2e}, rows with |sum|<1e-2: {(den.abs() < 1e-2).sum().item()}/{k}"
)
