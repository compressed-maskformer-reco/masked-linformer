import pytest
import torch

from masked_linformer import LinformerLM, LinformerSelfAttention, attend, project

DIM, SEQ, K, HEADS = 32, 64, 8, 4


def attn(renorm=True, seed=0, **kw):
    torch.manual_seed(seed)
    return LinformerSelfAttention(
        DIM, SEQ, k=K, heads=HEADS, renorm=renorm, **kw
    ).double()


def batch(lengths, n=SEQ, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(len(lengths), n, DIM, dtype=torch.float64)
    mask = torch.arange(n)[None] < torch.tensor(lengths)[:, None]
    return x, mask


def garbage_at_pad(x, mask, value=None):
    g = torch.randn_like(x) * 1e3 if value is None else torch.full_like(x, value)
    return torch.where(mask[..., None], x, g)


@pytest.mark.parametrize("renorm", [False, True])
def test_valid_rows_independent_of_padded_content(renorm):
    a = attn(renorm)
    x, mask = batch([11, 5, 40])
    out = a(x, mask=mask)
    for g in (
        garbage_at_pad(x, mask),
        garbage_at_pad(x, mask, float("nan")),
        garbage_at_pad(x, mask, float("inf")),
    ):
        out_g = a(g, mask=mask)
        assert torch.equal(out_g[mask], out[mask])
        assert not out_g.isnan().any()
    assert torch.equal(out[~mask], torch.zeros_like(out[~mask]))
    # control: the same perturbation on one valid row does move the output
    x2 = x.clone()
    x2[0, 3] += 1e3
    assert (a(x2, mask=mask)[mask] - out[mask]).abs().max() > 1e-3


def test_padded_positions_receive_zero_gradient():
    a = attn()
    x, mask = batch([11, 5, 40])
    x.requires_grad_(True)
    a(x, mask=mask).square().sum().backward()
    assert torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask]))
    assert x.grad[mask].norm() > 1.0  # instrument is live


@pytest.mark.parametrize("renorm", [False, True])
def test_output_independent_of_batch_padding_length(renorm):
    a = attn(renorm)
    x64, mask64 = batch([10, 20])
    x24, mask24 = x64[:, :24], mask64[:, :24]
    out64, out24 = a(x64, mask=mask64), a(x24, mask=mask24)
    torch.testing.assert_close(out64[mask64], out24[mask24], rtol=1e-5, atol=1e-8)


def length_stats(renorm, L, seeds=4, B=32):
    keys, outs = [], []
    for s in range(seeds):
        a = attn(renorm, seed=s).eval()
        x, mask = batch([L] * B, seed=100 + s)
        with torch.no_grad():
            keys.append(a._project(a.to_k(x), a.proj_k, mask).norm(dim=-1).mean())
            outs.append(a(x, mask=mask)[mask].pow(2).mean().sqrt())
    return torch.stack(keys).mean(), torch.stack(outs).mean()


def test_rescale_removes_length_dependence():
    for renorm, lo, hi in ((True, 0.9, 1.1), (False, 0.0, 0.6)):
        k_full, o_full = length_stats(renorm, SEQ)
        k_short, o_short = length_stats(renorm, 8)
        assert lo < k_short / k_full < hi, (renorm, (k_short / k_full).item())
        assert lo < o_short / o_full < hi, (renorm, (o_short / o_full).item())


def test_full_length_unmasked_matches_plain_linformer():
    a = attn()
    x, _ = batch([SEQ, SEQ])
    b, n, h, d_h = 2, SEQ, HEADS, DIM // HEADS
    q = a.to_q(x).reshape(b, n, h, d_h).transpose(1, 2)
    kv = [
        torch.einsum("bnd,nk->bkd", t, p).reshape(b, K, h, d_h).transpose(1, 2)
        for t, p in ((a.to_k(x), a.proj_k), (a.to_v(x), a.proj_v))
    ]
    p = torch.einsum("bhnd,bhkd->bhnk", q, kv[0]).mul(d_h**-0.5).softmax(-1)
    ref = a.to_out(
        torch.einsum("bhnk,bhkd->bhnd", p, kv[1]).transpose(1, 2).reshape(b, n, -1)
    )
    torch.testing.assert_close(a(x), ref, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(
        a(x, mask=torch.ones(b, n, dtype=torch.bool)), ref, rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_fully_padded_sequence_is_finite_zero(dtype):
    a = attn().to(dtype)
    x, mask = batch([0, 7])
    out = a(x.to(dtype), mask=mask)
    assert out.isfinite().all()
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert out[1, :7].abs().sum() > 0


def test_cross_attention_masks_context_and_queries():
    a = attn()
    torch.manual_seed(1)
    x = torch.randn(2, 10, DIM, dtype=torch.float64)
    ctx, cmask = batch([30, 3])
    qmask = torch.arange(10)[None] < torch.tensor([10, 6])[:, None]
    out = a(x, context=ctx, mask=qmask, context_mask=cmask)
    out_g = a(x, context=garbage_at_pad(ctx, cmask), mask=qmask, context_mask=cmask)
    assert torch.equal(out_g, out)
    assert torch.equal(out[~qmask], torch.zeros_like(out[~qmask]))
    assert (
        a(x, context=ctx, mask=qmask, context_mask=torch.ones_like(cmask)) - out
    ).abs().max() > 1e-3


def test_lm_logits_at_valid_positions_ignore_pad_tokens():
    torch.manual_seed(0)
    lm = (
        LinformerLM(
            num_tokens=50, dim=DIM, seq_len=SEQ, depth=2, k=K, heads=HEADS, pad_id=0
        )
        .double()
        .eval()
    )
    lengths = torch.tensor([12, 30])
    mask = torch.arange(SEQ)[None] < lengths[:, None]
    toks = torch.randint(1, 50, (2, SEQ))
    padded = torch.where(mask, toks, 0)
    logits = lm(padded)  # mask derived from pad_id
    logits_other = lm(
        toks, mask=mask
    )  # arbitrary tokens in the padded slots, explicit mask
    assert torch.equal(logits[mask], logits_other[mask])
    assert (
        lm(toks)[mask] - logits[mask]
    ).abs().max() > 1e-3  # unmasked: pad slots do matter


def test_matches_lucidrains_linformer_package():
    ld = pytest.importorskip("linformer")
    ref = ld.LinformerSelfAttention(DIM, SEQ, k=K, heads=HEADS).double().eval()
    x, _ = batch([SEQ, SEQ])
    xs = x[:, :24]
    mask24 = torch.arange(SEQ)[None].expand(2, -1) < 24
    for renorm in (False, True):
        a = attn(renorm).eval()
        a.load_state_dict(ref.state_dict())
        torch.testing.assert_close(a(x), ref(x), rtol=0, atol=1e-14)
        torch.testing.assert_close(
            a(xs), ref(xs), rtol=0, atol=1e-14
        )  # unmasked short input: sliced E
        # a mask of all ones on the short input rescales only with renorm on
        diff = (a(xs, mask=mask24[:, :24]) - ref(xs)).abs().max()
        assert (diff > 1e-2) if renorm else (diff == 0), (renorm, diff.item())

    torch.manual_seed(1)
    lm_ref = (
        ld.LinformerLM(num_tokens=50, dim=DIM, seq_len=SEQ, depth=2, k=K, heads=HEADS)
        .double()
        .eval()
    )
    lm = (
        LinformerLM(num_tokens=50, dim=DIM, seq_len=SEQ, depth=2, k=K, heads=HEADS)
        .double()
        .eval()
    )

    def remap(
        key,
    ):  # lucidrains linformer.net.layers.i.{0,1}.{norm,fn}.* -> layers.i.{0,1,2,3}.*
        if not key.startswith("linformer.net.layers."):
            return key
        _, _, _, i, slot, kind, *rest = key.split(".")
        idx = {("0", "norm"): 0, ("0", "fn"): 1, ("1", "norm"): 2, ("1", "fn"): 3}[
            (slot, kind)
        ]
        return ".".join(["linformer", "layers", i, str(idx), *rest])

    lm.load_state_dict(
        {remap(k): v for k, v in lm_ref.state_dict().items()}, strict=True
    )
    toks = torch.randint(0, 50, (2, SEQ))
    torch.testing.assert_close(lm(toks), lm_ref(toks), rtol=0, atol=1e-12)


@pytest.mark.parametrize("renorm", [False, True])
def test_project_leading_dims_match_per_slice(renorm):
    torch.manual_seed(0)
    proj = torch.randn(SEQ, K, dtype=torch.float64)
    x, mask = batch([11, 5, 40])
    t = torch.stack([x, 2 * x], dim=1)  # (b, h, n, d)
    out = project(garbage_at_pad(t, mask[:, None]), proj, mask, renorm)
    per_slice = torch.stack([project(t[:, i], proj, mask, renorm) for i in range(2)], 1)
    assert torch.equal(out, per_slice)
    assert not torch.equal(out[0], out[1])  # control: events are not mixed together


def per_query_reference(q, k, v, proj_k, proj_v, attn_mask, renorm):
    """Materialize each query's own projections E^T diag(M_i) K, F^T diag(M_i) V."""
    outs = []
    for i in range(q.shape[-2]):
        k_i = project(k, proj_k, attn_mask[:, i], renorm)
        v_i = project(v, proj_v, attn_mask[:, i], renorm)
        a = (q[..., i : i + 1, :] @ k_i.transpose(-1, -2) * q.shape[-1] ** -0.5).softmax(-1)
        outs.append(a @ v_i)
    return torch.cat(outs, dim=-2)


def qkv_masks(nq=7, n=SEQ - 3, seed=0):
    torch.manual_seed(seed)
    b, h, d = 3, 2, 8
    q = torch.randn(b, h, nq, d, dtype=torch.float64)
    k, v = torch.randn(2, b, h, n, d, dtype=torch.float64)
    attn_mask = torch.rand(b, nq, n) < 0.3
    attn_mask[:, :, -4:] = False  # padding: no query may see it
    attn_mask[0, 2] = False  # a query with nothing to attend to
    proj_k, proj_v = torch.randn(2, SEQ, K, dtype=torch.float64)
    return q, k, v, proj_k, proj_v, attn_mask


@pytest.mark.parametrize("renorm", [False, True])
def test_attend_matches_per_query_projection(renorm):
    q, k, v, pk, pv, m = qkv_masks()
    out = attend(q, k, v, pk, pv, m, renorm)
    torch.testing.assert_close(out, per_query_reference(q, k, v, pk, pv, m, renorm))
    assert torch.equal(out[0, :, 2], torch.zeros_like(out[0, :, 2]))
    # control: the per-query masks matter
    shared = m.any(1, keepdim=True).expand_as(m)
    assert (attend(q, k, v, pk, pv, shared, renorm) - out).abs().max() > 1e-3


def test_attend_hidden_keys_do_not_leak():
    q, k, v, pk, pv, m = qkv_masks()
    k.requires_grad_(True)
    out = attend(q, k, v, pk, pv, m)
    hidden = ~m.any(1)[:, None, :, None].expand_as(k)
    k_nan = torch.where(hidden, float("nan"), k.detach())
    v_nan = torch.where(hidden, float("nan"), v)
    assert torch.equal(attend(q, k_nan, v_nan, pk, pv, m), out.detach())
    out.square().sum().backward()
    assert torch.equal(k.grad[hidden], torch.zeros_like(k.grad[hidden]))
    assert k.grad[~hidden].norm() > 1e-3


@pytest.mark.parametrize("renorm", [False, True])
def test_attn_mask_of_padding_equals_padding_mask(renorm):
    a = attn(renorm)
    x, mask = batch([11, 5, 40])
    out = a(x, mask=mask)
    uniform = mask[:, None, :].expand(-1, SEQ, -1)
    torch.testing.assert_close(a(x, mask=mask, attn_mask=uniform)[mask], out[mask])
