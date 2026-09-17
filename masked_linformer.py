"""Linformer (arXiv:2006.04768) self-attention with padding masks.

Structure follows lucidrains/linformer. The difference is masking: Linformer projects keys and
values along the sequence axis (K' = E K), so a padded position cannot be removed from the
(n, k) score matrix, each projected key mixes every original position. The mask is applied
where the original position still exists: K and V rows are zeroed before the projection, padded
query rows are zeroed after. Autograd then gives exactly zero gradient into padded tokens; no
gradient hooks are needed.

Zeroing rows shrinks the projected keys with the number of valid tokens (a sum of fewer terms),
which turns sequence length into an attention temperature. With ``renorm=True`` each projection
column j is rescaled by ||E[:, j]|| / ||E[valid, j]|| so its norm matches the full-length case.
Measured on xavier-init E (n=256): plain zeroing gives 0.19x norm at 8 valid tokens, this
rescale gives 1.07x; count-based rescaling (n / n_valid) overshoots to 6x. The numerator is the
norm over the full ``seq_len`` so the result does not depend on how far a batch was padded. The
rescale applies only when a mask is passed; unmasked inputs of any length reproduce
lucidrains/linformer exactly (E sliced to the batch length), so a mask of all ones on a short
input is not the same as no mask.

``torch.where`` is used rather than a mask multiply: a NaN or inf in a padded row would otherwise
propagate through ``0 * nan`` into every output row and every gradient. E and F index absolute
positions, so pad on the same side the model was trained with.

Causal masking is not supported: every projected key mixes future positions.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def init_(tensor):
    std = 1 / math.sqrt(tensor.shape[-1])
    return tensor.uniform_(-std, std)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, glu=False):
        super().__init__()
        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x):
        if self.glu:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = F.gelu(x) * v
        else:
            x = F.gelu(self.w1(x))
        return self.w2(self.dropout(x))


class LinformerSelfAttention(nn.Module):
    """``mask`` / ``context_mask`` are bool ``(batch, len)``, True = real token.

    ``mask`` covers the queries (and the keys/values when ``context`` is None); ``context_mask``
    covers the keys/values of ``context``. Padded query rows of the output are zero.
    """

    def __init__(
        self,
        dim,
        seq_len,
        k=256,
        heads=8,
        dim_head=None,
        one_kv_head=False,
        share_kv=False,
        dropout=0.0,
        renorm=True,
        eps=1e-8,
    ):
        super().__init__()
        assert dim % heads == 0, "dimension must be divisible by the number of heads"
        self.seq_len, self.k, self.heads = seq_len, k, heads
        self.dim_head = dim_head or dim // heads
        self.renorm, self.eps = renorm, eps

        self.to_q = nn.Linear(dim, self.dim_head * heads, bias=False)
        kv_dim = self.dim_head if one_kv_head else self.dim_head * heads
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(self.dim_head * heads, dim)

    def _project(self, t, proj, mask):
        """(b, n, d) -> (b, k, d) along the sequence axis; padded rows of ``t`` contribute nothing."""
        n = t.shape[1]
        if mask is None:  # plain Linformer: E sliced to the batch length, no rescale
            return torch.einsum("bnd,nk->bkd", t, proj[:n])
        out = torch.einsum(
            "bnd,nk->bkd", torch.where(mask[..., None], t, 0.0), proj[:n]
        )
        if self.renorm:
            p = proj.to(
                torch.promote_types(proj.dtype, torch.float32)
            )  # eps must survive half precision
            full = (
                p.pow(2).sum(0).sqrt()
            )  # over all seq_len rows: batch-padding invariant
            kept = (
                torch.einsum("bn,nk->bk", mask.to(p.dtype), p[:n].pow(2))
                .sqrt()
                .clamp_min(self.eps)
            )
            out = out * (full / kept).to(out.dtype)[..., None]
        return out

    def forward(self, x, context=None, mask=None, context_mask=None):
        b, n, _ = x.shape
        kv_input = x if context is None else context
        kv_len = kv_input.shape[1]
        assert kv_len <= self.seq_len, (
            f"key/value length {kv_len} exceeds seq_len {self.seq_len}"
        )

        if context_mask is None and context is None:
            context_mask = mask

        queries = self.to_q(x)
        keys = self.to_k(kv_input)
        values = keys if self.share_kv else self.to_v(kv_input)
        proj_v = self.proj_k if self.share_kv else self.proj_v

        keys = self._project(keys, self.proj_k, context_mask)
        values = self._project(values, proj_v, context_mask)

        h, d_h, k = self.heads, self.dim_head, self.k
        queries = queries.reshape(b, n, h, d_h).transpose(1, 2)
        keys, values = (
            t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
            for t in (keys, values)
        )

        dots = torch.einsum("bhnd,bhkd->bhnk", queries, keys) * d_h**-0.5
        attn = self.dropout(dots.softmax(dim=-1))
        out = torch.einsum("bhnk,bhkd->bhnd", attn, values)
        out = self.to_out(out.transpose(1, 2).reshape(b, n, -1))
        if mask is not None:
            out = torch.where(mask[..., None], out, 0.0)
        return out


class Linformer(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        depth,
        k=256,
        heads=8,
        dim_head=None,
        one_kv_head=False,
        share_kv=False,
        dropout=0.0,
        renorm=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            attn = LinformerSelfAttention(
                dim,
                seq_len,
                k=k,
                heads=heads,
                dim_head=dim_head,
                one_kv_head=one_kv_head,
                share_kv=share_kv,
                dropout=dropout,
                renorm=renorm,
            )
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        attn,
                        nn.LayerNorm(dim),
                        FeedForward(dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for norm1, attn, norm2, ff in self.layers:
            x = x + attn(norm1(x), mask=mask)
            x = x + ff(norm2(x))
        return x


class LinformerLM(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        seq_len,
        depth,
        k=256,
        heads=8,
        dim_head=None,
        one_kv_head=False,
        share_kv=False,
        dropout=0.0,
        renorm=True,
        pad_id=None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.linformer = Linformer(
            dim,
            seq_len,
            depth,
            k=k,
            heads=heads,
            dim_head=dim_head,
            one_kv_head=one_kv_head,
            share_kv=share_kv,
            dropout=dropout,
            renorm=renorm,
        )
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, mask=None):
        if mask is None and self.pad_id is not None:
            mask = x != self.pad_id
        h = self.token_emb(x) + self.pos_emb(torch.arange(x.shape[1], device=x.device))
        return self.to_logits(self.linformer(h, mask=mask))
