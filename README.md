# masked-linformer

Linformer self-attention ([arXiv:2006.04768](https://arxiv.org/abs/2006.04768)) in the shape of
[lucidrains/linformer](https://github.com/lucidrains/linformer), plus the two things that
implementation lacks: padding masks and length-invariant scaling.

```python
from masked_linformer import LinformerSelfAttention, Linformer, LinformerLM

attn = LinformerSelfAttention(dim=512, seq_len=4096, k=256, heads=8)
mask = lengths[:, None] > torch.arange(4096)          # (batch, seq) bool, True = real token
y = attn(x, mask=mask)                                # padded query rows come back as zero
y = attn(x, context=mem, mask=mask, context_mask=mem_mask)

lm = LinformerLM(num_tokens=20000, dim=512, seq_len=4096, depth=12, k=256, heads=8, pad_id=0)
logits = lm(tokens)                                   # mask derived from pad_id
```

Why the mask is not applied to the attention matrix: Linformer projects keys and values along
the sequence axis (`K' = E K`), so every projected key mixes every original position and there
is no entry of the `(n, k)` score matrix that corresponds to a padded token. The mask has to be
applied to K and V before the projection (and to the output rows for padded queries). Autograd
then delivers exactly zero gradient into padded tokens; no hooks. The module uses `torch.where`
rather than a multiply so NaN or inf in padded rows cannot leak.

`renorm=True` (default) rescales each projection column by `||E[:, j]|| / ||E[valid, j]||`, so
the projected key and value norms do not shrink with the number of valid tokens. Measured over
8 seeds x 64 inputs at init (n=64, k=8): projected key norm at 1 valid token is 0.10x the
full-length value without the rescale and 1.01x with it; output RMS 0.43x vs 0.92x.

The rescale applies only when a mask is passed. With no mask the output is identical to
lucidrains/linformer with the same weights at any length (tested against the package), so the
module is a drop-in there; note that lucidrains' sliced projection then keeps the length
dependence across unmasked batches of different lengths (0.50x key norm at half length).

Per-query masks (`attn_mask`, bool `(batch, n, kv_len)`, e.g. MaskFormer's mask attention) are
exact: query `i` attends to its own projections `E^T diag(M_i) K` and `F^T diag(M_i) V`, each
with the column rescale of its own mask. `attend(q, k, v, proj_k, proj_v, attn_mask)` computes
the same numbers without building them, as `(S * M_i) E` for the scores and `((a F^T) * M_i) v`
for the output (`S = q k^T`), so memory is that of ordinary attention and so is the compute: the
saving Linformer gives the shared projection does not carry over to per-query masks.

```python
y = attn(x, context=mem, context_mask=mem_mask, attn_mask=query_to_mem_mask)
```

Not supported: causal masking (every projected key mixes future positions; the `(n, k)` triangular
mask some implementations offer leaves every query beyond position k unmasked).

```
uv run pytest
```
