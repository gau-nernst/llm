# Linear Attention

Resources:
- https://sustcsonglin.github.io/blog/2024/deltanet-1/
- https://www.youtube.com/watch?v=pUCWwGR5WmQ
- Songlin papers: [Gated Linear Attention](https://arxiv.org/abs/2312.06635), [DeltaNet](https://arxiv.org/abs/2406.06484), [Gated DeltaNet](https://arxiv.org/abs/2412.06464)

## Motivation

Starting from softmax attention. We have a query vector q `[dim]` and Key-Value matrices K `[L, dim]` and V `[L, dim]`. In code form:

```python
s = q @ K.T     # [L]
p = softmax(s)  # [L]
o = p @ V       # [dim]
```

In math form

```math
\vec o = \sum_i \frac{\exp(\vec q \cdot \vec k_i)}{\sum_j \exp(\vec q \cdot \vec k_j)} \vec v_i
```

```math
= \frac{\sum_i \exp(\vec q \cdot \vec k_i) \vec v_i}{\sum_j \exp(\vec q \cdot \vec k_j)}
```

Let's remove the softmax operator, which enables us to reorder the ops. This becomes **Linear attention**.

```python
o = (q @ K.T) @ V
  = q @ (K.T @ V)
```

We can define the **State matrix** `S = K.T @ V`, which has shape `[dim, dim]`. The reduction is done over the sequence length dimension. This can be seen as a **Recurrent network**.

```math
S_t = \sum_{i=1}^{i=t} \vec k_i \otimes \vec v_i
```

```math
= S_{t-1} + \vec k_t \otimes \vec v_t
```

Where $\otimes$ is **outer product** `[dim] x [dim] -> [dim,dim]`. The output at each time step is simply the query projected using the state at that time step `o_t = q_t @ S_t`. The equation above is also known as the Recurrent form of Linear attention.

By modifying the recurrent relation a bit, we obtain **DeltaNet**.

```math
S_t = S_{t-1} + \beta_t \vec k_t \otimes (\vec v_t - \vec k_t \cdot S_{t-1})
```

The term $\vec k_t \cdot S_{t-1}$ can be seen as the **predicted value** using the old state matrix $S_{t-1}$. Hence, the recurrence relation suggests an **online learning** process of $S$, which models value as a linear projection of key. The loss function is L2.

```math
L = \frac{1}{2} \lVert \vec k \cdot S - \vec v \rVert^2
```

```math
\nabla_S L = \vec k \otimes (\vec k \cdot S - \vec v)
```

The update only uses the latest key-value data point (hence, online learning). We can apply the same perspective to the original Linear attention, which turns out to be optimizing for cosine similarity.

**Gating** In the recurrent update equation, we can apply gating to the past state matrix to forget it faster.

Gated Linear attention

```math
S_t = g_t S_{t-1} + \vec k_t \otimes \vec v_t
```

Gated DeltaNet

```math
S_t = g_t S_{t-1} + \beta_t \vec k_t \otimes (\vec v_t - \vec k_t \cdot g_t S_{t-1})
```

## Linear Attention: Parallel Computation

During training, or prefill phase, we need to compute multiple queries over multiple key-values at the same time, while maintaining memory efficiency and performant use of Tensor Cores. As an example, let's look at the **Parallel form** of softmax attention, which additionally enforces the causal relation via an additive attention bias ($-\infty$ at positions we don't want to attend to).

```python
S = Q @ K.T + M  # [Lq, Lkv]
P = softmax(S)   # [Lq, Lkv]
O = P @ V        # [Lq, dim]
```

We can parallelize over the query tokens i.e. each GPU threadblock handles a disjoint chunk of queries. This limits the size of `Lq` for each threadblock. However, materializing the full matrix S and P of shape `[Lq, Lkv]` is still memory-expensive when `Lkv` is large. Flash Attention solves this by chunking `Lkv`:
- At each new chunk of Keys and Values, the parallel form above is used to compute the Outputs, using the given Queries.
- This new output, computed from the new KV chunk, is added to the old output, computed from the previous KV chunks. There are extra math to make sure the results remain the same (i.e. online softmax trick).

Similarly, let's analyze the Parallel form of Linear attention:

```python
P = (Q @ K.T) * M  # [Lq, Lkv]
O = P @ V          # [Lq, dim]
```

`M` is the multiplicative causal mask (0 at where we don't want to attend to). We group `Q @ K.T` (instead of `K.T @ V`) because we need to apply the causal mask for queries: queries can't see future keys! However, this order of computation incurs more FLOPs when the sequence length is large (ignoring the causal mask for now).

```python
O = (Q @ K.T) @ V  # FLOPs = 4 * Lq * Lkv * dim
O = Q @ (K.T @ V)  # FLOPs = 2 * (Lq + Lkv) * dim * dim
```

Hence, it's possible to reduce FLOPs by using the 2nd form when causal mask is not needed (i.e. all queries are AFTER all keys-values), and use the 1st form otherwise (i.e. some queries are BEFORE some keys-values).

```python
O[a:b] = Q[a:b] @ (K[0:a].T @ V[0:a])        # w/o mask
       + ((Q[a:b] @ K[a:b].T) * M) @ V[a:b]  # w/ mask
```

Notice that the accumulated state at `t=a` can be reused for that at `t=b`, and be shared across ALL queries.

```python
K[0:b].T @ V[0:b] = K[0:a].T @ V[0:a] + K[a:b].T @ V[a:b]
                  = sum(K[..].T @ V[..])
```

This is different from softmax attention: generally, for different query tokens, we need to recompute everything with respect to other keys and values.

Hence, we can iterate over the sequence dimension, and compute the output at each chunk naturally. We obtain this pseudocode implementation (known as "non-materialization version" in https://arxiv.org/abs/2312.06635)

```python
# initial state can be non-zero e.g. extend attention
S = torch.zeros(dim, dim)

for t in range(0, T, BLOCK_T):
    tile_Q = Q[t : t + BLOCK_T]  # [BLOCK_T, dim]
    tile_K = K[t : t + BLOCK_T]  # [BLOCK_T, dim]
    tile_V = V[t : t + BLOCK_T]  # [BLOCK_T, dim]

    # compute and store output
    tile_O = tile_Q @ S + ((tile_Q @ tile_K.T) * M) @ tile_V
    O[t : t + BLOCK_T] = tile_O

    # update state for the next state
    S += tile_K.T @ tile_V
```

When we have sufficiently large batch size (number of attention heads is fixed), the GPU should be well-utilized.

TODO: gated version

## Gated DeltaNet: Parallel Computation
