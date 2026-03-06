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

Where $\otimes$ is **outer product** `[dim_k] x [dim_v] -> [dim_k,dim_v]`. The output at each time step is simply the query projected using the state at that time step `o_t = q_t @ S_t`. The equation above is also known as the Recurrent form of Linear attention.

TODO: change the outer product to matmul notation.

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

### Parallel Computation for Gated Linear Attention

Update rule of Gated Linear Attention

```math
S_t = g_t S_{t-1} + \vec k_t \otimes \vec v_t
```

Unrolling the recursive updates from a to b, we get the following

```math
S_b = \left(g_{a+1} \dots g_b \right) S_a + \sum_{t=a+1}^b \left(g_{a+1} \dots g_t \right) \vec k_t \otimes \vec v_t
```

The cumulative gating factor is a bit annoying, but I suppose it's not the end of the world. We can compute the cumulative product first, pre-multiply it to keys, then use the tensor cores as usual.

One note on numerical stability. Multiplying a lot of <1 numbers together is usually not a good idea. We can use the well-known log trick.

```math
\prod g_t = \exp \left( \sum \log g_t \right)
```

Moreover, we can predict $\log g_t$ directly, hence only an extra exponential is involved.

To compute the output

```math
o_b = q_b S_b
```

```math
= \left(g_{a+1} \dots g_b \right) \vec q_b S_a + \sum_{t=a+1}^b \left(g_{a+1} \dots g_t \right) (\vec q_b \cdot \vec k_t) \vec v_t
```

Notice that we have removed the outer product $\vec k_t \otimes \vec v_t$ by computing the dot product $\vec q_b \cdot \vec k_t$ first. Next, we stack multiple consecutive queries together to form a matrix multiplication:
- The first term becomes `G * Q @ S`, where G is the cumulative gating factor having different values for each query position. It doesn't matter if we multiply G with Q first, or multiply G with the matmul result Q @ S.
  - G has shape `[BLOCK_T, 1]`, Q has shape `[BLOCK_T, dim]`, S has shape `[dim, dim]`.
  - Intuitively, given a query token, it means we "discount" the old state matrix by the (cumulative) gating factor, as determined by the distance `b-a`.
- The second term becomes `((Q @ (G * K).T) * M) @ V`, where M is the causal mask. We have to pre-multiply G with K for the math to work out correctly.

Note: GLA paper rewrites the equations above a bit in terms of how the cumulative gating factors are computed and fused into matmul inputs/outputs. I haven't tried to implement GLA, but it doesn't seem necessary.

## Gated DeltaNet: Parallel Computation

As usual, let's start with the Update rule

```math
S_t = g_t S_{t-1} + \beta_t \vec k_t \otimes (\vec v_t - \vec k_t \cdot g_t S_{t-1})
```

Let's group all the terms involving the previous state together.

```math
S_t = g_t (I - \beta_t \vec k_t \otimes \vec k_t) S_t + \beta_t \vec k_t \otimes \vec v_t
```

Uh oh, the gating term now involves a matrix multiplication with the previous state. If we unroll the recursive relations N times, there will be N extra matrix multiplications. Defining $H_t = I - \beta_t \vec k_t \otimes \vec k_t$, we have (I'm using a random letter H here, not following any conventions):

```math
S_b = \left(g_{a+1} \dots g_b \right) \left(H_{a+1} \dots H_b \right) S_a + \sum_{t=a+1}^b \left(g_{a+1} \dots g_b \right) \left(H_{a+1} \dots H_b \right) \vec k_t \otimes \vec v_t
```

Luckily, there are people very good at math and find ways to compute $H_{a+1} \dots H_b$ without actually doing repeated matmuls. Assume

```math
H_a \dots H_b = I - \sum_{t=a}^b \vec k_t \otimes \vec w_t
```

We don't know what $\vec w_t$ is yet, but we know it's a vector that we can compute. We prove this relation by induction. Consider $b=a$:

```math
H_a = I - \vec k_a \otimes \vec w_a
```

Which is correct by the definition of $H_t$, and if $\vec w_a = \beta_t \vec k_a$. Next, let's prove the inductive step.

```math
H_a \dots H_b = \left(I - \sum_{t=a}^{b-1} \vec k_t \otimes \vec w_t\right) \left(I - \beta_b \vec k_b \otimes \vec k_b\right)
```

```math
= I - \sum_{t=a}^{b-1} \vec k_t \otimes \vec w_t - \beta_b \vec k_b \otimes \vec k_b + \beta_b \left( \sum_{t=a}^{b-1} \vec k_t \otimes \vec w_t\right)\left(\vec k_b \otimes \vec k_b\right)
```
