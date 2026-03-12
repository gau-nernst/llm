# Linear Attention

Resources:
- https://sustcsonglin.github.io/blog/2024/deltanet-1/
- https://www.youtube.com/watch?v=pUCWwGR5WmQ
- Songlin papers: [Gated Linear Attention](https://arxiv.org/abs/2312.06635), [DeltaNet](https://arxiv.org/abs/2406.06484), [Gated DeltaNet](https://arxiv.org/abs/2412.06464)

## Motivation

Starting from softmax attention. We have a query vector q `[1, dim]` and Key-Value matrices K `[L, dim]` and V `[L, dim]`. We purposely treat vectors as row vectors so that when we stack multiple vectors together to form a matrix, the equations stay the same. In code form:

```python
s = q @ K.T     # [1, L]
p = softmax(s)  # [1, L]
o = p @ V       # [1, dim]
```

In math form

```math
\vec o = \sum_i \frac{\exp\left(\vec q \, \vec k_i^T\right)}{\sum_j \exp\left(\vec q \, \vec k_j^T\right)} \vec v_i
```

```math
= \frac{\sum_i \exp(\vec q \, \vec k_i^T) \vec v_i}{\sum_j \exp(\vec q \, \vec k_j^T)}
```

Note that because we define vectors as row vectors instead of the usual column vectors, dot product has the 2nd vector transposed instead of the 2nd one. Similarly, cross product in our analysis has the 1st vector transposed.

Let's remove the softmax operator, which enables us to reorder the ops. This becomes **Linear attention**.

```python
o = (q @ K.T) @ V
  = q @ (K.T @ V)
```

We can define the **State matrix** `S = K.T @ V`, which has shape `[dim_k, dim_v]`. The reduction is done over the sequence length dimension. This can be seen as a **Recurrent network**.

```math
S_t = \sum_{i=1}^{i=t} \vec k_i^T \vec v_i
```

```math
= S_{t-1} + \vec k_t^T \vec v_t
```

Notice that $\vec k_t^T \vec v_t$ is the **outer product** between the key and value vectors. The output at each time step is simply the query projected using the state at that time step $o_t = q_t S_t$. The equation above is also known as the Recurrent form of Linear attention.

By modifying the recurrent relation a bit, we obtain **DeltaNet**.

```math
S_t = S_{t-1} + \beta_t \vec k_t^T (\vec v_t - \vec k_t S_{t-1})
```

The term $\vec k_t S_{t-1}$ can be seen as the **predicted value** using the old state matrix $S_{t-1}$. Hence, the recurrence relation suggests an **online learning** process of $S$, which models value as a linear projection of key. The loss function is L2.

```math
L = \frac{1}{2} \lVert \vec k S - \vec v \rVert^2
```

```math
\nabla_S L = \vec k^T (\vec k S - \vec v)^T
```

The update only uses the latest key-value data point (hence, online learning). We can apply the same perspective to the original Linear attention, which turns out to be optimizing for cosine similarity.

**Gating** In the recurrent update equation, we can apply gating to the past state matrix to forget it faster.

Gated Linear attention

```math
S_t = g_t S_{t-1} + \vec k_t^T \vec v_t
```

Gated DeltaNet

```math
S_t = g_t S_{t-1} + \beta_t \vec k_t^T (\vec v_t - g_t \vec k_t S_{t-1})
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

<details>
<summary>Tile implementation in PyTorch</summary>

```python
import torch
from torch import Tensor


def la_recurrent(
    Q: Tensor,  # (num_tokens, qk_dim)
    K: Tensor,  # (num_tokens, qk_dim)
    V: Tensor,  # (num_tokens, v_dim)
    state: Tensor,  # (v_dim, qk_dim)
):
    num_tokens, _ = Q.shape
    out = torch.empty_like(V)

    for t in range(num_tokens):
        q_vec = Q[t : t + 1]  # row vectors
        k_vec = K[t : t + 1]
        v_vec = V[t : t + 1]

        state = state + v_vec.T @ k_vec  # outer product
        out[t : t + 1] = q_vec @ state.T

    return state, out


def la_parallel(
    Q: Tensor,  # (num_tokens, qk_dim)
    K: Tensor,  # (num_tokens, qk_dim)
    V: Tensor,  # (num_tokens, v_dim)
    state: Tensor,  # (v_dim, qk_dim)
    BLOCK_T: int = 16,
):
    num_tokens, _ = Q.shape
    assert num_tokens % BLOCK_T == 0

    out = torch.empty_like(V)
    mask = torch.tril(torch.ones(BLOCK_T, BLOCK_T))

    for t in range(0, num_tokens, BLOCK_T):
        q_tile = Q[t : t + BLOCK_T]
        k_tile = K[t : t + BLOCK_T]
        v_tile = V[t : t + BLOCK_T]

        out[t : t + BLOCK_T] = q_tile @ state.T + (q_tile @ k_tile.T * mask) @ v_tile
        state = state + v_tile.T @ k_tile

    return state, out


num_tokens = 64
qk_dim = 64
v_dim = 96

Q = torch.randn(num_tokens, qk_dim)
K = torch.randn(num_tokens, qk_dim) * (qk_dim**-0.5)
V = torch.randn(num_tokens, v_dim)
state = torch.randn(v_dim, qk_dim)

state_re, out_re = la_recurrent(Q, K, V, state)
state_pa, out_pa = la_parallel(Q, K, V, state)

torch.testing.assert_close(state_pa, state_re)
torch.testing.assert_close(out_pa, out_re)
```

</details>

### Parallel Computation for Gated Linear Attention

Update rule of Gated Linear Attention

```math
S_t = g_t S_{t-1} + \vec k_t^T \vec v_t
```

Let's unroll it one more time to observe the pattern.

```math
S_t = g_t \left( g_{t-1} S_{t-2} + \vec k_{t-1}^T \vec v_{t-1}\right) + \vec k_t^T \vec v_t
```

```math
= g_t g_{t-1} S_{t-2} + g_t \vec k_{t-1}^T \vec v_{t-1} + \vec k_t^T \vec v_t
```

Generalizing it ($a \leq b$)

```math
S_b = \left(g_b \dots g_{a+1}\right) S_a + \sum_{t=a+1}^b \left(g_b \dots g_{t+1}\right) \vec k_t^T \vec v_t
```

The cumulative gating factor is a bit annoying, but I suppose it's not the end of the world. Note that the scales vary along the reduction dim, hence we can only pre-apply this scaling to the inputs ($\vec k$ in this case) to convert this to a matmul.

One note on numerical stability. Multiplying a lot of <1 numbers together is usually not a good idea. We can use the well-known log trick.

```math
\prod g_t = \exp \left(\sum \log g_t\right)
```

Moreover, we can predict $\log g_t$ directly, hence only an extra exponential is involved.

To compute the output

```math
\vec o_b = \vec q_b S_b
```

```math
= \left(g_b \dots g_{a+1}\right) \vec q_b S_a + \sum_{t=a+1}^b \left(g_b \dots g_{t+1}\right) \vec q_b \vec k_t^T \vec v_t
```

Notice that we have replaced the outer product $\vec k_t^T \vec v_t$ with the dot product $\vec q_b \vec k_t^T$ (ignoring the extra scaling).
- For the 1st term, we can either pre-multiply $\vec q_b$ or post-multiply the result $\vec q_b S_a$.
- For the 2nd term, it's a bit more complicated: the scaling depends on the relative position between query and key vectors - key stays at $t$, and query stays at $b$. We can re-write the output equation as:

```math
\vec o_b = \left(g_b \dots g_{a+1}\right) \vec q_b S_a + \sum_{t=a+1}^b \left(g_b \dots g_{a+1}\right) \vec q_b \frac{1}{\left(g_t \dots g_{a+1}\right)} \vec k_t^T \vec v_t
```

We can either pre-scale queries and keys, OR post-scale `Q @ K.T`. Notice that `Q @ K.T` is doing reduction along the `qk_dim`, so scaling along the `BLOCK_T` dim can be fused at output.

Next, we stack multiple consecutive queries together to form a matrix multiplication. First, look at the pre-scaling approach.

```
Q' = G * Q
K' = (1/G) * K

S = G[-1] * (S + K'.T @ V)
O = Q' @ S + ((Q' @ K'.T) * M) @ V
```

- `G` is the cumulative product, shape `[BT, 1]`
- `(G * Q)` and `(1 / G) * K` are scaled queries and keys, computed in parallel.
- `M` is the causal mask.

There is a problem with input scaling with tensor cores: typically we want a fast loop of gmem->smem then MMA. If we need to scale the inputs in the same kernel, it can leave the MMA pipeline idle (BF16 -> FP32 -> scale -> BF16). One way to solve this is to have a separate pre-scaling kernel, so that data in gmem is already scaled.

A possibly better way is that we can scale **MMA output** instead, which is already in FP32.

```
M' = (1/G) * M
O = G[-1] * (Q @ S + ((Q @ K.T) * M') @ V)
```

Notice that we have fused scaling of `K` to the causal mask `M`. For `S` equation, we can't avoid input pre-scaling.

<details>
<summary>Tile implementation in PyTorch</summary>

```python
import torch
from torch import Tensor

def gla_recurrent(
    Q: Tensor,  # (num_tokens, qk_dim)
    K: Tensor,  # (num_tokens, qk_dim)
    V: Tensor,  # (num_tokens, v_dim)
    gate: Tensor,  # (num_tokens)
    state: Tensor,  # (v_dim, qk_dim)
):
    num_tokens, _ = Q.shape
    out = torch.empty_like(V)

    for t in range(num_tokens):
        q_vec = Q[t : t + 1]  # row vectors
        k_vec = K[t : t + 1]
        v_vec = V[t : t + 1]
        g = gate[t].exp()

        state = g * state + v_vec.T @ k_vec  # outer product
        out[t : t + 1] = q_vec @ state.T

    return state, out


def gla_parallel(
    Q: Tensor,  # (num_tokens, qk_dim)
    K: Tensor,  # (num_tokens, qk_dim)
    V: Tensor,  # (num_tokens, v_dim)
    gate: Tensor,  # (num_tokens)
    state: Tensor,  # (v_dim, qk_dim)
    BLOCK_T: int = 16,
):
    num_tokens, _ = Q.shape
    assert num_tokens % BLOCK_T == 0
    mask = torch.tril(torch.ones(BLOCK_T, BLOCK_T))
    out = torch.empty_like(V)

    for t in range(0, num_tokens, BLOCK_T):
        q_tile = Q[t : t + BLOCK_T]
        k_tile = K[t : t + BLOCK_T]
        v_tile = V[t : t + BLOCK_T]
        g_tile = gate[t : t + BLOCK_T].view(-1, 1)

        cu_gate = g_tile.cumsum(dim=0).exp()
        scaled_mask = mask / cu_gate.view(1, -1)
        out[t : t + BLOCK_T] = cu_gate * (q_tile @ state.T + ((q_tile @ k_tile.T) * scaled_mask) @ v_tile)

        # last element in tile cumsum = tile sum
        state = cu_gate[-1] * (state + v_tile.T @ (k_tile / cu_gate))

    return state, out


num_tokens = 256
qk_dim = 64
v_dim = 96

Q = torch.randn(num_tokens, qk_dim)
K = torch.randn(num_tokens, qk_dim) * (qk_dim**-0.5)
V = torch.randn(num_tokens, v_dim)
gate = torch.rand(num_tokens).log()
state = torch.randn(v_dim, qk_dim)

state_re, out_re = gla_recurrent(Q, K, V, gate, state)
state_pa, out_pa = gla_parallel(Q, K, V, gate, state)

torch.testing.assert_close(state_pa, state_re)
torch.testing.assert_close(out_pa, out_re)
```

</details>

## Gated DeltaNet: Parallel Computation

As usual, let's start with the Update rule

```math
S_t = g_t S_{t-1} + \beta_t \vec k_t^T (\vec v_t - g_t \vec k_t S_{t-1})
```

Let's group all the terms involving the previous state together.

```math
S_t = g_t (I - \beta_t \vec k_t^T \vec k_t) S_{t-1} + \beta_t \vec k_t^T \vec v_t
```

Uh oh, the gating term now involves a matrix multiplication with the previous state. If we unroll the recursive relations N times, there will be N extra matrix multiplications. Defining $H_t = I - \beta_t \vec k_t^T \vec k_t$, we have (I'm using a random letter H here, not following any conventions):

```math
S_t = g_t H_t S_{t-1} + \beta_t \vec k_t^T \vec v_t
```

```math
= g_t g_{t-1} H_t H_{t-1} S_{t-2} + g_t H_t \beta_{t-1} \vec k_{t-1}^T \vec v_{t-1} + \beta_t \vec k_t^T \vec v_t
```

Generalizing it ($a \leq b$)

```math
S_b = \left(g_b \dots g_{a+1}\right) \left(H_b \dots H_{a+1}\right) S_a + \sum_{t=a+1}^b \left(g_b \dots g_{t+1}\right) \left(H_b \dots H_{t+1}\right) \beta_t \vec k_t^T \vec v_t
```

Luckily, there are people very good at math and find ways to compute $H_b \dots H_a$ without actually doing repeated matmuls. Assume

```math
H_b \dots H_a = I - \sum_{t=a}^b \vec k_t^T \vec w_t
```

We don't know what $\vec w_t$ is yet, but we know it's a vector that we can compute. We prove this relation by induction. Consider $b=a$:

```math
H_a = I - \vec k_a^T \vec w_a
```

Which is correct by the definition of $H_t$, and if $\vec w_a = \beta_a \vec k_a$. Next, let's prove the inductive step.

```math
H_b \dots H_a = \left(I - \beta_b \vec k_b^T \vec k_b\right) \left(I - \sum_{t=a}^{b-1} \vec k_t^T \vec w_t\right)
```

```math
= I - \sum_{t=a}^{b-1} \vec k_t^T \vec w_t - \beta_b \vec k_b^T \vec k_b \left(I - \sum_{t=a}^{b-1} \vec k_t^T \vec w_t\right)
```

```math
= I - \sum_{t=a}^{b-1} \vec k_t^T \vec w_t - \vec k_b^T \beta_b \left(\vec k_b - \sum_{t=a}^{b-1} \vec k_b \vec k_t^T \vec w_t\right)
```

If we let $\vec w_b = \beta_b \left( \vec k_b - \sum_{t=a}^{b-1} \vec k_b \vec k_t^T \vec w_t\right)$, then everything works out correctly, though it feels like cheating! Note that $\vec k_b \vec k_t^T$ is a dot product. We can verify again that if we set $b=a$, the recursive relation of $\vec w_t$ also works out correctly.

We want to prove the following (2nd term in $S_b$ equation), and also find the recurrent relation for $\vec u$

```math
\sum_{t=a}^b \left(g_b \dots g_{t+1}\right) \left(H_b \dots H_{t+1}\right) \beta_t \vec k_t^T \vec v_t = \sum_{t=a}^b \left(g_b \dots g_{t+1}\right) \vec k_t^T \vec u_t
```

At $b=a$, we get $\vec u_a = \beta_a \vec v_a$. Proving the inductive step

```math
\sum_{t=a}^b \left(g_b \dots g_{t+1}\right) \left(H_b \dots H_{t+1}\right) \beta_t \vec k_t^T \vec v_t
```

```math
= g_b H_b \sum_{t=a}^{b-1} \left(g_{b-1} \dots g_{t+1}\right) \left(H_{b-1} \dots H_{t+1}\right) \beta_t \vec k_t^T \vec v_t + \beta_b \vec k_b^T \vec v_b
```

```math
= g_b (I - \beta_b \vec k_b^T \vec k_b) \sum_{t=a}^{b-1} \left(g_{b-1} \dots g_{t+1}\right) \vec k_t^T \vec u_t + \beta_b \vec k_b^T \vec v_b
```

```math
= \sum_{t=a}^{b-1} \left(g_b \dots g_{t+1}\right) \vec k_t^T \vec u_t + \vec k_b^T \beta_b \left(\vec v_b - \sum_{t=a}^{b-1} \left(g_b \dots g_{t+1}\right) \vec k_b \vec k_t^T \vec u_t\right)
```

Hence, by setting $\vec u_b = \beta_b \left(\vec v_b - \sum_{t=a}^{b-1} \left(g_b \dots g_{t+1}\right) \vec k_b \vec k_t^T \vec u_t\right)$, the relation is correct. Note that this is very similar to the recurrent relation for $\vec w$. Also pay attention that the inner cumulative product is from $t+1$ to $b$, while the outer summation is only up to $b-1$.

Putting everything back to the state update equation.

```math
S_b = \left(g_b \dots g_{a+1}\right) \left(I - \sum_{t=a+1}^b \vec k_t^T \vec w_t\right) S_a + \sum_{t=a+1}^b \left(g_b \dots g_{t+1}\right) \vec k_t^T \vec u_t
```

Rewritting this a bit for parallel computations

```math
S_b = \left(g_b \dots g_{a+1}\right) \left[S_a + \sum_{t=a+1}^b \left(\frac{1}{g_t \dots g_{a+1}}\right) \vec k_t^T \left(\vec u_t - \left(g_t \dots g_{a+1}\right)\vec w_t S_a\right)\right]
```

Notice that this looks very similar to GLA's equations, with $\vec v_t$ being replaced with $\vec u_t - \left(g_t \dots g_{a+1}\right)\vec w_t S_a$. Hence, we can define this expression as $\vec v_t' = \vec u_t - \left(g_t \dots g_{a+1}\right)\vec w_t S_a$

```math
S_b = \left(g_b \dots g_{a+1}\right) \left[S_a + \sum_{t=a+1}^b \left(\frac{1}{g_t \dots g_{a+1}}\right) \vec k_t^T \vec v_t'\right]
```

For the output:

```math
\vec o_b = \left(g_b \dots g_{a+1}\right) \vec q_b S_a + \sum_{t=a+1}^b \left(g_b \dots g_{a+1}\right) \vec q_b \left(\frac{1}{g_t \dots g_{a+1}}\right) \vec k_t^T \vec v_t'
```

Similar to GLA, we can either pre-scale the inputs, or post-scale MMA outputs. Looking at the matrix form of input pre-scaling approach.

```
Q' = G * Q
K' = (1 / G) * K
W' = G * W
V' = U - W' @ S

S = G[-1] * (S + K'.T @ V')
O = Q' @ S + ((Q' @ K'.T) * M) @ V'
```

Note: looks like the equations we have derived are slightly different from the author's. Hopefully it's still correct...

For the MMA output post-scaling approach.

```
V' = U - G * (W @ S)
M' = (1/G) * M

S = G[-1] * (S + K.T @ ((1/G) * V'))
O = G * (Q @ S + ((Q @ K.T) * M') @ V')
```

Notice that in state update equation, we fuse `(1/G)` to `V'`. This is ok since `V'` is originally computed in FP32 (1st line), hence adding this scaling won't incur too much overheads.

**Compute w and u.** The last piece is how to compute $\vec w$ and $\vec u$ efficiently in parallel. The author provides these equations, in which perhaps we just take it at the face value.

```
T_W = (I + strictLower(B * (K @ K.T)))^(-1)
W = T_W @ (B * K)

T_U = (I + strictLower(B * Gamma * (K @ K.T)))^(-1)
U = T_U @ (B * V)
```

Size of `T_W` and `T_U` is `[BLOCK_T, BLOCK_T]`, so with sufficiently small `BLOCK_T`, calculting the inverse might not be too slow? [FLA repo](https://github.com/fla-org/flash-linear-attention/blob/v0.4.1/fla/ops/gated_delta_rule/chunk.py#L27) uses `BLOCK_T=64`.

`Gamma` is a `[BLOCK_T, BLOCK_T]` matrix that encodes cumulative product of gate scales. It's defined as `Gamma[i,j] = cumprod(g[:i]) / cumprod(g[:j])`.

There is a problem with this: we need to compute 2 separate matrix inverses for $\vec w$ and $\vec u$! Turns out, we can rewrite the recursive relation for $\vec w$ so that it follows the similar form as $\vec u$, hence sharing the same inverse. The new equation we need to prove:

```math
\left(g_b \dots g_a\right) \left(H_b \dots H_a\right) = \left(g_b \dots g_a\right) I - \sum_{t=a}^b \left(g_b \dots g_{t+1}\right) \vec k_t^T \vec w_t
```

At $b=a$, we get $\vec w_a = g_a \beta_a \vec k_a$. Next, proving the inductive step:

```math
\left(g_b \dots g_a\right) \left(H_b \dots H_a\right)
```

```math
= g_b \left(I - \beta_b \vec k_b^T \vec k_b\right) \left[\left(g_{b-1} \dots g_a\right) I - \sum_{t=a}^{b-1} \left(g_{b-1} \dots g_{t+1}\right) \vec k_t^T \vec w_t\right]
```

```math
= \left(I - \beta_b \vec k_b^T \vec k_b\right) \left[\left(g_b \dots g_a\right) I - \sum_{t=a}^{b-1} \left(g_b \dots g_{t+1}\right) \vec k_t^T \vec w_t\right]
```

```math
= \left(g_b \dots g_a\right) I - \sum_{t=a}^{b-1} \left(g_b \dots g_{t+1}\right) \vec k_t^T \vec w_t - \vec k_b^T \beta_b \left[\left(g_b \dots g_a\right) \vec k_b - \sum_{t=a}^{b-1} \left(g_b \dots g_{t+1}\right) \vec k_b \vec k_t^T \vec w_t\right]
```

The equations look a lot more complicated, but then we can set

```math
\vec w_b = \beta_b \left[\left(g_b \dots g_a\right) \vec k_b - \sum_{t=a}^{b-1} \left(g_b \dots g_{t+1}\right) \vec k_b \vec k_t^T \vec w_t\right]
```

Examining the state update and output equations again

```math
\vec v_t' = \vec u_t - \vec w_t S_a
```

```math
S_b = \left(g_b \dots g_{a+1}\right) \left[S_a + \sum_{t=a+1}^b \left(\frac{1}{g_t \dots g_{a+1}}\right) \vec k_t^T \vec v_t'\right]
```

```math
\vec o_b = \left(g_b \dots g_{a+1}\right) \vec q_b S_a + \sum_{t=a+1}^b \left(g_b \dots g_{a+1}\right) \vec q_b \left(\frac{1}{g_t \dots g_{a+1}}\right) \vec k_t^T \vec v_t'
```

Matrix notation

```
T = (I + strictLower(B * Gamma (K @ K.T)))^(-1)
W = T @ (B * G * K)
U = T @ (B * V)

V' = U - W @ S
M' = (1/G) * M

S = G[-1] * (S + K.T @ ((1/G) * V'))
O = G * (Q @ S + ((Q @ K.T) * M') @ V')
```

<details>
<summary>Tile implementation in PyTorch</summary>

```python
import torch
from torch import Tensor

def gdn_recurrent(
    Q: Tensor,  # (num_tokens, qk_dim)
    K: Tensor,  # (num_tokens, qk_dim)
    V: Tensor,  # (num_tokens, v_dim)
    gate: Tensor,  # (num_tokens)
    beta: Tensor,  # (num_tokens)
    state: Tensor,  # (v_dim, qk_dim)
):
    num_tokens, _ = Q.shape
    out = torch.empty_like(V)

    for t in range(num_tokens):
        q_vec = Q[t : t + 1]  # row vectors
        k_vec = K[t : t + 1]
        v_vec = V[t : t + 1]
        g = gate[t].exp()
        b = beta[t]

        v_err = v_vec - g * k_vec @ state.T
        state = g * state + b * v_err.T @ k_vec  # outer product

        out[t : t + 1] = q_vec @ state.T

    return state, out


def gdn_parallel(
    Q: Tensor,  # (num_tokens, qk_dim)
    K: Tensor,  # (num_tokens, qk_dim)
    V: Tensor,  # (num_tokens, v_dim)
    gate: Tensor,  # (num_tokens)
    beta: Tensor,  # (num_tokens)
    state: Tensor,  # (v_dim, qk_dim)
    BLOCK_T: int = 16,
):
    num_tokens, _ = Q.shape
    assert num_tokens % BLOCK_T == 0

    mask = torch.tril(torch.ones(BLOCK_T, BLOCK_T))
    eye = torch.eye(BLOCK_T)
    out = torch.empty_like(V)

    for t in range(0, num_tokens, BLOCK_T):
        q_tile = Q[t : t + BLOCK_T]
        k_tile = K[t : t + BLOCK_T]
        v_tile = V[t : t + BLOCK_T]

        g_tile = gate[t : t + BLOCK_T].view(-1, 1)
        b_tile = beta[t : t + BLOCK_T].view(-1, 1)

        # just follow the equation...
        cu_gate = g_tile.cumsum(dim=0)
        GAMMA = (cu_gate - cu_gate.view(1, -1)).exp()  # [BLOCK_T, BLOCK_T]
        cu_gate = cu_gate.exp()

        b_gamma_kkt = b_tile * GAMMA * (k_tile @ k_tile.T)
        T = torch.linalg.inv(eye + b_gamma_kkt.tril(-1)) * b_tile.view(1, -1)
        w_tile = (T * cu_gate.view(1, -1)) @ k_tile
        u_tile = T @ v_tile

        scaled_mask = mask / cu_gate.view(1, -1)
        v_err = u_tile - w_tile @ state.T
        out[t : t + BLOCK_T] = cu_gate * (q_tile @ state.T + ((q_tile @ k_tile.T) * scaled_mask) @ v_err)

        # last element in tile cumsum = tile sum
        state = cu_gate[-1] * (state + (v_err / cu_gate).T @ k_tile)

    return state, out


num_tokens = 16 * 5
qk_dim = 64
v_dim = 96

Q = torch.randn(num_tokens, qk_dim)
K = torch.randn(num_tokens, qk_dim) * (qk_dim**-0.5)
V = torch.randn(num_tokens, v_dim)
gate = torch.rand(num_tokens).log()
beta = torch.randn(num_tokens)
state = torch.randn(v_dim, qk_dim)

state_re, out_re = gdn_recurrent(Q, K, V, gate, beta, state)
state_pa, out_pa = gdn_parallel(Q, K, V, gate, beta, state, BLOCK_T=16)

torch.testing.assert_close(state_pa, state_re)
torch.testing.assert_close(out_pa, out_re)
```

</details>
