# Reinforcement Learning for LLMs

Resources:
- https://spinningup.openai.com/
- https://rlhfbook.com/c/11-policy-gradients.html

## Basic concepts

Term | Explanation
---|---
Agent | Us
Environment | The world
Action | Environment changes upon an agent's action, but it can also change on its own. Current action is acted **after**/**on** current state.
State | Complete view of the world.
Observation | What agent can see. Partial or full state of the world.
Policy | Agent's rule to decide which action to take. It can be probabilistic / stochastic. Usually it's parameterized so that we can optimize it.
Trajectory (aka episodes, rollouts) | Sequence of states and actions
State transition | What happens to the world in the next time step. Depends on the laws of the environment, and only the current action. It can be stochastic.
Reward | Response from the environment to the agent. Only depends on **current state, current action, and next state**. It can be simplified to only depend on current state, or current state and current action.
Return | Cumulative reward over a trajectory. The agent's goal is to maximize this. For infinite-horizon, we can include a discount factor.
Expected return | Since policy and transition can be stochastic, trajectory is stochastic as well. Hence, expected return is computed over the distribution of trajectories. The **optimal policy** maximizes expected return.
Q-function | (aka Action-value function) Expected return at a particular state **and taking a particular action**, following a particular policy. **Optimal action** corresponds to optimal Q-function.
Value function | Expected return at a particular state, following a particular policy. It follows that Value function is expected value of Q-function over action distribution (policy at current state).
Advantage function | The difference between Q-function and Value function i.e. Q - V -> how much a particular action is better than average, given a policy.

## Model-free vs Model-based

Model-based: Agent learns to model the environment. This allows the agent to plan ahead, but the learned model might not generalise well.

### Model-free RL

Learning | Description
---|---
Policy optimization | Models the policy. Optimize via stochastic gradient ascent on (approximate) Expected return. Typically **on-policy** - use data according to latest policy. Usually more stable. e.g. A2C/A3C, PPO.
Q-Learning | Models the Q-function. Optimize based on Bellman equation. Typically **off-policy** - use any data, not necessarily with the latest policy. The corresponding policy is choosing the action with the largest Q-function value. e.g. DQN.

## Policy gradient

**Policy gradient** is the gradient of Expected return $J(\pi_\theta)$ (our objective) with respect to Policy's parameters $\theta$. We can derive a nice form for it.

```math
\nabla_\theta J(\pi_\theta) = \nabla_\theta \mathbb{E}_{\tau\sim \pi_\theta}[G(\tau)]
```
```math
= \nabla_\theta \int_\tau P(\tau|\theta) G(\tau)
```
```math
= \int_\tau \nabla_\theta P(\tau|\theta) G(\tau)
```
```math
= \int_\tau P(\tau|\theta) \nabla_\theta\log P(\tau|\theta) G(\tau)
```
```math
= \mathbb{E}_{\tau\sim\pi_\theta}\left[\nabla_\theta\log P(\tau|\theta) G(\tau)\right]
```
```math
= \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t) G(\tau)\right]
```

Note: A trajectory's log-prob is equal to log-prob of initial state, plus the log-prob sum of (action + state transition). Since initial state and state transition are properties of the environment, they do not depend on policy's parameters. Therefore, the derivative of trajectory's log-prob (wrt policy's parameters) is equal to the sum of log-policy's derivatives.

```math
\nabla_\theta\log P(\tau|\theta) = \sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t)
```

Note that the total number of timesteps $T$ needs not to be the same across trajectories.

In the sampling form (estimator)

```math
\hat g = \frac{1}{|D|} \sum_{\tau\in D}\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t) G(\tau)
```

Notice that once we sample the trajectories (using current policy), we can bring the gradient operator out of the sum. We can define the objective function:

```math
L(\theta) = \frac{1}{|D|} \sum_{\tau\in D}\sum_{t=0}^T \log\pi_\theta(a_t|s_t) G(\tau)
```

Please note that this loss function is only defined to simplify the computation of policy gradient using an autograd engine. **THERE IS NO MEANING** behind this loss function. Do not try to interpret its value.

**Other forms of Policy gradient** It turns out that we can replace $G(\tau)$ in Policy gradient equation with other terms that result in the same expected value. These alternatives can offer a tighter variance of Policy gradient estimator, making training more data-efficient. Here are some alternative forms (proofs won't be covered).

- Using on-policy Q-function

```math
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t) Q^{\pi_\theta}(s_t,a_t)\right]
```

- Using Advantage function - this is the most common since there are many ways to estimate the advantage function and it results in lowest variance.

```math
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t,a_t)\right]
```

Notice in the above two forms, all the terms inside the summation depends only on current $(s_t,a_t)$. Hence, we can specify a loss depending only on those.

### Estimate Advantage function

```math
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
```

One common way to estimate the Value function is with a neural network. We will train the Value function network together with the Policy network, using the same rollout data.

```math
L_{VF} = \mathbb{E}_{s\sim D}\left[(V_\phi(s)-R(s))^2\right]
```

Thanks to this, we can estimate the Value function at current state without integrating all possible trajectories.

**Generalized Advantage Estimator** (GAE) https://arxiv.org/abs/1506.02438

```math
\hat A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}
```

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```

Note that the Value function above is estimated as well, using least mean squares optimization mentioned previously.

When $\lambda=0$

```math
\hat A_t^{GAE(\gamma,0)} = r_t + \gamma V(s_{t+1}) - V(s_t)
```

This expression is TD-residual for the estimated Value function. $r_t + \gamma V(s_{t+1})$ is basically the Q-function if the $V(s_t)$ is a perfect estimate. This Advantage estimator has high bias since the Value function estimator is not accurate.

When $\lambda=1$

```math
\hat A_t^{GAE(\gamma,1)} = \sum_{l=0}^{\infty}\gamma^l r_{t+l} - V(s_t)
```

The summation term is Return of a particular trajectory sample starting at current state and action. This is an unbiased estimator of Q-function, but it has high variance, which we are trying to avoid.

### Trust Region Policy Estimation (TRPO)

https://arxiv.org/abs/1502.05477

Impose an additional constraint not to deviate too much from the old policy -> avoid big update that can collapse training.

**Surrogate advantage** is a measure of how policy $\pi_\theta$ performs relative to the old policy $\pi_{\theta_{old}}$, **using data from the old policy**.

```math
L(\theta_{old},\theta) = \mathbb{E}_{s,a\sim\pi_{\theta_{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}A^{\pi_{\theta_{old}}}(s,a)\right]
```

**Average KL-divergence** over states visited by the old policy

```math
\bar D_{KL}(\theta|\theta_{old}) = \mathbb{E}_{s\sim\pi_{\theta_{old}}}\left[D_{KL}\left(\pi_\theta(\cdot|s)|\pi_{\theta_{old}}(\cdot|s)\right)\right]
```

Theoretical TRPO update

```math
\theta_{k+1} = \arg\max_\theta L(\theta_k,\theta), \bar D_{KL}(\theta||\theta_k)\leq\delta
```

Note
- Gradient of surrogate advantage function (wrt policy's parameters) is still equal to policy gradient.
- Surrogate advantage is taking expectation over $(s,a)\sim\pi_{\theta_{old}}$, instead of summing over the time steps then taking expectation over trajectories. Not sure if the author is assuming $T$ is the same across different trajectories. TODO: double-check this

### Proximal Policy Optimization (PPO)

https://arxiv.org/abs/1707.06347

**PPO-clip** Loss function

```math
L_{PPO-Clip} = \mathbb{E}_{s,a\sim\pi_{\theta_{old}}}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}A^{\pi_{\theta_{old}}},\mathrm{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)},1-\epsilon,1+\epsilon\right)A^{\pi_{\theta_{old}}}\right)\right]
```

Note that since Advantage $A^{\pi_{\theta_{old}}}$ can be either positive or negative, we can't factor out the Advantage term.

**PPO-Penalty** (not used much) Loss function

```math
L_{PPO-Penalty} = \mathbb{E}_{s,a\sim\pi_{\theta_{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}A^{\pi_{\theta_{old}}}-\beta D_{KL}\left(\pi_{\theta_{old}}(\cdot|s)|\pi_\theta(\cdot|s)\right)\right]
```

Where the penalty coefficient $\beta$ is adjusted automatically over the course of training.

## RL algorithms for LLM fine-tuning

**Reinforcement Learning from Human Feedback (RLHF) + PPO** (OpenAI)

- https://arxiv.org/abs/1909.08593 / https://github.com/openai/lm-human-preferences
- https://arxiv.org/abs/2009.01325 / https://github.com/openai/summarize-from-feedback
- https://arxiv.org/abs/2203.02155 (InstructGPT)

Given prompt $x$ and completion $y$. We consider this to be a **contextual bandit environment** i.e. single-step RL, since (1) user prompt $x$ is the context, and (2) we only have reward on the full response. Action space is all possible $y\in V^{|y|}$, where $V$ is the vocabulary of the LLM.

The Reward function is set as

```math
R(x,y) = r_\phi(x,y) - \beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
```

Where $r_\phi(x,y)$ is the **Reward model**, and the KL divergence term is used to keep the RL model not deviate too much from the SFT model.

The Reward model is frozen during RL fine-tuning. InstructGPT uses the following objective (very similar to Discriminator loss in Relativistic GAN), where $y_1$ is preferred over $y_0$. TODO: check how data is collected

```math
L_{RM}(\phi) = -\mathbb{E}_{(x,y_0,y_1)\sim D}\left[\log\sigma\left(r_\phi(x,y_1)-r_\phi(x,y_0)\right)\right]
```

The OpenAI papers above then use PPO-Clip to RL-finetune the model with this engineered reward. GAE is used to estimate the Advantage function, which requires estimating the **Value function** with a neural network. This Value function is initialized from the Reward model.

TODO: check OpenAI code to see how they normalize the loss across tokens / samples.

**Direct Preference Optimization** (DPO) https://arxiv.org/abs/2305.18290

Use Reward model's objective to optimize the LLM directly

```math
L_{DPO}(\theta) = -\mathbb{E}_{(x,y_0,y_1)\sim D}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_1|x)}{\pi_{ref}(y_1|x)}-\beta\log\frac{\pi_\theta(y_0|x)}{\pi_{ref}(y_0|x)}\right)\right]
```

**Reinforcement Learning with Verifiable Rewards** (RLVR) First coined in Tulu3 (https://arxiv.org/abs/2411.15124). The reward function comes from an easily verifiable procedure e.g. answer to a math problem.

**Group Relative Policy Optimization** (GRPO) https://arxiv.org/abs/2402.03300

In this setup, state is prefix (prompt $x$ + tokens generated so far $y_{1..t-1}$), and action is next token $y_t$. Number of timesteps is equal to number of generated tokens.

For a given prompt $x$, sample a few responses $\{y\}$ using the old policy. GRPO loss defined on a single $(x,y)$ pair is:

```math
L_{GRPO}(x,y,\theta) = \frac{1}{|y|}\sum_{t=1}^{|y|}\left\{\min\left[\frac{\pi_\theta(y_t|x,y_{1..t-1})}{\pi_{\theta_{old}}(y_t|x,y_{1..t-1})}\hat A_t,\mathrm{clip}\left(\frac{\pi_\theta(y_t|x,y_{1..t-1})}{\pi_{\theta_{old}}(y_t|x,y_{1..t-1})},1-\epsilon,1+\epsilon\right)\hat A_t\right]\right\}-\beta D_{KL}(\pi_\theta||\pi_{ref})
```

This is basically PPO-Clip loss with extra KL penalty term (the KL term is opposite direction of PPO-Penalty). Advantage is estimated based on **relative rewards within a group** (hence the name). Also notice that KL penalty is not part of the reward, like RLHF, but part of the loss. TODO: check how KL term is computed.

If we only have reward at the end of each rollout $r(x,y)$, Advantage is estimated as normalized rewards.

```math
\hat A_t = \frac{r(x,y) - \mathrm{mean}(\{r(x,y_i)\}_i)}{\mathrm{std}(\{r(x,y_i)\}_i)}
```

If we have rewards at intermediate steps (process supervision), Advantage is estimated as the sum of current and future normalized rewards. TODO: check how reward normalization is done in this case. Each rollout can have different number of (intermediate) rewards.

The full GRPO loss is then averaged over sampled responses $y\sim\pi_{\theta_{old}}$ for a single $x$, then averaged over sampled prompts $x$.

TODO: check DeepSeek-R1 as well

**GRPO Done Right** (Dr. GRPO) https://arxiv.org/abs/2503.20783

Length normalization is removed

```math
L_{Dr.GRPO}(x,y,\theta) = \sum_{t=1}^{|y|}\left\{\min\left[\frac{\pi_\theta(y_t|x,y_{1..t-1})}{\pi_{\theta_{old}}(y_t|x,y_{1..t-1})}\hat A_t,\mathrm{clip}\left(\frac{\pi_\theta(y_t|x,y_{1..t-1})}{\pi_{\theta_{old}}(y_t|x,y_{1..t-1})},1-\epsilon,1+\epsilon\right)\hat A_t\right]\right\}-\beta D_{KL}(\pi_\theta||\pi_{ref})
```

Reward normalization only shifts mean to zero. Don't rescale it to unit variance

```math
\hat A_t = r(x,y) - \mathrm{mean}(\{r(x,y_i)\}_i)
```

**Decoupled Clip and Dynamic Sampling Policy Optimization** (DAPO) https://arxiv.org/abs/2503.14476

On a prompt-response pair $(x,y)$, DAPO modifies PPO loss to have 2 separate clipping values

```math
L_{DAPO}(x,y,\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}\hat A_{t},\mathrm{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)},1-\epsilon_\mathrm{low},1+\epsilon_\mathrm{high}\right)\hat A_t\right)
```

Similar to GRPO, Advantage is estimated as normalized rewards. The other important difference is
1. How sampling is done: Remove all prompts that have 0 or 1 accuracy i.e. sampled responses are all wrong or all correct. This is to avoid normalized reward being zero, which will contribute to training instability.
2. How expectation is computed: Loss is averaged over **all tokens in a batch**, instead of:
  - GRPO: loss for each response is averged over its length (per-sample), then averaged over rollouts.
  - Dr. GRPO: loss for each response is **summed** over its length (per-sample), then averaged over rollouts.
3. Special treatment for truncated responses: Long responses are truncated. DAPO will either ignore truncated samples (**overlong filtering**) or apply a penalty to the reward function (**soft overlong punishment**).

Unlike GRPO and OpenAI RLHF, DAPO doesn't include any KL penalty term at all, whether in its reward function or loss function.
