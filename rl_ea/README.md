# RL-EA EvoX Template

This folder contains a runnable template for adding a reinforcement-learning search module
on top of an EvoX evolutionary algorithm. The provided concrete example is **RL-enhanced PSO**.

## Directory layout

- `algorithms/rl_wrapper.py`  
  Generic wrapper that can enhance any EvoX algorithm exposing `pop`, `fit`, `lb`, `ub`, `step`, and `init_step`.
- `algorithms/pso_rl.py`  
  Convenience factory for RL-enhanced PSO.
- `rl/networks.py`  
  Actor/Critic networks.
- `rl/agent.py`  
  TD3 agent.
- `replay/buffer.py`  
  Prioritized replay buffer specialized for optimization traces.
- `features/state_builder.py`  
  Builds state = `[x, Δx, f, Δf]`.
- `utils/optimization.py`  
  Reward helpers.
- `run_example.py`  
  Minimal runnable example on the Sphere problem.

## Installation assumptions

This code assumes:
1. Python 3.10+
2. PyTorch installed
3. EvoX installed and importable

## How to run

```bash
cd /mnt/data/rl_ea_evox
python run_example.py
```

## How to use with your own EvoX workflow

If you already have an EvoX workflow, the usual pattern is:

```python
from rl_ea_evox.algorithms.pso_rl import make_rl_pso

algo = make_rl_pso(pop_size=64, lb=lb, ub=ub, device=device)
workflow = YourEvoXWorkflow(algorithm=algo, problem=problem, ...)
workflow.init_step()
for _ in range(num_generations):
    workflow.step()
```

The wrapper relies on EvoX's usual behavior that `Algorithm.evaluate` is proxied by the workflow.

## How to adapt to other algorithms

If you want to enhance DE/GA/CMA-ES:
1. Construct the base EvoX algorithm instance.
2. Create `TD3Agent` with `state_dim = 2 * dim + 2`, `action_dim = dim`.
3. Wrap it with `RLEnhancedAlgorithm`.

Example:

```python
base = SomeEvoXAlgorithm(...)
agent = TD3Agent(state_dim=2*dim+2, action_dim=dim, max_action=(ub-lb)*0.1, device=device)
replay = PrioritizedReplayBuffer(capacity=200000, device=device)
algo = RLEnhancedAlgorithm(base, agent, replay, rl_candidate_ratio=0.3, ...)
```

## Important design notes

- The RL module does **not** replace the EA. It adds RL-generated candidates on top of EA candidates.
- The replay buffer stores both EA-generated and RL-generated transitions.
- Rewards use relative fitness improvement for minimization:

```math
r_t = rac{f(x_t)-f(x_{t+1})}{|f(x_t)| + \epsilon} + bonus
```

- The current code is a practical baseline, not the final most advanced version. Good next upgrades are:
  - n-step returns
  - step-reuse augmentation
  - separate elite / normal / exploration buffers
  - adaptive EA/RL mixing ratio

## Suggested next upgrade path

1. Verify this baseline runs end-to-end.
2. Add n-step return computation in the replay sampler.
3. Add dynamic `rl_candidate_ratio` based on recent RL vs EA contribution.
4. Add `step reuse` data augmentation with true re-evaluation.
