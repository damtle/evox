"""Minimal example for using the RL-enhanced PSO in EvoX.

This script assumes you already have EvoX installed and available in the environment.
It uses a simple Sphere problem through an EvoX-compatible workflow. If your local
workflow API differs slightly, keep the algorithm construction unchanged and adapt
only the workflow section.
"""

from __future__ import annotations

import torch

from rl_ea.algorithms.pso_rl import make_rl_pso


class SphereProblem:
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * x, dim=1)


class SimpleWorkflow:
    """A lightweight stand-in workflow showing how EvoX injects evaluate into the algorithm.

    If you already use EvoX workflows, you likely won't need this class. It is provided so the
    example is runnable even without remembering the exact high-level workflow API.
    """

    def __init__(self, algorithm, problem):
        self.algorithm = algorithm
        self.problem = problem
        # In EvoX, Algorithm.evaluate is normally proxied by workflow.
        self.algorithm.evaluate = self.problem.evaluate

    def init_step(self):
        self.algorithm.init_step()

    def step(self):
        self.algorithm.step()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 10
    pop_size = 64
    lb = -5.0 * torch.ones(dim, device=device)
    ub = 5.0 * torch.ones(dim, device=device)

    algo = make_rl_pso(pop_size=pop_size, lb=lb, ub=ub, device=device)
    problem = SphereProblem()
    workflow = SimpleWorkflow(algo, problem)

    workflow.init_step()
    for gen in range(200):
        workflow.step()
        best = float(algo.base.global_best_fit.item())
        if gen % 10 == 0:
            print(f'gen={gen:04d}, best={best:.6f}')
