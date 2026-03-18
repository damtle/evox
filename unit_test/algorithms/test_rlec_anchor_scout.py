import torch

from rlec.control.intent_vector import IntentVector
from rlec.control.interpreter import ControlInterpreter
from rlec.control.niche_manager import NicheManager
from rlec.features.macro_state import MacroStateBuilder
from rlec.features.stage_reward import StageRewardCalculator


def _make_manager() -> NicheManager:
    return NicheManager(
        pop_size=8,
        dim=2,
        device=torch.device("cpu"),
        complete_patience=2,
        max_anchor_niches=2,
        merge_dist=0.5,
        merge_fit_eps=0.2,
    )


def test_completion_patience_requires_consecutive_stages():
    mgr = _make_manager()

    pop = torch.tensor(
        [
            [0.00, 0.00],
            [0.01, 0.00],
            [0.00, 0.01],
            [1.00, 1.00],
            [1.01, 1.00],
            [1.00, 1.01],
            [2.00, 2.00],
            [2.01, 2.00],
        ],
        dtype=torch.float32,
    )
    fit = torch.tensor([0.1, 0.12, 0.11, 0.2, 0.21, 0.19, 0.4, 0.41], dtype=torch.float32)

    stage_start_pop = pop * 1.5
    stage_start_fit = fit + 1e-4
    stage_start_stag = torch.zeros(pop.shape[0])
    stag = torch.zeros(pop.shape[0])
    intent_eff = torch.tensor([0.4, 0.5, 0.2, 0.3, 0.2, 0.5], dtype=torch.float32)

    plan1 = mgr.plan(pop, fit, stage_start_pop, stage_start_fit, stage_start_stag, stag, intent_eff, initial_div=1.0)
    assert not bool(plan1.completed_mask.any().item())

    plan2 = mgr.plan(pop, fit, stage_start_pop, stage_start_fit, stage_start_stag, stag, intent_eff, initial_div=1.0)
    assert bool(plan2.completed_mask.any().item())


def test_overlap_group_uses_distance_and_fitness_conditions():
    mgr = _make_manager()
    centers = torch.tensor([[0.0, 0.0], [0.1, 0.1], [3.0, 3.0]], dtype=torch.float32)
    best_fit = torch.tensor([1.0, 1.05, 1.01], dtype=torch.float32)

    gids = mgr._detect_overlap_groups(centers, best_fit, merge_dist=0.5, merge_fit_eps=0.1)

    assert int(gids[0].item()) == int(gids[1].item())
    assert int(gids[0].item()) != int(gids[2].item())


def test_anchor_selection_respects_max_count():
    mgr = _make_manager()
    rows = [
        {"status": "refining", "is_overlap_redundant": False, "best_fit": 1.0, "niche_age": 3, "radius_rel": 0.02, "best_improve_rel": 0.001},
        {"status": "completed", "is_overlap_redundant": False, "best_fit": 0.9, "niche_age": 2, "radius_rel": 0.01, "best_improve_rel": 0.0002},
        {"status": "refining", "is_overlap_redundant": False, "best_fit": 0.95, "niche_age": 1, "radius_rel": 0.03, "best_improve_rel": 0.002},
    ]
    selected = mgr._select_anchor_niches(rows, max_anchor_niches=2)
    assert len(selected) == 2


def test_interpreter_anchor_blocks_rescue_and_scales_down():
    interp = ControlInterpreter(pop_size=6)
    intent = IntentVector(e_t=0.6, x_t=0.4, d_t=0.5, b_t=0.6, r_t=0.6, p_t=0.2)
    fit = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
    stag = torch.tensor([0.0, 12.0, 10.0, 5.0, 8.0, 7.0], dtype=torch.float32)

    anchor = torch.tensor([True, True, False, False, False, False])
    scout = torch.tensor([False, False, True, True, False, False])
    completed = torch.tensor([False, False, False, False, True, False])
    guardian = torch.tensor([True, False, False, False, False, False])

    cmd = interp.interpret(
        intent,
        fit,
        stag,
        initial_div=1.0,
        niche_roles={
            "anchor_mask": anchor,
            "scout_mask": scout,
            "completed_mask": completed,
            "guardian_mask": guardian,
        },
    )

    assert not bool(cmd.rescue_mask[anchor].any().item())
    assert bool((cmd.step_scales[anchor] <= 0.5).all().item())


def test_macro_state_and_reward_dimensions():
    builder = MacroStateBuilder(dim=2, pop_size=8)
    assert builder.state_dim == 30

    pop = torch.rand((8, 2))
    fit = torch.rand(8)
    stagnation = torch.zeros(8)
    state = builder.build(pop, fit, pop, fit, stagnation, torch.zeros(6), torch.zeros(4), torch.zeros(12))
    assert state.shape[-1] == 30

    calc = StageRewardCalculator()
    reward = calc.calculate(
        fit_start=torch.ones(8),
        fit_end=torch.ones(8) * 0.9,
        div_start=1.0,
        div_end=0.9,
        target_div=0.8,
        initial_div=1.0,
        stagnation_start=torch.zeros(8),
        stag_ratio_start=0.2,
        stag_ratio_end=0.1,
        action=torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        feedbacks=torch.tensor([0.2, 0.3, 0.1, 0.0]),
        niche_stats=torch.tensor([0.1, 0.1, 0.1, 0.9, 0.5, 0.8, 0.3, 0.2]),
    )
    assert isinstance(reward, float)
