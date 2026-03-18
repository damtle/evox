import torch

from rlec.control.intent_vector import IntentVector
from rlec.control.interpreter import ControlInterpreter
from rlec.control.subpopulation_manager import SubpopulationManager
from rlec.features.macro_state import MacroStateBuilder
from rlec.features.stage_reward import StageRewardCalculator


def _make_manager() -> SubpopulationManager:
    return SubpopulationManager(
        pop_size=20,
        dim=2,
        device=torch.device("cpu"),
        exploit_ratio=0.20,
        bridge_ratio=0.30,
        explore_ratio=0.50,
        migration_period=4,
    )


def test_subpop_init_ratio_and_unique_membership():
    mgr = _make_manager()
    mgr.init_subpops()

    ids = mgr.subpop_ids
    assert ids.shape[0] == 20
    assert bool(((ids >= 0) & (ids <= 2)).all().item())

    n_exploit = int((ids == SubpopulationManager.EXPLOIT).sum().item())
    n_bridge = int((ids == SubpopulationManager.BRIDGE).sum().item())
    n_explore = int((ids == SubpopulationManager.EXPLORE).sum().item())
    assert n_exploit + n_bridge + n_explore == 20
    assert n_exploit >= 1 and n_bridge >= 1 and n_explore >= 1


def test_migration_and_apply_updates_ids_and_age():
    mgr = _make_manager()
    mgr.init_subpops()

    pop = torch.rand((20, 2))
    fit = torch.linspace(0.0, 1.0, 20)
    prev_fit = fit + 0.1
    stagnation = torch.zeros(20)

    # Force exploit stagnation to enable x2e.
    exploit_idx = torch.where(mgr.subpop_ids == SubpopulationManager.EXPLOIT)[0]
    if exploit_idx.numel() > 0:
        stagnation[exploit_idx[:1]] = 15.0

    intent = torch.tensor([0.7, 0.3, 0.6, 0.8, 0.7, 0.5])
    plan = mgr.plan_stage(pop, fit, prev_fit, stagnation, stage_id=4, intent_eff=intent)

    moved_before = int(
        plan.migrate_to_exploit.sum().item()
        + plan.migrate_to_bridge.sum().item()
        + plan.migrate_to_explore.sum().item()
    )
    mgr.apply_migration(plan, stage_id=4)

    assert mgr.subpop_ids.shape[0] == 20
    if moved_before > 0:
        assert mgr.last_migration_stage == 4


def test_interpreter_subpop_role_behavior():
    interp = ControlInterpreter(pop_size=6)
    intent = IntentVector(e_t=0.7, x_t=0.4, d_t=0.5, b_t=0.6, r_t=0.6, p_t=0.4)
    fit = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
    stag = torch.tensor([12.0, 10.0, 8.0, 6.0, 5.0, 3.0], dtype=torch.float32)

    exploit = torch.tensor([True, True, False, False, False, False])
    bridge = torch.tensor([False, False, True, True, False, False])
    explore = torch.tensor([False, False, False, False, True, True])
    guardian = torch.tensor([True, False, False, False, False, False])

    cmd = interp.interpret(
        intent,
        fit,
        stag,
        initial_div=1.0,
        subpop_roles={
            "exploit_mask": exploit,
            "bridge_mask": bridge,
            "explore_mask": explore,
            "exploit_guardian_mask": guardian,
        },
    )

    assert not bool(cmd.rescue_mask[exploit].any().item())
    assert bool((cmd.step_scales[explore] >= cmd.step_scales[bridge]).all().item())


def test_macro_state_and_reward_interfaces():
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
        subpop_stats=torch.tensor([0.8, 0.6, 0.5, 0.7, 0.2]),
    )
    assert isinstance(reward, float)
