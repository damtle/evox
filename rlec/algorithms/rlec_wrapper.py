from typing import Optional
import torch

from evox.core import Algorithm, Mutable
from evox.utils import clamp

from rlec.features.macro_state import MacroStateBuilder
from rlec.features.stage_reward import StageRewardCalculator
from rlec.control.intent_vector import IntentVector
from rlec.control.interpreter import ControlInterpreter
from rlec.control.niche_manager import NicheManager
from rlec.rl.ppo import PPO
from rlec.utils.population_metrics import compute_diversity


class RLECWrapper(Algorithm):
    """
    RLEC 寮哄寲瀛︿範杩涘寲闃舵鎺у埗鍣?
    灏嗗簳灞?EA 瑙嗕负鎴樻湳鎵ц灞傦紝RL 瑙嗕负鎴樼暐璋冨害灞傘€?
    """

    def __init__(
            self,
            base_algo: Algorithm,
            stage_length: int = 10,  # K 浠ｄ负涓€涓喅绛栧懆鏈?
            update_stages: int = 2,  # 鏀堕泦澶氬皯涓樁娈电殑鏁版嵁鏇存柊涓€娆?PPO
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.base = base_algo
        self.K = stage_length
        self.update_stages = update_stages
        self.device = device if device is not None else base_algo.fit.device

        pop_shape = base_algo.pop.shape
        self.pop_size, self.dim = pop_shape[0], pop_shape[1]

        # 鍒濆鍖栦笁澶т欢锛氶浄杈俱€佽瘎鍒嗐€佺炕璇戝畼
        self.state_builder = MacroStateBuilder(self.dim, self.pop_size)
        self.reward_calc = StageRewardCalculator()
        self.interpreter = ControlInterpreter(self.pop_size)
        self.niche_manager = NicheManager(self.pop_size, self.dim, self.device)

        # 鍒濆鍖?PPO 澶ц剳 (鐘舵€佺淮搴?12锛屽姩浣滅淮搴?6)
        self.rl_agent = PPO(
            state_dim=self.state_builder.state_dim,
            action_dim=6,
            device=self.device
        )
        self.initial_div = Mutable(torch.tensor(0.0, device=self.device))
        self.current_cmd_alpha_margin = Mutable(torch.tensor(0.0, device=self.device))
        self.current_cmd_beta_margin = Mutable(torch.tensor(0.0, device=self.device))

        self.current_cmd_scales = Mutable(torch.ones(self.pop_size, device=self.device))

        # ================= 鏃跺簭杩借釜鐘舵€?=================
        self.step_counter = Mutable(torch.tensor(0, device=self.device))
        self.stagnation = Mutable(torch.zeros(self.pop_size, device=self.device))

        # 璁板綍涓婁竴浠ｇ殑绉嶇兢 (鐢ㄤ簬绠楁彁鍗囧拰鏇存柊 stagnation)
        self.prev_pop = Mutable(base_algo.pop.clone())
        self.prev_fit = Mutable(torch.full_like(base_algo.fit, torch.inf))

        # 璁板綍鈥滈樁娈靛垵鈥濈殑绉嶇兢锛岀敤浜庡湪 K 浠ｇ粨鏉熷悗绠楁€昏处
        self.stage_start_pop = Mutable(base_algo.pop.clone())
        self.stage_start_fit = Mutable(torch.full_like(base_algo.fit, torch.inf))
        self.stage_start_div = Mutable(torch.tensor(0.0, device=self.device))
        self.stage_start_stagnation = Mutable(torch.zeros(self.pop_size, device=self.device))

        # ================= RL 璁板繂杩借釜 =================
        self.hidden_a = Mutable(torch.zeros(1, 1, 64, device=self.device))
        self.hidden_c = Mutable(torch.zeros(1, 1, 64, device=self.device))
        self.last_state = Mutable(torch.zeros(1, self.state_builder.state_dim, device=self.device))
        self.last_action_raw = Mutable(torch.zeros(6, device=self.device))
        self.last_intent = Mutable(torch.zeros(6, device=self.device))
        self.last_logprob = Mutable(torch.tensor(0.0, device=self.device))
        self.last_value = Mutable(torch.tensor(0.0, device=self.device))
        self.last_roi = Mutable(torch.tensor(0.0, device=self.device))

        # ================= 鎴樻湳鎸囦护缂撳瓨 =================
        # 鐢ㄤ簬瀛樻斁褰撳墠 K 浠ｄ腑锛屽簳灞?EA 蹇呴』鏈嶄粠鐨勬寚浠?
        self.current_cmd_protected = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_cmd_rescue = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        # 銆愭柊澧炪€戯細蹇呴』缂撳瓨瑙ｉ噴鍣ㄨ绠楀ソ鐨?middle_mask锛屼笉瑕佽嚜宸遍噸绠楋紒
        self.current_cmd_middle = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_cmd_quiet = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_cmd_target_div = Mutable(torch.tensor(0.0, device=self.device))
        self.current_niche_guardian = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_niche_converged = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_gate_refine = Mutable(torch.tensor(0.0, device=self.device))
        self.current_gate_recover = Mutable(torch.tensor(0.0, device=self.device))
        self.last_intent_eff = Mutable(torch.zeros(6, device=self.device))

        # 鍖呰鍣ㄨ嚜宸辩淮鎶ょ殑鏈€浼樿褰?
        self.global_best_fit = Mutable(torch.tensor(torch.inf, device=self.device))
        self.global_best_location = Mutable(torch.zeros(self.dim, device=self.device))

        # 鍦?__init__ 涓柊澧炴墽琛屽弽棣堢殑 Mutable 杩借釜鍙橀噺
        self.last_intervene_ratio = Mutable(torch.tensor(0.0, device=self.device))
        self.last_accept_ratio = Mutable(torch.tensor(0.0, device=self.device))
        self.last_rescue_success = Mutable(torch.tensor(0.0, device=self.device))
        self.last_elite_damage = Mutable(torch.tensor(0.0, device=self.device))


    # [姝ゅ鏆撮湶 pop, fit 绛夊睘鎬э紝淇濇寔涓?EVOX 鐨勫吋瀹规€
    @property
    def pop(self):
        return self.base.pop

    @pop.setter
    def pop(self, value):
        self.base.pop = value

    @property
    def fit(self):
        return self.base.fit

    @fit.setter
    def fit(self, value):
        self.base.fit = value

    @property
    def lb(self):
        return self.base.lb

    @property
    def ub(self):
        return self.base.ub

    def init_step(self):
        if hasattr(self, 'evaluate'):
            self.base.evaluate = self.evaluate
        self.base.init_step()

        self.prev_pop = self.base.pop.clone()
        self.prev_fit = self.base.fit.clone()
        self.stage_start_pop = self.base.pop.clone()
        self.stage_start_fit = self.base.fit.clone()
        self.stage_start_div = compute_diversity(self.base.pop)

        best_val, best_idx = torch.min(self.base.fit, dim=0)
        self.global_best_fit = best_val.clone()
        self.global_best_location = self.base.pop[best_idx].clone()

        # 鍒濆鍖栨椂瑙﹀彂绗?0 娆¤皟搴?
        self.initial_div = compute_diversity(self.base.pop)
        self._dispatch_new_stage()

    def _dispatch_new_stage(self):
        """鍙镐护閮ㄤ笅杈炬柊闃舵鎸囦护"""
        current_pop = self.base.pop
        current_fit = self.base.fit
        current_div = compute_diversity(current_pop)

        # 1. 鎻愬彇 12 缁村畯瑙傛€佸娍闆疯揪鍥?
        state = self.state_builder.build(
            pop_t=current_pop, fit_t=current_fit,
            pop_t_minus_k=self.stage_start_pop, fit_t_minus_k=self.stage_start_fit,
            stagnation=self.stagnation,
            last_action=self.last_intent_eff,
            feedback_stats=torch.stack([
                self.last_intervene_ratio,
                self.last_accept_ratio,
                self.last_rescue_success,
                self.last_elite_damage
            ]).to(self.device)
        )

        # 2. PPO-LSTM 澶ц剳鎬濊€冿紝鐢熸垚鏂版寚浠?
        # state: [1, 12]
        action_raw, intent_np, log_prob, value = self.rl_agent.select_action(
            state, deterministic=False
        )

        self.last_state = state.clone()

        # 3. 缈昏瘧瀹樼炕璇戞寚浠?
        intent_tensor = torch.tensor(intent_np, device=self.device, dtype=torch.float32)
        intent_obj = IntentVector.from_tensor(intent_tensor)

        # Continuous gates: do not overwrite actions, only rescale their effects.
        stage_best_start = torch.min(self.stage_start_fit)
        stage_best_now = torch.min(current_fit)
        stage_progress = (stage_best_start - stage_best_now) / (torch.abs(stage_best_start) + 1e-8)
        div_ratio = current_div / (self.initial_div + 1e-8)
        best_now = torch.min(current_fit)
        median_now = torch.median(current_fit)
        spread_ratio = (median_now - best_now) / (torch.abs(best_now) + 1e-8)
        stag_ratio = torch.mean((self.stagnation >= 10).float())

        refine_div = torch.clamp((0.15 - div_ratio) / 0.15, 0.0, 1.0)
        refine_spread = torch.clamp((0.20 - spread_ratio) / 0.20, 0.0, 1.0)
        refine_prog = torch.clamp((1e-4 - stage_progress) / 1e-4, 0.0, 1.0)
        refine_stag = torch.clamp((stag_ratio - 0.2) / 0.6, 0.0, 1.0)
        g_refine = torch.clamp(
            0.35 * refine_div
            + 0.25 * refine_spread
            + 0.20 * refine_prog
            + 0.20 * refine_stag,
            0.0,
            1.0,
        )

        recover_div = torch.clamp((0.18 - div_ratio) / 0.18, 0.0, 1.0)
        recover_prog = torch.clamp((1e-4 - stage_progress) / 1e-4, 0.0, 1.0)
        recover_spread = torch.clamp((spread_ratio - 0.20) / 0.40, 0.0, 1.0)
        g_recover = torch.clamp(
            0.40 * recover_div
            + 0.30 * recover_prog
            + 0.30 * recover_spread,
            0.0,
            1.0,
        ) * (1.0 - g_refine)

        e_eff = torch.clamp(intent_tensor[0] * (1.0 - 0.85 * g_refine) + 0.50 * g_recover * (1.0 - intent_tensor[0]), 0.0, 1.0)
        x_eff = torch.clamp(intent_tensor[1] + 0.25 * g_refine * (1.0 - intent_tensor[1]), 0.0, 1.0)
        d_eff = torch.clamp(intent_tensor[2] * (1.0 - 0.50 * g_refine) + 0.30 * g_recover * (1.0 - intent_tensor[2]), 0.0, 1.0)
        b_eff = torch.clamp(intent_tensor[3] * (1.0 - 0.75 * g_refine) + 0.45 * g_recover * (1.0 - intent_tensor[3]), 0.0, 1.0)
        r_eff = torch.clamp(intent_tensor[4] * (1.0 - 0.90 * g_refine) + 0.35 * g_recover * (1.0 - intent_tensor[4]), 0.0, 1.0)
        p_eff = torch.clamp(intent_tensor[5] + 0.30 * g_refine * (1.0 - intent_tensor[5]), 0.0, 1.0)

        intent_eff_tensor = torch.stack([e_eff, x_eff, d_eff, b_eff, r_eff, p_eff]).to(self.device)

        intent_eff_obj = IntentVector.from_tensor(intent_eff_tensor)

        self.current_gate_refine = g_refine.detach()
        self.current_gate_recover = g_recover.detach()

        cmd = self.interpreter.interpret(intent_eff_obj, current_fit, self.stagnation, self.initial_div.item())

        self.last_action_raw = torch.tensor(action_raw, device=self.device)
        self.last_intent = intent_tensor.clone()
        self.last_intent_eff = intent_eff_tensor.clone()
        self.last_logprob = torch.tensor(log_prob, device=self.device)
        self.last_value = torch.tensor(value, device=self.device)

        self.current_cmd_protected = cmd.protected_mask
        self.current_cmd_middle = cmd.middle_mask
        self.current_cmd_quiet = cmd.quiet_mask
        self.current_cmd_rescue = cmd.rescue_mask
        self.current_cmd_scales = cmd.step_scales.clone()
        # 璁板綍鍙屽蹇嶅害
        self.current_cmd_alpha_margin = torch.tensor(cmd.alpha_margin, device=self.device)
        self.current_cmd_beta_margin = torch.tensor(cmd.beta_margin, device=self.device)
        self.current_cmd_target_div = torch.tensor(cmd.target_diversity, device=self.device)

        # Niche-level lifecycle plan: classify subpopulations and allocate release/rescue budgets.
        niche_plan = self.niche_manager.plan(
            pop=current_pop,
            fit=current_fit,
            stage_start_pop=self.stage_start_pop,
            stage_start_fit=self.stage_start_fit,
            stage_start_stagnation=self.stage_start_stagnation,
            stagnation=self.stagnation,
            intent_eff=intent_eff_tensor,
            initial_div=float(self.initial_div.item()),
        )
        self.current_niche_guardian = niche_plan.guardian_mask
        self.current_niche_converged = niche_plan.converged_mask
        self.current_cmd_protected = self.current_cmd_protected | niche_plan.guardian_mask
        self.current_cmd_quiet = self.current_cmd_quiet | niche_plan.guardian_mask
        self.current_cmd_middle = self.current_cmd_middle & (niche_plan.exploring_mask | niche_plan.refining_mask) & (~niche_plan.guardian_mask)
        self.current_cmd_rescue = self.current_cmd_rescue & niche_plan.rescue_mask & niche_plan.exploring_mask & (~niche_plan.guardian_mask)
        # Niche-aware scale correction after role slicing.
        self.current_cmd_scales[niche_plan.refining_mask] = self.current_cmd_scales[niche_plan.refining_mask] * 0.6
        self.current_cmd_scales[niche_plan.converged_mask & (~niche_plan.guardian_mask)] = 0.0

        release_idx = torch.where(niche_plan.release_mask)[0]
        if release_idx.numel() > 0:
            new_pos = self.niche_manager.sample_release_positions(
                int(release_idx.numel()),
                self.lb,
                self.ub,
                avoid_centers=niche_plan.active_centers,
            )
            self.base.pop[release_idx] = new_pos
            self.base.fit[release_idx] = self.evaluate(new_pos)
            self.stagnation[release_idx] = 0.0
            self.prev_pop[release_idx] = self.base.pop[release_idx]
            self.prev_fit[release_idx] = self.base.fit[release_idx]
            # Clear stale lifecycle labels from released particles immediately.
            self.current_niche_converged[release_idx] = False
            self.current_niche_guardian[release_idx] = False
            self.current_cmd_quiet[release_idx] = False
            self.current_cmd_protected[release_idx] = False
            self.current_cmd_rescue[release_idx] = False
            self.current_cmd_middle[release_idx] = False
            self.current_cmd_scales[release_idx] = 1.0
            current_div = compute_diversity(self.base.pop)

        # Sync any potential interpreter-side normalized clipping back to effective action.
        intent_eff_tensor = torch.tensor(
            [
                intent_eff_obj.e_t, intent_eff_obj.x_t, intent_eff_obj.d_t,
                intent_eff_obj.b_t, intent_eff_obj.r_t, intent_eff_obj.p_t
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self.last_intent_eff = intent_eff_tensor.clone()

        # # 5. 銆愮‖鏍告垬鏈姩浣滐細鐮村眬鏁戞彺銆?
        # # 鍦ㄩ樁娈靛垵锛屽琚爣璁颁负 rescue 鐨勭矑瀛愭柦鍔犳棤瑙嗙墿鐞嗚寰嬬殑闅忔満浼犻€侊紒
        # rescue_idx = torch.where(cmd.rescue_mask)[0]
        # if len(rescue_idx) > 0:
        #     new_pos = self.lb + torch.rand(len(rescue_idx), self.dim, device=self.device) * (self.ub - self.lb)
        #     self.base.pop[rescue_idx] = new_pos
        #     self.base.fit[rescue_idx] = self.evaluate(new_pos)
        #     # 閲嶆柊璁＄畻鏁戞彺鍚庣殑鐪熷疄澶氭牱鎬?
        #     current_div = compute_diversity(self.base.pop)

        # 6. 鏇存柊鏂伴樁娈电殑璧风偣閿氱偣 (姝ゆ椂绉嶇兢宸茬粡鏄晳鎻村悗鐨勫共鍑€鐘舵€佷簡)
        self.stage_start_pop = self.base.pop.clone()
        self.stage_start_fit = self.base.fit.clone()
        self.stage_start_div = current_div
        self.stage_start_stagnation = self.stagnation.clone()

    def step(self):
        step_val = int(self.step_counter.item())

        # =====================================================================
        # 绗竴闃舵锛氭竻绠椾笂涓€闃舵濂栧姳锛屽苟涓嬭揪鏂伴樁娈垫寚浠?(姣?K 浠ｈЕ鍙戜竴娆?
        # =====================================================================
        if step_val > 0 and step_val % self.K == 0:
            current_div = compute_diversity(self.base.pop)

            # 1. 缁撶畻杩欎釜鍛ㄦ湡鐨勬€昏处 (K浠ｅ鍔?
            stag_ratio_start = (self.stage_start_stagnation >= 10).sum().float() / self.pop_size
            stag_ratio_end = (self.stagnation >= 10).sum().float() / self.pop_size

            # 銆愭柊澧炪€戯細灏嗘敹闆嗗埌鐨勬墽琛屽弽棣堟墦鍖呬紶缁欏鍔卞嚱鏁?
            feedbacks = torch.stack([
                self.last_intervene_ratio,
                self.last_accept_ratio,
                self.last_rescue_success,
                self.last_elite_damage
            ]).to(self.device)

            # 銆愪慨鏀广€戯細鍙帴鏀?reward锛堝墺绂?ROI锛夛紝骞惰ˉ榻?feedbacks 鍙傛暟
            reward = self.reward_calc.calculate(
                fit_start=self.stage_start_fit, fit_end=self.base.fit,
                div_start=self.stage_start_div.item(), div_end=current_div.item(),
                target_div=self.current_cmd_target_div.item(),
                initial_div=self.initial_div.item(),
                stagnation_start=self.stage_start_stagnation,
                stag_ratio_start=stag_ratio_start.item(),
                stag_ratio_end=stag_ratio_end.item(),
                action=self.last_intent_eff,
                feedbacks=feedbacks
            )
            self.rl_agent.buffer.add(
                state=self.last_state, action=self.last_action_raw,
                reward=reward, value=self.last_value.item(),
                log_prob=self.last_logprob.item(), done=0.0
            )

            # 3. 濡傛灉鏀掑浜嗛樁娈垫暟锛岄€氱煡 PPO 杩涜鍙嶅悜浼犳挱鏇存柊澶ц剳
            if len(self.rl_agent.buffer.states) >= self.update_stages:
                self.rl_agent.update()

            # 4. 鍙戝竷鏂颁竴闃舵鐨勬帶鍒舵剰鍥?
            self._dispatch_new_stage()

        # =====================================================================
        # 绗簩闃舵锛氬簳灞傛垬鏈墽琛?(寮辫€﹀悎娈嬪樊鎺у埗)
        # =====================================================================
        self.base.step()

        curr_pop = self.base.pop.clone()
        curr_fit = self.base.fit.clone()

        # Guardian hard-freeze (stable first version).
        guardian_mask = self.current_niche_guardian
        if guardian_mask.any():
            curr_pop[guardian_mask] = self.prev_pop[guardian_mask]
            curr_fit[guardian_mask] = self.prev_fit[guardian_mask]

        # 璁＄畻灞€閮ㄥ昂搴﹂敋鐐?
        pop_std = torch.std(curr_pop, dim=0) + 1e-8
        trial_pop = curr_pop.clone()

        intent = IntentVector.from_tensor(self.last_intent_eff)
        middle_mask = self.current_cmd_middle
        rescue_mask = self.current_cmd_rescue
        quiet_mask = self.current_cmd_quiet
        refine_gate = float(self.current_gate_refine.item())

        # Keep good stagnation particles quiet when refinement gate is high.
        if refine_gate > 0.5:
            middle_mask = middle_mask & (~quiet_mask)
            rescue_mask = rescue_mask & (~quiet_mask)

        # 1. 鎺㈢储鎵板姩 (e)
        if middle_mask.any() and intent.e_t > 0.01:
            mid_scales = self.current_cmd_scales[middle_mask].unsqueeze(1)
            noise = torch.randn_like(trial_pop[middle_mask]) * pop_std * intent.e_t * mid_scales
            # Ignore numerically tiny perturbations in late fine-search.
            small_noise = torch.norm(noise, dim=1, keepdim=True) < 1e-6
            noise = torch.where(small_noise, torch.zeros_like(noise), noise)
            trial_pop[middle_mask] += noise

        # 2. 寮€鍙戞媺鎷?(x)
        if middle_mask.any() and intent.x_t > 0.01:
            dir_to_best = self.global_best_location - trial_pop[middle_mask]
            mid_scales = self.current_cmd_scales[middle_mask].unsqueeze(1)
            pull = torch.rand(middle_mask.sum().item(), 1, device=self.device) * intent.x_t * mid_scales
            trial_pop[middle_mask] += dir_to_best * pull

        # 3. 鍋滄粸鏁戞彺 (r) - 淇¤禆鍩熷紡鎭㈠
        if rescue_mask.any():
            rescue_scales = self.current_cmd_scales[rescue_mask].unsqueeze(1)
            rescue_strength = 1.0 - 0.8 * refine_gate
            base_pos = curr_pop[rescue_mask]
            to_best = self.global_best_location - base_pos
            noise = torch.randn_like(base_pos) * pop_std * 0.5 * rescue_strength * rescue_scales
            trial_pop[rescue_mask] = base_pos + 0.3 * to_best + noise

        trial_pop = clamp(trial_pop, self.lb, self.ub)

        # =====================================================================
        # 绗笁闃舵锛氳椽蹇冧繚搴曢獙鏀朵笌鍙嶉鏀堕泦
        # =====================================================================
        modified_mask = middle_mask | rescue_mask
        n_mod = modified_mask.sum().float()

        if n_mod > 0:
            trial_fit = curr_fit.clone()
            trial_fit[modified_mask] = self.evaluate(trial_pop[modified_mask])

            # 绾椽蹇冿細鍙姣?EA 鑷繁璧扮殑涓€姝ュソ锛屽氨鎺ュ彈
            strict_accept = trial_fit < curr_fit
            margin_fit = curr_fit + torch.abs(curr_fit) * self.current_cmd_alpha_margin + self.current_cmd_beta_margin
            soft_accept = trial_fit < margin_fit
            if refine_gate > 0.65:
                accept_mask = strict_accept
            else:
                accept_mask = strict_accept | (rescue_mask & soft_accept)

            curr_pop[accept_mask] = trial_pop[accept_mask]
            curr_fit[accept_mask] = trial_fit[accept_mask]

            # 鏀堕泦鎺у埗鎵ц鍙嶉 (渚涗笅涓€闃舵 State 鍜?Reward 浣跨敤)
            self.last_intervene_ratio = n_mod / self.pop_size
            self.last_accept_ratio = accept_mask.sum().float() / n_mod
            if rescue_mask.any():
                self.last_rescue_success = (accept_mask & rescue_mask).sum().float() / rescue_mask.sum().float()
            else:
                self.last_rescue_success = torch.tensor(0.0, device=self.device)

        else:
            self.last_intervene_ratio = torch.tensor(0.0, device=self.device)
            self.last_accept_ratio = torch.tensor(0.0, device=self.device)
            self.last_rescue_success = torch.tensor(0.0, device=self.device)

        # 銆愭柊澧炪€戯細閲嶆柊瀹氫箟 Elite Damage銆傚鏌ュ簳灞?EA 鏄惁鐮村潖浜嗙簿鑻憋紒
        protected_mask = self.current_cmd_protected
        if protected_mask.any():
            # 缁熻鍦?base.step() 涔嬪悗鍙樺樊鐨勭簿鑻辩兢浣?
            worsened = (curr_fit > self.prev_fit) & protected_mask
            self.last_elite_damage = worsened.sum().float() / protected_mask.sum().float()

            # 鏃㈢劧鍙€滀繚鎶も€濓紝鎴戜滑灏辫疮褰诲埌搴曪細鎶婂彉宸殑绮捐嫳鐩存帴鍥炴粴鍒颁笂涓€浠ｏ紒
            curr_pop[worsened] = self.prev_pop[worsened]
            curr_fit[worsened] = self.prev_fit[worsened]
        else:
            self.last_elite_damage = torch.tensor(0.0, device=self.device)

        self.base.pop = curr_pop
        self.base.fit = curr_fit

        # =====================================================================
        # 鏀跺熬鐘舵€佹洿鏂?
        # =====================================================================
        improved = curr_fit < self.prev_fit
        self.stagnation = torch.where(improved, torch.zeros_like(self.stagnation), self.stagnation + 1)

        self.prev_pop = curr_pop.clone()
        self.prev_fit = curr_fit.clone()

        current_best_val, current_best_idx = torch.min(self.base.fit, dim=0)
        if current_best_val < self.global_best_fit:
            self.global_best_fit = current_best_val.clone()
            self.global_best_location = self.base.pop[current_best_idx].clone()

        # 鍏煎鎬у悓姝?
        if hasattr(self.base, 'global_best_fit'):
            self.base.global_best_fit = self.global_best_fit.clone()
            self.base.global_best_location = self.global_best_location.clone()

        self.step_counter = self.step_counter + 1
