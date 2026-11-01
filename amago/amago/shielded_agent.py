"""
Shielded AMAGO agent that performs action-time safety filtering for discrete actions.
"""

import itertools
from typing import Any, Tuple

import gin
import torch.nn as nn
import torch.nn.functional as F
import torch

from .agent import Agent


@gin.configurable
class ShieldedAMAGOAgent(Agent):
    """AMAGO Agent with a lightweight runtime shield based on cost critics.

    The shield is intentionally minimal:
    - keeps AMAGO training unchanged
    - only modifies rollout action selection
    - uses conservative cost-to-go from `cost_critics` to filter unsafe actions
    """

    def __init__(
        self,
        *args,
        shield_alpha: float = 1.25,
        shield_safety_margin: float = 0.0,
        shield_fallback: str = "policy",
        shield_warmup_epochs: int = 0,
        shield_critic_agg: str = "mean",
        shield_filter_mode: str = "soft",
        shield_soft_beta: float = 1.0,
        shield_score_mode: str = "barrier",
        shield_log_margins: bool = False,
        shield_critic_quantile: float = 0.75,
        shield_uncertainty_mode: str = "none",
        shield_uncertainty_kappa: float = 0.0,
        shield_uncertainty_q_hi: float = 0.9,
        shield_uncertainty_q_lo: float = 0.1,
        shield_uncertainty_min: float | None = None,
        shield_uncertainty_max: float | None = None,
        shield_continuous_num_samples: int = 32,
        shield_continuous_sampling_scheme: str = "structured",
        shield_continuous_wide_scale: float = 2.5,
        shield_final_sampling_temperature: float = 1.0,
        shield_continuous_enable_fast_path: bool = True,
        shield_continuous_mean_exit_margin: float = 0.0,
        shield_continuous_stage1_num_samples: int = 3,
        use_world_model_aux: bool = True,
        wm_hidden_dim: int = 256,
        wm_nll_weight: float = 1.0,
        wm_loss_weight: float = 1.0,
        cbf_smoothness_weight: float = 0.1,
        latent_distill_weight: float = 0.1,
        rnd_loss_weight: float = 0.01,
        epsilon_calib_weight: float = 0.05,
        epsilon_calib_quantile: float = 0.95,
        gamma_conjugacy_weight: float = 0.1,
        cbf_collapse_weight: float = 0.1,
        detach_structural_losses: bool = True,
        # Backward-compatible aliases.
        mem_loss_weight: float | None = None,
        mem_collapse_weight: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if shield_fallback not in {"min_cost", "policy"}:
            raise ValueError(
                f"Unsupported shield_fallback={shield_fallback}. "
                "Use one of ['min_cost', 'policy']."
            )
        if shield_critic_agg not in {"max", "mean", "min", "quantile"}:
            raise ValueError(
                f"Unsupported shield_critic_agg={shield_critic_agg}. "
                "Use one of ['max', 'mean', 'min', 'quantile']."
            )
        if shield_filter_mode not in {"hard", "soft"}:
            raise ValueError(
                f"Unsupported shield_filter_mode={shield_filter_mode}. "
                "Use one of ['hard', 'soft']."
            )
        if shield_score_mode not in {"barrier", "cost_penalty", "budget_penalty"}:
            raise ValueError(
                f"Unsupported shield_score_mode={shield_score_mode}. "
                "Use one of ['barrier', 'cost_penalty', 'budget_penalty']."
            )
        if shield_uncertainty_mode not in {"none", "std", "quantile_spread"}:
            raise ValueError(
                f"Unsupported shield_uncertainty_mode={shield_uncertainty_mode}. "
                "Use one of ['none', 'std', 'quantile_spread']."
            )
        if shield_continuous_sampling_scheme not in {"structured", "iid"}:
            raise ValueError(
                "Unsupported shield_continuous_sampling_scheme="
                f"{shield_continuous_sampling_scheme}. "
                "Use one of ['structured', 'iid']."
            )
        self.shield_alpha = shield_alpha
        self.shield_safety_margin = shield_safety_margin
        self.shield_fallback = shield_fallback
        self.shield_warmup_epochs = max(int(shield_warmup_epochs), 0)
        self.shield_critic_agg = shield_critic_agg
        self.shield_filter_mode = shield_filter_mode
        self.shield_soft_beta = float(shield_soft_beta)
        self.shield_score_mode = str(shield_score_mode)
        # Instrumentation: record the EXECUTED action's barrier margin at action-selection time,
        # on exactly the scale the shield uses (no offline reconstruction). Off by default.
        self.shield_log_margins = bool(shield_log_margins)
        self._margin_log = []
        self.shield_critic_quantile = float(shield_critic_quantile)
        self.shield_uncertainty_mode = shield_uncertainty_mode
        self.shield_uncertainty_kappa = float(shield_uncertainty_kappa)
        self.shield_uncertainty_q_hi = float(shield_uncertainty_q_hi)
        self.shield_uncertainty_q_lo = float(shield_uncertainty_q_lo)
        self.shield_uncertainty_min = (
            None if shield_uncertainty_min is None else float(shield_uncertainty_min)
        )
        self.shield_uncertainty_max = (
            None if shield_uncertainty_max is None else float(shield_uncertainty_max)
        )
        self.shield_continuous_num_samples = max(int(shield_continuous_num_samples), 1)
        self.shield_continuous_sampling_scheme = str(shield_continuous_sampling_scheme)
        self.shield_continuous_wide_scale = float(shield_continuous_wide_scale)
        self.shield_final_sampling_temperature = max(float(shield_final_sampling_temperature), 1e-6)
        self.shield_continuous_enable_fast_path = bool(shield_continuous_enable_fast_path)
        self.shield_continuous_mean_exit_margin = float(shield_continuous_mean_exit_margin)
        self.shield_continuous_stage1_num_samples = max(int(shield_continuous_stage1_num_samples), 1)
        self.current_epoch = 0
        self.use_world_model_aux = use_world_model_aux
        self.wm_nll_weight = wm_nll_weight
        self.wm_loss_weight = wm_loss_weight
        self.cbf_smoothness_weight = (
            cbf_smoothness_weight if mem_loss_weight is None else mem_loss_weight
        )
        self.latent_distill_weight = latent_distill_weight
        self.rnd_loss_weight = rnd_loss_weight
        self.epsilon_calib_weight = epsilon_calib_weight
        self.epsilon_calib_quantile = epsilon_calib_quantile
        self.gamma_conjugacy_weight = gamma_conjugacy_weight
        self.detach_structural_losses = bool(detach_structural_losses)
        self.cbf_collapse_weight = (
            cbf_collapse_weight
            if mem_collapse_weight is None
            else mem_collapse_weight
        )

        d = self.traj_encoder.emb_dim
        a = self.action_dim
        self.proj_policy = nn.Linear(d, d)
        self.proj_world = nn.Linear(d, d)
        self.proj_cbf = nn.Linear(d, d)
        self.dyn_z_w1 = nn.Linear(d + a, wm_hidden_dim)
        self.dyn_z_w2 = nn.Linear(wm_hidden_dim, d)
        self.dyn_r = nn.Linear(d + a, 1)
        self.dyn_c1 = nn.Linear(d + a, 1)
        self.dyn_c2 = nn.Linear(d + a, 1)
        self.dyn_logvar = nn.Linear(wm_hidden_dim, d)
        # Keep param names stable for checkpoint compatibility.
        self.mem_w1 = nn.Linear(d, 128)
        self.mem_w2 = nn.Linear(128, 64)
        self.rnd_target_w1 = nn.Linear(d, 128)
        self.rnd_target_w2 = nn.Linear(128, 64)
        self.rnd_pred_w1 = nn.Linear(d, 128)
        self.rnd_pred_w2 = nn.Linear(128, 64)
        self.eps_cal_scale_raw = nn.Parameter(torch.tensor(1.0))
        self.eps_cal_bias_raw = nn.Parameter(torch.tensor(-20.0))
        for p in self.rnd_target_w1.parameters():
            p.requires_grad = False
        for p in self.rnd_target_w2.parameters():
            p.requires_grad = False

    @property
    def trainable_params(self):
        aux_modules = (
            self.proj_policy,
            self.proj_world,
            self.proj_cbf,
            self.dyn_z_w1,
            self.dyn_z_w2,
            self.dyn_r,
            self.dyn_c1,
            self.dyn_c2,
            self.dyn_logvar,
            self.mem_w1,
            self.mem_w2,
            self.rnd_pred_w1,
            self.rnd_pred_w2,
        )
        return itertools.chain(
            super().trainable_params,
            *(m.parameters() for m in aux_modules),
            [self.eps_cal_scale_raw, self.eps_cal_bias_raw],
        )

    def project_latent(self, z: torch.Tensor, task: str) -> torch.Tensor:
        if task == "policy":
            return F.gelu(self.proj_policy(z))
        if task == "world":
            return F.gelu(self.proj_world(z))
        if task == "cbf":
            return F.gelu(self.proj_cbf(z))
        raise ValueError(f"Unknown projection task={task}")

    def world_model_step(self, z: torch.Tensor, a: torch.Tensor):
        inp = torch.cat([z, a], dim=-1)
        h = F.gelu(self.dyn_z_w1(inp))
        z_next = self.dyn_z_w2(h)
        logvar = self.dyn_logvar(h).clamp(-10.0, 5.0)
        r = self.dyn_r(inp).squeeze(-1)
        c1 = self.dyn_c1(inp).squeeze(-1)
        c2 = self.dyn_c2(inp).squeeze(-1)
        c = torch.maximum(c1, c2)
        return z_next, r, c, logvar, c1, c2

    def cbf_latent_smoothness_loss(
        self,
        z_t: torch.Tensor,
        z_next: torch.Tensor,
        transition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e_t = F.normalize(self.mem_w2(F.gelu(self.mem_w1(z_t))), dim=-1, eps=1e-6)
        e_next = F.normalize(
            self.mem_w2(F.gelu(self.mem_w1(z_next))), dim=-1, eps=1e-6
        )
        align = 1.0 - (e_t * e_next).sum(dim=-1)
        if transition_mask is not None:
            m = transition_mask.float()
            align = (align * m).sum() / m.sum().clamp_min(1.0)
        else:
            align = align.mean()
        emb = e_t.reshape(-1, e_t.shape[-1])
        collapse = (0.2 - emb.std(dim=0)).clamp_min(0.0).mean()
        return align + self.cbf_collapse_weight * collapse

    def rnd_error(self, z: torch.Tensor) -> torch.Tensor:
        z_w = self.project_latent(z, "world")
        with torch.no_grad():
            t = self.rnd_target_w2(F.gelu(self.rnd_target_w1(z_w)))
        p = self.rnd_pred_w2(F.gelu(self.rnd_pred_w1(z_w)))
        return (p - t).pow(2).mean(dim=-1)

    def epsilon_from_sigma(self, sigma: torch.Tensor, l_h: float = 1.0) -> torch.Tensor:
        scale = F.softplus(self.eps_cal_scale_raw) + 1e-8
        bias = F.softplus(self.eps_cal_bias_raw)
        return max(l_h, 0.0) * (scale * sigma.clamp_min(0.0) + bias)

    def epsilon_calibration_loss(
        self,
        pred_z: torch.Tensor,
        z_next: torch.Tensor,
        pred_logvar: torch.Tensor,
        state_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = min(max(self.epsilon_calib_quantile, 0.5), 0.999)
        sigma_pred = torch.sqrt(torch.exp(pred_logvar).mean(dim=-1))
        residual = torch.sqrt((pred_z - z_next).pow(2).mean(dim=-1))
        eps_pred = self.epsilon_from_sigma(sigma_pred, l_h=1.0)
        diff = residual - eps_pred
        calib_err = torch.maximum(q * diff, (q - 1.0) * diff).unsqueeze(-1)
        if state_mask is None:
            return calib_err.mean()
        return (calib_err * state_mask).sum() / state_mask.sum().clamp_min(1.0)

    def _safe_info_tensor(self, batch, key: str, default: float = 0.0) -> torch.Tensor:
        if hasattr(batch, "infos") and isinstance(batch.infos, dict) and key in batch.infos:
            return batch.infos[key].float().to(batch.actions.device)
        return torch.full(
            (batch.actions.shape[0], batch.actions.shape[1]),
            default,
            dtype=torch.float32,
            device=batch.actions.device,
        )

    def forward(self, batch, log_step: bool):
        critic_loss, actor_loss = super().forward(batch=batch, log_step=log_step)
        if not self.use_world_model_aux:
            return critic_loss, actor_loss

        # Re-derive the same mask used in base Agent.forward
        # 1.0 where loss at this index should count, 0.0 where it should be ignored
        # (B, L-1, 1)
        state_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).float()[:, 1:, ...]
        
        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s, log_dict=None)
        s_rep, _ = self.traj_encoder(
            seq=o, time_idxs=batch.time_idxs, hidden_state=None, log_dict=None
        )
        z_t = s_rep[:, :-1, :].contiguous()
        z_next = s_rep[:, 1:, :].contiguous()
        z_t_world = self.project_latent(z_t, "world")
        z_next_world = self.project_latent(z_next, "world")
        actions = batch.actions.float()
        rewards = batch.rews.squeeze(-1).float()
        costs = self._safe_info_tensor(batch, "cost", default=0.0)

        # 1. World Model Loss (Masked)
        pred_z, pred_r, _, logvar, c1, c2 = self.world_model_step(z_t_world, actions)
        inv_var = torch.exp(-logvar)
        z_nll = 0.5 * (inv_var * (pred_z - z_next_world).pow(2) + logvar)
        
        # We use masked_avg logic manually here
        def masked_mean(tensor, mask):
            return (tensor * mask).sum() / mask.sum().clamp_min(1.0)

        wm_z_loss = masked_mean(z_nll.mean(dim=-1, keepdim=True), state_mask) * self.wm_nll_weight
        wm_r_loss = masked_mean((pred_r - rewards).pow(2).unsqueeze(-1), state_mask)
        wm_c_loss = masked_mean(((c1 - costs).pow(2) + (c2 - costs).pow(2)).unsqueeze(-1), state_mask)
        wm_loss = wm_z_loss + wm_r_loss + wm_c_loss

        # 2. CBF-Latent Smoothness Loss (Masked)
        soft_reset = self._safe_info_tensor(batch, "soft_reset", default=0.0)
        # soft_reset is (B, L-1). state_mask is (B, L-1, 1). 
        # align output from cbf_latent_smoothness_loss will be (B, L-1).
        transition_mask = state_mask.squeeze(-1) * (1.0 - soft_reset)
        z_t_cbf = self.project_latent(z_t, "cbf")
        z_next_cbf = self.project_latent(z_next, "cbf")
        cbf_smoothness_loss = self.cbf_latent_smoothness_loss(
            z_t_cbf, z_next_cbf, transition_mask=transition_mask
        )

        # 3. Latent Distillation (Masked)
        z_t_policy_in = z_t.detach() if self.detach_structural_losses else z_t
        z_t_world_anchor = z_t_world.detach() if self.detach_structural_losses else z_t_world
        z_t_policy = self.project_latent(z_t_policy_in, "policy")
        distill_err = (z_t_policy - z_t_world_anchor).pow(2).mean(dim=-1, keepdim=True)
        latent_distill = masked_mean(distill_err, state_mask)

        # 4. RND Loss (Masked) - pass raw z_t to avoid double projection
        rnd_err = self.rnd_error(z_t).unsqueeze(-1)
        rnd_loss = masked_mean(rnd_err, state_mask)

        # 5. Epsilon Calibration (Masked)
        eps_calib = self.epsilon_calibration_loss(
            pred_z=pred_z,
            z_next=z_next_world,
            pred_logvar=logvar,
            state_mask=state_mask,
        )

        # 6. Gamma Conjugacy (Masked)
        gamma_test = float(self.gammas[-1].item())
        z_t_conj_in = z_t.detach() if self.detach_structural_losses else z_t
        z_next_conj_in = z_next.detach() if self.detach_structural_losses else z_next
        delta_world_raw = z_next_world - gamma_test * z_t_world
        delta_world = (
            delta_world_raw.detach()
            if self.detach_structural_losses
            else delta_world_raw
        )
        delta_policy = self.project_latent(z_next_conj_in, "policy") - gamma_test * self.project_latent(
            z_t_conj_in, "policy"
        )
        conj_err = (delta_policy - delta_world).pow(2).mean(dim=-1, keepdim=True)
        gamma_conjugacy = masked_mean(conj_err, state_mask)

        total_aux = (
            self.wm_loss_weight * wm_loss
            + self.cbf_smoothness_weight * cbf_smoothness_loss
            + self.latent_distill_weight * latent_distill
            + self.rnd_loss_weight * rnd_loss
            + self.epsilon_calib_weight * eps_calib
            + self.gamma_conjugacy_weight * gamma_conjugacy
        )
        # Adding to actor_loss (B, L-1, G, 1). Scalar total_aux broadcasts.
        actor_loss = actor_loss + total_aux

        self.update_info["shield/wm_loss"] = wm_loss.detach()
        self.update_info["shield/cbf_smoothness_loss"] = cbf_smoothness_loss.detach()
        # Backward-compatible metric key.
        self.update_info["shield/mem_loss"] = cbf_smoothness_loss.detach()
        self.update_info["shield/latent_distill"] = latent_distill.detach()
        self.update_info["shield/rnd_loss"] = rnd_loss.detach()
        self.update_info["shield/eps_calib_loss"] = eps_calib.detach()
        self.update_info["shield/gamma_conjugacy_loss"] = gamma_conjugacy.detach()
        self.update_info["shield/aux_total_loss"] = total_aux.detach()
        return critic_loss, actor_loss

    @staticmethod
    def _budget_from_obs(cost_to_go: torch.Tensor) -> torch.Tensor:
        # Common shapes are [B, L], [B, L, 1], or [B, 1, 1] for single-step rollout.
        if cost_to_go.ndim >= 3 and cost_to_go.shape[-1] == 1:
            cost_to_go = cost_to_go.squeeze(-1)
        if cost_to_go.ndim == 1:
            cost_to_go = cost_to_go.unsqueeze(-1)
        if cost_to_go.ndim != 2:
            # Last-resort flatten while preserving batch.
            cost_to_go = cost_to_go.reshape(cost_to_go.shape[0], -1)
        return cost_to_go

    def _predicted_costs_per_action(
        self, state_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cost estimate for each discrete action.

        Returns:
            Tuple[pred_costs, delta] where each has shape [B, L, A]
        """
        B, L, _ = state_seq.shape
        G = len(self.gammas)
        A = self.action_dim
        eye = torch.eye(A, device=state_seq.device, dtype=state_seq.dtype)
        per_action_costs = []
        per_action_uncertainty = []
        for action_idx in range(A):
            # Avoid zero-stride expanded views before calling compiled critics.
            action = eye[action_idx].reshape(1, 1, 1, 1, A).repeat(1, B, L, G, 1)
            q_cost = self.cost_critics(state_seq, action)
            q_cost = self.cost_popart(q_cost, normalized=False)
            # [1, B, L, C, G, 1] -> [B, L, C, G]
            q_cost = q_cost.squeeze(0).squeeze(-1)
            q_cost_agg = self._aggregate_cost_critics(q_cost)[..., -1]
            delta = self._uncertainty_radius(q_cost)
            per_action_costs.append(q_cost_agg)
            per_action_uncertainty.append(delta)
        pred_costs = torch.stack(per_action_costs, dim=-1)
        uncertainty = torch.stack(per_action_uncertainty, dim=-1)
        return pred_costs, uncertainty

    def _uncertainty_radius(self, q_cost: torch.Tensor) -> torch.Tensor:
        """Compute per-action uncertainty proxy at test gamma.

        Args:
            q_cost: Tensor with shape [B, L, C, G]

        Returns:
            Tensor with shape [B, L]
        """
        mode = self.shield_uncertainty_mode
        if mode == "none" or self.shield_uncertainty_kappa <= 0.0:
            delta = torch.zeros_like(q_cost[..., 0, -1])
        elif mode == "std":
            # Ensemble disagreement over critics at each gamma; use test-time gamma.
            delta = q_cost.std(dim=2, correction=0)[..., -1]
        else:
            q_hi = min(max(self.shield_uncertainty_q_hi, 0.0), 1.0)
            q_lo = min(max(self.shield_uncertainty_q_lo, 0.0), 1.0)
            if q_lo > q_hi:
                q_lo, q_hi = q_hi, q_lo
            spread = torch.quantile(q_cost, q=q_hi, dim=2) - torch.quantile(
                q_cost, q=q_lo, dim=2
            )
            delta = spread[..., -1]
        delta = self.shield_uncertainty_kappa * delta
        if self.shield_uncertainty_min is not None:
            delta = torch.clamp_min(delta, self.shield_uncertainty_min)
        if self.shield_uncertainty_max is not None:
            delta = torch.clamp_max(delta, self.shield_uncertainty_max)
        return delta

    def _aggregate_cost_critics(self, q_cost: torch.Tensor) -> torch.Tensor:
        if self.shield_critic_agg == "max":
            return q_cost.max(dim=2).values
        if self.shield_critic_agg == "mean":
            return q_cost.mean(dim=2)
        if self.shield_critic_agg == "min":
            return q_cost.min(dim=2).values
        q = min(max(self.shield_critic_quantile, 0.0), 1.0)
        return torch.quantile(q_cost, q=q, dim=2)

    def _aggregate_cost_critics_candidates(self, q_cost: torch.Tensor) -> torch.Tensor:
        if self.shield_critic_agg == "max":
            return q_cost.max(dim=3).values
        if self.shield_critic_agg == "mean":
            return q_cost.mean(dim=3)
        if self.shield_critic_agg == "min":
            return q_cost.min(dim=3).values
        q = min(max(self.shield_critic_quantile, 0.0), 1.0)
        return torch.quantile(q_cost, q=q, dim=3)

    def _uncertainty_radius_candidates(self, q_cost: torch.Tensor) -> torch.Tensor:
        mode = self.shield_uncertainty_mode
        if mode == "none" or self.shield_uncertainty_kappa <= 0.0:
            delta = torch.zeros_like(q_cost[..., 0, -1])
        elif mode == "std":
            delta = q_cost.std(dim=3, correction=0)[..., -1]
        else:
            q_hi = min(max(self.shield_uncertainty_q_hi, 0.0), 1.0)
            q_lo = min(max(self.shield_uncertainty_q_lo, 0.0), 1.0)
            if q_lo > q_hi:
                q_lo, q_hi = q_hi, q_lo
            spread = torch.quantile(q_cost, q=q_hi, dim=3) - torch.quantile(
                q_cost, q=q_lo, dim=3
            )
            delta = spread[..., -1]
        delta = self.shield_uncertainty_kappa * delta
        if self.shield_uncertainty_min is not None:
            delta = torch.clamp_min(delta, self.shield_uncertainty_min)
        if self.shield_uncertainty_max is not None:
            delta = torch.clamp_max(delta, self.shield_uncertainty_max)
        return delta

    @staticmethod
    def _continuous_log_prob(action_dists, candidates: torch.Tensor) -> torch.Tensor:
        logp = action_dists.log_prob(candidates)
        if logp.ndim == candidates.ndim:
            logp = logp.sum(dim=-1)
        return logp

    def _select_continuous_actions(
        self,
        candidate_actions: torch.Tensor,
        scores: torch.Tensor,
        fallback_actions: torch.Tensor,
        has_safe: torch.Tensor | None,
        sample: bool,
    ) -> torch.Tensor:
        candidate_actions = candidate_actions[..., -1, :]
        B, L, A = fallback_actions.shape
        K = candidate_actions.shape[0]
        flat_candidates = candidate_actions.permute(1, 2, 0, 3).reshape(B * L, K, A)
        flat_scores = scores.permute(1, 2, 0).reshape(B * L, K)
        flat_fallback = fallback_actions.reshape(B * L, A)
        chosen = flat_fallback.clone()
        if has_safe is None:
            valid_mask = torch.ones(B * L, dtype=torch.bool, device=flat_scores.device)
        else:
            valid_mask = has_safe.reshape(B * L)
        if valid_mask.any():
            valid_scores = flat_scores[valid_mask]
            if sample:
                probs = torch.softmax(
                    valid_scores / self.shield_final_sampling_temperature,
                    dim=-1,
                )
                idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                idx = valid_scores.argmax(dim=-1)
            chosen[valid_mask] = flat_candidates[valid_mask, idx, :]
        return chosen.view(B, L, A)

    def _continuous_candidate_subset(
        self,
        action_dists,
        num_samples: int,
        base_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_samples = max(int(num_samples), 1)
        if self.shield_continuous_sampling_scheme == "iid":
            return self._continuous_candidate_actions_iid(
                action_dists, num_samples, first_candidate=base_action
            )
        return self._continuous_candidate_actions_structured(
            action_dists, num_samples, base_action=base_action
        )

    def _score_continuous_candidates(
        self,
        action_dists,
        state_seq: torch.Tensor,
        safe_threshold: torch.Tensor,
        candidate_actions: torch.Tensor,
        fallback_policy: torch.Tensor,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_cost = self.cost_critics(state_seq, candidate_actions)
        q_cost = self.cost_popart(q_cost, normalized=False).squeeze(-1)
        pred_costs = self._aggregate_cost_critics_candidates(q_cost)[..., -1]
        uncertainty = self._uncertainty_radius_candidates(q_cost)
        robust_costs = pred_costs + uncertainty

        if self.shield_log_margins:
            # Real per-step candidate feasibility (how many sampled candidates satisfy the
            # barrier), stashed for the logger in get_actions. NOT `has_safe`, which is trivially true
            # in soft mode. The last call before the action is returned is the deciding one.
            with torch.no_grad():
                feas = (robust_costs <= safe_threshold.unsqueeze(0))
                self._last_feas_frac = feas.float().mean(dim=0).reshape(-1).float().cpu()
                self._last_feas_any = feas.any(dim=0).reshape(-1).float().cpu()

        if self.shield_fallback == "policy":
            fallback_actions = fallback_policy
        else:
            min_idx = robust_costs.argmin(dim=0)
            fallback_actions = self._select_continuous_actions(
                candidate_actions=candidate_actions,
                scores=pred_costs.neg(),
                fallback_actions=fallback_policy,
                has_safe=torch.ones_like(min_idx, dtype=torch.bool),
                sample=False,
            )

        logp = self._continuous_log_prob(action_dists, candidate_actions)[..., -1]
        if self.shield_filter_mode == "soft":
            if self.shield_score_mode in ("cost_penalty", "budget_penalty"):
                # Simpler baselines: exp(-beta*Q_C) (budget-blind) or exp(-beta*(Q_C - B))
                # (budget-aware linear shift). No barrier-margin construction. Same cost critic.
                if self.shield_score_mode == "cost_penalty":
                    penalty = robust_costs
                else:  # budget_penalty
                    penalty = robust_costs - safe_threshold.unsqueeze(0)
                # numerical stability: shift by per-state min over candidates (selection is shift-invariant)
                penalty = penalty - penalty.min(dim=0, keepdim=True).values.detach()
                scores = logp - self.shield_soft_beta * penalty
                has_safe = torch.ones_like(robust_costs[0], dtype=torch.bool)
            else:  # barrier (default, released behavior)
                excess = (robust_costs - safe_threshold.unsqueeze(0)).clamp_min(0.0)
                scores = logp - self.shield_soft_beta * excess
                has_safe = (excess <= 0.0).any(dim=0)
        else:
            safe_mask = robust_costs <= safe_threshold.unsqueeze(0)
            scores = logp.masked_fill(~safe_mask, float("-inf"))
            has_safe = safe_mask.any(dim=0)
        actions = self._select_continuous_actions(
            candidate_actions=candidate_actions,
            scores=scores,
            fallback_actions=fallback_actions,
            has_safe=None if self.shield_filter_mode == "soft" else has_safe,
            sample=sample,
        )
        return actions, has_safe, pred_costs, robust_costs

    def _continuous_shielded_actions(
        self,
        action_dists,
        state_seq: torch.Tensor,
        safe_threshold: torch.Tensor,
        sample: bool,
    ) -> torch.Tensor:
        policy_actions = action_dists.sample() if sample else action_dists.mean
        fallback_policy = policy_actions[..., -1, :]

        if self.shield_continuous_enable_fast_path:
            gate_candidate = policy_actions if sample else action_dists.mean
            gate_candidates = gate_candidate.unsqueeze(0)
            gate_actions, _, gate_pred_costs, _ = self._score_continuous_candidates(
                action_dists=action_dists,
                state_seq=state_seq,
                safe_threshold=safe_threshold,
                candidate_actions=gate_candidates,
                fallback_policy=fallback_policy,
                sample=sample,
            )
            gate_slack = safe_threshold - gate_pred_costs.squeeze(0)
            if torch.all(gate_slack >= self.shield_continuous_mean_exit_margin):
                return gate_actions

            stage1_num_samples = min(
                max(int(self.shield_continuous_stage1_num_samples), 1),
                int(self.shield_continuous_num_samples),
            )
            if stage1_num_samples > 1:
                stage1_base = policy_actions if sample else None
                stage1_candidates = self._continuous_candidate_subset(
                    action_dists,
                    stage1_num_samples,
                    base_action=stage1_base,
                )
                stage1_actions, stage1_has_safe, _, _ = self._score_continuous_candidates(
                    action_dists=action_dists,
                    state_seq=state_seq,
                    safe_threshold=safe_threshold,
                    candidate_actions=stage1_candidates,
                    fallback_policy=fallback_policy,
                    sample=sample,
                )
                if torch.all(stage1_has_safe) or stage1_num_samples >= int(self.shield_continuous_num_samples):
                    return stage1_actions

        if sample:
            candidate_actions = self._continuous_candidate_actions_iid(
                action_dists,
                int(self.shield_continuous_num_samples),
                first_candidate=policy_actions,
            )
        else:
            candidate_actions = self._continuous_candidate_subset(
                action_dists,
                int(self.shield_continuous_num_samples),
            )
        actions, _, _, _ = self._score_continuous_candidates(
            action_dists=action_dists,
            state_seq=state_seq,
            safe_threshold=safe_threshold,
            candidate_actions=candidate_actions,
            fallback_policy=fallback_policy,
            sample=sample,
        )
        return actions

    def _continuous_candidate_actions(self, action_dists) -> torch.Tensor:
        num_samples = max(int(self.shield_continuous_num_samples), 1)
        if self.shield_continuous_sampling_scheme == "iid":
            return self._continuous_candidate_actions_iid(action_dists, num_samples)
        return self._continuous_candidate_actions_structured(action_dists, num_samples)

    def _continuous_candidate_actions_iid(
        self,
        action_dists,
        num_samples: int,
        first_candidate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lead = action_dists.mean if first_candidate is None else first_candidate
        candidates = [lead.unsqueeze(0)]
        remaining = num_samples - 1
        if remaining > 0:
            differentiable = getattr(self.actor, "actions_differentiable", True)
            sampler = action_dists.rsample if differentiable else action_dists.sample
            candidates.extend([sampler().unsqueeze(0) for _ in range(remaining)])
        return torch.cat(candidates, dim=0)

    def _continuous_candidate_actions_structured(
        self,
        action_dists,
        num_samples: int,
        base_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean = action_dists.mean
        center = mean if base_action is None else base_action
        if num_samples <= 1:
            return center.unsqueeze(0)

        std = self._continuous_action_std(action_dists, mean)
        candidates = [center.unsqueeze(0)]
        eps_local = torch.rand_like(mean)

        if num_samples >= 2:
            plus_local = (center + std * eps_local).clamp(-1.0, 1.0)
            candidates.append(plus_local.unsqueeze(0))
        if num_samples >= 3:
            minus_local = (center - std * eps_local).clamp(-1.0, 1.0)
            candidates.append(minus_local.unsqueeze(0))
        if num_samples >= 4:
            eps_wide = torch.randn_like(mean)
            wide = (mean + self.shield_continuous_wide_scale * std * eps_wide).clamp(-1.0, 1.0)
            candidates.append(wide.unsqueeze(0))

        remaining = num_samples - len(candidates)
        if remaining > 0:
            differentiable = getattr(self.actor, "actions_differentiable", True)
            sampler = action_dists.rsample if differentiable else action_dists.sample
            candidates.extend([sampler().unsqueeze(0) for _ in range(remaining)])
        return torch.cat(candidates, dim=0)

    @staticmethod
    def _continuous_action_std(action_dists, mean: torch.Tensor) -> torch.Tensor:
        def safe_getattr(obj, attr):
            try:
                return getattr(obj, attr)
            except (AttributeError, NotImplementedError):
                return None

        std = safe_getattr(action_dists, "stddev")
        if std is None:
            std = safe_getattr(action_dists, "scale")
        base_dist = safe_getattr(action_dists, "base_dist")
        if std is None and base_dist is not None:
            std = safe_getattr(base_dist, "stddev")
        if std is None and base_dist is not None:
            std = safe_getattr(base_dist, "scale")
        if std is None:
            std = torch.ones_like(mean)
        return std.clamp_min(1e-6)

    def _shielded_action_probs(
        self,
        probs: torch.Tensor,
        pred_costs: torch.Tensor,
        uncertainty_delta: torch.Tensor,
        safe_threshold: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robust_costs = pred_costs + uncertainty_delta
        safe_mask = robust_costs <= safe_threshold.unsqueeze(-1)
        if self.shield_filter_mode == "soft":
            if self.shield_score_mode == "cost_penalty":
                # Baseline: plain pessimistic-cost penalty exp(-beta*Q_C), i.e. no budget-aware
                # barrier margin (no safe_threshold subtraction / clamp). Same cost critic, simpler rule.
                penalty = robust_costs
            elif self.shield_score_mode == "budget_penalty":
                # Budget-aware simple baseline: exp(-beta*(Q_C - B)); linear budget shift, no clamp
                # and no barrier-margin construction.
                penalty = robust_costs - safe_threshold.unsqueeze(-1)
            else:  # barrier (default, released behavior)
                penalty = (robust_costs - safe_threshold.unsqueeze(-1)).clamp_min(0.0)
            if self.shield_score_mode in ("cost_penalty", "budget_penalty"):
                # numerical stability: subtract the per-state min so exp(.) does not underflow when Q_C is
                # large. The subsequent normalization is shift-invariant, so this does not change results.
                penalty = penalty - penalty.min(dim=-1, keepdim=True).values.detach()
            safe_probs = probs * torch.exp(-self.shield_soft_beta * penalty)
        else:
            safe_probs = probs * safe_mask.float()
        safe_sum = safe_probs.sum(dim=-1, keepdim=True)
        normalized_safe_probs = safe_probs / safe_sum.clamp_min(1e-8)
        has_safe = safe_sum.squeeze(-1) > 0
        return normalized_safe_probs, has_safe

    @staticmethod
    def _init_hidden_state_compatible(agent: "ShieldedAMAGOAgent", batch_size: int, device: torch.device, dtype: torch.dtype):
        hidden_state = agent.traj_encoder.init_hidden_state(batch_size, device)
        # FlashAttention path may initialize cache as bfloat16. On some GPUs (and CPU),
        # q/k/v compute runs in float32, which makes indexed cache writes fail.
        # Keep KV cache dtype aligned with runtime embedding dtype.
        hidden_state.key_cache.data = hidden_state.key_cache.data.to(dtype=dtype)
        hidden_state.val_cache.data = hidden_state.val_cache.data.to(dtype=dtype)
        return hidden_state

    def get_actions(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state=None,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        # Shield is only defined for discrete AMAGO + constrained setup.
        if (
            self.current_epoch < self.shield_warmup_epochs
            or self.constrained_config is None
            or "cost_to_go" not in obs
            or not hasattr(self, "cost_critics")
        ):
            return super().get_actions(
                obs=obs,
                rl2s=rl2s,
                time_idxs=time_idxs,
                hidden_state=hidden_state,
                sample=sample,
            )

        tstep_emb = self.tstep_encoder(obs=obs, rl2s=rl2s)
        if (not self.training) and hidden_state is None and tstep_emb.shape[1] > 1:
            # Eval-time cache path expects L==1 per call. For batched sequence inputs
            # (common in tests), roll the encoder token-by-token.
            hidden_state = ShieldedAMAGOAgent._init_hidden_state_compatible(
                self, tstep_emb.shape[0], tstep_emb.device, tstep_emb.dtype
            )
            chunks = []
            for i in range(tstep_emb.shape[1]):
                out_i, hidden_state = self.traj_encoder(
                    tstep_emb[:, i : i + 1, :],
                    time_idxs=time_idxs[:, i : i + 1, :],
                    hidden_state=hidden_state,
                )
                chunks.append(out_i)
            state_seq = torch.cat(chunks, dim=1)
        else:
            if hidden_state is None and (not self.training):
                hidden_state = ShieldedAMAGOAgent._init_hidden_state_compatible(
                    self, tstep_emb.shape[0], tstep_emb.device, tstep_emb.dtype
                )
            state_seq, hidden_state = self.traj_encoder(
                tstep_emb, time_idxs=time_idxs, hidden_state=hidden_state
            )
        action_dists = self.actor(state_seq)

        budget = self._budget_from_obs(obs["cost_to_go"]).to(
            dtype=state_seq.dtype, device=state_seq.device
        )
        scaled_budget = torch.clamp(
            budget * float(self.cost_multiplier), min=0.0
        )
        safe_threshold = torch.clamp(
            float(self.shield_alpha) * scaled_budget - float(self.shield_safety_margin),
            min=0.0,
        )

        if not self.discrete:
            actions = self._continuous_shielded_actions(
                action_dists=action_dists,
                state_seq=state_seq,
                safe_threshold=safe_threshold,
                sample=sample,
            )
            if self.shield_log_margins:
                # Continuous actions: margin of the action ACTUALLY executed. Instrumented here, after
                # selection, so it is independent of which internal branch (fast path / staged sampling)
                # produced it. Same critic + PopArt denorm + aggregation the shield itself uses.
                with torch.no_grad():
                    # critic expects [K, B, L, G, A]; the executed action is gamma-reduced [B, L, A].
                    a_x = actions
                    if a_x.ndim == 3:
                        a_x = a_x.unsqueeze(-2)          # [B, L, 1, A]
                    a_x = a_x.unsqueeze(0)               # [1, B, L, 1, A]
                    n_g = len(self.gammas)
                    if a_x.shape[-2] == 1 and n_g > 1:
                        a_x = a_x.repeat(1, 1, 1, n_g, 1)  # [1, B, L, G, A]
                    q_x = self.cost_critics(state_seq, a_x)
                    q_x = self.cost_popart(q_x, normalized=False).squeeze(-1)
                    pc_x = self._aggregate_cost_critics_candidates(q_x)[..., -1]
                    unc_x = self._uncertainty_radius_candidates(q_x)
                    exec_cost = (pc_x + unc_x).reshape(-1)
                    exec_bq = safe_threshold.reshape(-1) - exec_cost
                    n = exec_cost.numel()
                    fa = getattr(self, "_last_feas_any", None)
                    ff = getattr(self, "_last_feas_frac", None)
                    nan = torch.full((n,), float("nan"))
                    any_feas = fa if (fa is not None and fa.numel() == n) else nan
                    frac_feas = ff if (ff is not None and ff.numel() == n) else nan
                    self._margin_log.append(
                        torch.stack(
                            [exec_bq.float().cpu(), budget.reshape(-1).float().cpu(),
                             exec_cost.float().cpu(), any_feas.float(), frac_feas.float()], dim=1
                        )
                    )
            return actions.to(dtype=torch.float32), hidden_state

        probs = action_dists.probs[..., -1, :]  # [B, L, A]
        pred_costs, uncertainty_delta = self._predicted_costs_per_action(state_seq)  # [B, L, A]
        normalized_safe_probs, has_safe = self._shielded_action_probs(
            probs=probs,
            pred_costs=pred_costs,
            uncertainty_delta=uncertainty_delta,
            safe_threshold=safe_threshold,
        )

        if self.shield_fallback == "policy":
            fallback_action = probs.argmax(dim=-1)
        else:
            fallback_action = pred_costs.argmin(dim=-1)

        flat_has_safe = has_safe.reshape(-1)
        flat_fallback = fallback_action.reshape(-1)
        flat_probs = normalized_safe_probs.reshape(-1, normalized_safe_probs.shape[-1])
        chosen = torch.zeros_like(flat_fallback)

        if sample:
            if flat_has_safe.any():
                sampled = torch.multinomial(flat_probs[flat_has_safe], num_samples=1).squeeze(-1)
                chosen[flat_has_safe] = sampled
            chosen = torch.where(flat_has_safe, chosen, flat_fallback)
        else:
            greedy_safe = flat_probs.argmax(dim=-1)
            chosen = torch.where(flat_has_safe, greedy_safe, flat_fallback)

        if self.shield_log_margins:
            # Executed-action barrier margin, computed HERE on the shield's own scale:
            #   b_Q = safe_threshold - (pred_cost + uncertainty) for the action actually taken.
            # Also record the remaining budget so violations (budget < 0) can be correlated offline.
            with torch.no_grad():
                robust_a = pred_costs + uncertainty_delta                      # [B, L, A]
                robust = robust_a.reshape(-1, pred_costs.shape[-1])
                exec_cost = robust.gather(1, chosen.view(-1, 1).long()).squeeze(1)
                exec_bq = safe_threshold.reshape(-1) - exec_cost
                # Real feasibility over the action set (not `has_safe`, which is trivially
                # true in soft mode).
                feas = (robust_a <= safe_threshold.unsqueeze(-1))
                any_feas = feas.any(dim=-1).reshape(-1).float().cpu()
                frac_feas = feas.float().mean(dim=-1).reshape(-1).float().cpu()
                self._margin_log.append(
                    torch.stack(
                        [exec_bq.float().cpu(), budget.reshape(-1).float().cpu(),
                         exec_cost.float().cpu(), any_feas, frac_feas], dim=1
                    )
                )

        actions = chosen.view_as(has_safe).unsqueeze(-1)
        return actions.to(dtype=torch.uint8), hidden_state

    def dump_margin_log(self, path: str) -> int:
        """Write the executed-action margin log to an .npz. Returns #rows written."""
        if not self._margin_log:
            return 0
        import numpy as np

        arr = torch.cat(self._margin_log, dim=0).numpy()
        np.savez(
            path,
            exec_b_q=arr[:, 0],       # executed action's barrier margin (shield's own scale)
            budget=arr[:, 1],         # remaining budget at that step (violation if < 0)
            exec_cost=arr[:, 2],      # executed action's robust predicted cost
            any_feasible=arr[:, 3],   # 1 if >=1 candidate satisfied the barrier at that step
            frac_feasible=arr[:, 4],  # fraction of the candidate set that satisfied it
        )
        return int(arr.shape[0])
