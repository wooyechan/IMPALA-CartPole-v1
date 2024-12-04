import torch
from config import Config

def log_probs_from_logits_and_actions(policy_logits, actions):
    """Compute action log probabilities from policy logits and actions."""
    dist = torch.distributions.Categorical(logits=policy_logits)
    return dist.log_prob(actions)

def from_importance_weights(
    actor_policy_logits,  # [B, T, A]
    learner_policy_logits,  # [B, T, A]
    actions,  # [B, T]
    discounts,  # [B, T]
    rewards,  # [B, T]
    values,  # [B, T]
    bootstrap_value  # [B]
):
    """
    IMPALA V-trace implementation following the paper's equations exactly.
    All inputs have batch dimension first: [B, T, ...]
    
    V-trace targets v_s = V(x_s) + Σ_t γ^(t-s) Π_i c_i [δ_tρ_t]
    where δ_t = ρ_t(r_t + γV(x_{t+1}) - V(x_t)) is the temporal difference
    """
    # Compute importance weights (ρ_t in the paper)
    actor_log_probs = torch.log_softmax(actor_policy_logits, dim=-1)
    learner_log_probs = torch.log_softmax(learner_policy_logits, dim=-1)
    
    # Gather the log probs for the taken actions
    actions_expanded = actions.unsqueeze(-1)  # [B, T, 1]
    actor_log_probs = actor_log_probs.gather(-1, actions_expanded).squeeze(-1)  # [B, T]
    learner_log_probs = learner_log_probs.gather(-1, actions_expanded).squeeze(-1)  # [B, T]
    
    # ρ_t = π(a_t|x_t) / μ(a_t|x_t)
    # In log space: log(ρ_t) = log(π(a_t|x_t)) - log(μ(a_t|x_t))
    log_rhos = learner_log_probs - actor_log_probs # 순서 수정: actor/learner
    rhos = torch.exp(log_rhos)  # [B, T]

    # Compute truncated importance sampling weights (c_t in the paper)
    cs = torch.clamp(rhos, max=Config.COEF_MAX)  # [B, T]
    
    # Compute truncated rhos for immediate advantages (ρ_t in the paper)
    clipped_rhos = torch.clamp(rhos, max=Config.RHO_MAX)  # [B, T]

    # Compute temporal differences (δ_t in the paper)
    bootstrap_value = bootstrap_value.unsqueeze(1)  # [B, 1]
    next_values = torch.cat([values[:, 1:], bootstrap_value], dim=1)  # [B, T]
    deltas = clipped_rhos * (rewards + discounts * next_values - values)  # [B, T]

    # Compute V-trace targets recursively
    vs = torch.zeros_like(values)  # [B, T]
    last_v = bootstrap_value.squeeze(1)  # [B]
    
    # V-trace targets are computed backwards in time
    for t in reversed(range(values.shape[1])):
        # v_s = V(x_s) + Σ_t γ^(t-s) Π_i c_i [δ_tρ_t]
        vs[:, t] = values[:, t] + deltas[:, t]
        if t < values.shape[1] - 1:
            vs[:, t] += discounts[:, t] * cs[:, t] * (vs[:, t + 1] - values[:, t + 1])
    
    # Compute advantages for policy gradient
    # A_t = ρ_t(r_t + γv_{t+1} - V(x_t))
    vs_t_plus_1 = torch.cat([vs[:, 1:], bootstrap_value], dim=1)
    pg_advantages = clipped_rhos * (rewards + discounts * vs_t_plus_1 - values)

    return vs, pg_advantages
