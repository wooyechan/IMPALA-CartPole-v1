import torch
from config import Config

def compute_vtrace(actor_policy_logits, learner_policy_logits, actions, discounts, rewards, values, bootstrap_value):
    # Compute importance weights (ρ_t)
    actor_log_probs = torch.log_softmax(actor_policy_logits, dim=-1)
    learner_log_probs = torch.log_softmax(learner_policy_logits, dim=-1)
    
    actions_expanded = actions.unsqueeze(-1)
    actor_log_probs = actor_log_probs.gather(-1, actions_expanded).squeeze(-1)
    learner_log_probs = learner_log_probs.gather(-1, actions_expanded).squeeze(-1)
    
    log_rhos = learner_log_probs - actor_log_probs
    rhos = torch.exp(log_rhos)

    # Compute truncated importance sampling weights (c_t)
    cs = torch.clamp(rhos, max=Config.COEF_MAX)
    clipped_rhos = torch.clamp(rhos, max=Config.RHO_MAX)

    # Compute temporal differences
    bootstrap_value = bootstrap_value.unsqueeze(1)
    next_values = torch.cat([values[:, 1:], bootstrap_value], dim=1)
    deltas = clipped_rhos * (rewards + discounts * next_values - values)

    # Compute V-trace targets recursively
    vs = torch.zeros_like(values)
    last_v = bootstrap_value.squeeze(1)
    
    for t in reversed(range(values.shape[1])):
        curr_discount = discounts[:, t]
        curr_c = cs[:, t]
        curr_delta = deltas[:, t]
        
        vs_t = values[:, t] + curr_delta + curr_c * curr_discount * (last_v - values[:, t])
        vs[:, t] = vs_t
        last_v = vs_t

    # Advantage for policy gradient
    vs_tp1 = torch.cat([vs[:, 1:], bootstrap_value], dim=1)
    pg_advantages = clipped_rhos * (rewards + discounts * vs_tp1 - values)

    return vs, pg_advantages, rhos
