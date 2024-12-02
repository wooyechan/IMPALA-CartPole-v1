import torch

def log_probs_from_logits_and_actions(policy_logits, actions):
    """Compute action log probabilities from policy logits and actions."""
    dist = torch.distributions.Categorical(logits=policy_logits)
    return dist.log_prob(actions)

def from_importance_weights(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0
):
    """
    Optimized V-trace implementation.
    """
    # Compute log importance weights all at once
    behavior_log_probs = torch.log_softmax(behavior_policy_logits, dim=-1)
    target_log_probs = torch.log_softmax(target_policy_logits, dim=-1)
    
    # Gather the log probs for the taken actions
    behavior_log_probs = behavior_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    target_log_probs = target_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    
    log_rhos = target_log_probs - behavior_log_probs
    rhos = torch.exp(log_rhos)

    # Compute clipped importance sampling weights
    clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    cs = torch.clamp(rhos, max=1.0)

    # Compute temporal differences
    next_values = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
    deltas = clipped_rhos * (rewards + discounts * next_values - values)

    # Compute v-trace targets and advantages in a single pass
    result = torch.zeros_like(values)
    acc = 0
    for t in range(len(values) - 1, -1, -1):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        result[t] = acc

    vs = values + result
    
    # Compute policy gradient advantages
    vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
    pg_advantages = clipped_rhos * (rewards + discounts * vs_t_plus_1 - values)

    return {
        "vs": vs,
        "pg_advantages": pg_advantages,
    }
