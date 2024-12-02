import torch

def log_probs_from_logits_and_actions(policy_logits, actions):
    """로짓과 행동으로 로그 확률 계산."""
    dist = torch.distributions.Categorical(logits=policy_logits)
    return dist.log_prob(actions)


def from_importance_weights(
    log_rhos, discounts, rewards, values, bootstrap_value, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0
):
    """V-Trace 계산."""
    rhos = torch.exp(log_rhos)

    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos

    cs = torch.clamp(rhos, max=1.0)
    deltas = clipped_rhos * (rewards + discounts * bootstrap_value - values)

    acc = 0
    vs_minus_v_xs = []
    for delta, c in zip(reversed(deltas), reversed(cs)):
        acc = delta + discounts * c * acc
        vs_minus_v_xs.append(acc)

    vs_minus_v_xs.reverse()
    vs = values + torch.stack(vs_minus_v_xs)

    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos

    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs[1:] - values)

    return {"vs": vs[:-1], "pg_advantages": pg_advantages}
