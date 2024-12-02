import torch
import numpy as np
from config import Config

class Learner:
    def __init__(self, model, optimizer, stop_event):
        self.shared_model = model.to("cpu")
        self.optimizer = optimizer
        self.gamma = Config.GAMMA
        self.stop_event = stop_event

    def compute_vtrace(self, rewards, values, log_mu, log_pi):
        rho = torch.clamp(torch.exp(log_pi - log_mu), max=Config.RHO_MAX)
        coef = torch.clamp(rho, max=Config.COEF_MAX)

        T, B = rewards.size(0), rewards.size(1)
        vs = torch.zeros((T + 1, B), device=rewards.device)
        advantages = torch.zeros((T, B), device=rewards.device)
        next_values = torch.cat([values, torch.zeros(1, B, device=values.device)], dim=0)

        for rev_step in reversed(range(T)):
            delta = rho[rev_step] * (rewards[rev_step] + self.gamma * next_values[rev_step + 1] - values[rev_step])
            advantages[rev_step] = delta
            vs[rev_step] = (
                values[rev_step]
                + delta
                + self.gamma * coef[rev_step] * (vs[rev_step + 1] - next_values[rev_step + 1])
            )

        return vs[:-1], advantages

    def learn(self, queue):
        step = 0
        while not self.stop_event.is_set():
            if not queue.empty():
                batch = queue.get()

                states = torch.tensor(np.array(batch["states"]), dtype=torch.float32).to("cpu")
                actions = torch.tensor(np.array(batch["actions"]), dtype=torch.long).to("cpu")
                rewards = torch.tensor(np.array(batch["rewards"]), dtype=torch.float32).to("cpu")
                logits = torch.tensor(np.array(batch["logits"]), dtype=torch.float32).to("cpu")

                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_mu = dist.log_prob(actions)

                current_logits, values = self.shared_model(states)
                values = values.squeeze(-1)
                dist_learner = torch.distributions.Categorical(logits=current_logits)
                log_pi = dist_learner.log_prob(actions).unsqueeze(-1)

                vs, advantages = self.compute_vtrace(
                    rewards=rewards.unsqueeze(-1),
                    values=values.unsqueeze(-1),
                    log_mu=log_mu.unsqueeze(-1),
                    log_pi=log_pi,
                )

                policy_loss = -(log_pi * advantages.detach()).sum()
                baseline_loss = 0.5 * ((vs.detach() - values.unsqueeze(-1)) ** 2).sum()
                entropy_loss = -0.01 * dist_learner.entropy().sum()

                total_loss = policy_loss + Config.BASELINE_LOSS_WEIGHT * baseline_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.shared_model.parameters(), Config.GRAD_CLIP)
                self.optimizer.step()

                if step % Config.LOG_INTERVAL == 0:
                    print(
                        f"[Learner] Step: {step}, Loss: {total_loss.item():.3f}, "
                        f"Policy Loss: {policy_loss.item():.3f}, "
                        f"Baseline Loss: {baseline_loss.item():.3f}, "
                        f"Entropy Loss: {entropy_loss.item():.3f}"
                    )
                step += 1
