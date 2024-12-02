import torch
from config import Config
from vtrace import from_importance_weights

class Learner:
    def __init__(self, model, optimizer, stop_event):
        self.device = torch.device("cpu")
        self.shared_model = model.to(self.device)
        self.optimizer = optimizer
        self.gamma = Config.GAMMA
        self.stop_event = stop_event
        self.baseline_cost = Config.BASELINE_LOSS_WEIGHT
        self.entropy_cost = Config.ENTROPY_COST

    @torch.no_grad()
    def _prepare_batch(self, batch):
        """Prepare batch data efficiently"""
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        values = torch.tensor(batch["values"], dtype=torch.float32, device=self.device)
        behavior_logits = torch.tensor(batch["logits"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["done"], dtype=torch.bool, device=self.device)
        return states, actions, rewards, values, behavior_logits, done

    def compute_loss(self, states, actions, behavior_logits, target_logits, rewards, values, done):
        """Compute IMPALA loss efficiently"""
        # Convert done flags to discount factors (0 if done, gamma otherwise)
        discounts = self.gamma * (~done).float()
        
        # Get bootstrap value for last state
        with torch.no_grad():
            _, bootstrap_value = self.shared_model(states[-1].unsqueeze(0))
            bootstrap_value = bootstrap_value.squeeze()

        # Compute vtrace targets and advantages
        vtrace_returns = from_importance_weights(
            behavior_policy_logits=behavior_logits,
            target_policy_logits=target_logits,
            actions=actions,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=Config.RHO_MAX,
            clip_pg_rho_threshold=Config.COEF_MAX
        )

        # Compute all losses in one go
        policy_dist = torch.distributions.Categorical(logits=target_logits)
        log_policy = policy_dist.log_prob(actions)
        entropy = policy_dist.entropy()

        policy_loss = -(log_policy * vtrace_returns["pg_advantages"].detach()).mean()
        value_loss = 0.5 * ((values - vtrace_returns["vs"].detach()) ** 2).mean()
        entropy_bonus = entropy.mean()

        total_loss = (
            policy_loss +
            self.baseline_cost * value_loss -
            self.entropy_cost * entropy_bonus
        )

        return total_loss, policy_loss, value_loss, entropy_bonus

    def learn(self, queue):
        step = 0
        while not self.stop_event.is_set():
            if not queue.empty():
                try:
                    # Process batch
                    batch = queue.get()
                    states, actions, rewards, values, behavior_logits, done = self._prepare_batch(batch)
                    
                    # Get current policy predictions
                    target_logits, current_values = self.shared_model(states)
                    current_values = current_values.squeeze(-1)

                    # Compute losses
                    total_loss, policy_loss, value_loss, entropy = self.compute_loss(
                        states=states,
                        actions=actions,
                        behavior_logits=behavior_logits,
                        target_logits=target_logits,
                        rewards=rewards,
                        values=values,
                        done=done
                    )

                    # Optimize
                    self.optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.shared_model.parameters(), Config.GRAD_CLIP)
                    self.optimizer.step()

                    if step % Config.LOG_INTERVAL == 0:
                        print(
                            f"[Learner] Step: {step}, Loss: {total_loss.item():.3f}, "
                            f"Policy Loss: {policy_loss.item():.3f}, "
                            f"Value Loss: {value_loss.item():.3f}, "
                            f"Entropy: {entropy.item():.3f}"
                        )
                    step += 1
                except Exception as e:
                    print(f"Error in learner: {str(e)}")
                    continue
