import torch
from config import Config
from vtrace import from_importance_weights

class Learner:
    def __init__(self, model, optimizer, stop_event):
        self.device = torch.device("cpu")
        self.shared_model = model.to(self.device)
        self.optimizer = optimizer
        self.stop_event = stop_event

    @torch.no_grad()
    def _prepare_batch(self, batch):
        """Prepare batch data efficiently"""
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        actor_logits = torch.tensor(batch["logits"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["done"], dtype=torch.bool, device=self.device)
        return states, actions, rewards, actor_logits, done

    def compute_loss(self, states, actions, actor_logits, learner_logits, rewards, values, done):
        """Compute IMPALA loss efficiently"""
        # Convert done flags to discount factors (0 if done, gamma otherwise)
        discounts = Config.GAMMA * (~done).float()
        
        # Get bootstrap value for last state
        with torch.no_grad():
            _, bootstrap_value = self.shared_model(states[-1].unsqueeze(0))
            bootstrap_value = bootstrap_value.squeeze()

        # Compute vtrace targets and advantages
        vs, pg_advantages = from_importance_weights(actor_logits, learner_logits, actions, discounts, rewards, values, bootstrap_value)

        # Compute all losses in one go
        policy_dist = torch.distributions.Categorical(logits=learner_logits)
        log_policy = policy_dist.log_prob(actions)
        entropy = policy_dist.entropy()

        policy_loss = -(log_policy * pg_advantages.detach()).mean()
        value_loss = 0.5 * ((values - vs.detach()) ** 2).mean()
        entropy_bonus = entropy.mean()

        total_loss = policy_loss + Config.BASELINE_LOSS_WEIGHT * value_loss - Config.ENTROPY_COST * entropy_bonus

        return total_loss, policy_loss, value_loss, entropy_bonus

    def learn(self, queue):
        step = 0
        while not self.stop_event.is_set():
            if not queue.empty():
                # Process batch
                batch = queue.get()
                states, actions, rewards, actor_logits, done = self._prepare_batch(batch)
                
                # Get current policy predictions
                learner_logits, values = self.shared_model(states)
                values = values.squeeze(-1)

                # Compute losses
                total_loss, policy_loss, value_loss, entropy = self.compute_loss(states, actions, actor_logits, learner_logits, rewards, values, done)

                # Optimize
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.shared_model.parameters(), Config.GRAD_CLIP)
                self.optimizer.step()
                step += 1