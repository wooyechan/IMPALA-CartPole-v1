import torch
from config import Config
from vtrace import from_importance_weights
from queue_manager import QueueManager

class Learner:
    def __init__(self, model, optimizer, stop_event):
        self.device = torch.device("cpu")
        self.shared_model = model.to(self.device)
        self.optimizer = optimizer
        self.stop_event = stop_event
        self.queue_manager = QueueManager(device=self.device)

    def compute_loss(self, states, actions, actor_logits, learner_logits, rewards, values, done):
        """
        배치 단위로 IMPALA loss 계산
        모든 입력은 [B, T, ...] 형태
        """
        # Convert done flags to discount factors (0 if done, gamma otherwise)
        discounts = Config.GAMMA * (~done).float()  # [B, T]
        
        # Get bootstrap value for last state of each trajectory
        with torch.no_grad():
            _, bootstrap_value = self.shared_model(states[:, -1])  # [B, 1]
            bootstrap_value = bootstrap_value.squeeze(-1)  # [B]

        # Compute vtrace targets and advantages
        vs, pg_advantages = from_importance_weights(
            actor_logits, learner_logits, actions, 
            discounts, rewards, values, bootstrap_value
        )

        # Compute policy loss for each trajectory
        policy_dist = torch.distributions.Categorical(logits=learner_logits)
        log_policy = policy_dist.log_prob(actions)  # [B, T]
        entropy = policy_dist.entropy().mean(dim=1)  # [B]

        # Mean over time and batch dimensions
        policy_loss = -(log_policy * pg_advantages.detach()).mean()
        value_loss = 0.5 * ((values - vs.detach()) ** 2).mean()
        entropy_loss = entropy.mean()

        total_loss = policy_loss + Config.BASELINE_LOSS_WEIGHT * value_loss - Config.ENTROPY_COST * entropy_loss

        return total_loss, policy_loss, value_loss, entropy_loss

    def learn(self, queue):
        step = 0
        while not self.stop_event.is_set():
            # Queue에서 배치 크기만큼의 trajectory를 가져와서 처리
            batch = self.queue_manager.get_batch(queue)
            if batch is None:
                continue
                
            states, actions, rewards, actor_logits, done = batch  # [B, T, ...]
            
            # Get current policy predictions (배치 처리)
            B, T = states.shape[:2]
            states_flat = states.reshape(-1, states.shape[-1])  # [B*T, state_dim]
            learner_logits, values = self.shared_model(states_flat)
            
            # Reshape back to [B, T, ...]
            learner_logits = learner_logits.reshape(B, T, -1)
            values = values.reshape(B, T)

            # Compute losses with proper batch handling
            total_loss, policy_loss, value_loss, entropy = self.compute_loss(
                states, actions, actor_logits, learner_logits, rewards, values, done
            )

            # Optimize
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.shared_model.parameters(), Config.GRAD_CLIP)
            self.optimizer.step()
            step += 1