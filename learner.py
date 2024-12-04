import torch
from torch.utils.tensorboard import SummaryWriter
import time
from config import Config
from vtrace import compute_vtrace
from queue_manager import QueueManager

class Learner:
    def __init__(self, model, optimizer, stop_event):
        self.device = torch.device("cpu")
        self.shared_model = model.to(self.device)
        self.optimizer = optimizer
        self.stop_event = stop_event
        self.queue_manager = QueueManager(device=self.device)
        self.writer = SummaryWriter('logs/runs')
        self.step = 0
        self.log_interval = 100
        self.metric_buffer = {
            'Loss/total': [], 'Loss/policy': [], 'Loss/value': [], 'Loss/entropy': [],
            'IS_Ratio/min': [], 'IS_Ratio/max': [], 'IS_Ratio/avg': [],
            'Timing/batch_time': [], 'Timing/forward_time': [], 'Timing/backward_time': [],
            'Timing/vtrace_time': []
        }

    def compute_loss(self, states, actions, actor_logits, learner_logits, rewards, values, done):
        # Convert done flags to discount factors
        discounts = Config.GAMMA * (~done).float()
        
        # Get bootstrap value for last state
        with torch.no_grad():
            _, bootstrap_value = self.shared_model(states[:, -1])
            bootstrap_value = bootstrap_value.squeeze(-1)

        # Compute vtrace targets and advantages
        vs, pg_advantages, rhos = compute_vtrace(actor_logits, learner_logits, actions, discounts, rewards, values, bootstrap_value)

        policy_dist = torch.distributions.Categorical(logits=learner_logits)
        log_policy = policy_dist.log_prob(actions)
        entropy = policy_dist.entropy().mean(dim=1)

        policy_loss = -(log_policy * pg_advantages.detach()).mean()
        value_loss = 0.5 * ((values - vs.detach()) ** 2).mean()
        entropy_loss = entropy.mean()

        total_loss = policy_loss + Config.BASELINE_LOSS_WEIGHT * value_loss - Config.ENTROPY_COST * entropy_loss

        return total_loss, policy_loss, value_loss, entropy_loss, rhos

    def log_metrics(self):
        for metric_name, values in self.metric_buffer.items():
            if values:
                avg_value = sum(values) / len(values)
                self.writer.add_scalar(metric_name, avg_value, self.step)
        self.metric_buffer = {k: [] for k in self.metric_buffer.keys()}

    def learn(self, queue):
        while not self.stop_event.is_set():
            batch_start_time = time.time()
            batch = self.queue_manager.get_batch(queue)
            if batch is None:
                continue
                
            states, actions, rewards, actor_logits, done = batch
            batch_time = time.time() - batch_start_time
            
            # Forward pass
            forward_start_time = time.time()
            B, T = states.shape[:2]
            states_flat = states.reshape(-1, states.shape[-1])
            learner_logits, values = self.shared_model(states_flat)
            learner_logits = learner_logits.reshape(B, T, -1)
            values = values.reshape(B, T)
            forward_time = time.time() - forward_start_time

            # Loss computation with vtrace
            vtrace_start_time = time.time()
            total_loss, policy_loss, value_loss, entropy, rhos = self.compute_loss(states, actions, actor_logits, learner_logits, rewards, values, done)
            vtrace_time = time.time() - vtrace_start_time

            # Optimization step
            backward_start_time = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.shared_model.parameters(), Config.GRAD_CLIP)
            self.optimizer.step()
            backward_time = time.time() - backward_start_time

            # Calculate importance sampling ratio statistics
            with torch.no_grad():
                min_ratio = rhos.min().item()
                max_ratio = rhos.max().item()
                avg_ratio = rhos.mean().item()

            # Store metrics in buffer
            metrics = {
                'Loss/total': total_loss.item(),
                'Loss/policy': policy_loss.item(),
                'Loss/value': value_loss.item(),
                'Loss/entropy': entropy.item(),
                'IS_Ratio/min': min_ratio,
                'IS_Ratio/max': max_ratio,
                'IS_Ratio/avg': avg_ratio,
                'Timing/batch_time': batch_time,
                'Timing/forward_time': forward_time,
                'Timing/backward_time': backward_time,
                'Timing/vtrace_time': vtrace_time
            }
            
            for name, value in metrics.items():
                self.metric_buffer[name].append(value)
            
            self.step += 1
            
            if self.step % self.log_interval == 0:
                self.log_metrics()

    def cleanup(self):
        if hasattr(self, 'writer'):
            self.writer.close()