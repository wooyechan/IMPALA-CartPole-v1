import torch
from config import Config

class QueueManager:
    def __init__(self, device="cpu"):
        self.device = device
        self.batch_size = Config.BATCH_SIZE
    
    def create_batch(self, trajectories):
        states, actions, rewards, logits, dones = zip(*trajectories)
        
        batch_states = torch.stack([torch.tensor(s, dtype=torch.float32, device=self.device) for s in states])
        batch_actions = torch.stack([torch.tensor(a, dtype=torch.long, device=self.device) for a in actions])
        batch_rewards = torch.stack([torch.tensor(r, dtype=torch.float32, device=self.device) for r in rewards])
        batch_logits = torch.stack([torch.tensor(l, dtype=torch.float32, device=self.device) for l in logits])
        batch_dones = torch.stack([torch.tensor(d, dtype=torch.bool, device=self.device) for d in dones])
        
        return batch_states, batch_actions, batch_rewards, batch_logits, batch_dones
    
    def get_batch(self, queue):
        trajectories = []
        
        for _ in range(self.batch_size):
            if queue.empty():
                break
            trajectories.append(queue.get())
            
        if len(trajectories) < self.batch_size:
            return None
            
        return self.create_batch(trajectories)
