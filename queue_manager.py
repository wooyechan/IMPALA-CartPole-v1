import torch
import numpy as np
from config import Config

class QueueManager:
    def __init__(self, device="cpu"):
        self.device = device
        self.batch_size = Config.BATCH_SIZE
    
    def create_batch(self, trajectories):
        """
        여러 trajectory를 배치 차원으로 쌓음 [T, ...] -> [B, T, ...]
        Args:
            trajectories: [(states, actions, rewards, logits, dones), ...]
            각 trajectory는 UNROLL_LENGTH 길이의 시퀀스
        Returns:
            states: [batch_size, unroll_length, state_dim]
            actions: [batch_size, unroll_length]
            rewards: [batch_size, unroll_length]
            logits: [batch_size, unroll_length, action_dim]
            dones: [batch_size, unroll_length]
        """
        # Unzip trajectories
        states, actions, rewards, logits, dones = zip(*trajectories)
        
        # Convert to tensors with proper batch and time dimensions
        batch_states = torch.stack([
            torch.tensor(s, dtype=torch.float32, device=self.device) 
            for s in states
        ])  # [B, T, state_dim]
        
        batch_actions = torch.stack([
            torch.tensor(a, dtype=torch.long, device=self.device)
            for a in actions
        ])  # [B, T]
        
        batch_rewards = torch.stack([
            torch.tensor(r, dtype=torch.float32, device=self.device)
            for r in rewards
        ])  # [B, T]
        
        batch_logits = torch.stack([
            torch.tensor(l, dtype=torch.float32, device=self.device)
            for l in logits
        ])  # [B, T, action_dim]
        
        batch_dones = torch.stack([
            torch.tensor(d, dtype=torch.bool, device=self.device)
            for d in dones
        ])  # [B, T]
        
        return batch_states, batch_actions, batch_rewards, batch_logits, batch_dones
    
    def get_batch(self, queue):
        """
        Queue에서 batch_size만큼의 trajectory를 가져와서 배치로 변환
        """
        trajectories = []
        
        # Collect trajectories
        for _ in range(self.batch_size):
            if queue.empty():
                break
            trajectories.append(queue.get())
            
        if len(trajectories) < self.batch_size:
            return None
            
        return self.create_batch(trajectories)
