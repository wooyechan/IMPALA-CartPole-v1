import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        # Simple but effective architecture for CartPole
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head
        self.policy = nn.Linear(hidden_dim, action_dim)
        
        # Value head
        self.value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights for faster convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.policy(features)
        value = self.value(features)
        return logits, value
