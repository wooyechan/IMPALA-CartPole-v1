import torch.nn as nn
import torch

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 256)
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 입력을 평면화 (batch_size, input_dim)
        x = torch.relu(self.fc(x))
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value
