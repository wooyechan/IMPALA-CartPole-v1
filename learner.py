import torch
from vtrace import compute_vtrace
import numpy as np

class Learner:
    def __init__(self, model, optimizer, gamma=0.99, device="cpu"):
        self.shared_model = model.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device

    def learn(self, queue):
        cnt = 0
        while True:
            if not queue.empty():
                batch = queue.get()
                states = torch.tensor(np.array(batch["states"]), dtype=torch.float32).to(self.device)
                actions = torch.tensor(batch["actions"], dtype=torch.long).to(self.device)
                rewards = torch.tensor(batch["rewards"], dtype=torch.float32).to(self.device)
                values = torch.tensor(batch["values"], dtype=torch.float32).to(self.device)

                logits, current_values = self.shared_model(states)
                current_values = current_values.squeeze()

                # V-trace 계산
                vtrace_returns = compute_vtrace(rewards, current_values, self.gamma)

                # 행동 선택 확률의 로그 계산
                action_probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(action_probs[range(len(actions)), actions])

                # 크기 일치 확인 및 조정
                log_probs = log_probs[:len(vtrace_returns.advantages)]

                # 손실 계산
                policy_loss = -(log_probs * vtrace_returns.advantages).mean()
                value_loss = (vtrace_returns.target_values - current_values[:len(vtrace_returns.target_values)]).pow(2).mean()
                entropy_loss = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

                # 모델 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
