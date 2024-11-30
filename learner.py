import torch
from vtrace import compute_vtrace
import numpy as np
import matplotlib.pyplot as plt

class Learner:
    def __init__(self, model, optimizer, gamma=0.99, device="cpu"):
        self.shared_model = model.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.episode_rewards = []  # 에피소드별 총 보상 저장
        self.avg_rewards = []  # 10개씩 평균 계산된 보상 저장

    def learn(self, queue):
        # 그래프 설정
        plt.ion()
        plt.figure()

        while True:
            if not queue.empty():
                batch = queue.get()
                states = torch.tensor(np.array(batch["states"]), dtype=torch.float32).to(self.device)
                actions = torch.tensor(batch["actions"], dtype=torch.long).to(self.device)
                rewards = torch.tensor(batch["rewards"], dtype=torch.float32).to(self.device)
                values = torch.tensor(batch["values"], dtype=torch.float32).to(self.device)

                # 에피소드 총 보상 계산 및 저장
                episode_reward = rewards.sum().item()
                self.episode_rewards.append(episode_reward)

                # 매 10개 에피소드의 평균 계산
                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])  # 마지막 10개의 평균
                    self.avg_rewards.append(avg_reward)

                # V-trace 계산
                logits, current_values = self.shared_model(states)
                current_values = current_values.squeeze()
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

                # 그래프 업데이트
                if len(self.avg_rewards) > 0:  # 평균 보상이 계산된 경우만 그래프 그리기
                    plt.clf()
                    plt.plot(self.avg_rewards, label="10-Episode Avg Reward")
                    plt.xlabel("Batch (10 Episodes)")
                    plt.ylabel("Average Reward")
                    plt.title("Training Performance")
                    plt.legend()
                    plt.pause(0.1)
