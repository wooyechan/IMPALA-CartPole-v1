import torch
import gym
import numpy as np

class Actor:
    def __init__(self, actor_id, model, queue, t_max, shared_model):
        self.actor_id = actor_id
        self.env = gym.make("CartPole-v1")
        self.local_model = model
        self.shared_model = shared_model
        self.queue = queue
        self.t_max = t_max
        state, _ = self.env.reset()
        self.state = np.array(state, dtype=np.float32)
        self.done = False
        self.trajectory = {"states": [], "actions": [], "rewards": [], "values": []}

    def sync_model(self):
        """Synchronize local model with the shared model."""
        self.local_model.load_state_dict(self.shared_model.state_dict())

    def run(self):
        while True:
            self.sync_model()  # Get latest model from Learner
            for _ in range(self.t_max):
                state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, value = self.local_model(state_tensor)
                action = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # 보상 커스터마이징
                if terminated or truncated:
                    reward -= 10  # 에피소드 종료 시 패널티
                else:
                    reward += 0.1 * (1 - abs(next_state[2]))  # 기울기에 따라 보상 조정

                done = terminated or truncated
                print(reward) 
                self.trajectory["states"].append(self.state)
                self.trajectory["actions"].append(action)
                self.trajectory["rewards"].append(reward)
                self.trajectory["values"].append(value.item())

                self.state = np.array(next_state, dtype=np.float32)
                if done:
                    state, _ = self.env.reset()
                    self.state = np.array(state, dtype=np.float32)
                    self.queue.put(self.trajectory)
                    self.trajectory = {"states": [], "actions": [], "rewards": [], "values": []}
                    break

           
