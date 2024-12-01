import torch
import gym
import numpy as np

class Actor:
    def __init__(self, actor_id, model, queue, shared_model):
        self.actor_id = actor_id
        self.env = gym.make("CartPole-v1")
        self.local_model = model
        self.shared_model = shared_model
        self.queue = queue
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
            while True:
                state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, value = self.local_model(state_tensor)
                action = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated
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

           
