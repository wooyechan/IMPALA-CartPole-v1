import gym
import torch
import numpy as np
np.bool8 = bool
from torch.utils.tensorboard import SummaryWriter

class Actor:
    def __init__(self, actor_id, model, queue, shared_model, stop_event):
        self.actor_id = actor_id
        self.env = gym.make("CartPole-v1")
        self.local_model = model.to("cpu")
        self.shared_model = shared_model
        self.queue = queue
        self.stop_event = stop_event
        
    def sync_model(self):
        self.local_model.load_state_dict(self.shared_model.state_dict())
        
    def reset(self):
        """
        환경을 초기화하고 초기 상태 및 trajectory를 반환합니다.
        """
        total = 0
        state = np.array(self.env.reset()[0], dtype=np.float32)
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "logits": [],  # action logits
            "done": []     # episode termination flags
        }
        return total, state, trajectory

    def run(self):
        writer = SummaryWriter(log_dir=f"logs/actor_{self.actor_id}")
        episode = 0
        total, state, trajectory = self.reset()
        
        while not self.stop_event.is_set():
            self.sync_model() 
            trajectory_length = 0
            max_steps = 5  # Maximum trajectory length
            
            while trajectory_length < max_steps and not self.stop_event.is_set():
                with torch.no_grad():
                    logits, _ = self.local_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                done = bool(terminated) or bool(truncated)

                # Trajectory 업데이트
                trajectory["states"].append(state)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["logits"].append(logits.squeeze(0).numpy())
                trajectory["done"].append(done)
                
                trajectory_length += 1
                state = np.array(next_state, dtype=np.float32)
                
                if done:
                    break

            # If we have collected any steps, send the trajectory
            if trajectory_length > 0:
                # Send trajectory to learner
                self.queue.put({
                    "states": np.array(trajectory["states"]),
                    "actions": np.array(trajectory["actions"]),
                    "rewards": np.array(trajectory["rewards"]),
                    "logits": np.array(trajectory["logits"]),
                    "done": np.array(trajectory["done"])
                })

            if done:
                # TensorBoard에 에피소드 보상 기록
                episode += 1
                writer.add_scalar("score", total, episode)
                print(f"Actor {self.actor_id}: Episode {episode}, Reward: {total}")
                total, state, trajectory = self.reset()
