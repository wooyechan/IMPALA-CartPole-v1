import gym
import torch
import numpy as np
np.bool8 = bool
from torch.utils.tensorboard import SummaryWriter
from config import Config

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

    def run(self):
        writer = SummaryWriter(log_dir=f"logs/actor_{self.actor_id}")
        episode = 0
        total = 0
        state = np.array(self.env.reset()[0], dtype=np.float32)

        while not self.stop_event.is_set():
            self.sync_model() 
            states, actions, rewards, logits, dones = [], [], [], [], []
            
            # Collect fixed-length trajectory
            for _ in range(Config.UNROLL_LENGTH):
                with torch.no_grad():
                    logits_tensor, _ = self.local_model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                probs = torch.softmax(logits_tensor, dim=-1)
                action = torch.multinomial(probs, 1).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                done = bool(terminated) or bool(truncated)

                # Trajectory 업데이트
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                logits.append(logits_tensor.squeeze(0).numpy())
                dones.append(done)
                
                state = np.array(next_state, dtype=np.float32)
                
                if done:
                    # TensorBoard에 에피소드 보상 기록
                    total = 0
                    episode += 1
                    writer.add_scalar("score", total, episode)
                    print(f"Actor {self.actor_id}: Episode {episode}, Reward: {total}")
                    
            # 항상 trajectory를 queue에 전송
            self.queue.put((
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(logits),
                np.array(dones)
            ))

            if done:
                state = np.array(self.env.reset()[0], dtype=np.float32)
