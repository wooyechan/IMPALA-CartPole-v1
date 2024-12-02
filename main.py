import gym
from multiprocessing import Process, Queue, Event
from actor import Actor
from learner import Learner
from networks import PolicyNetwork
from config import Config
import torch
import os
import shutil

if __name__ == "__main__":
    log_dir = "logs"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir) 
    
    torch.multiprocessing.set_start_method("spawn")
    
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    shared_model = PolicyNetwork(obs_space, action_space)
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=Config.LEARNING_RATE)
    queue = Queue(maxsize=Config.QUEUE_SIZE)
    stop_event = Event()
    
    actors = [
        Process(target=Actor(i + 1, PolicyNetwork(obs_space, action_space), queue, shared_model, stop_event).run)
        for i in range(4)
    ]

    learner = Learner(shared_model, optimizer, stop_event)

    for actor in actors:
        actor.start()

    try:
        learner.learn(queue) 
    except KeyboardInterrupt:
        print("Learning terminated.")
        stop_event.set() 
        
    for actor in actors:
        actor.terminate()
        actor.join()
