import gym
from multiprocessing import Process, Queue
from actor import Actor
from learner import Learner
from networks import PolicyNetwork
from config import Config
import torch
import multiprocessing as mp
import numpy as np

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    env.close()
    
    shared_model = PolicyNetwork(obs_space, action_space)
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=Config.LEARNING_RATE)
    queue = mp.Queue(maxsize=Config.BATCH_SIZE)

    actors = []
    for i in range(Config.NUM_ACTORS):
        local_model = PolicyNetwork(obs_space, action_space)
        actor = Actor(i, local_model, queue, shared_model)
        process = mp.Process(target=actor.run)
        actors.append(process)

    learner = Learner(shared_model, optimizer, Config.GAMMA)
    
    for actor in actors:
        actor.start()

    learner.learn(queue)

    for actor in actors:
        actor.join()

