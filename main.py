import gym
import torch
import torch.multiprocessing as mp
from actor import Actor
from learner import Learner
from networks import PolicyNetwork
from config import Config
import os
import shutil

def run_actor(actor_id, obs_space, action_space, queue, shared_model, stop_event):
    # Create model inside the process
    model = PolicyNetwork(obs_space, action_space)
    actor = Actor(actor_id, model, queue, shared_model, stop_event)
    actor.run()

def main():
    # Create necessary directories
    log_dir = os.path.join(os.getcwd(), "logs")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize environment to get dimensions
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    env.close()

    # Initialize shared model and optimizer
    shared_model = PolicyNetwork(obs_space, action_space)
    shared_model.share_memory()
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=Config.LEARNING_RATE)

    # Initialize multiprocessing components
    mp.set_start_method('spawn', force=True)
    processes = []
    queue = mp.Queue(maxsize=Config.QUEUE_SIZE)
    stop_event = mp.Event()

    # Start actor processes
    for i in range(Config.NUM_ACTORS):
        p = mp.Process(
            target=run_actor,
            args=(i + 1, obs_space, action_space, queue, shared_model, stop_event)
        )
        p.start()
        processes.append(p)

    # Start learner
    learner = Learner(shared_model, optimizer, stop_event)
    try:
        learner.learn(queue)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Cleaning up...")
    finally:
        stop_event.set()
        learner.cleanup()  # Close TensorBoard writer
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    main()
