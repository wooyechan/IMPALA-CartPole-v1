import numpy as np

def discount_rewards(rewards, gamma):
    discounted = []
    running_add = 0
    for r in reversed(rewards):
        running_add = r + gamma * running_add
        discounted.insert(0, running_add)
    return np.array(discounted)
