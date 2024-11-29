import torch

class VTraceReturns:
    def __init__(self, target_values, advantages):
        self.target_values = target_values
        self.advantages = advantages

def compute_vtrace(rewards, values, gamma):
    """
    V-trace 계산. rewards와 values는 같은 길이를 가져야 함.
    """
    # values와 rewards는 같은 길이
    target_values = rewards[:-1] + gamma * values[1:]
    advantages = target_values - values[:-1]

    return VTraceReturns(
        target_values=target_values, 
        advantages=advantages
    )
