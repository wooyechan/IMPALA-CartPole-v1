class Config:
    # Training hyperparameters
    GAMMA = 0.99                    # discount factor for rewards
    LEARNING_RATE = 0.001          # learning rate for optimizer
    GRAD_CLIP = 0.7                # gradient clipping threshold
    BASELINE_LOSS_WEIGHT = 0.4     # weight for value function loss
    ENTROPY_COST = 0.008           # weight for entropy regularization

    # V-trace parameters
    COEF_MAX = 0.95                # max importance sampling weight for value trace
    RHO_MAX = 0.95                 # max importance sampling weight for policy gradient

    # Architecture parameters
    HIDDEN_SIZE = 64               # size of hidden layers in networks
    NUM_ACTORS = 4                 # number of parallel actors
    QUEUE_SIZE = 100               # size of trajectory queue
    BATCH_SIZE = 6                 # number of trajectories per batch
    UNROLL_LENGTH = 4              # length of trajectory segments
