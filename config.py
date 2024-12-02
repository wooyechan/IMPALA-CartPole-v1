class Config:
    # Training hyperparameters
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    ENTROPY_COST = 0.0005
    BASELINE_LOSS_WEIGHT = 0.5
    GRAD_CLIP = 40.0

    # V-trace parameters
    RHO_MAX = 1.0  # Importance weight clipping for advantage calculation
    COEF_MAX = 1.0  # Importance weight clipping for value function

    # Architecture parameters
    HIDDEN_SIZE = 256
    NUM_ACTORS = 4
    QUEUE_SIZE = 20
    BATCH_SIZE = 32

    # Logging
    LOG_INTERVAL = 100  # Steps between logging
