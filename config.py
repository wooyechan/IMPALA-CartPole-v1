class Config:
    # Training hyperparameters
    LEARNING_RATE = 0.0001  # 학습률을 낮춤
    GAMMA = 0.99
    ENTROPY_COST = 0.001  # 엔트로피 코스트를 약간 증가
    BASELINE_LOSS_WEIGHT = 0.5  # value network의 영향력 증가
    GRAD_CLIP = 40.0  # gradient clipping을 더 강하게

    # V-trace parameters
    RHO_MAX = 1  # importance weight clipping을 더 강하게
    COEF_MAX = 1  # importance weight clipping을 더 강하게

    # Architecture parameters
    HIDDEN_SIZE = 256
    NUM_ACTORS = 4
    QUEUE_SIZE = 200
    BATCH_SIZE = 4
    UNROLL_LENGTH = 5  # Length of trajectories before sending to learner
    BATCH_TRAJECTORIES = 4  # Number of trajectories to process in one batch
