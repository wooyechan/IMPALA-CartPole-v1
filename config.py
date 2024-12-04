class Config:
    # Training hyperparameters
    LEARNING_RATE = 0.001  # 빠른 학습 유지
    GAMMA = 0.99
    ENTROPY_COST = 0.008  # 두 설정의 중간값
    BASELINE_LOSS_WEIGHT = 0.4  # value와 policy의 균형
    GRAD_CLIP = 0.7  # 적당한 gradient 제한

    # V-trace parameters
    RHO_MAX = 0.95  # 약간의 제한만 유지
    COEF_MAX = 0.95  # 약간의 제한만 유지

    # Architecture parameters
    HIDDEN_SIZE = 64  # 작은 네트워크 유지
    NUM_ACTORS = 4
    QUEUE_SIZE = 100  # 빠른 경험 순환
    BATCH_SIZE = 6  # 두 설정의 중간값
    UNROLL_LENGTH = 4  # 짧은 trajectory로 빠른 학습
    BATCH_TRAJECTORIES = 6  # 두 설정의 중간값
