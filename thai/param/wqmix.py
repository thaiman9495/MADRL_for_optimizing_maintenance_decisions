import torch
from dataclasses import dataclass


@dataclass
class WqmixParam:
    run_mode: int
    gpu: str
    is_stochastic_dependence: bool

    n_steps_training: int
    buffer_capacity: int
    batch_size: int
    discount_factor: float
    episode_len_train: int
    grad_norm_clip: float
    reward_normalization: float

    # Exploration
    epsilon_start: float
    epsilon_end: float
    epsilon_anneal_time: int

    # Learning rate (lr)
    lr_start: float
    lr_end: float
    lr_step_size: int
    lr_decay_constant: float

    # Network
    hidden_conf_1: tuple
    hidden_conf_2: tuple
    rnn_conf: int

    n_neurons_mixer: int
    n_neurons_hyper: int
    mixer_true_conf: tuple

    target_update_freq: int
    policy_update_freq: int
    log_freq: int

    # Weighting
    weighting_constant: float

    # Evaluation
    n_runs: int
    episode_len_eval: int

    # Path for saving data
    path_log: str
    path_model: str

    n_agents: int = 11
    input_max: float = 3.0
    n_actions: int = 3
