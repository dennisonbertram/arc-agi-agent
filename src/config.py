"""Configuration for ARC-AGI-3 RL Agent."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    arc_api_key: str = ""
    arc_base_url: str = "https://three.arcprize.org"
    operation_mode: str = "OFFLINE"
    grid_size: int = 64
    num_colors: int = 16
    num_actions: int = 8
    aux_feature_dim: int = 15
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_cnn_layers: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 64
    epochs_per_update: int = 4
    buffer_size: int = 10000
    max_actions_per_game: int = 200
    improvement_iterations: int = 10
    games_per_iteration: int = 5
    eval_games: int = 3
    improvement_threshold: float = 0.05
    checkpoint_dir: Path = Path("checkpoints")
    recording_dir: Path = Path("recordings")
    log_dir: Path = Path("logs")

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}


config = Config()
