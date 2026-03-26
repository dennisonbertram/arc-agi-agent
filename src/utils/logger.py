"""Structured logging for training."""
import json
import time
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir: "str | Path" = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training.jsonl"

    def log(self, data: dict):
        data["timestamp"] = time.time()
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def log_episode(self, episode: int, reward: float, length: int, **kwargs):
        self.log({"type": "episode", "episode": episode, "reward": reward, "length": length, **kwargs})

    def log_update(self, step: int, **kwargs):
        self.log({"type": "update", "step": step, **kwargs})
