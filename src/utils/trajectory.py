"""Trajectory recording and loading."""
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class TrajectoryStep:
    action: int
    x: int
    y: int
    reward: float
    state: str
    levels_completed: int


class TrajectoryRecorder:
    def __init__(self):
        self.steps: list = []
        self.game_id: str = ""

    def record(self, step: TrajectoryStep):
        self.steps.append(step)

    def save(self, path: "str | Path"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"game_id": self.game_id, "steps": [asdict(s) for s in self.steps]}, f)

    @classmethod
    def load(cls, path: "str | Path") -> "TrajectoryRecorder":
        with open(path) as f:
            data = json.load(f)
        rec = cls()
        rec.game_id = data["game_id"]
        rec.steps = [TrajectoryStep(**s) for s in data["steps"]]
        return rec
