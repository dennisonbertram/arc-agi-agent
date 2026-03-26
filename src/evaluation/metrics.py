"""ARC-AGI-3 scoring metrics."""


def compute_rhae(ai_actions: int, human_baseline: int) -> float:
    """Compute Relative Human Action Efficiency.

    RHAE = (human_baseline / ai_actions)^2, capped at 1.0
    """
    if ai_actions <= 0:
        return 0.0
    score = (human_baseline / ai_actions) ** 2
    return min(score, 1.0)


def compute_game_score(level_scores: list) -> float:
    """Compute weighted game score.

    Later levels weighted more: weight = level_number
    """
    if not level_scores:
        return 0.0
    weighted_sum = sum(score * (i + 1) for i, score in enumerate(level_scores))
    weight_total = sum(i + 1 for i in range(len(level_scores)))
    return weighted_sum / weight_total


def compute_overall_score(game_scores: list) -> float:
    """Average of game scores."""
    return sum(game_scores) / len(game_scores) if game_scores else 0.0
