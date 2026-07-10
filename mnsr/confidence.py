from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfidenceWeights:
    validation_score: float = 0.30
    reasoning_consistency: float = 0.20
    answer_consistency: float = 0.20
    error_penalty: float = 0.15
    memory_similarity: float = 0.10
    self_eval: float = 0.05
    retry_decay: float = 0.05


class ConfidenceEstimator:
    def __init__(self, weights: ConfidenceWeights = ConfidenceWeights()):
        self.weights = weights

    def estimate(
        self,
        validation_score: float = 1.0,
        reasoning_consistency: float = 1.0,
        answer_consistency: float = 1.0,
        num_errors: int = 0,
        memory_similarity: float = 0.0,
        retry_count: int = 0,
        self_eval_score: Optional[float] = None,
    ) -> float:
        w = self.weights

        error_term = 1.0 / (1.0 + num_errors)
        retry_term = 1.0 / (1.0 + retry_count)

        components = (
            w.validation_score * self._clip(validation_score)
            + w.reasoning_consistency * self._clip(reasoning_consistency)
            + w.answer_consistency * self._clip(answer_consistency)
            + w.error_penalty * error_term
            + w.memory_similarity * self._clip(memory_similarity)
            - w.retry_decay * (1.0 - retry_term)
        )

        if self_eval_score is not None:
            components += w.self_eval * self._clip(self_eval_score)
        else:
            components += w.self_eval * self._clip(validation_score)

        return self._clip(components)

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
