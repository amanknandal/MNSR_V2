from dataclasses import dataclass
from typing import Dict
from mnsr.cognitive_state import CognitiveState


@dataclass
class ControllerConfig:
    confidence_threshold: float = 0.55
    validation_threshold: float = 0.60
    answer_consistency_threshold: float = 0.60
    reasoning_consistency_threshold: float = 0.60
    memory_threshold: float = 0.65
    risk_threshold: float = 0.75
    max_reasoning_lines_for_decompose: int = 3
    max_steps_per_query: int = 4
    max_retry_count: int = 6


class MetaCognitiveController:

    def __init__(self, config: ControllerConfig = ControllerConfig()):
        self.config = config

    def evaluate(self, state: CognitiveState) -> str:
        cfg = self.config

        if state.current_step >= cfg.max_steps_per_query or state.retry_count >= cfg.max_retry_count:
            return self._select(state, "TERMINATE")

        severe_risk = state.contradiction or state.risk_score >= cfg.risk_threshold

        if severe_risk:
            if state.was_recently_attempted("BACKTRACK", window=len(state.correction_history)):
                return self._select(state, "REPLAN")
            return self._select(state, "BACKTRACK")

        if (
            state.validation_score >= cfg.validation_threshold
            and state.reasoning_consistency >= cfg.reasoning_consistency_threshold
            and state.answer_consistency < cfg.answer_consistency_threshold
        ):
            return self._select(state, "ANSWER_REPAIR")

        if state.validation_score < cfg.validation_threshold and len(state.symbolic_errors) > 0:
            return self._select(state, "REASONING_REPAIR")

        if (
            state.total_steps <= cfg.max_reasoning_lines_for_decompose
            and state.confidence < cfg.confidence_threshold
        ):
            return self._select(state, "DECOMPOSE")

        if (
            state.memory_match_type == "failure"
            and state.memory_similarity >= cfg.memory_threshold
        ):
            return self._select(state, "MEMORY_REFLECTION")

        if (
            state.confidence < cfg.confidence_threshold
            and state.validation_score >= cfg.validation_threshold
            and state.retry_count == 0
        ):
            return self._select(state, "SELF_VERIFY")

        if state.reasoning_consistency < cfg.reasoning_consistency_threshold:
            return self._select(state, "SELF_CRITIQUE")

        if state.confidence < cfg.confidence_threshold and state.retry_count >= 1:
            return self._select(state, "MULTI_PATH_REASONING")

        return self._select(state, "CONTINUE")

    @staticmethod
    def _select(state: CognitiveState, action: str) -> str:
        state.set_action(action)
        return action

    def explain(self, state: CognitiveState) -> Dict:
        return {
            "chosen_action": state.action,
            "metrics": {
                "confidence": state.confidence,
                "uncertainty": state.uncertainty,
                "validation_score": state.validation_score,
                "reasoning_consistency": state.reasoning_consistency,
                "answer_consistency": state.answer_consistency,
                "contradiction": state.contradiction,
                "memory_similarity": state.memory_similarity,
                "memory_match_type": state.memory_match_type,
                "systemic_risk": state.risk_score,
            },
            "execution": {
                "step": state.current_step,
                "retry_count": state.retry_count,
                "max_allowed_steps": self.config.max_steps_per_query,
                "loop_pct": round(state.current_step / self.config.max_steps_per_query, 2),
                "correction_history": state.correction_history,
            },
        }
