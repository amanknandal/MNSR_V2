from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


@dataclass
class CognitiveState:

    reasoning: str = ""
    final_answer: Optional[str] = None

    confidence: float = 1.0
    uncertainty: float = 0.0

    contradiction: bool = False
    validation_score: float = 1.0
    symbolic_errors: List[Dict] = field(default_factory=list)
    symbolic_warnings: List[Dict] = field(default_factory=list)

    reasoning_consistency: float = 1.0
    answer_consistency: float = 1.0

    memory_similarity: float = 0.0
    memory_match_type: Optional[str] = None

    risk_score: float = 0.0
    action: Optional[str] = None
    correction_history: List[str] = field(default_factory=list)

    current_step: int = 0
    total_steps: int = 0
    retry_count: int = 0
    max_reflection_depth: int = 4

    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0

    def update_reasoning(self, reasoning: str):
        self.reasoning = reasoning
        self.total_steps = len([line for line in reasoning.split("\n") if line.strip()])
        self._calculate_derived_risk()

    def update_confidence(self, confidence: float):
        self.confidence = max(0.0, min(1.0, confidence))
        self.uncertainty = round(1.0 - self.confidence, 4)
        self._calculate_derived_risk()

    def update_symbolic_result(self, report: Dict):
        self.symbolic_errors = report.get("errors", [])
        self.symbolic_warnings = report.get("warnings", [])
        self.validation_score = report.get("score", 1.0 if report.get("valid", True) else 0.0)
        self.reasoning_consistency = report.get("reasoning_consistency", self.reasoning_consistency)
        self.answer_consistency = report.get("answer_consistency", self.answer_consistency)
        self.contradiction = not report.get("valid", True)
        self._calculate_derived_risk()

    def update_memory_similarity(self, similarity: float, match_type: Optional[str] = None):
        self.memory_similarity = max(0.0, min(1.0, similarity))
        self.memory_match_type = match_type

    def set_action(self, action: str):
        self.action = action
        if action not in ("CONTINUE", "TERMINATE"):
            self.correction_history.append(action)

    def set_answer(self, answer: str):
        self.final_answer = answer

    def next_step(self):
        self.current_step += 1
        self.elapsed_time = round(time.time() - self.start_time, 4)

    def increment_retry(self):
        self.retry_count += 1

    def was_recently_attempted(self, action: str, window: int = 2) -> bool:
        return action in self.correction_history[-window:]

    def _calculate_derived_risk(self):
        base_risk = self.uncertainty
        inconsistency_penalty = (1.0 - self.reasoning_consistency) * 0.3 + (
            1.0 - self.answer_consistency
        ) * 0.3
        base_risk = max(base_risk, inconsistency_penalty)
        if self.contradiction:
            base_risk = max(base_risk, 0.7) + (0.1 * len(self.symbolic_errors))
        self.risk_score = max(0.0, min(1.0, round(base_risk, 4)))

    def reset(self):
        self.reasoning = ""
        self.final_answer = None
        self.confidence = 1.0
        self.uncertainty = 0.0
        self.contradiction = False
        self.validation_score = 1.0
        self.symbolic_errors = []
        self.symbolic_warnings = []
        self.reasoning_consistency = 1.0
        self.answer_consistency = 1.0
        self.memory_similarity = 0.0
        self.memory_match_type = None
        self.risk_score = 0.0
        self.action = None
        self.correction_history = []
        self.current_step = 0
        self.total_steps = 0
        self.retry_count = 0
        self.start_time = time.time()
        self.elapsed_time = 0.0

    def to_dict(self) -> Dict:
        return {
            "reasoning": self.reasoning,
            "answer": self.final_answer,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "contradiction": self.contradiction,
            "validation_score": self.validation_score,
            "reasoning_consistency": self.reasoning_consistency,
            "answer_consistency": self.answer_consistency,
            "memory_similarity": self.memory_similarity,
            "memory_match_type": self.memory_match_type,
            "risk_score": self.risk_score,
            "action": self.action,
            "correction_history": self.correction_history,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "retry_count": self.retry_count,
            "elapsed_time": self.elapsed_time,
            "symbolic_errors": self.symbolic_errors,
            "symbolic_warnings": self.symbolic_warnings,
        }

    def __repr__(self) -> str:
        return (
            f"CognitiveState(step={self.current_step}, confidence={self.confidence}, "
            f"contradiction={self.contradiction}, risk={self.risk_score}, "
            f"action='{self.action}', retries={self.retry_count})"
        )
