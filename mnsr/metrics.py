"metrics.py"
from dataclasses import dataclass, asdict, field
from typing import Dict, List
ACTION_TYPES = [
    "REASONING_REPAIR",
    "ANSWER_REPAIR",
    "MEMORY_REFLECTION",
    "SELF_CRITIQUE",
    "SELF_VERIFY",
    "DECOMPOSE",
    "REPLAN",
    "MULTI_PATH_REASONING",
    "BACKTRACK",
]
@dataclass
class ExperimentMetrics:
    total_questions: int = 0
    baseline_correct: int = 0
    mnsr_correct: int = 0
    arithmetic_errors_detected: int = 0
    arithmetic_errors_fixed: int = 0
    action_counts: Dict[str, int] = field(default_factory=lambda: {a: 0 for a in ACTION_TYPES})
    total_steps: int = 0
    total_reflection_depth: int = 0
    total_confidence: float = 0.0
    validation_successes: int = 0
    memory_retrievals: int = 0
    correction_attempts: int = 0
    correction_successes: int = 0
    def update(
        self,
        baseline_correct: bool,
        mnsr_correct: bool,
        validation_report: Dict,
        final_action: str,
        steps: int,
        correction_history: List[str] = None,
        confidence: float = 0.0,
        had_any_correction: bool = False,
    ):
        correction_history = correction_history or []

        self.total_questions += 1
        self.total_steps += steps
        self.total_reflection_depth += len(correction_history)
        self.total_confidence += confidence

        if baseline_correct:
            self.baseline_correct += 1
        if mnsr_correct:
            self.mnsr_correct += 1

        num_errors = validation_report.get("num_errors", 0)
        if num_errors > 0:
            self.arithmetic_errors_detected += num_errors
        if num_errors > 0 and mnsr_correct and not baseline_correct:
            self.arithmetic_errors_fixed += 1

        if validation_report.get("valid", False):
            self.validation_successes += 1

        for action in correction_history:
            if action in self.action_counts:
                self.action_counts[action] += 1
            if action == "MEMORY_REFLECTION":
                self.memory_retrievals += 1

        if had_any_correction or correction_history:
            self.correction_attempts += 1
            if mnsr_correct and not baseline_correct:
                self.correction_successes += 1

    def summary(self) -> Dict:
        if self.total_questions == 0:
            return {}
        return {
            "baseline_accuracy": round(self.baseline_correct / self.total_questions, 4),
            "mnsr_accuracy": round(self.mnsr_correct / self.total_questions, 4),
            "error_detection_rate": round(self.arithmetic_errors_detected / self.total_questions, 4),
            "error_fix_rate": round(
                self.arithmetic_errors_fixed / max(1, self.arithmetic_errors_detected), 4
            ),
            "avg_steps": round(self.total_steps / self.total_questions, 2),
            "avg_reflection_depth": round(self.total_reflection_depth / self.total_questions, 2),
            "avg_confidence": round(self.total_confidence / self.total_questions, 4),
            "validation_success_rate": round(self.validation_successes / self.total_questions, 4),
            "memory_retrieval_rate": round(self.memory_retrievals / self.total_questions, 4),
            "correction_success_rate": round(
                self.correction_successes / max(1, self.correction_attempts), 4
            ),
            "action_counts": dict(self.action_counts),
        }

    def export(self) -> Dict:
        return asdict(self)
