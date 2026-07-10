import re
from typing import Dict, List, Optional


class SymbolicValidator:

    ASSUMPTION_CUES = [
        r"\blet'?s assume\b",
        r"\bassuming that\b",
        r"\bi (?:will )?assume\b",
        r"\bsuppose that\b",
        r"\bmust be\b(?!.*because)",
    ]

    CONTRADICTION_CUES = [
        r"\bhowever, (?:actually|in fact)\b",
        r"\bwait,? (?:that'?s|this is) (?:wrong|incorrect)\b",
        r"\bactually,? (?:no|that'?s wrong)\b",
        r"\bon second thought\b",
        r"\bi made a mistake\b",
    ]

    UNSUPPORTED_CLAIM_CUES = [
        r"\bit is (?:well[- ]known|a fact) that\b",
        r"\bstudies show\b",
        r"\baccording to (?:research|experts)\b(?!.*\[)",
    ]

    def __init__(self, min_reasoning_lines: int = 2):
        self.min_reasoning_lines = min_reasoning_lines
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []

    def validate(self, reasoning: str, dataset_type: Optional[str] = None) -> Dict:
        self.errors = []
        self.warnings = []

        self.arithmetic_check(reasoning)
        self.contradiction_check(reasoning)
        self.unsupported_assumption_check(reasoning)
        self.hallucination_heuristic_check(reasoning)
        self.missing_steps_check(reasoning)
        self.answer_format_check(reasoning)
        final_consistency = self.final_answer_consistency_check(reasoning)
        if dataset_type:
            self.dataset_specific_check(reasoning, dataset_type)

        reasoning_consistency = self._score_reasoning_consistency(reasoning)

        return self.final_report(reasoning_consistency, final_consistency)

    def arithmetic_check(self, reasoning: str):
        pattern = r"(-?\d+\.?\d*)\s*([\+\-\*/])\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)"
        for m in re.finditer(pattern, reasoning):
            a, op, b, claimed = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
            if op == "+":
                actual = a + b
            elif op == "-":
                actual = a - b
            elif op == "*":
                actual = a * b
            elif op == "/":
                if b == 0:
                    continue
                actual = a / b
            else:
                continue
            if abs(actual - claimed) > 1e-6:
                self.errors.append({
                    "type": "Arithmetic Error",
                    "expression": m.group(0),
                    "expected": int(actual) if actual.is_integer() else actual,
                    "found": int(claimed) if claimed.is_integer() else claimed,
                    "position": m.start(),
                })

    def contradiction_check(self, reasoning: str):
        for cue in self.CONTRADICTION_CUES:
            for m in re.finditer(cue, reasoning, re.IGNORECASE):
                self.errors.append({
                    "type": "Contradiction",
                    "cue": m.group(0),
                    "position": m.start(),
                })

    def unsupported_assumption_check(self, reasoning: str):
        for cue in self.ASSUMPTION_CUES:
            for m in re.finditer(cue, reasoning, re.IGNORECASE):
                self.warnings.append({
                    "type": "Unsupported Assumption",
                    "cue": m.group(0),
                    "position": m.start(),
                })

    def hallucination_heuristic_check(self, reasoning: str):
        for cue in self.UNSUPPORTED_CLAIM_CUES:
            for m in re.finditer(cue, reasoning, re.IGNORECASE):
                self.warnings.append({
                    "type": "Potential Hallucinated Claim",
                    "cue": m.group(0),
                    "position": m.start(),
                })

    def missing_steps_check(self, reasoning: str):
        lines = [l for l in reasoning.split("\n") if l.strip()]
        if len(lines) < self.min_reasoning_lines:
            self.warnings.append({
                "type": "Missing Reasoning Steps",
                "detail": f"Only {len(lines)} non-empty line(s) of reasoning.",
            })

    def answer_format_check(self, reasoning: str):
        if not re.search(r"final\s*answer\s*:?", reasoning, re.IGNORECASE):
            self.errors.append({
                "type": "Answer Format Error",
                "detail": "No 'Final Answer:' marker found.",
            })

    def final_answer_consistency_check(self, reasoning: str) -> float:
        m = re.search(r"final\s*answer\s*:?\s*(.*)", reasoning, re.IGNORECASE)
        if not m:
            return 0.5

        final_answer = m.group(1).strip()
        body = reasoning[: m.start()]

        final_numbers = re.findall(r"-?\d+\.?\d*", final_answer)
        body_numbers = re.findall(r"-?\d+\.?\d*", body)

        if not final_numbers or not body_numbers:
            return 1.0

        if final_numbers[-1] != body_numbers[-1]:
            self.warnings.append({
                "type": "Answer/Reasoning Mismatch",
                "detail": f"Final answer '{final_numbers[-1]}' differs from last "
                          f"derived value '{body_numbers[-1]}' in the reasoning body.",
            })
            return 0.3

        return 1.0

    def dataset_specific_check(self, reasoning: str, dataset_type: str):
        m = re.search(r"final\s*answer\s*:?\s*(.*)", reasoning, re.IGNORECASE)
        answer = m.group(1).strip().lower() if m else ""

        if dataset_type == "numeric":
            if not re.search(r"-?\d+\.?\d*", answer):
                self.errors.append({
                    "type": "Dataset Constraint Violation",
                    "detail": "Expected a numeric final answer for a numeric dataset.",
                })
        elif dataset_type == "boolean":
            if not re.search(r"\b(yes|no|true|false)\b", answer):
                self.errors.append({
                    "type": "Dataset Constraint Violation",
                    "detail": "Expected a yes/no final answer for a boolean dataset.",
                })
        elif dataset_type == "multiple_choice":
            if not re.search(r"\b[a-eA-E]\b", answer) and len(answer) > 40:
                self.warnings.append({
                    "type": "Dataset Constraint Violation",
                    "detail": "Expected a short option letter/label for a multiple-choice dataset.",
                })

    def _score_reasoning_consistency(self, reasoning: str) -> float:
        contradiction_hits = sum(
            1 for cue in self.CONTRADICTION_CUES if re.search(cue, reasoning, re.IGNORECASE)
        )
        score = 1.0 - 0.25 * contradiction_hits - 0.15 * len(self.errors)
        return max(0.0, min(1.0, round(score, 4)))

    def final_report(self, reasoning_consistency: float = 1.0, answer_consistency: float = 1.0) -> Dict:
        num_errors = len(self.errors)
        num_warnings = len(self.warnings)
        score = 1.0 - (0.25 * num_errors) - (0.05 * num_warnings)
        score = max(0.0, min(1.0, round(score, 4)))
        return {
            "valid": num_errors == 0,
            "score": score,
            "num_errors": num_errors,
            "errors": self.errors,
            "warnings": self.warnings,
            "reasoning_consistency": reasoning_consistency,
            "answer_consistency": answer_consistency,
        }
