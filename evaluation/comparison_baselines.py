# "comparison_baselines.py"
# from typing import Dict, Any, Optional
# from evaluation.baseline import BaseReasoner
# from models.phi3 import Phi3Mini
# class SelfConsistencyReasoner(BaseReasoner):
#     def __init__(self, model: Optional[Phi3Mini] = None, num_paths: int = 5, temperature: float = 1.2):
#         super().__init__(model=model)
#         self.num_paths = num_paths
#         self.temperature = temperature
#     def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
#         paths = self.model.generate_multi_path(question, num_paths=self.num_paths, temperature=self.temperature)
#         answers = [p["answer"] for p in paths if p.get("answer")]
#         if not answers:
#             return {
#                 "method": "SelfConsistency",
#                 "question": question,
#                 "reasoning": "",
#                 "answer": "",
#                 "num_paths": 0,
#             }
#         vote_counts: Dict[str, int] = {}
#         for a in answers:
#             key = a.strip().lower()
#             vote_counts[key] = vote_counts.get(key, 0) + 1
#         winner = max(vote_counts, key=vote_counts.get)
#         winning_path = next((p for p in paths if p["answer"].strip().lower() == winner), paths[0])
#         return {
#             "method": "SelfConsistency",
#             "question": question,
#             "reasoning": winning_path["reasoning"],
#             "answer": winning_path["answer"],
#             "num_paths": len(paths),
#             "vote_counts": vote_counts,
#         }


# class ReflexionReasoner(BaseReasoner):

#     def __init__(
#         self,
#         model: Optional[Phi3Mini] = None,
#         max_iterations: int = 2,
#         confidence_threshold: float = 0.85,
#     ):
#         super().__init__(model=model)
#         self.max_iterations = max_iterations
#         self.confidence_threshold = confidence_threshold

#     def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
#         result = self.model.reasoning(question)
#         reasoning = result.get("reasoning", "")
#         answer = result.get("answer", "")

#         confidence = self.model.self_evaluate(question, reasoning, answer)
#         iterations = 0

#         while iterations < self.max_iterations and (
#             confidence is None or confidence < self.confidence_threshold
#         ):
#             critique_prompt = f"""Critique your own previous answer to the question below.
# List concrete issues with the reasoning or the final answer, if any exist.
# If you find no issues, respond with exactly: No issues found.

# Question:
# {question}

# Previous reasoning:
# {reasoning}

# Previous answer:
# {answer}

# Critique:"""
#             critique_result = self.model.generate(critique_prompt)
#             critique_text = critique_result.get("text", "")

#             if self._no_issues(critique_text):
#                 break

#             revise_prompt = f"""You previously answered the question below, then critiqued
# your own answer. Use the critique to produce an improved, corrected answer.

# Question:
# {question}

# Previous reasoning:
# {reasoning}

# Self-critique:
# {critique_text}

# Provide the corrected step-by-step reasoning.
# After your reasoning, end with:
# Final Answer: <answer>"""
#             revised_result = self.model.generate(revise_prompt)
#             reasoning = revised_result.get("text", "")
#             answer = self.model.extract_answer(reasoning)

#             confidence = self.model.self_evaluate(question, reasoning, answer)
#             iterations += 1

#         return {
#             "method": "Reflexion",
#             "question": question,
#             "reasoning": reasoning,
#             "answer": answer,
#             "iterations": iterations,
#             "final_confidence": confidence,
#         }

#     @staticmethod
#     def _no_issues(critique_text: str) -> bool:
#         t = critique_text.lower()
#         return any(
#             phrase in t
#             for phrase in ("no issues", "no errors", "no concrete issues", "looks correct", "is correct")
#         )
import re
from typing import Dict, Any, Optional
from evaluation.baseline import BaseReasoner
from models.phi3 import Phi3Mini


def _normalize_vote_key(answer: str) -> str:
    if not answer:
        return ""
    a = answer.strip().lower()
    a = re.sub(r"[,$%]", "", a)
    a = a.rstrip(".")
    try:
        f = float(a)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return a


class SelfConsistencyReasoner(BaseReasoner):
    def __init__(
        self,
        model: Optional[Phi3Mini] = None,
        num_paths: int = 5,
        temperature: float = 0.8,
    ):
        super().__init__(model=model)
        self.num_paths = num_paths
        self.temperature = temperature

    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
        paths = self.model.generate_multi_path(
            question, num_paths=self.num_paths, temperature=self.temperature
        )
        answers = [p["answer"] for p in paths if p.get("answer")]
        if not answers:
            return {
                "method": "SelfConsistency",
                "question": question,
                "reasoning": "",
                "answer": "",
                "num_paths": len(paths),
                "num_answered_paths": 0,
            }

        vote_counts: Dict[str, int] = {}
        for a in answers:
            key = _normalize_vote_key(a)
            vote_counts[key] = vote_counts.get(key, 0) + 1

        winner_key = max(vote_counts, key=vote_counts.get)
        winning_path = next(
            (p for p in paths if _normalize_vote_key(p.get("answer", "")) == winner_key),
            paths[0],
        )

        return {
            "method": "SelfConsistency",
            "question": question,
            "reasoning": winning_path["reasoning"],
            "answer": winning_path["answer"],
            "num_paths": len(paths),
            "num_answered_paths": len(answers),
            "vote_counts": vote_counts,
        }


class ReflexionReasoner(BaseReasoner):

    _NO_ISSUES_PHRASE = "no issues found"

    def __init__(
        self,
        model: Optional[Phi3Mini] = None,
        max_iterations: int = 2,
        confidence_threshold: float = 0.85,
    ):
        super().__init__(model=model)
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
        result = self.model.reasoning(question)
        reasoning = result.get("reasoning", "")
        answer = result.get("answer", "")

        confidence = self.model.self_evaluate(question, reasoning, answer)
        iterations = 0

        while iterations < self.max_iterations and (
            confidence is None or confidence < self.confidence_threshold
        ):
            critique_prompt = f"""Critique your own previous answer to the question below.
List concrete issues with the reasoning or the final answer, if any exist.
If, and only if, you find no issues at all, respond with EXACTLY this and
nothing else: No issues found.

Question:
{question}

Previous reasoning:
{reasoning}

Previous answer:
{answer}

Critique:"""
            critique_result = self.model.generate(critique_prompt)
            critique_text = critique_result.get("text", "")

            if self._no_issues(critique_text):
                break

            revise_prompt = f"""You previously answered the question below, then critiqued
your own answer. Use the critique to produce an improved, corrected answer.

Question:
{question}

Previous reasoning:
{reasoning}

Self-critique:
{critique_text}

Provide the corrected step-by-step reasoning.
After your reasoning, end with:
Final Answer: <answer>"""
            revised_result = self.model.generate(revise_prompt)
            revised_text = revised_result.get("text", "")

            if revised_text.startswith("Error:"):
                break

            reasoning = revised_text
            answer = self.model.extract_answer(reasoning)

            confidence = self.model.self_evaluate(question, reasoning, answer)
            iterations += 1

        return {
            "method": "Reflexion",
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
            "iterations": iterations,
            "final_confidence": confidence,
        }

    @classmethod
    def _no_issues(cls, critique_text: str) -> bool:
        if not critique_text:
            return True
        t = critique_text.strip().lower().rstrip(".")
        return t == cls._NO_ISSUES_PHRASE