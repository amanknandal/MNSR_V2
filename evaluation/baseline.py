import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.phi3 import Phi3Mini
from mnsr.symbolic_validator import SymbolicValidator
from mnsr.pipeline import MNSRPipeline


class BaseReasoner(ABC):

    def __init__(self, model: Optional[Phi3Mini] = None):
        self.model = model if model is not None else Phi3Mini()

    @abstractmethod
    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
        pass


class BaselineCoT(BaseReasoner):

    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
        result = self.model.reasoning(question)
        return {
            "method": "ChainOfThought",
            "question": question,
            "reasoning": result.get("reasoning", ""),
            "answer": result.get("answer", ""),
        }


class ValidatorBaseline(BaseReasoner):

    def __init__(self, model: Optional[Phi3Mini] = None):
        super().__init__(model=model)
        self.validator = SymbolicValidator()

    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
        dataset_type = MNSRPipeline.infer_dataset_type(question, dataset_hint)
        result = self.model.reasoning(question)
        report = self.validator.validate(result.get("reasoning", ""), dataset_type=dataset_type)
        return {
            "method": "ValidatorOnly",
            "question": question,
            "reasoning": result.get("reasoning", ""),
            "answer": result.get("answer", ""),
            "validation": report,
        }


class RevisionBaseline(BaseReasoner):

    def __init__(self, model: Optional[Phi3Mini] = None):
        super().__init__(model=model)
        self.validator = SymbolicValidator()

    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict[str, Any]:
        dataset_type = MNSRPipeline.infer_dataset_type(question, dataset_hint)
        result = self.model.reasoning(question)

        reasoning = result.get("reasoning", "")
        answer = result.get("answer", "")

        report = self.validator.validate(reasoning, dataset_type=dataset_type)

        if not report.get("valid", True):
            prompt = f"""The following reasoning contains arithmetic or logical errors.
Question:
{question}
Previous reasoning:
{reasoning}
Please correct the reasoning.
Finish with:
Final Answer: <answer>"""
            revised = self.model.generate(prompt)
            if isinstance(revised, dict):
                reasoning = revised.get("text", revised.get("generation", revised.get("reasoning", "")))
            else:
                reasoning = str(revised)
            if hasattr(self.model, "extract_answer"):
                answer = self.model.extract_answer(reasoning)
            elif isinstance(revised, dict) and "answer" in revised:
                answer = revised["answer"]

        return {
            "method": "ValidatorRevision",
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
            "validation": report,
        }


def get_all_baselines(model: Optional[Phi3Mini] = None) -> List[BaseReasoner]:
    return [BaselineCoT(model=model), ValidatorBaseline(model=model), RevisionBaseline(model=model)]
