from typing import Dict, Optional
from models.phi3 import Phi3Mini
from mnsr.symbolic_validator import SymbolicValidator
from mnsr.cognitive_state import CognitiveState
from mnsr.controller import MetaCognitiveController
from mnsr.router import StrategyRouter
from mnsr.memory import ReflectionMemory
from mnsr.confidence import ConfidenceEstimator


class MNSRPipeline:

    def __init__(
        self,
        max_reflection_depth: int = 4,
        use_self_eval: bool = True,
        model: Optional[Phi3Mini] = None,
        cache: Optional[Dict] = None,
    ):
        self.model = model if model is not None else Phi3Mini(cache=cache)
        self.validator = SymbolicValidator()
        self.controller = MetaCognitiveController()
        self.router = StrategyRouter(self.model)
        self.memory = ReflectionMemory()
        self.confidence_estimator = ConfidenceEstimator()
        self.max_reflection_depth = max_reflection_depth
        self.use_self_eval = use_self_eval

    @staticmethod
    def infer_dataset_type(question: str, hint: Optional[str] = None) -> str:
        if hint:
            hint = hint.lower()
            if any(k in hint for k in ("gsm8k", "math", "arith")):
                return "numeric"
            if any(k in hint for k in ("strategyqa", "boolq", "yesno", "yes_no")):
                return "boolean"
            if any(k in hint for k in ("mmlu", "arc", "multiple_choice", "mc")):
                return "multiple_choice"
            if any(k in hint for k in ("truthfulqa", "halueval")):
                return "freeform"
        q = question.lower()
        if q.strip().startswith(("is ", "are ", "was ", "were ", "does ", "did ", "can ", "will ")):
            return "boolean"
        return "freeform"

    def solve(self, question: str, dataset_hint: Optional[str] = None) -> Dict:
        dataset_type = self.infer_dataset_type(question, dataset_hint)
        state = CognitiveState(max_reflection_depth=self.max_reflection_depth)

        result = self.model.reasoning(question)
        state.update_reasoning(result["reasoning"])
        state.set_answer(result["answer"])

        report = self.validator.validate(state.reasoning, dataset_type=dataset_type)
        state.update_symbolic_result(report)

        self_eval_score = None
        if self.use_self_eval:
            self_eval_score = self.model.self_evaluate(question, state.reasoning, state.final_answer)

        initial_confidence = self.confidence_estimator.estimate(
            validation_score=report["score"],
            reasoning_consistency=report["reasoning_consistency"],
            answer_consistency=report["answer_consistency"],
            num_errors=report["num_errors"],
            memory_similarity=0.0,
            retry_count=0,
            self_eval_score=self_eval_score,
        )
        state.update_confidence(initial_confidence)

        best = {
            "reasoning": state.reasoning,
            "answer": state.final_answer,
            "confidence": state.confidence,
            "validation": report,
        }

        trace_log = []
        action = "CONTINUE"

        while state.current_step < self.max_reflection_depth:
            state.next_step()

            retrieved = self.memory.retrieve(question)
            success_hint, failure_hint = "", ""
            if retrieved["success"] is not None:
                state.update_memory_similarity(retrieved["success"]["similarity"], "success")
                success_hint = retrieved["success"]["episode"]["corrected_reasoning"]
            if retrieved["failure"] is not None:
                failure_ep = retrieved["failure"]["episode"]
                failure_hint = f"Question: {failure_ep['question']}\nMistake: {failure_ep['reasoning']}"
                if state.memory_match_type != "success" or (
                    retrieved["failure"]["similarity"] > (retrieved["success"] or {}).get("similarity", 0.0)
                ):
                    state.update_memory_similarity(retrieved["failure"]["similarity"], "failure")
            if retrieved["success"] is None and retrieved["failure"] is None:
                state.update_memory_similarity(0.0, None)

            action = self.controller.evaluate(state)

            trace_log.append({
                "step": state.current_step,
                "action": action,
                "confidence": state.confidence,
                "validation_score": state.validation_score,
                "risk_score": state.risk_score,
            })

            if action in ("CONTINUE", "TERMINATE"):
                break

            state.increment_retry()
            execution = self.router.execute(
                action=action,
                question=question,
                state=state,
                memory_hint=success_hint,
                failure_hint=failure_hint,
            )

            state.update_reasoning(execution["reasoning"])
            state.set_answer(execution["answer"])

            report = self.validator.validate(state.reasoning, dataset_type=dataset_type)
            state.update_symbolic_result(report)

            if self.use_self_eval:
                self_eval_score = self.model.self_evaluate(question, state.reasoning, state.final_answer)

            new_confidence = self.confidence_estimator.estimate(
                validation_score=report["score"],
                reasoning_consistency=report["reasoning_consistency"],
                answer_consistency=report["answer_consistency"],
                num_errors=report["num_errors"],
                memory_similarity=state.memory_similarity,
                retry_count=state.retry_count,
                self_eval_score=self_eval_score,
            )
            state.update_confidence(new_confidence)

            if state.confidence >= best["confidence"]:
                best = {
                    "reasoning": state.reasoning,
                    "answer": state.final_answer,
                    "confidence": state.confidence,
                    "validation": report,
                }

            if report["score"] >= 0.85 and state.confidence >= 0.75:
                break

        final_report = self.validator.validate(best["reasoning"], dataset_type=dataset_type)

        self.memory.add(
            question=question,
            reasoning=state.reasoning,
            answer=state.final_answer,
            errors=final_report["errors"],
            corrected_reasoning=best["reasoning"],
            success=final_report["valid"],
            controller_actions=list(state.correction_history),
            retry_count=state.retry_count,
            confidence=best["confidence"],
            validation_report=final_report,
        )

        return {
            "question": question,
            "dataset_type": dataset_type,
            "reasoning": best["reasoning"],
            "answer": best["answer"],
            "confidence": best["confidence"],
            "final_action": action,
            "validation": final_report,
            "trace_log": trace_log,
            "state": state.to_dict(),
        }
