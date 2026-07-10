from typing import Dict
from mnsr.cognitive_state import CognitiveState


class StrategyRouter:

    def __init__(self, model):
        self.model = model

    def execute(
        self,
        action: str,
        question: str,
        state: CognitiveState,
        memory_hint: str = "",
        failure_hint: str = "",
    ) -> Dict:
        dispatch = {
            "CONTINUE": self._passthrough,
            "TERMINATE": self._passthrough,
            "ANSWER_REPAIR": self._answer_repair,
            "REASONING_REPAIR": self._reasoning_repair,
            "BACKTRACK": self._backtrack,
            "REPLAN": self._replan,
            "DECOMPOSE": self._decompose,
            "MEMORY_REFLECTION": self._memory_reflection,
            "SELF_CRITIQUE": self._self_critique,
            "SELF_VERIFY": self._self_verify,
            "MULTI_PATH_REASONING": self._multi_path_reasoning,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"Unknown system execution action: {action}")

        if action in ("CONTINUE", "TERMINATE"):
            return handler(state)
        if action == "MEMORY_REFLECTION":
            return handler(question, state, memory_hint, failure_hint)
        return handler(question, state)

    def _passthrough(self, state: CognitiveState) -> Dict:
        return {"reasoning": state.reasoning, "answer": state.final_answer}

    def _answer_repair(self, question: str, state: CognitiveState) -> Dict:
        prompt = f"""The reasoning below correctly solves the question, but the
stated "Final Answer" line does not match the value actually derived in the
reasoning. Do not change the reasoning. Simply restate the final answer that
is consistent with the derivation.

Question:
{question}

Reasoning:
{state.reasoning}

Respond with exactly:
Final Answer: <answer>"""
        return self._run(prompt)

    def _reasoning_repair(self, question: str, state: CognitiveState) -> Dict:
        errors_desc = "\n".join(
            f"- {e.get('type')}: {e.get('expression', e.get('detail', ''))}"
            for e in state.symbolic_errors
        ) or "No specific errors listed."
        prompt = f"""You are reviewing a mathematical/logical problem solver. The
following reasoning contains specific flagged errors. Correct only the
flawed steps; keep the rest of the reasoning intact.

Question:
{question}

Previous reasoning trail:
{state.reasoning}

Flagged errors:
{errors_desc}

Provide the corrected step-by-step reasoning.
After your reasoning, end with:
Final Answer: <answer>"""
        return self._run(prompt)

    def _backtrack(self, question: str, state: CognitiveState) -> Dict:
        prompt = f"""The previous reasoning trail contained fatal symbolic
contradictions or arithmetic errors. Discard the failed thinking strategy
entirely and solve the question completely fresh from the beginning.

Question:
{question}

Provide your completely new step-by-step reasoning.
After your reasoning, end with:
Final Answer: <answer>"""
        return self._run(prompt)

    def _replan(self, question: str, state: CognitiveState) -> Dict:
        prompt = f"""A prior fresh attempt at this question still failed
validation. Before solving, first write a short numbered PLAN describing
the distinct steps needed. Then execute the plan step by step.

Question:
{question}

Reasoning that still failed:
{state.reasoning}

Format:
Plan:
1. ...
2. ...

Execution:
<step-by-step reasoning following the plan>

Final Answer: <answer>"""
        return self._run(prompt)

    def _decompose(self, question: str, state: CognitiveState) -> Dict:
        prompt = f"""The question below may require multiple sub-steps that were
skipped. Break it into 2-4 smaller sub-questions, answer each sub-question in
order, then combine the sub-answers into the final answer.

Question:
{question}

Format:
Sub-question 1: ...
Sub-answer 1: ...
Sub-question 2: ...
Sub-answer 2: ...
...

Final Answer: <answer>"""
        return self._run(prompt)

    def _memory_reflection(
        self, question: str, state: CognitiveState, memory_hint: str, failure_hint: str
    ) -> Dict:
        hint_block = memory_hint or "No successful reference episode available."
        failure_block = failure_hint or "No failure precedent available."
        prompt = f"""You have access to reference memory of past similar
problems. One shows a past MISTAKE to avoid; the other shows a past
CORRECT / successful resolution pattern. Use both to avoid repeating the
mistake on the current target problem.

Past mistake to avoid:
{failure_block}

Past successful resolution pattern:
{hint_block}

Target Question:
{question}

Provide your step-by-step reasoning using the reference knowledge.
After your reasoning, end with:
Final Answer: <answer>"""
        return self._run(prompt)

    def _self_critique(self, question: str, state: CognitiveState) -> Dict:
        critique_prompt = f"""Critique the following reasoning for internal
contradictions, unjustified leaps, or unclear logic. List concrete issues
only; do not solve the question yet.

Question:
{question}

Reasoning:
{state.reasoning}

Critique:"""
        critique_result = self.model.generate(critique_prompt)
        critique_text = critique_result.get("text", "")

        revise_prompt = f"""Using the critique below, rewrite the reasoning to
address every issue raised.

Question:
{question}

Original reasoning:
{state.reasoning}

Critique:
{critique_text}

Provide the corrected step-by-step reasoning.
After your reasoning, end with:
Final Answer: <answer>"""
        return self._run(revise_prompt)

    def _self_verify(self, question: str, state: CognitiveState) -> Dict:
        prompt = f"""Independently re-derive the answer to the question below
without looking at the previous reasoning. Then compare your new derivation
to the previous answer. If they agree, restate that answer. If they
disagree, explain which is correct and use it as the final answer.

Question:
{question}

Previous answer:
{state.final_answer}

Provide your independent step-by-step reasoning, the comparison, and then:
Final Answer: <answer>"""
        return self._run(prompt)

    def _multi_path_reasoning(self, question: str, state: CognitiveState) -> Dict:
        paths = self.model.generate_multi_path(question, num_paths=3, temperature=0.7)
        answers = [p["answer"] for p in paths if p.get("answer")]

        if not answers:
            return {"reasoning": state.reasoning, "answer": state.final_answer}

        vote_counts: Dict[str, int] = {}
        for a in answers:
            key = a.strip().lower()
            vote_counts[key] = vote_counts.get(key, 0) + 1

        winner = max(vote_counts, key=vote_counts.get)
        winning_path = next((p for p in paths if p["answer"].strip().lower() == winner), paths[0])

        combined_reasoning = "\n\n---\n\n".join(
            f"[Path {i+1}] {p['reasoning']}" for i, p in enumerate(paths)
        )
        combined_reasoning += f"\n\n[Majority Vote Result] {winning_path['answer']}"

        return {"reasoning": combined_reasoning, "answer": winning_path["answer"]}

    def _run(self, prompt: str) -> Dict:
        result = self.model.generate(prompt)
        text = result.get("text", "")
        return {"reasoning": text, "answer": self.model.extract_answer(text)}
