import re
import threading
from typing import Dict, Any, List, Optional
from ollama import Client


class Phi3Mini:

    def __init__(
        self,
        model: str = "phi3:mini",
        host: str = "http://localhost:11434",
        cache: Optional[Dict] = None,
    ):
        self.client = Client(host=host)
        self.model = model
        self.cache = cache if cache is not None else {}
        self._lock = threading.Lock()

    def extract_answer(self, text: str) -> str:
        if not text:
            return ""
        m = re.search(r"final\s*answer\s*:?\s*(.*)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            return numbers[-1]
        return text.strip()

    def _chat(self, prompt: str, temperature: float = 0.0) -> str:
        cacheable = temperature == 0.0
        key = (self.model, prompt)
        if cacheable:
            with self._lock:
                if key in self.cache:
                    return self.cache[key]
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        text = response["message"]["content"]
        if cacheable:
            with self._lock:
                self.cache[key] = text
        return text

    def reasoning(self, question: str, temperature: float = 0.0) -> Dict[str, Any]:
        prompt = f"""
Solve the following question carefully.

Question:
{question}

Explain your reasoning step by step.

Finish with exactly:

Final Answer: <answer>
"""
        try:
            text = self._chat(prompt, temperature=temperature)
            return {"reasoning": text, "answer": self.extract_answer(text)}
        except Exception as e:
            return {"reasoning": f"Error: {e}", "answer": ""}

    def generate(self, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        try:
            text = self._chat(prompt, temperature=temperature)
            return {"text": text, "answer": self.extract_answer(text)}
        except Exception as e:
            return {"text": f"Error: {e}", "answer": ""}

    def self_evaluate(self, question: str, reasoning: str, answer: str) -> Optional[float]:
        prompt = f"""You previously answered a question. Rate your confidence
that the final answer below is correct, as a single number between 0 and 1
(0 = certainly wrong, 1 = certainly correct). Respond with ONLY the number.

Question:
{question}

Reasoning:
{reasoning}

Final Answer:
{answer}

Confidence (0-1):"""
        try:
            text = self._chat(prompt, temperature=0.0)
            m = re.search(r"(\d*\.?\d+)", text)
            if not m:
                return None
            score = float(m.group(1))
            return max(0.0, min(1.0, score))
        except Exception:
            return None

    def generate_multi_path(
        self, question: str, num_paths: int = 3, temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        paths = []
        for _ in range(max(1, num_paths)):
            paths.append(self.reasoning(question, temperature=temperature))
        return paths

    def process_single_item(self, question: str):
        return self.reasoning(question)

    def generate_all_parallel(self, questions: List[str]):
        return [self.reasoning(q) for q in questions]
