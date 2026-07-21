# import re
# import threading
# from typing import Dict, Any, List, Optional, Tuple
# from concurrent.futures import ThreadPoolExecutor
# from ollama import Client


# class Phi3Mini:
#     def __init__(
#         self,
#         model: str = "phi3:mini",
#         host: str = "http://localhost:11434",
#         cache: Optional[Dict] = None,
#     ):
#         self.client = Client(host=host)
#         self.model = model
#         self.cache = cache if cache is not None else {}
#         self._lock = threading.Lock()
#         # Per-key locks to deduplicate concurrent identical API calls
#         self._key_locks: Dict[Tuple[str, str], threading.Lock] = {}
#         # Pre-compiled regex patterns (avoid recompiling on every call)
#         self._answer_re = re.compile(r"final\s*answer\s*:?\s*(.*)", re.IGNORECASE)
#         self._number_re = re.compile(r"-?\d+\.?\d*")
#         self._confidence_re = re.compile(r"(\d*\.?\d+)")

#     def extract_answer(self, text: str) -> str:
#         if not text:
#             return ""
#         m = self._answer_re.search(text)
#         if m:
#             return m.group(1).strip()
#         numbers = self._number_re.findall(text)
#         if numbers:
#             return numbers[-1]
#         return text.strip()

#     def _chat(self, prompt: str, temperature: float = 0.0) -> str:
#         """
#         Cached chat with per-key deduplication.
#         If two threads request the same prompt simultaneously, only one
#         API call is made; the second thread waits and receives the cached result.
#         """
#         cacheable = temperature == 0.0
#         key = (self.model, prompt)

#         if cacheable:
#             with self._lock:
#                 if key in self.cache:
#                     return self.cache[key]
#                 key_lock = self._key_locks.get(key)
#                 if key_lock is None:
#                     key_lock = threading.Lock()
#                     self._key_locks[key] = key_lock
#             with key_lock:
#                 with self._lock:
#                     if key in self.cache:
#                         return self.cache[key]

#                 response = self.client.chat(
#                     model=self.model,
#                     messages=[{"role": "user", "content": prompt}],
#                     options={"temperature": temperature},
#                 )
#                 text = response["message"]["content"]

#                 with self._lock:
#                     self.cache[key] = text
#                 return text
#         response = self.client.chat(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             options={"temperature": temperature},
#         )
#         return response["message"]["content"]
#     def reasoning(self, question: str, temperature: float = 0.0) -> Dict[str, Any]:
#         prompt = f"""
# Solve the following question carefully.

# Question:
# {question}

# Explain your reasoning step by step.

# Finish with exactly:

# Final Answer: <answer>
# """
#         try:
#             text = self._chat(prompt, temperature=temperature)
#             return {"reasoning": text, "answer": self.extract_answer(text)}
#         except Exception as e:
#             return {"reasoning": f"Error: {e}", "answer": ""}

#     def generate(self, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
#         try:
#             text = self._chat(prompt, temperature=temperature)
#             return {"text": text, "answer": self.extract_answer(text)}
#         except Exception as e:
#             return {"text": f"Error: {e}", "answer": ""}

#     def self_evaluate(self, question: str, reasoning: str, answer: str) -> Optional[float]:
#         prompt = f"""You previously answered a question. Rate your confidence
# that the final answer below is correct, as a single number between 0 and 1
# (0 = certainly wrong, 1 = certainly correct). Respond with ONLY the number.

# Question:
# {question}

# Reasoning:
# {reasoning}

# Final Answer:
# {answer}

# Confidence (0-1):"""
#         try:
#             text = self._chat(prompt, temperature=0.0)
#             m = self._confidence_re.search(text)
#             if not m:
#                 return None
#             score = float(m.group(1))
#             return max(0.0, min(1.0, score))
#         except Exception:
#             return None
#     def generate_multi_path(
#         self, question: str, num_paths: int = 3, temperature: float = 0.7
#     ) -> List[Dict[str, Any]]:
#         """
#         Generate multiple reasoning paths SEQUENTIALLY.
#         Running these in parallel caused 30+ concurrent Ollama requests,
#         which overwhelmed the CPU and RAM. Sequential is safer and often
#         faster for small models on local hardware.
#         """
#         paths = []
#         for _ in range(max(1, num_paths)):
#             paths.append(self.reasoning(question, temperature=temperature))
#         return paths
#     def process_single_item(self, question: str):
#         return self.reasoning(question)
#     def generate_all_parallel(self, questions: List[str]):
#         return [self.reasoning(q) for q in questions]



# ollama pull phi4-mini-reasoning


"""
Phi3Mini: local Ollama-backed reasoning model wrapper for MNSR.
Class name and public method signatures are preserved for backward
compatibility with the rest of the MNSR pipeline. The default model has
been upgraded to `phi4-mini-reasoning`, which is better suited to
multi-step logical/mathematical reasoning than the original Phi-3 Mini,
while remaining small enough (3.8B) to run CPU-only on an 8 GB machine.
"""

import re
import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from ollama import Client


class Phi3Mini:
    """
    Thread-safe wrapper around a local Ollama chat model, with caching,
    retry logic, and robust answer/confidence parsing tuned for
    reasoning-heavy evaluation workloads (GSM8K, StrategyQA, etc.).
    """

    def __init__(
        self,
        model: str = "phi4-mini-reasoning",
        host: str = "http://localhost:11434",
        cache: Optional[Dict[Tuple[str, str], str]] = None,
        timeout: Optional[float] = 120.0,
        max_cache_size: int = 5000,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        """
        Args:
            model: Ollama model tag to use. Defaults to a reasoning-tuned
                small model well-suited to CPU-only, low-RAM machines.
            host: Ollama server URL.
            cache: Optional externally-supplied cache dict (e.g. to share
                across reasoner instances or persist across runs). If not
                given, an in-memory dict is created.
            timeout: Per-request timeout (seconds) passed to the Ollama
                client. None disables the timeout.
            max_cache_size: Maximum number of (model, prompt) -> response
                entries kept in the cache before the oldest is evicted
                (FIFO). Prevents unbounded memory growth on long eval runs.
            max_retries: Number of attempts for a single Ollama call before
                giving up and raising the underlying exception.
            retry_base_delay: Base delay (seconds) for exponential backoff
                between retries (delay = retry_base_delay * 2 ** attempt).
        """
        client_kwargs: Dict[str, Any] = {"host": host}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self.client = Client(**client_kwargs)

        self.model = model
        self.cache: Dict[Tuple[str, str], str] = cache if cache is not None else {}
        self.max_cache_size = max_cache_size
        self.max_retries = max(1, max_retries)
        self.retry_base_delay = retry_base_delay

        # FIFO eviction order for the cache, kept in sync with self.cache.
        self._cache_order: deque = deque()
        # Guards self.cache, self._cache_order, and self._key_locks.
        self._lock = threading.Lock()
        # Per-(model, prompt) locks so concurrent identical requests
        # de-duplicate into a single Ollama call instead of firing N times.
        self._key_locks: Dict[Tuple[str, str], threading.Lock] = {}

        # Precompiled regexes, reused across all calls.
        self._think_tag_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        self._markdown_strip_re = re.compile(r"[*_`#]")
        self._boxed_re = re.compile(r"\\boxed\{([^{}]*)\}")
        self._final_answer_re = re.compile(r"final\s*answer\s*:?\s*(.*)", re.IGNORECASE)
        self._answer_is_re = re.compile(
            r"(?:answer\s*(?:is|=)|=)\s*(-?\d[\d,]*\.?\d*)", re.IGNORECASE
        )
        self._number_re = re.compile(r"-?\d[\d,]*\.?\d*")
        self._confidence_re = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b")
        self._percent_re = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------
    def extract_answer(self, text: str) -> str:
        """
        Extract a final answer from raw model output.

        Priority order:
          1. `\\boxed{...}` (common in math-reasoning-tuned models)
          2. `Final Answer: <x>` (the format we explicitly prompt for)
          3. `answer is X` / `= X` near the end of the response
          4. The last standalone number in the response
          5. The raw (cleaned) text, as a last resort
        """
        if not text:
            return ""

        cleaned = self._think_tag_re.sub("", text)

        boxed_matches = self._boxed_re.findall(cleaned)
        if boxed_matches:
            return boxed_matches[-1].strip()

        cleaned = self._markdown_strip_re.sub("", cleaned)

        final_match = self._final_answer_re.search(cleaned)
        if final_match:
            answer = final_match.group(1).strip().rstrip(".").strip()
            if answer:
                return answer

        tail = cleaned[-300:]
        alt_match = self._answer_is_re.search(tail)
        if alt_match:
            return alt_match.group(1).strip()

        numbers = self._number_re.findall(cleaned)
        if numbers:
            return numbers[-1]

        return cleaned.strip()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def _cache_get(self, key: Tuple[str, str]) -> Optional[str]:
        with self._lock:
            return self.cache.get(key)

    def _cache_put(self, key: Tuple[str, str], value: str) -> None:
        with self._lock:
            if key not in self.cache and len(self._cache_order) >= self.max_cache_size:
                oldest = self._cache_order.popleft()
                self.cache.pop(oldest, None)
            self.cache[key] = value
            self._cache_order.append(key)
            self._key_locks.pop(key, None)

    # ------------------------------------------------------------------
    # Low-level model call with retry
    # ------------------------------------------------------------------
    def _call_model(self, prompt: str, temperature: float) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature},
                )
                return response["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - want to retry any failure
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_base_delay * (2 ** attempt))
        assert last_error is not None
        raise last_error

    def _chat(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Cached, thread-safe, retrying chat call.

        Only temperature == 0.0 responses are cached (deterministic calls),
        since sampled (temperature > 0) calls are expected to vary between
        invocations, e.g. for self-consistency multi-path generation.
        """
        cacheable = temperature == 0.0
        if not cacheable:
            return self._call_model(prompt, temperature)

        key = (self.model, prompt)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        with self._lock:
            key_lock = self._key_locks.get(key)
            if key_lock is None:
                key_lock = threading.Lock()
                self._key_locks[key] = key_lock

        with key_lock:
            cached = self._cache_get(key)
            if cached is not None:
                return cached
            text = self._call_model(prompt, temperature)
            self._cache_put(key, text)
            return text

    # ------------------------------------------------------------------
    # High-level reasoning calls
    # ------------------------------------------------------------------
    def reasoning(self, question: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Run one chain-of-thought pass and extract its final answer."""
        prompt = (
            "Solve the following question carefully but concisely.\n\n"
            f"Question:\n{question}\n\n"
            "Explain your reasoning step by step, without unnecessary "
            "repetition or restating the question.\n\n"
            "Finish with exactly:\n\n"
            "Final Answer: <answer>"
        )
        try:
            text = self._chat(prompt, temperature=temperature)
            return {"reasoning": text, "answer": self.extract_answer(text)}
        except Exception as exc:  # noqa: BLE001
            return {"reasoning": f"Error: {exc}", "answer": ""}

    def generate(self, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Run an arbitrary prompt (e.g. critique/revision) through the model."""
        try:
            text = self._chat(prompt, temperature=temperature)
            return {"text": text, "answer": self.extract_answer(text)}
        except Exception as exc:  # noqa: BLE001
            return {"text": f"Error: {exc}", "answer": ""}

    # ------------------------------------------------------------------
    # Confidence parsing
    # ------------------------------------------------------------------
    def _parse_confidence(self, text: str) -> Optional[float]:
        """
        Parse a self-reported confidence score, accepting only values that
        are already probability-shaped (0-1) or an explicit percentage.

        Deliberately does NOT grab "the first number anywhere in the text"
        and clamp it -- doing so can silently turn an unrelated number
        (e.g. "step 2") into a fake high confidence, which breaks any
        confidence-threshold logic downstream. Returns None when no
        reliable value can be found, so callers can treat "unknown" as
        "low confidence" rather than a fabricated one.
        """
        if not text:
            return None

        candidates = self._confidence_re.findall(text)
        valid_scores = [float(c) for c in candidates if 0.0 <= float(c) <= 1.0]
        if valid_scores:
            return valid_scores[-1]

        percent_match = self._percent_re.search(text)
        if percent_match:
            value = float(percent_match.group(1)) / 100.0
            return max(0.0, min(1.0, value))

        return None

    def self_evaluate(self, question: str, reasoning: str, answer: str) -> Optional[float]:
        """Ask the model to rate its own confidence in a previous answer."""
        prompt = (
            "You previously answered a question. Rate your confidence "
            "that the final answer below is correct, as a single number "
            "between 0 and 1 (0 = certainly wrong, 1 = certainly correct). "
            "Respond with ONLY the number.\n\n"
            f"Question:\n{question}\n\n"
            f"Reasoning:\n{reasoning}\n\n"
            f"Final Answer:\n{answer}\n\n"
            "Confidence (0-1):"
        )
        try:
            text = self._chat(prompt, temperature=0.0)
            return self._parse_confidence(text)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Multi-path / batch helpers
    # ------------------------------------------------------------------
    def generate_multi_path(
        self, question: str, num_paths: int = 3, temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple reasoning paths SEQUENTIALLY.

        Kept sequential rather than parallel intentionally: running
        several concurrent requests against a local Ollama instance on an
        8 GB CPU-only machine causes severe resource contention and is
        typically slower (and less stable) than running them one at a
        time.
        """
        return [self.reasoning(question, temperature=temperature) for _ in range(max(1, num_paths))]

    def process_single_item(self, question: str) -> Dict[str, Any]:
        return self.reasoning(question)

    def generate_all_parallel(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Despite the name (kept for backward compatibility), this runs
        sequentially for the same reason as `generate_multi_path`: a
        single low-end local Ollama instance handles one request at a
        time far more reliably than several concurrent ones.
        """
        return [self.reasoning(q) for q in questions]