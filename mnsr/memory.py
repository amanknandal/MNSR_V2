from typing import List, Dict, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ReflectionMemory:
    def __init__(self):
        self.memory: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None

    def add(
        self,
        question: str,
        reasoning: str,
        answer: str,
        errors: List[Dict],
        corrected_reasoning: str = "",
        success: bool = False,
        controller_actions: Optional[List[str]] = None,
        retry_count: int = 0,
        confidence: float = 0.0,
        validation_report: Optional[Dict] = None,
    ):
        episode = {
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
            "errors": errors,
            "corrected_reasoning": corrected_reasoning,
            "success": success,
            "controller_actions": controller_actions or [],
            "retry_count": retry_count,
            "confidence": confidence,
            "validation_report": validation_report or {},
        }
        self.memory.append(episode)
        self._rebuild_index()

    def _rebuild_index(self):
        questions = [ep["question"].lower() for ep in self.memory]
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(questions)
        except ValueError:
            self.tfidf_matrix = None

    def retrieve(self, question: str, threshold: float = 0.60) -> Dict[str, Optional[Dict]]:
        result = {"success": None, "failure": None}
        if not self.memory or self.tfidf_matrix is None:
            return result

        query_vec = self.vectorizer.transform([question.lower()])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        for outcome_key, want_success in (("success", True), ("failure", False)):
            candidate_indices = [
                i for i, ep in enumerate(self.memory) if ep["success"] == want_success
            ]
            if not candidate_indices:
                continue
            best_idx = max(candidate_indices, key=lambda i: similarities[i])
            best_score = float(similarities[best_idx])
            if best_score >= threshold:
                result[outcome_key] = {
                    "similarity": round(best_score, 3),
                    "episode": self.memory[best_idx],
                }
        return result

    def retrieve_best(self, question: str, threshold: float = 0.60) -> Optional[Dict]:
        both = self.retrieve(question, threshold=threshold)
        candidates = [v for v in both.values() if v is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c["similarity"])

    def size(self) -> int:
        return len(self.memory)

    def success_count(self) -> int:
        return sum(1 for ep in self.memory if ep["success"])

    def failure_count(self) -> int:
        return sum(1 for ep in self.memory if not ep["success"])

    def clear(self):
        self.memory = []
        self.tfidf_matrix = None

    def export(self) -> List[Dict]:
        return self.memory
