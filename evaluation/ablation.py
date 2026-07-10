import copy
import json
from typing import Dict, Any, Optional
from evaluation.experiment import MNSRExperiment
from models.phi3 import Phi3Mini


class MockValidator:
    def validate(self, reasoning: str, dataset_type=None) -> Dict[str, Any]:
        return {
            "valid": True,
            "score": 1.0,
            "errors": [],
            "warnings": [],
            "num_errors": 0,
            "reasoning_consistency": 1.0,
            "answer_consistency": 1.0,
        }


class MockMemory:
    def retrieve(self, question: str, threshold: float = 0.60):
        return {"success": None, "failure": None}

    def add(self, *args, **kwargs):
        pass


class MockController:
    def evaluate(self, state) -> str:
        state.set_action("CONTINUE")
        return "CONTINUE"


class MockConfidenceEstimator:
    def estimate(self, *args, **kwargs) -> float:
        return 0.80


class AblationStudy:

    def __init__(self, dataset_path: str, model: Optional[Phi3Mini] = None, max_workers: int = 1):
        self.dataset_path = dataset_path
        self.results: Dict[str, Dict[str, Any]] = {}
        self.cache: Dict = {}
        self.model = model if model is not None else Phi3Mini(cache=self.cache)
        self.max_workers = max_workers

    def _run_pipeline(
        self,
        disable_validator: bool = False,
        disable_memory: bool = False,
        disable_controller: bool = False,
        disable_dynamic_confidence: bool = False,
    ) -> Dict[str, Any]:
        experiment = MNSRExperiment(
            self.dataset_path,
            model=self.model,
            cache=self.cache,
            max_workers=self.max_workers,
        )
        if disable_validator:
            experiment.mnsr.validator = MockValidator()
        if disable_memory:
            experiment.mnsr.memory = MockMemory()
        if disable_controller:
            experiment.mnsr.controller = MockController()
        if disable_dynamic_confidence:
            experiment.mnsr.confidence_estimator = MockConfidenceEstimator()
        experiment.run()
        return copy.deepcopy(experiment.metrics.summary())

    def run(self) -> Dict[str, Dict[str, Any]]:
        print("\n[Ablation] Running Full MNSR Baseline Configuration...")
        self.results["Full MNSR"] = self._run_pipeline()

        print("\n[Ablation] Running Configuration: (- Symbolic Validator)...")
        self.results["- Validator"] = self._run_pipeline(disable_validator=True)

        print("\n[Ablation] Running Configuration: (- Reflection Memory Index)...")
        self.results["- Memory"] = self._run_pipeline(disable_memory=True)

        print("\n[Ablation] Running Configuration: (- Meta-Cognitive Controller)...")
        self.results["- Controller"] = self._run_pipeline(disable_controller=True)

        print("\n[Ablation] Running Configuration: (- Dynamic Confidence Estimation)...")
        self.results["- Dynamic Confidence"] = self._run_pipeline(disable_dynamic_confidence=True)

        return self.results

    def print_report(self):
        print("\n" + "=" * 16 + " ABLATION STUDY ANALYSIS REPORT " + "=" * 16 + "\n")
        print(f"{'System Configuration':28} {'Accuracy':>12} {'Avg Steps':>12}")
        print("-" * 55)
        for name, metrics in self.results.items():
            acc = metrics.get("mnsr_accuracy", 0.0)
            steps = metrics.get("avg_steps", 0.0)
            print(f"{name:28} {acc * 100:>11.2f}% {steps:>12.2f}")
        print("\n" + "=" * 64 + "\n")

    def save(self, filename="ablation_results.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        print(f"\nAblation study metrics data matrix exported -> {filename}")
