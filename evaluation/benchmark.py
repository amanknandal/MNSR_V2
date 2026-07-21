"benchmark.py"
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any
class BenchmarkReport:
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results = self._load()
    def _load(self) -> list:
        if not self.results_file.exists():
            print(f"Warning: Target payload file {self.results_file} does not exist.")
            return []
        with open(self.results_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Malformed execution logs found at {self.results_file}")
                return []
    def generate(self) -> Dict[str, Any]:
        total = len(self.results)
        if total == 0:
            return {"Error": "No data available to calculate academic report metrics."}

        baseline_correct = sum(1 for x in self.results if x.get("baseline_correct") is True)
        mnsr_correct = sum(1 for x in self.results if x.get("mnsr_correct") is True)

        action_counter = Counter(x.get("action_path", "none") for x in self.results)

        avg_baseline_latency = sum(x.get("baseline_latency_sec", 0.0) for x in self.results) / total
        avg_mnsr_latency = sum(x.get("mnsr_latency_sec", 0.0) for x in self.results) / total
        avg_confidence = sum(x.get("mnsr_confidence", 0.0) for x in self.results) / total

        exact_match = sum(
            1 for x in self.results
            if str(x.get("mnsr_answer", "")).strip().lower() == str(x.get("gold", "")).strip().lower()
        )

        corrected_questions = sum(1 for x in self.results if x.get("correction_history"))

        report = {
            "Total Evaluation Samples": total,
            "Baseline System Accuracy": round(baseline_correct / total, 4),
            "MNSR System Accuracy": round(mnsr_correct / total, 4),
            "Delta Accuracy Gain": round((mnsr_correct - baseline_correct) / total, 4),
            "Exact Match Rate": round(exact_match / total, 4),
            "Avg Baseline Latency (sec)": round(avg_baseline_latency, 3),
            "Avg MNSR Latency (sec)": round(avg_mnsr_latency, 3),
            "Avg MNSR Confidence": round(avg_confidence, 4),
            "Questions Requiring Correction": corrected_questions,
            "Correction Rate": round(corrected_questions / total, 4),
        }
        for action, count in sorted(action_counter.items()):
            report[f"Action Count [{action}]"] = count
        return report

    def print_report(self):
        report = self.generate()
        print("\n" + "=" * 15 + " IEEE BENCHMARK METRICS REPORT " + "=" * 15 + "\n")
        for k, v in report.items():
            if isinstance(v, float) and ("Accuracy" in k or "Delta" in k or "Rate" in k or "Confidence" in k):
                print(f"{k:32}: {v * 100:.2f}%")
            else:
                print(f"{k:32}: {v}")
        print("\n" + "=" * 61 + "\n")

    def save(self, filename="benchmark_report.json"):
        report = self.generate()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        print(f"Benchmark metric matrix saved to -> {filename}")
