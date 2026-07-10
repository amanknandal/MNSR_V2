import json
from pathlib import Path
from typing import Dict, Any, List
from statsmodels.stats.contingency_tables import mcnemar


class ComparisonReport:

    def __init__(self, results_file: str = "results.json"):
        self.results_file = Path(results_file)
        self.results = self._load()
        self.methods = self._detect_methods()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.results_file.exists():
            print(f"Warning: '{self.results_file}' not found.")
            return []
        with open(self.results_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Malformed JSON in '{self.results_file}'.")
                return []

    def _detect_methods(self) -> List[str]:
        if not self.results:
            return []
        methods = []
        sample = self.results[0]
        for key in sample.keys():
            if key.endswith("_correct"):
                methods.append(key[: -len("_correct")])
        ordered = []
        for preferred in ("baseline", "mnsr"):
            if preferred in methods:
                ordered.append(preferred)
        for m in sorted(methods):
            if m not in ordered:
                ordered.append(m)
        return ordered

    def accuracy_table(self) -> Dict[str, Dict[str, float]]:
        total = len(self.results)
        table = {}
        if total == 0:
            return table
        for method in self.methods:
            correct = sum(1 for x in self.results if x.get(f"{method}_correct") is True)
            latencies = [x.get(f"{method}_latency_sec", 0.0) for x in self.results]
            avg_latency = sum(latencies) / total if total else 0.0
            table[method] = {
                "accuracy": round(correct / total, 4),
                "correct": correct,
                "total": total,
                "avg_latency_sec": round(avg_latency, 3),
            }
        return table

    def mcnemar_vs_mnsr(self) -> Dict[str, Dict[str, float]]:
        results_out = {}
        if "mnsr" not in self.methods:
            return results_out
        for method in self.methods:
            if method == "mnsr":
                continue
            both_correct = mnsr_only = other_only = both_wrong = 0
            for x in self.results:
                m = bool(x.get("mnsr_correct", False))
                o = bool(x.get(f"{method}_correct", False))
                if m and o:
                    both_correct += 1
                elif m and not o:
                    mnsr_only += 1
                elif not m and o:
                    other_only += 1
                else:
                    both_wrong += 1
            table = [[both_correct, other_only], [mnsr_only, both_wrong]]
            try:
                stat = mcnemar(table, exact=False, correction=True)
                results_out[method] = {
                    "chi_square": round(float(stat.statistic), 4),
                    "p_value": round(float(stat.pvalue), 6),
                    "mnsr_only_correct": mnsr_only,
                    "other_only_correct": other_only,
                }
            except Exception as e:
                results_out[method] = {"error": str(e)}
        return results_out

    def print_report(self):
        table = self.accuracy_table()
        sig = self.mcnemar_vs_mnsr()

        print("\n" + "=" * 18 + " METHOD COMPARISON REPORT " + "=" * 18 + "\n")
        print(f"{'Method':22} {'Accuracy':>10} {'Avg Latency (s)':>18}")
        print("-" * 55)
        for method, stats in table.items():
            print(f"{method:22} {stats['accuracy'] * 100:>9.2f}% {stats['avg_latency_sec']:>18.3f}")
        print("\n" + "-" * 55)
        print("McNemar's Test: MNSR vs each method")
        print("-" * 55)
        for method, stats in sig.items():
            if "error" in stats:
                print(f"MNSR vs {method:15}: error ({stats['error']})")
                continue
            print(
                f"MNSR vs {method:15}: chi2={stats['chi_square']:<8} "
                f"p={stats['p_value']:<10} "
                f"(MNSR-only correct={stats['mnsr_only_correct']}, "
                f"{method}-only correct={stats['other_only_correct']})"
            )
        print("\n" + "=" * 63 + "\n")

    def save(self, filename: str = "comparison_report.json"):
        payload = {
            "accuracy_table": self.accuracy_table(),
            "mcnemar_vs_mnsr": self.mcnemar_vs_mnsr(),
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        print(f"Comparison report saved to -> {filename}")
