"statistical_tests.py"
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
class StatisticalAnalyzer:
    def __init__(self, results_file: str = "results.json"):
        self.results_file = Path(results_file)
        self.results = self._load()
    def _load(self) -> List[Dict[str, Any]]:
        if not self.results_file.exists():
            print(f"Warning: Evaluation results file '{self.results_file}' not found.")
            return []
        with open(self.results_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Malformed JSON payload in '{self.results_file}'.")
                return []
    def wilcoxon_test(self) -> Dict[str, float]:
        if not self.results:
            return {"statistic": 0.0, "p_value": 1.0}
        baseline = [int(x.get("baseline_correct", False)) for x in self.results]
        mnsr = [int(x.get("mnsr_correct", False)) for x in self.results]
        if baseline == mnsr:
            return {"statistic": 0.0, "p_value": 1.0}
        try:
            statistic, p = wilcoxon(baseline, mnsr, zero_method="zsplit")
            return {"statistic": round(float(statistic), 4), "p_value": round(float(p), 6)}
        except ValueError:
            return {"statistic": 0.0, "p_value": 1.0}

    def mcnemar_test(self) -> Dict[str, float]:
        if not self.results:
            return {"chi_square": 0.0, "p_value": 1.0}

        both_correct = baseline_only = mnsr_only = both_wrong = 0

        for sample in self.results:
            b = bool(sample.get("baseline_correct", False))
            m = bool(sample.get("mnsr_correct", False))
            if b and m:
                both_correct += 1
            elif b and not m:
                baseline_only += 1
            elif not b and m:
                mnsr_only += 1
            else:
                both_wrong += 1

        table = [[both_correct, baseline_only], [mnsr_only, both_wrong]]
        result = mcnemar(table, exact=False, correction=True)
        return {"chi_square": round(float(result.statistic), 4), "p_value": round(float(result.pvalue), 6)}

    def confidence_interval(self) -> Dict[str, Any]:
        if not self.results:
            return {"accuracy": 0.0, "95CI": (0.0, 0.0)}
        scores = np.array([int(x.get("mnsr_correct", False)) for x in self.results])
        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1) if n > 1 else 0.0
        ci = 1.96 * std / np.sqrt(n) if n > 0 else 0.0
        return {
            "accuracy": round(float(mean), 4),
            "95CI": (
                round(float(max(0.0, mean - ci)), 4),
                round(float(min(1.0, mean + ci)), 4),
            ),
        }

    def report(self):
        print("\n" + "=" * 16 + " STATISTICAL SIGNIFICANCE REPORT " + "=" * 16 + "\n")
        w_test = self.wilcoxon_test()
        print("Wilcoxon Signed-Rank Test:")
        print(f"  Statistic  : {w_test['statistic']}")
        print(f"  P-Value    : {w_test['p_value']}")
        print("-" * 65)

        m_test = self.mcnemar_test()
        print("McNemar's Test (Contingency Marginal Homogeneity):")
        print(f"  Chi-Square : {m_test['chi_square']}")
        print(f"  P-Value    : {m_test['p_value']}")
        print("-" * 65)

        ci = self.confidence_interval()
        print("MNSR Accuracy 95% Confidence Interval (Wald):")
        print(f"  Mean Acc   : {ci['accuracy'] * 100:.2f}%")
        print(f"  95% CI     : ({ci['95CI'][0] * 100:.2f}%, {ci['95CI'][1] * 100:.2f}%)")
        print("\n" + "=" * 65 + "\n")
