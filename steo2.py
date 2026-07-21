# evaluation/comparison_and_ablation.py
"""
Combined module for Step 4 + Step 5 of the MNSR pipeline.

Step 4 -> ComparisonReport
    Compares MNSR against Self-Consistency, Reflexion, and plain CoT in
    terms of accuracy, latency, token efficiency, and calibrated confidence.

Step 5 -> AblationStudy
    Strips MNSR components one at a time and re-runs the engine to measure
    each component's contribution. Components ablated:
        - metacognitive_monitor
        - neural_symbolic_solver
        - self_reflection
        - retrieval_augmentation
"""
import os
import json
import time
import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ===================================================================== #
# Step 4 — ComparisonReport
# ===================================================================== #
class ComparisonReport:
    def __init__(self, results_file: str, reference_method: str = "mnsr"):
        self.results_file = results_file
        self.reference = reference_method
        self.records = self._load(results_file)
        self.summary = {}
        self.head_to_head = []

    @staticmethod
    def _load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Results file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "records" in data:
            return data["records"]
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _safe_avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    def _aggregate(self):
        agg = defaultdict(lambda: {"n": 0, "correct": 0,
                                   "latency": [], "tokens": [],
                                   "confidence": []})
        for rec in self.records:
            for m, p in (rec.get("predictions") or {}).items():
                if p is None:
                    continue
                a = agg[m]
                a["n"] += 1
                a["correct"] += int(bool(p.get("correct", False)))
                a["latency"].append(float(p.get("latency", 0.0) or 0.0))
                a["tokens"].append(int(p.get("tokens", 0) or 0))
                a["confidence"].append(float(p.get("confidence", 0.0) or 0.0))
        out = {}
        for m, a in agg.items():
            n = a["n"]
            out[m] = {
                "n": n,
                "accuracy": a["correct"] / n if n else 0.0,
                "mean_latency_s": self._safe_avg(a["latency"]),
                "mean_tokens": self._safe_avg(a["tokens"]),
                "mean_confidence": self._safe_avg(a["confidence"]),
                "tokens_per_correct": (
                    self._safe_avg(a["tokens"]) / (a["correct"] or 1)
                ),
            }
        return out

    def _build_head_to_head(self, agg):
        if self.reference not in agg:
            return []
        ref = agg[self.reference]
        rows = []
        for m, a in agg.items():
            if m == self.reference:
                continue
            rows.append({
                "baseline": m,
                "reference": self.reference,
                "accuracy_delta": ref["accuracy"] - a["accuracy"],
                "latency_delta_s": ref["mean_latency_s"] - a["mean_latency_s"],
                "token_delta": ref["mean_tokens"] - a["mean_tokens"],
                "confidence_delta": ref["mean_confidence"] - a["mean_confidence"],
                "relative_accuracy_gain_pct": (
                    (ref["accuracy"] - a["accuracy"]) / a["accuracy"] * 100.0
                    if a["accuracy"] > 0 else 0.0
                ),
            })
        rows.sort(key=lambda r: r["accuracy_delta"], reverse=True)
        return rows

    def build(self):
        self.summary = self._aggregate()
        self.head_to_head = self._build_head_to_head(self.summary)
        return self.to_dict()

    def to_dict(self):
        return {
            "dataset": os.path.basename(os.path.dirname(self.results_file)),
            "reference_method": self.reference,
            "summary": self.summary,
            "head_to_head": self.head_to_head,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[ComparisonReport] Saved comparison -> {path}")

    def print_report(self):
        if not self.summary:
            self.build()
        print("\n" + "=" * 78)
        print(f"COMPARISON REPORT  |  reference={self.reference}")
        print("=" * 78)
        hdr = (f"{'Method':<18}{'Acc':>8}{'Latency(s)':>13}"
               f"{'Tokens':>10}{'Conf':>9}{'Tok/Correct':>13}")
        print(hdr)
        print("-" * 78)
        for m, a in self.summary.items():
            print(f"{m:<18}{a['accuracy']:>8.4f}"
                  f"{a['mean_latency_s']:>13.3f}"
                  f"{a['mean_tokens']:>10.1f}"
                  f"{a['mean_confidence']:>9.3f}"
                  f"{a['tokens_per_correct']:>13.1f}")
        print("-" * 78)
        print(f"\nHead-to-head deltas vs {self.reference}:")
        for r in self.head_to_head:
            print(f"  vs {r['baseline']:<16} "
                  f"Δacc={r['accuracy_delta']:+.4f} "
                  f"({r['relative_accuracy_gain_pct']:+.2f}%)  "
                  f"Δlat={r['latency_delta_s']:+.3f}s  "
                  f"Δtok={r['token_delta']:+.1f}  "
                  f"Δconf={r['confidence_delta']:+.3f}")
        print("=" * 78)


# ===================================================================== #
# Step 5 — AblationStudy
# ===================================================================== #
# Ablation configurations. Each entry disables one MNSR component by
# patching its config on the model instance before re-running inference.
ABLATION_COMPONENTS = [
    "full",
    "no_metacognitive_monitor",
    "no_neural_symbolic_solver",
    "no_self_reflection",
    "no_retrieval_augmentation",
]


class AblationStudy:
    def __init__(self, dataset_path, model, max_workers: int = 4):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.model = model
        self.max_workers = max_workers
        self.samples = self._load_dataset()
        self.results = {}    # variant -> per-sample list
        self.summary = {}    # variant -> aggregate metrics

    def _load_dataset(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(self.dataset_path)
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "samples" in data:
            return data["samples"]
        return data

    # ----------------- ablation configuration ----------------- #
    def _apply_variant(self, variant: str):
        """Patch model config flags to disable a single MNSR component.
        Resets state for a fresh variant run."""
        # Restore baseline config first.
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            cfg = {}
            try:
                self.model.config = cfg
            except Exception:
                pass

        cfg["use_metacognitive_monitor"] = True
        cfg["use_neural_symbolic_solver"] = True
        cfg["use_self_reflection"] = True
        cfg["use_retrieval_augmentation"] = True
        cfg["variant"] = variant

        if variant == "no_metacognitive_monitor":
            cfg["use_metacognitive_monitor"] = False
        elif variant == "no_neural_symbolic_solver":
            cfg["use_neural_symbolic_solver"] = False
        elif variant == "no_self_reflection":
            cfg["use_self_reflection"] = False
        elif variant == "no_retrieval_augmentation":
            cfg["use_retrieval_augmentation"] = False
        # "full" leaves everything enabled.

    # ----------------- inference ----------------- #
    def _infer_one(self, sample, variant):
        """Run a single MNSR inference under the active variant config.
        Expects model.solve / model.predict / model.reason to exist; we
        try several common names to remain backend-agnostic."""
        question = sample.get("question") or sample.get("prompt") or ""
        ground_truth = sample.get("answer") or sample.get("ground_truth") or ""
        sid = sample.get("id")

        t0 = time.time()
        out = None
        for fn_name in ("solve", "predict", "reason", "infer"):
            fn = getattr(self.model, fn_name, None)
            if callable(fn):
                try:
                    out = fn(question, config_override=None) \
                        if "config_override" in fn.__code__.co_varnames \
                        else fn(question)
                except TypeError:
                    out = fn(question)
                break
        if out is None:
            out = {"answer": "", "confidence": 0.0, "tokens": 0}
        latency = time.time() - t0

        pred_answer = ""
        confidence = 0.0
        tokens = 0
        if isinstance(out, dict):
            pred_answer = out.get("answer", "") or out.get("prediction", "")
            confidence = float(out.get("confidence", 0.0) or 0.0)
            tokens = int(out.get("tokens", 0) or 0)
        elif isinstance(out, str):
            pred_answer = out

        correct = self._score(pred_answer, ground_truth)
        return {
            "sample_id": sid,
            "variant": variant,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": pred_answer,
            "correct": correct,
            "confidence": confidence,
            "tokens": tokens,
            "latency": latency,
        }

    @staticmethod
    def _score(prediction, ground_truth):
        if prediction is None or ground_truth is None:
            return False
        p = str(prediction).strip().lower()
        g = str(ground_truth).strip().lower()
        if not p or not g:
            return False
        if g in p or p in g:
            return True
        # numeric match fallback
        try:
            return abs(float(p) - float(g)) < 1e-3
        except Exception:
            return False

    # ----------------- run ----------------- #
    def run(self):
        print(f"\n[AblationStudy] Dataset: {self.dataset_name}  "
              f"n_samples={len(self.samples)}  "
              f"variants={len(ABLATION_COMPONENTS)}")
        for variant in ABLATION_COMPONENTS:
            self._apply_variant(variant)
            rows = []
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(self._infer_one, s, variant)
                           for s in self.samples]
                for fut in as_completed(futures):
                    try:
                        rows.append(fut.result())
                    except Exception as e:
                        print(f"  [warn] {variant} sample failed: {e}")
            self.results[variant] = rows
            n = len(rows)
            n_correct = sum(1 for r in rows if r["correct"])
            acc = n_correct / n if n else 0.0
            mean_lat = sum(r["latency"] for r in rows) / n if n else 0.0
            mean_tok = sum(r["tokens"] for r in rows) / n if n else 0.0
            mean_conf = sum(r["confidence"] for r in rows) / n if n else 0.0
            self.summary[variant] = {
                "n": n,
                "accuracy": acc,
                "correct": n_correct,
                "mean_latency_s": mean_lat,
                "mean_tokens": mean_tok,
                "mean_confidence": mean_conf,
                "elapsed_s": time.time() - t0,
            }
            print(f"  -> {variant:<30} acc={acc:.4f}  "
                  f"lat={mean_lat:.3f}s  tok={mean_tok:.1f}  "
                  f"conf={mean_conf:.3f}")

        # Restore full configuration after ablation is done.
        self._apply_variant("full")
        return self.to_dict()

    def to_dict(self):
        return {
            "dataset": self.dataset_name,
            "components": ABLATION_COMPONENTS,
            "summary": self.summary,
            "results": self.results,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[AblationStudy] Saved ablation results -> {path}")

    def print_report(self):
        print("\n" + "=" * 78)
        print("COMPONENT ABLATION REPORT")
        print("=" * 78)
        print(f"{'Variant':<32}{'Acc':>8}{'Latency(s)':>13}"
              f"{'Tokens':>10}{'Conf':>9}")
        print("-" * 78)
        for v in ABLATION_COMPONENTS:
            s = self.summary.get(v, {})
            print(f"{v:<32}{s.get('accuracy', 0):>8.4f}"
                  f"{s.get('mean_latency_s', 0):>13.3f}"
                  f"{s.get('mean_tokens', 0):>10.1f}"
                  f"{s.get('mean_confidence', 0):>9.3f}")
        print("-" * 78)
        # Contribution deltas relative to full
        if "full" in self.summary:
            base = self.summary["full"]["accuracy"]
            print("\nComponent contribution (Δacc vs full):")
            for v in ABLATION_COMPONENTS:
                if v == "full":
                    continue
                delta = base - self.summary.get(v, {}).get("accuracy", 0.0)
                print(f"  {v:<30} Δacc = {delta:+.4f}")
        print("=" * 78)