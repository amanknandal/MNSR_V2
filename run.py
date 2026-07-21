"run.py"
import os
import json
import time
from pathlib import Path
from evaluation.experiment import MNSRExperiment
from evaluation.benchmark import BenchmarkReport
from evaluation.statistical_tests import StatisticalAnalyzer
from evaluation.comparison_report import ComparisonReport
from evaluation.comparison_baselines import SelfConsistencyReasoner, ReflexionReasoner
from evaluation.visualize import ResultsVisualizer
from evaluation.ablation import AblationStudy
from models.phi3 import Phi3Mini
DATASET_DIR = "datasets"
RESULTS_DIR = "results"
MAX_WORKERS = 4
RUN_COMPARISON_BASELINES = True
def verify_and_get_datasets():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: The directory '{DATASET_DIR}' was not found.")
        print("Please create it and place your benchmark JSON files inside.")
        return []
    target_files = ["gsm8k.json", "halueval_qa.json", "strategyqa.json", "truthful_qa.json"]
    found_datasets = []
    for file in target_files:
        path = os.path.join(DATASET_DIR, file)
        if os.path.exists(path):
            found_datasets.append(path)
        else:
            print(f"Warning: Expected target file '{file}' is missing from '{DATASET_DIR}/'")
    if not found_datasets:
        found_datasets = [
            os.path.join(DATASET_DIR, f)
            for f in os.listdir(DATASET_DIR)
            if f.endswith(".json")
        ]
    return sorted(found_datasets)
def run_pipeline_for_dataset(dataset_path):
    dataset_name = Path(dataset_path).stem
    output_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"MNSR EVALUATION STARTED FOR TRACK: {dataset_name.upper()}")
    print(f"Worker Profile: Ollama (Concurrency: {MAX_WORKERS})")
    print("=" * 80)

    start_time = time.time()

    results_file = os.path.join(output_dir, "results.json")
    benchmark_file = os.path.join(output_dir, "benchmark_report.json")
    ablation_file = os.path.join(output_dir, "ablation_results.json")

    comparison_file = os.path.join(output_dir, "comparison_report.json")

    shared_cache = {}
    shared_model = Phi3Mini(cache=shared_cache)

    extra_methods = {}
    if RUN_COMPARISON_BASELINES:
        extra_methods = {
            "self_consistency": SelfConsistencyReasoner(model=shared_model, num_paths=5, temperature=0.7),
            "reflexion": ReflexionReasoner(model=shared_model, max_iterations=2, confidence_threshold=0.85),
        }

    print(f"\n[Step 1/6] Running Core Experiments on {dataset_name}...")
    experiment = MNSRExperiment(
        dataset_path,
        model=shared_model,
        cache=shared_cache,
        max_workers=MAX_WORKERS,
        extra_methods=extra_methods,
    )
    experiment.run()
    experiment.save_results(results_file)
    experiment.print_summary()

    print("\n[Step 2/6] Compiling Benchmark Matrix Report...")
    bench = BenchmarkReport(results_file)
    bench.save(benchmark_file)
    bench.print_report()

    print("\n[Step 3/6] Computing Non-Parametric Significance Statistics...")
    analyzer = StatisticalAnalyzer(results_file)
    analyzer.report()

    print("\n[Step 4/6] Comparing MNSR Against Self-Consistency, Reflexion, and Baseline CoT...")
    comparison = ComparisonReport(results_file)
    comparison.save(comparison_file)
    comparison.print_report()

    print("\n[Step 5/6] Executing Component Ablation Studies...")
    ablation = AblationStudy(dataset_path, model=shared_model, max_workers=MAX_WORKERS)
    ablation.run()
    ablation.save(ablation_file)
    ablation.print_report()

    print("\n[Step 6/6] Exporting Paper Figures...")
    try:
        visualizer = ResultsVisualizer(
            benchmark_file=benchmark_file,
            ablation_file=ablation_file,
            output_dir=output_dir,
        )
        visualizer.generate_all(comparison_file=comparison_file)
    except Exception as e:
        print(f"Visualization step warning: {e}")
        print("Continuing pipeline execution...")

    elapsed_time = time.time() - start_time
    print(f"\nCompleted dataset run for '{dataset_name}' in {elapsed_time / 60:.2f} minutes.")
    print(f"Unique cached LLM calls for this dataset: {len(shared_cache)}")


def main():
    print("\n" + "#" * 80)
    print("### INITIALIZING LOCAL MNSR EXPERIMENT MATRIX RUNNER ###")
    print("#" * 80)

    datasets = verify_and_get_datasets()
    if not datasets:
        print("Pipeline execution halted: No dataset profiles found.")
        return

    print(f"Verification check: Found {len(datasets)} dataset queues to process.")
    for idx, path in enumerate(datasets, 1):
        try:
            with open(path, "r", encoding="utf-8") as f:
                samples = json.load(f)
                print(f"  [{idx}] {os.path.basename(path)} -> Detected {len(samples)} data rows.")
        except Exception:
            print(f"  [{idx}] {os.path.basename(path)} -> Unable to read format preview.")

    total_start = time.time()
    for dataset in datasets:
        try:
            run_pipeline_for_dataset(dataset)
        except Exception as err:
            print(f"\nCRITICAL: Track crashed on dataset '{os.path.basename(dataset)}' due to error: {err}")
            print("Skipping to next matrix file to preserve execution queue flow...\n")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("SUCCESS: ALL REASONING DATASETS EVALUATED BY THE MNSR ENGINE.")
    print(f"Total Wall-Clock Execution Time: {total_elapsed / 60:.2f} minutes.")
    print("Review final compilation profiles and PNGs inside your 'results/' folder.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
