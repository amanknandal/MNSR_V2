import os
import json
import time
from pathlib import Path

from evaluation.comparison_report import ComparisonReport
from evaluation.visualize import ResultsVisualizer
from evaluation.ablation import AblationStudy
from models.phi3 import Phi3Mini

DATASET_DIR = "datasets"
RESULTS_DIR = "results"
MAX_WORKERS = 4


def verify_and_get_datasets():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: The directory '{DATASET_DIR}' was not found.")
        return []

    target_files = [
        "gsm8k.json",
        "halueval_qa.json",
        "strategyqa.json",
        "truthful_qa.json",
    ]

    found_datasets = []

    for file in target_files:
        path = os.path.join(DATASET_DIR, file)
        if os.path.exists(path):
            found_datasets.append(path)

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

    print("\n" + "=" * 80)
    print(f"CONTINUING PIPELINE FOR: {dataset_name.upper()}")
    print("=" * 80)

    start_time = time.time()

    results_file = os.path.join(output_dir, "results.json")
    benchmark_file = os.path.join(output_dir, "benchmark_report.json")
    ablation_file = os.path.join(output_dir, "ablation_results.json")
    comparison_file = os.path.join(output_dir, "comparison_report.json")

    # ------------------------------------------------------------------
    # Verify previous outputs exist
    # ------------------------------------------------------------------

    if not os.path.exists(results_file):
        print(f"ERROR: Missing {results_file}")
        print("Run Step 1 first to generate results.json")
        return

    if not os.path.exists(benchmark_file):
        print(f"ERROR: Missing {benchmark_file}")
        print("Run Step 2 first to generate benchmark_report.json")
        return

    shared_cache = {}
    shared_model = Phi3Mini(cache=shared_cache)

    # ------------------------------------------------------------------
    # Step 4
    # ------------------------------------------------------------------

    print("\n[Step 4/6] Comparing MNSR Against Self-Consistency, Reflexion, and Baseline CoT...")

    comparison = ComparisonReport(results_file)
    comparison.save(comparison_file)
    comparison.print_report()

    # ------------------------------------------------------------------
    # Step 5
    # ------------------------------------------------------------------

    print("\n[Step 5/6] Executing Component Ablation Studies...")

    ablation = AblationStudy(
        dataset_path,
        model=shared_model,
        max_workers=MAX_WORKERS,
    )

    ablation.run()
    ablation.save(ablation_file)
    ablation.print_report()

    # ------------------------------------------------------------------
    # Step 6
    # ------------------------------------------------------------------

    print("\n[Step 6/6] Exporting Paper Figures...")

    try:
        visualizer = ResultsVisualizer(
            benchmark_file=benchmark_file,
            ablation_file=ablation_file,
            output_dir=output_dir,
        )

        visualizer.generate_all(
            comparison_file=comparison_file
        )

    except Exception as e:
        print(f"Visualization warning: {e}")

    elapsed = (time.time() - start_time) / 60

    print("\n" + "=" * 80)
    print(f"Finished '{dataset_name}' in {elapsed:.2f} minutes.")
    print("=" * 80)


def main():
    print("\n" + "#" * 80)
    print("### RUNNING MNSR STEPS 4, 5 AND 6 ONLY ###")
    print("#" * 80)

    datasets = verify_and_get_datasets()

    if not datasets:
        print("No datasets found.")
        return

    print(f"Found {len(datasets)} datasets.")

    total_start = time.time()

    for dataset in datasets:
        try:
            run_pipeline_for_dataset(dataset)
        except Exception as e:
            print(f"\nFailed on {dataset}")
            print(e)

    total_elapsed = (time.time() - total_start) / 60

    print("\n" + "=" * 80)
    print(f"ALL DONE. Total Time: {total_elapsed:.2f} minutes.")
    print("=" * 80)


if __name__ == "__main__":
    main()