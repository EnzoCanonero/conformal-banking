from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np

from llm_scorekit import (
    conformal_quantile,
    lac_prediction_sets,
    lac_scores,
    socop_prediction_sets,
    socop_scores,
)
from examples.banking77.scores._artifacts import load_manifest, save_csv, save_manifest
from examples.banking77.scores._data import PartitionPredictions
from examples.banking77.scores._evaluation import naive_metrics
from examples.banking77.scores._socop_tuning import select_regularizations

from ._comparison_data import load_classifier
from ._data import load_predictions
from ._evaluation import (
    ALPHAS,
    CONFIDENCE_LEVEL,
    CONFIDENCE_THRESHOLDS,
    SOCOP_REGULARIZATIONS,
    evaluate_sets,
)
from ._plots import save_comparison_figures
from .prepare import CACHE_PATH, STUDY_DIRECTORY


CLASSIFIER_DIRECTORIES = {
    "tfidf": Path("outputs/banking77/score_comparison/prepared"),
    "encoder": Path("outputs/banking77/representation_comparison/encoder/prepared"),
}
OUTPUT_DIRECTORY = STUDY_DIRECTORY / "comparison"


# Recalibrate each method after choosing SOCOP lambda only on the tuning halves.
def evaluate_conformal(
    partitions: dict[str, PartitionPredictions],
    regularizations: dict[float, float],
) -> list[dict[str, str | float]]:
    calibration = partitions["calibration"]
    test = partitions["test"]
    lac_calibration_scores = lac_scores(calibration.probabilities, calibration.labels)
    rows: list[dict[str, str | float]] = []

    for alpha in ALPHAS:
        regularization = regularizations[alpha]
        socop_calibration_scores = socop_scores(
            calibration.probabilities, calibration.labels, regularization=regularization
        )
        lac_threshold = conformal_quantile(lac_calibration_scores, alpha)
        socop_threshold = conformal_quantile(socop_calibration_scores, alpha)
        lac_sets = lac_prediction_sets(test.probabilities, lac_threshold)
        socop_sets = socop_prediction_sets(
            test.probabilities, socop_threshold, regularization=regularization
        )

        for method, sets, threshold, penalty in (
            ("lac", lac_sets, lac_threshold, float("nan")),
            ("socop_tuned", socop_sets, socop_threshold, regularization),
        ):
            row: dict[str, str | float] = {
                "method": method,
                "alpha": alpha,
                "target_coverage": 1.0 - alpha,
                "regularization": penalty,
                "threshold": threshold,
                "calibration_count": calibration.labels.size,
            }
            row.update(evaluate_sets(sets, test))
            rows.append(row)

    return rows


# Compare all models on matched cached records, without training or LLM inference.
def run_comparison(
    cache_path: Path = CACHE_PATH,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> None:
    llm_partitions, metadata = load_predictions(cache_path)
    models = {"qwen": llm_partitions}
    sources = {
        "qwen": {
            "archive": str(cache_path),
            "sha256": sha256(cache_path.read_bytes()).hexdigest(),
            "model_id": metadata["model_id"],
            "model_revision": metadata["model_revision"],
            "scoring": metadata["scoring"],
            "normalization": "softmax of summed scores, temperature 1",
            "task_specific_training": False,
        }
    }
    for model, directory in CLASSIFIER_DIRECTORIES.items():
        models[model], sources[model] = load_classifier(
            directory, llm_partitions, metadata
        )

    # Use the union of already-declared grids, not cutoffs chosen from test errors.
    baseline_manifest = load_manifest(CLASSIFIER_DIRECTORIES["tfidf"])
    baseline_cutoffs = baseline_manifest["confidence_thresholds"]
    confidence_thresholds = sorted(set(baseline_cutoffs) | set(CONFIDENCE_THRESHOLDS))
    conformal_rows: list[dict[str, str | float]] = []
    naive_rows: list[dict[str, str | float]] = []
    tuning_rows: list[dict[str, str | float]] = []
    accuracy_rows: list[dict[str, str | float]] = []

    for model, partitions in models.items():
        print(f"Evaluating {model} from saved predictions")
        regularizations, candidates = select_regularizations(
            partitions["tuning_a"],
            partitions["tuning_b"],
            ALPHAS,
            SOCOP_REGULARIZATIONS,
        )
        model_rows = evaluate_conformal(partitions, regularizations)
        for row in model_rows:
            row["model"] = model
            conformal_rows.append(row)
        for candidate in candidates:
            tuning_rows.append({"model": model, **candidate})

        test = partitions["test"]
        for naive_row in naive_metrics(test, confidence_thresholds, CONFIDENCE_LEVEL):
            naive_rows.append({"model": model, "method": "naive", **naive_row})
        predictions = np.argmax(test.probabilities, axis=1)
        correct_count = int(np.sum(predictions == test.labels))
        accuracy_rows.append(
            {
                "model": model,
                "sample_count": test.labels.size,
                "correct_count": correct_count,
                "accuracy": correct_count / test.labels.size,
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    save_csv(output_directory / "conformal_metrics.csv", conformal_rows)
    save_csv(output_directory / "naive_metrics.csv", naive_rows)
    save_csv(output_directory / "tuning_candidates.csv", tuning_rows)
    save_csv(output_directory / "accuracy.csv", accuracy_rows)
    save_comparison_figures(conformal_rows, naive_rows, output_directory / "figures")
    manifest = {
        "sources": sources,
        "seed": metadata["random_seed"],
        "sample_sizes": metadata["sample_sizes"],
        "alignment": "identical IDs, labels and intent order; rows aligned to LLM cache",
        "alphas": ALPHAS,
        "socop_regularizations": SOCOP_REGULARIZATIONS,
        "socop_tuning": "separate per model and alpha; swap A/B, maximize singletons",
        "socop_tie_breaks": ["smaller total set size", "smaller regularization"],
        "confidence_thresholds": confidence_thresholds,
        "confidence_threshold_source": "union of existing TF-IDF and LLM grids",
        "confidence_level": CONFIDENCE_LEVEL,
        "interval_method": "Wilson, pointwise for this fixed split",
        "aggregation": "one seed only; no historical five-run averages",
        "deferred_set_size": "mean over non-singleton sets, including empty sets",
        "versions": {"numpy": version("numpy"), "matplotlib": version("matplotlib")},
    }
    save_manifest(output_directory, manifest)
    print(f"Comparison saved to {output_directory}")


if __name__ == "__main__":
    run_comparison()
