from collections.abc import Sequence
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.pipeline import Pipeline

from conformal_selective_prediction.data import load_banking77
from conformal_selective_prediction.models import fit_tfidf_classifier

from ._artifacts import save_manifest, save_prepared_seed
from ._data import PreparedData, predict_partition, split_calibration_pool


DATA_DIRECTORY = Path("data/raw/banking77")
OUTPUT_DIRECTORY = Path("outputs/banking77/score_comparison/prepared")
DATASET_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
RANDOM_SEEDS = (7, 21, 42, 84, 123)
MIS_COVERAGE_RATES = (0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
CONFIDENCE_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SOCOP_REGULARIZATIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
SOCOP_REFERENCE_REGULARIZATION = 0.25
CONFIDENCE_LEVEL = 0.95


# Identify the actual local files as well as the documented source revision.
def _dataset_hashes(data_directory: Path) -> dict[str, str]:
    hashes = {}

    for filename in ("train.csv", "test.csv", "categories.json"):
        file_path = data_directory / filename
        file_contents = file_path.read_bytes()
        hashes[filename] = sha256(file_contents).hexdigest()

    return hashes


# Record the baseline's explicit settings from the fitted pipeline.
def _model_settings(model: Pipeline) -> dict[str, object]:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    return {
        "tfidf": {"ngram_range": vectorizer.ngram_range},
        "logistic_regression": {
            "C": classifier.C,
            "l1_ratio": classifier.l1_ratio,
            "solver": classifier.solver,
            "max_iter": classifier.max_iter,
        },
    }


# Fit the historical model split once and predict each held-out partition.
def prepare_seed(
    data_directory: Path,
    random_seed: int,
) -> tuple[PreparedData, dict[str, object]]:
    data = load_banking77(data_directory, random_seed=random_seed)
    tuning_a, tuning_b, calibration = split_calibration_pool(
        data.calibration, random_seed
    )
    model = fit_tfidf_classifier(data.train)

    # Scores use label IDs as probability-column indices.
    expected_classes = np.arange(len(data.class_names))
    if not np.array_equal(model.classes_, expected_classes):
        raise ValueError("model classes must match the dataset label indices")

    tuning_a_predictions = predict_partition(model, tuning_a)
    tuning_b_predictions = predict_partition(model, tuning_b)
    calibration_predictions = predict_partition(model, calibration)
    test_predictions = predict_partition(model, data.test)

    prepared_data = PreparedData(
        train_ids=data.train.sample_ids,
        class_names=data.class_names,
        tuning_a=tuning_a_predictions,
        tuning_b=tuning_b_predictions,
        calibration=calibration_predictions,
        test=test_predictions,
    )
    model_settings = _model_settings(model)

    return prepared_data, model_settings


# Prepare shared inputs only; no score selection or test evaluation happens here.
def prepare_study(
    data_directory: Path,
    output_directory: Path,
    random_seeds: Sequence[int] = RANDOM_SEEDS,
) -> None:
    if not random_seeds:
        raise ValueError("random_seeds must contain at least one seed")

    output_directory.mkdir(parents=True, exist_ok=True)
    preparation_id = uuid4().hex
    dataset_hashes = _dataset_hashes(data_directory)
    sample_counts = {}

    for random_seed in random_seeds:
        data, model_settings = prepare_seed(data_directory, random_seed)
        save_prepared_seed(output_directory, random_seed, preparation_id, data)
        sample_counts[str(random_seed)] = {
            "train": data.train_ids.size,
            "tuning_a": data.tuning_a.labels.size,
            "tuning_b": data.tuning_b.labels.size,
            "calibration": data.calibration.labels.size,
            "test": data.test.labels.size,
        }
        print(f"Prepared seed {random_seed}")

    manifest = {
        "preparation_id": preparation_id,
        "data_directory": str(data_directory),
        "dataset_source_revision": DATASET_REVISION,
        "dataset_sha256": dataset_hashes,
        "class_names": data.class_names,
        "random_seeds": list(random_seeds),
        "alphas": MIS_COVERAGE_RATES,
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        "socop_regularizations": SOCOP_REGULARIZATIONS,
        "socop_reference_regularization": SOCOP_REFERENCE_REGULARIZATION,
        "confidence_level": CONFIDENCE_LEVEL,
        "interval_method": "Wilson (pointwise, per split)",
        "split_protocol": {
            "model_training_fraction": 0.75,
            "tuning_fraction": 0.125,
            "final_calibration_fraction": 0.125,
            "stratification": "intent",
            "split_random_state": "run seed for both tuning splits",
            "test": "official split, unchanged",
        },
        "socop_tuning": {
            "scope": "separately for each seed and alpha",
            "validation": "swap calibration and selection roles of tuning halves",
            "objective": "highest combined held-out singleton rate",
            "tie_breaks": ["smaller average set size", "smaller regularization"],
        },
        "model_settings": model_settings,
        "sample_counts": sample_counts,
        "package_versions": {
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
        },
    }

    # Publish the manifest only after all seed archives are ready.
    save_manifest(output_directory, manifest)
    print(f"Prepared predictions saved to {output_directory}")


def main() -> None:
    prepare_study(DATA_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
