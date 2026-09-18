from functools import partial

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal_selective_prediction import (
    aps_prediction_sets,
    aps_scores,
    automated_error_rate,
    automation_rate,
    average_set_size,
    conformal_quantile,
    empirical_coverage,
    lac_prediction_sets,
    lac_scores,
    singleton_mask,
    socop_prediction_sets,
    socop_scores,
)


RANDOM_SEED = 42
MIS_COVERAGE_RATE = 0.1
SOCOP_REGULARIZATION = 0.25
SOCOP_REGULARIZATIONS = (0.05, SOCOP_REGULARIZATION, 1.0)


# Compare LAC, APS and SOCOP using the same synthetic data and fitted classifier.
def run_experiment() -> dict[str, dict[str, float]]:

    # Generate synthetic classification data.
    features, labels = make_classification(
        n_samples=8_000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.0,
        flip_y=0.05,
        random_state=RANDOM_SEED,
    )

    # Split the data into training, calibration, and test sets.
    train_features, rest_features, train_labels, rest_labels = train_test_split(
        features,
        labels,
        test_size=0.5,
        random_state=RANDOM_SEED,
    )

    cal_features, test_features, cal_labels, test_labels = train_test_split(
        rest_features,
        rest_labels,
        test_size=0.5,
        random_state=RANDOM_SEED,
    )

    # Train the classifier.
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500),
    )
    classifier.fit(train_features, train_labels)

    # All methods use the same probabilities; only the conformal score changes.
    calibration_probabilities = classifier.predict_proba(cal_features)
    test_probabilities = classifier.predict_proba(test_features)
    predicted_class_indices = test_probabilities.argmax(axis=1)
    test_predictions = classifier.classes_[predicted_class_indices]

    methods = [
        ("LAC", lac_scores, lac_prediction_sets),
        ("APS", aps_scores, aps_prediction_sets),
    ]

    # Compare fixed SOCOP settings, keeping each value the same for scores and sets.
    for regularization in SOCOP_REGULARIZATIONS:
        socop_score_function = partial(socop_scores, regularization=regularization)
        socop_set_function = partial(
            socop_prediction_sets,
            regularization=regularization,
        )
        method_name = f"SOCOP (lambda={regularization})"
        methods.append((method_name, socop_score_function, socop_set_function))

    results: dict[str, dict[str, float]] = {}

    for method_name, score_function, set_function in methods:
        # Each score needs its own threshold, learned from calibration data only.
        calibration_scores = score_function(calibration_probabilities, cal_labels)
        threshold = conformal_quantile(calibration_scores, MIS_COVERAGE_RATE)
        prediction_sets = set_function(test_probabilities, threshold)

        coverage = empirical_coverage(prediction_sets, test_labels)
        mean_set_size = average_set_size(prediction_sets)

        # Only one-label sets are automated; empty and multi-label sets are deferred.
        automation_mask = singleton_mask(prediction_sets)
        automated_fraction = automation_rate(automation_mask)
        automated_error = automated_error_rate(
            test_predictions,
            test_labels,
            automation_mask,
        )

        results[method_name] = {
            "coverage": coverage,
            "average_set_size": mean_set_size,
            "automation_rate": automated_fraction,
            "automated_error_rate": automated_error,
        }

    return results


def main() -> None:
    target_coverage = 1.0 - MIS_COVERAGE_RATE

    results = run_experiment()
    print(f"Target coverage: {target_coverage:.3f}")
    print(f"SOCOP central regularization: {SOCOP_REGULARIZATION}")

    for method_name, metrics in results.items():
        coverage = metrics["coverage"]
        mean_set_size = metrics["average_set_size"]
        automated_fraction = metrics["automation_rate"]
        automated_error = metrics["automated_error_rate"]

        print(f"\n{method_name}")
        print(f"Empirical coverage: {coverage:.3f}")
        print(f"Average set size: {mean_set_size:.3f}")
        print(f"Automation rate: {automated_fraction:.3f}")
        print(f"Automated-case error rate: {automated_error:.3f}")


if __name__ == "__main__":
    main()
