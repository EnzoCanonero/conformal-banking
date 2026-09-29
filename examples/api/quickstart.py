from pathlib import Path

import numpy as np

from conformal_selective_prediction import (
    automated_error_rate,
    automation_rate,
    average_set_size,
    calibrate,
    empirical_coverage,
    load_policy,
    save_policy,
)


# Demonstrate the public workflow with hand-written weights, not a model benchmark.
def main() -> None:
    class_names = ("billing", "delivery", "returns")

    # Columns follow class_names; each label is the correct column's integer index.
    calibration_probabilities = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.75, 0.15],
            [0.10, 0.20, 0.70],
            [0.65, 0.20, 0.15],
            [0.25, 0.60, 0.15],
            [0.20, 0.25, 0.55],
            [0.50, 0.30, 0.20],
            [0.30, 0.45, 0.25],
            [0.35, 0.25, 0.40],
            [0.35, 0.40, 0.25],
        ]
    )
    calibration_labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    policy = calibrate(
        calibration_probabilities,
        calibration_labels,
        class_names=class_names,
        method="lac",
        alpha=0.20,
    )

    policy_path = Path("outputs/quickstart/policy.json")
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    save_policy(policy, policy_path)
    restored_policy = load_policy(policy_path)

    # These separate examples illustrate singleton, multi-label and empty sets.
    evaluation_probabilities = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.45, 0.45, 0.10],
            [0.34, 0.33, 0.33],
            [0.10, 0.75, 0.15],
            [0.15, 0.20, 0.65],
        ]
    )
    evaluation_labels = np.array([0, 1, 2, 2, 2])
    decisions = restored_policy.predict(evaluation_probabilities)

    print(f"Policy saved to {policy_path}")
    for sample_index, prediction in enumerate(decisions.predictions):
        included_indices = np.flatnonzero(decisions.prediction_sets[sample_index])
        included_names = [restored_policy.class_names[index] for index in included_indices]

        if prediction is None:
            decision = "defer"
        else:
            predicted_name = restored_policy.class_names[prediction]
            decision = f"accept {predicted_name}"

        true_class = evaluation_labels[sample_index]
        true_name = restored_policy.class_names[true_class]
        print(
            f"Request {sample_index + 1}: {included_names} -> {decision}; "
            f"true={true_name}"
        )

    # Evaluation labels measure correctness; they do not affect the routing decisions.
    coverage = empirical_coverage(decisions.prediction_sets, evaluation_labels)
    automated_fraction = automation_rate(decisions.accepted)
    automated_error = automated_error_rate(
        decisions.predictions,
        evaluation_labels,
        decisions.accepted,
    )
    mean_set_size = average_set_size(decisions.prediction_sets)

    print(f"Coverage: {coverage:.1%}")
    print(f"Automation: {automated_fraction:.1%}")
    print(f"Automated-case error: {automated_error:.1%}")
    print(f"Average set size: {mean_set_size:.2f}")
    print("Hand-written examples illustrate the API, not expected performance.")


if __name__ == "__main__":
    main()
