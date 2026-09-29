import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from llm_scorekit import CalibratedPolicy, calibrate, load_policy, save_policy


@pytest.mark.parametrize(
    ("method", "regularization", "expected_threshold"),
    [
        ("lac", None, 0.875),
        ("aps", None, 0.9375),
        ("socop", 0.5, 32 / 7),
    ],
)
def test_calibrate_uses_the_selected_method(
    method: Literal["lac", "aps", "socop"],
    regularization: float | None,
    expected_threshold: float,
) -> None:
    calibration_probabilities = np.array(
        [
            [0.3125, 0.0625, 0.5, 0.125],
            [0.3125, 0.0625, 0.5, 0.125],
            [0.3125, 0.0625, 0.5, 0.125],
            [0.3125, 0.0625, 0.5, 0.125],
        ]
    )
    calibration_labels = np.array([0, 1, 2, 3])
    class_names = ("returns", "delivery", "billing", "account")

    policy = calibrate(
        calibration_probabilities,
        calibration_labels,
        class_names=class_names,
        method=method,
        alpha=0.4,
        regularization=regularization,
    )
    evaluation_probabilities = np.array([[0.3125, 0.0625, 0.5, 0.125]])
    decisions = policy.predict(evaluation_probabilities)

    assert policy.method == method
    assert policy.alpha == 0.4
    assert policy.threshold == pytest.approx(expected_threshold)
    assert policy.class_names == class_names
    assert policy.regularization == regularization
    expected_sets = np.array([[True, False, True, True]])
    np.testing.assert_array_equal(decisions.prediction_sets, expected_sets)


def test_predict_returns_class_indices_only_for_singletons() -> None:
    policy = CalibratedPolicy(
        method="lac",
        alpha=0.2,
        threshold=0.6,
        class_names=("returns", "billing", "delivery"),
    )
    probabilities = np.array(
        [
            [0.10, 0.80, 0.10],
            [0.45, 0.45, 0.10],
            [0.34, 0.33, 0.33],
            [0.10, 0.10, 0.80],
        ]
    )

    decisions = policy.predict(probabilities)

    expected_acceptance = np.array([True, False, False, True])
    np.testing.assert_array_equal(decisions.accepted, expected_acceptance)
    assert decisions.predictions == [1, None, None, 2]


@pytest.mark.parametrize("threshold", [0.75, float("inf")])
def test_save_and_load_preserve_policy_and_decisions(
    tmp_path: Path,
    threshold: float,
) -> None:
    policy = CalibratedPolicy(
        method="socop",
        alpha=0.1,
        threshold=threshold,
        class_names=("returns", "billing", "delivery"),
        regularization=0.5,
    )
    probabilities = np.array([[0.8, 0.1, 0.1], [0.45, 0.45, 0.1]])
    expected_decisions = policy.predict(probabilities)
    policy_path = tmp_path / "policy.json"

    save_policy(policy, policy_path)
    restored_policy = load_policy(policy_path)
    restored_decisions = restored_policy.predict(probabilities)

    assert restored_policy == policy
    np.testing.assert_array_equal(
        restored_decisions.prediction_sets,
        expected_decisions.prediction_sets,
    )
    np.testing.assert_array_equal(restored_decisions.accepted, expected_decisions.accepted)
    assert restored_decisions.predictions == expected_decisions.predictions

    # Infinity is a valid conformal threshold, but not a JSON number.
    if np.isinf(threshold):
        metadata = json.loads(policy_path.read_text())
        assert metadata["threshold"] == "inf"
