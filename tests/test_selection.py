import numpy as np

from llm_scorekit import (
    automated_error_rate,
    automation_rate,
    singleton_mask,
)


def test_singleton_mask_selects_only_single_class_sets() -> None:
    prediction_sets = np.array(
        [
            [False, True, False],
            [False, False, False],
            [True, False, True],
            [False, False, True],
        ]
    )

    selected_samples = singleton_mask(prediction_sets)

    expected_selection = np.array([True, False, False, True])
    np.testing.assert_array_equal(selected_samples, expected_selection)


def test_selective_metrics_measure_automation_and_automated_error() -> None:
    predictions = np.array([1, 0, 2, 0])
    labels = np.array([1, 2, 0, 0])
    automation_mask = np.array([True, False, True, True])

    selected_rate = automation_rate(automation_mask)
    selected_error = automated_error_rate(predictions, labels, automation_mask)

    assert selected_rate == 0.75
    assert selected_error == 1.0 / 3.0


def test_automated_error_rate_is_undefined_without_automated_samples() -> None:
    predictions = np.array([0, 1])
    labels = np.array([0, 1])
    automation_mask = np.array([False, False])

    selected_error = automated_error_rate(predictions, labels, automation_mask)

    assert np.isnan(selected_error)
