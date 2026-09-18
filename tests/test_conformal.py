import math

import numpy as np

from conformal_selective_prediction import (
    aps_prediction_sets,
    aps_scores,
    average_set_size,
    binomial_confidence_interval,
    conformal_quantile,
    empirical_coverage,
    lac_prediction_sets,
    lac_scores,
    socop_prediction_sets,
    socop_scores,
)


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    scores = np.array([0.4, 1.0, 0.2, 0.8, 0.1, 0.6, 0.9, 0.3, 0.7, 0.5])

    threshold = conformal_quantile(scores, alpha=0.15)

    assert threshold == 1.0


def test_conformal_quantile_returns_infinity_for_unavailable_rank() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    threshold = conformal_quantile(scores, alpha=0.1)

    assert math.isinf(threshold)


def test_lac_scores_use_the_true_class_probabilities() -> None:
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.2, 0.5, 0.3],
        ]
    )
    labels = np.array([0, 2, 1])

    scores = lac_scores(probabilities, labels)

    expected_scores = np.array([0.3, 0.4, 0.5])
    np.testing.assert_allclose(scores, expected_scores)


def test_lac_prediction_sets_include_scores_at_the_threshold() -> None:
    probabilities = np.array(
        [
            [0.75, 0.50, 0.25],
            [0.625, 0.375, 0.00],
        ]
    )

    prediction_sets = lac_prediction_sets(probabilities, threshold=0.5)

    expected_sets = np.array(
        [
            [True, True, False],
            [True, False, False],
        ]
    )
    np.testing.assert_array_equal(prediction_sets, expected_sets)


def test_aps_scores_accumulate_in_rank_order_with_stable_ties() -> None:
    probabilities = np.array(
        [
            [0.25, 0.50, 0.25],
            [0.25, 0.50, 0.25],
            [0.25, 0.50, 0.25],
        ]
    )
    labels = np.array([0, 1, 2])

    scores = aps_scores(probabilities, labels)

    expected_scores = np.array([0.75, 0.50, 1.00])
    np.testing.assert_allclose(scores, expected_scores)


def test_aps_prediction_sets_use_the_inclusive_score_threshold() -> None:
    probabilities = np.array(
        [
            [0.25, 0.50, 0.25],
            [0.875, 0.125, 0.00],
            [0.25, 0.125, 0.625],
        ]
    )

    prediction_sets = aps_prediction_sets(probabilities, threshold=0.75)

    expected_sets = np.array(
        [
            [True, True, False],
            [False, False, False],
            [False, False, True],
        ]
    )
    np.testing.assert_array_equal(prediction_sets, expected_sets)


def test_socop_scores_follow_hull_slopes_with_stable_ties() -> None:
    probabilities = np.array(
        [
            [0.3125, 0.0625, 0.5, 0.125],
            [0.3125, 0.0625, 0.5, 0.125],
            [0.3125, 0.0625, 0.5, 0.125],
            [0.3125, 0.0625, 0.5, 0.125],
        ]
    )
    labels = np.array([0, 1, 2, 3])

    scores = socop_scores(probabilities, labels, regularization=0.25)

    # The hull skips size two, so the second and third ranks share a score.
    expected_scores = np.array([24 / 7, 4.0, 0.5, 24 / 7])
    np.testing.assert_allclose(scores, expected_scores)

    higher_regularization_scores = socop_scores(
        probabilities,
        labels,
        regularization=0.5,
    )
    expected_higher_regularization_scores = np.array([32 / 7, 8.0, 1.0, 32 / 7])
    np.testing.assert_allclose(
        higher_regularization_scores,
        expected_higher_regularization_scores,
    )

    tied_probabilities = np.array([[0.5, 0.5], [0.5, 0.5]])
    tied_labels = np.array([0, 1])
    tied_scores = socop_scores(
        tied_probabilities,
        tied_labels,
        regularization=0.25,
    )

    expected_tied_scores = np.array([0.5, 2.5])
    np.testing.assert_allclose(tied_scores, expected_tied_scores)


def test_socop_prediction_sets_include_thresholds_and_remain_nested() -> None:
    probabilities = np.array([[0.3125, 0.0625, 0.5, 0.125]])
    thresholds = [0.25, 0.5, 24 / 7, 4.0, float("inf")]
    expected_sets = [
        [False, False, False, False],
        [False, False, True, False],
        [True, False, True, True],
        [True, True, True, True],
        [True, True, True, True],
    ]
    previous_sets = np.zeros(probabilities.shape, dtype=np.bool_)

    for threshold, expected_set in zip(thresholds, expected_sets, strict=True):
        prediction_sets = socop_prediction_sets(
            probabilities,
            threshold,
            regularization=0.25,
        )

        np.testing.assert_array_equal(prediction_sets[0], expected_set)
        removed_labels = previous_sets & ~prediction_sets
        assert not np.any(removed_labels)
        previous_sets = prediction_sets


def test_prediction_set_metrics_measure_coverage_and_size() -> None:
    prediction_sets = np.array(
        [
            [True, False, False],
            [False, True, True],
            [False, False, True],
            [True, False, True],
        ]
    )
    labels = np.array([0, 0, 2, 1])

    coverage = empirical_coverage(prediction_sets, labels)
    average_size = average_set_size(prediction_sets)

    assert coverage == 0.5
    assert average_size == 1.5


def test_lac_aps_and_socop_coverage_across_iid_simulations() -> None:
    number_of_simulations = 30
    number_of_calibration_samples = 1_000
    number_of_test_samples = 5_000
    number_of_classes = 3
    alpha = 0.1
    lower_target_alpha = 0.2
    socop_regularization = 0.25

    number_of_samples = number_of_calibration_samples + number_of_test_samples
    class_concentration = np.ones(number_of_classes)
    lac_coverages: list[float] = []
    aps_coverages: list[float] = []
    aps_lower_target_coverages: list[float] = []
    socop_coverages: list[float] = []
    socop_lower_target_coverages: list[float] = []

    for seed in range(number_of_simulations):
        random_generator = np.random.default_rng(seed)

        probabilities = random_generator.dirichlet(
            class_concentration,
            size=number_of_samples,
        )
        sampled_classes = random_generator.multinomial(1, probabilities)
        labels = np.argmax(sampled_classes, axis=1)

        calibration_probabilities = probabilities[:number_of_calibration_samples]
        calibration_labels = labels[:number_of_calibration_samples]
        test_probabilities = probabilities[number_of_calibration_samples:]
        test_labels = labels[number_of_calibration_samples:]

        calibration_scores = lac_scores(
            calibration_probabilities,
            calibration_labels,
        )
        threshold = conformal_quantile(calibration_scores, alpha)
        prediction_sets = lac_prediction_sets(test_probabilities, threshold)
        coverage = empirical_coverage(prediction_sets, test_labels)
        lac_coverages.append(coverage)

        # APS is calibrated separately, using exactly the same simulated requests.
        aps_calibration_scores = aps_scores(
            calibration_probabilities,
            calibration_labels,
        )
        aps_threshold = conformal_quantile(aps_calibration_scores, alpha)
        aps_sets = aps_prediction_sets(test_probabilities, aps_threshold)
        aps_coverage = empirical_coverage(aps_sets, test_labels)
        aps_coverages.append(aps_coverage)

        lower_target_threshold = conformal_quantile(
            aps_calibration_scores,
            lower_target_alpha,
        )
        lower_target_sets = aps_prediction_sets(
            test_probabilities,
            lower_target_threshold,
        )
        lower_target_coverage = empirical_coverage(lower_target_sets, test_labels)
        aps_lower_target_coverages.append(lower_target_coverage)

        # Raising the coverage target from 80% to 90% must not remove any label.
        removed_labels = lower_target_sets & ~aps_sets
        assert not np.any(removed_labels)

        # SOCOP uses its own calibrated scores with fixed regularization.
        socop_calibration_scores = socop_scores(
            calibration_probabilities,
            calibration_labels,
            regularization=socop_regularization,
        )
        socop_threshold = conformal_quantile(socop_calibration_scores, alpha)
        socop_sets = socop_prediction_sets(
            test_probabilities,
            socop_threshold,
            regularization=socop_regularization,
        )
        socop_coverage = empirical_coverage(socop_sets, test_labels)
        socop_coverages.append(socop_coverage)

        socop_lower_target_threshold = conformal_quantile(
            socop_calibration_scores,
            lower_target_alpha,
        )
        socop_lower_target_sets = socop_prediction_sets(
            test_probabilities,
            socop_lower_target_threshold,
            regularization=socop_regularization,
        )
        socop_lower_target_coverage = empirical_coverage(
            socop_lower_target_sets,
            test_labels,
        )
        socop_lower_target_coverages.append(socop_lower_target_coverage)

        socop_removed_labels = socop_lower_target_sets & ~socop_sets
        assert not np.any(socop_removed_labels)

    average_coverage = float(np.mean(lac_coverages))
    target_coverage = 1.0 - alpha
    coverage_tolerance = 0.01
    coverage_difference = abs(average_coverage - target_coverage)

    assert coverage_difference <= coverage_tolerance

    # Non-randomized APS may exceed its target; check for undercoverage only.
    average_aps_coverage = float(np.mean(aps_coverages))
    average_lower_target_coverage = float(np.mean(aps_lower_target_coverages))
    lower_coverage_target = 1.0 - lower_target_alpha

    assert average_aps_coverage >= target_coverage - coverage_tolerance
    assert average_lower_target_coverage >= lower_coverage_target - coverage_tolerance

    average_socop_coverage = float(np.mean(socop_coverages))
    average_socop_lower_target_coverage = float(
        np.mean(socop_lower_target_coverages)
    )

    assert average_socop_coverage >= target_coverage - coverage_tolerance
    assert (
        average_socop_lower_target_coverage
        >= lower_coverage_target - coverage_tolerance
    )


def test_binomial_interval_handles_observed_and_empty_counts() -> None:
    interval = binomial_confidence_interval(successes=50, total=100)
    expected_interval = (0.4038315303659956, 0.5961684696340044)

    np.testing.assert_allclose(interval, expected_interval)

    lower_bound, upper_bound = binomial_confidence_interval(successes=0, total=100)

    assert lower_bound == 0.0
    assert upper_bound > 0.0

    empty_interval = binomial_confidence_interval(successes=0, total=0)

    assert all(math.isnan(bound) for bound in empty_interval)
