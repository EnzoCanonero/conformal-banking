from importlib.metadata import version
from pathlib import Path

from ..banking77_scores._artifacts import load_manifest, save_csv, save_manifest
from ..banking77_scores._comparison_data import summarize_metrics
from ._comparison_data import load_results, summarize_results
from ._naive import evaluate_naive
from ._plots import (
    save_automation_figure,
    save_coverage_figure,
    save_encoder_automation_figure,
)


STUDY_DIRECTORY = Path("outputs/banking77/representation_comparison")
ENCODER_DIRECTORY = STUDY_DIRECTORY / "encoder"
OUTPUT_DIRECTORY = STUDY_DIRECTORY / "comparison"


# Reuse conformal results and compare naive routing on the cached encoder predictions.
def run_comparison(encoder_directory: Path, output_directory: Path) -> None:
    manifest, rows = load_results(encoder_directory)
    summary_rows = summarize_results(rows)

    # Keep the earlier confidence grid; do not choose a cutoff on the test results.
    prepared_manifest = manifest["encoder_preparation"]
    tfidf_prepared_directory = Path(prepared_manifest["tfidf_reference"]["prepared_directory"])
    tfidf_manifest = load_manifest(tfidf_prepared_directory)
    confidence_thresholds = tfidf_manifest["confidence_thresholds"]
    naive_rows, archive_hashes = evaluate_naive(
        encoder_directory / "prepared", prepared_manifest, confidence_thresholds
    )
    naive_summary = summarize_metrics(naive_rows, "confidence_threshold")
    for summary in naive_summary:
        summary["representation"] = "encoder"

    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    save_csv(output_directory / "metrics_summary.csv", summary_rows)
    save_csv(output_directory / "encoder_naive_metrics.csv", naive_rows)
    save_csv(output_directory / "encoder_naive_summary.csv", naive_summary)
    for method, filename_suffix in (("lac", "lac"), ("socop_tuned", "socop")):
        automation_path = figure_directory / f"automation_vs_error_{filename_suffix}.png"
        coverage_path = figure_directory / f"coverage_vs_target_{filename_suffix}.png"
        save_automation_figure(rows, summary_rows, method, automation_path)
        save_coverage_figure(rows, summary_rows, method, coverage_path)

    save_encoder_automation_figure(
        rows + naive_rows,
        summary_rows + naive_summary,
        figure_directory / "encoder_automation_vs_error.png",
    )

    manifest["representations"] = ["tfidf", "encoder"]
    manifest["methods"] = ["lac", "socop_tuned"]
    manifest["aggregation"] = "arithmetic mean of per-split metrics; no pooled counts"
    manifest["ranges"] = "per-split minimum and maximum, not confidence intervals"
    manifest["deferred_set_size"] = "mean size among non-singleton sets, including empty sets"
    manifest["encoder_policy_comparison"] = {
        "methods": ["lac", "socop_tuned", "naive"],
        "confidence_thresholds": confidence_thresholds,
        "confidence_threshold_source": str(tfidf_prepared_directory / "manifest.json"),
        "prepared_directory": str(encoder_directory / "prepared"),
        "prepared_archives_sha256": archive_hashes,
    }
    manifest["comparison_versions"] = {
        "numpy": version("numpy"),
        "matplotlib": version("matplotlib"),
    }
    save_manifest(output_directory, manifest)
    print(f"Comparison saved to {output_directory}")


def main() -> None:
    run_comparison(ENCODER_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
