from importlib.metadata import version
from pathlib import Path

from ._artifacts import save_csv, save_manifest
from ._comparison_data import load_results, summarize_metrics
from ._plots import save_automation_figure, save_coverage_figure, save_set_size_figure


STUDY_DIRECTORY = Path("outputs/banking77/score_comparison")
OUTPUT_DIRECTORY = STUDY_DIRECTORY / "comparison"


# Compare saved experiments without fitting, recalibrating or selecting new settings.
def run_comparison(study_directory: Path, output_directory: Path) -> None:
    manifest, conformal_rows, naive_rows = load_results(study_directory)
    conformal_summary = summarize_metrics(conformal_rows, "alpha")
    naive_summary = summarize_metrics(naive_rows, "confidence_threshold")

    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    save_csv(output_directory / "conformal_summary.csv", conformal_summary)
    save_csv(output_directory / "naive_summary.csv", naive_summary)

    # The naive baseline appears once, even though each experiment saved its own copy.
    all_rows = conformal_rows + naive_rows
    all_summary = conformal_summary + naive_summary
    figure_methods = {
        "aps_vs_naive.png": ("aps", "naive"),
        "socop_vs_naive.png": ("socop_fixed", "socop_tuned", "naive"),
        "automation_vs_error.png": ("lac", "aps", "socop_fixed", "socop_tuned", "naive"),
    }
    for filename, methods in figure_methods.items():
        save_automation_figure(
            all_rows, all_summary, methods, figure_directory / filename
        )
    save_coverage_figure(
        conformal_rows, conformal_summary, figure_directory / "coverage_vs_target.png"
    )
    save_set_size_figure(
        conformal_rows, conformal_summary, figure_directory / "set_size_vs_coverage.png"
    )

    result_manifest = manifest.copy()
    result_manifest["methods"] = ["lac", "aps", "socop_fixed", "socop_tuned", "naive"]
    result_manifest["study_directory"] = str(study_directory)
    result_manifest["aggregation"] = "arithmetic mean of per-split metrics; no pooled counts"
    result_manifest["ranges"] = "per-split minimum and maximum, not confidence intervals"
    result_manifest["comparison_versions"] = {
        "numpy": version("numpy"),
        "matplotlib": version("matplotlib"),
    }
    save_manifest(output_directory, result_manifest)
    print(f"Comparison saved to {output_directory}")


def main() -> None:
    run_comparison(STUDY_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
