from pathlib import Path

from ..banking77_scores.lac import run_experiment as evaluate_lac
from ._reference import save_reference, validate_reference


PREPARED_DIRECTORY = Path("outputs/banking77/representation_comparison/encoder/prepared")
OUTPUT_DIRECTORY = Path("outputs/banking77/representation_comparison/encoder/lac")


# Calibrate encoder LAC independently and retain the compatible TF-IDF reference.
def run_experiment(prepared_directory: Path, output_directory: Path) -> None:
    reference = validate_reference(prepared_directory, "lac")
    evaluate_lac(prepared_directory, output_directory, include_naive=False)
    save_reference(output_directory, reference)
    print(f"Reusing TF-IDF LAC results from {reference['directory']}")


def main() -> None:
    run_experiment(PREPARED_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
