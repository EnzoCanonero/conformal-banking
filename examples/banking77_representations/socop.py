from pathlib import Path

from ..banking77_scores.socop import run_experiment as evaluate_socop
from ._reference import save_reference, validate_reference


PREPARED_DIRECTORY = Path("outputs/banking77/representation_comparison/encoder/prepared")
OUTPUT_DIRECTORY = Path("outputs/banking77/representation_comparison/encoder/socop")


# Retune encoder SOCOP on its tuning halves, then calibrate on the reserved data.
def run_experiment(prepared_directory: Path, output_directory: Path) -> None:
    reference = validate_reference(prepared_directory, "socop")
    evaluate_socop(
        prepared_directory, output_directory, include_naive=False, include_fixed=False
    )
    save_reference(output_directory, reference)
    print(f"Reusing tuned TF-IDF SOCOP results from {reference['directory']}")


def main() -> None:
    run_experiment(PREPARED_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
