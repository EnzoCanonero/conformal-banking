import json
from math import isfinite
from pathlib import Path

from .policies import CalibratedPolicy


FORMAT_VERSION = 1


# Save calibrated settings as JSON, without model weights or calibration data.
def save_policy(policy: CalibratedPolicy, path: str | Path) -> None:
    threshold: float | str = float(policy.threshold)
    if threshold == float("inf"):
        threshold = "inf"

    regularization = policy.regularization
    if regularization is not None:
        regularization = float(regularization)

    metadata = {
        "format_version": FORMAT_VERSION,
        "method": policy.method,
        "alpha": float(policy.alpha),
        "threshold": threshold,
        "regularization": regularization,
        "class_names": list(policy.class_names),
    }

    # Serialize before writing so invalid values do not truncate an existing file.
    content = json.dumps(metadata, indent=2, allow_nan=False)
    policy_path = Path(path)
    policy_path.write_text(content + "\n", encoding="utf-8")


# Restore a policy; the caller must keep the same model and scoring procedure.
def load_policy(path: str | Path) -> CalibratedPolicy:
    policy_path = Path(path)
    content = policy_path.read_text(encoding="utf-8")
    metadata = json.loads(content)

    if not isinstance(metadata, dict):
        raise ValueError("policy file must contain a JSON object")

    format_version = metadata["format_version"]
    if type(format_version) is not int or format_version != FORMAT_VERSION:
        raise ValueError("unsupported policy format version")

    method = metadata["method"]
    if method not in ("lac", "aps", "socop"):
        raise ValueError("method must be 'lac', 'aps' or 'socop'")

    alpha = metadata["alpha"]
    if type(alpha) not in (int, float) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be a number between 0 and 1")

    # JSON has no infinity value; only the explicit string represents this case.
    stored_threshold = metadata["threshold"]
    if stored_threshold == "inf":
        threshold = float("inf")
    else:
        if type(stored_threshold) not in (int, float):
            raise ValueError("threshold must be a finite number or 'inf'")
        threshold = float(stored_threshold)
        if not isfinite(threshold):
            raise ValueError("threshold must be a finite number or 'inf'")

    class_names = metadata["class_names"]
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("class_names must be a nonempty list")
    if not all(isinstance(name, str) for name in class_names):
        raise ValueError("class_names must contain strings")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique")

    regularization = metadata["regularization"]
    if method == "socop":
        if type(regularization) not in (int, float):
            raise ValueError("SOCOP requires numeric regularization")
        if not isfinite(regularization) or regularization <= 0.0:
            raise ValueError("regularization must be finite and greater than zero")
        regularization = float(regularization)
    elif regularization is not None:
        raise ValueError("regularization is only used by SOCOP")

    return CalibratedPolicy(
        method=method,
        alpha=float(alpha),
        threshold=threshold,
        class_names=tuple(class_names),
        regularization=regularization,
    )
