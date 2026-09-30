from pathlib import Path, PurePosixPath
from tarfile import open as open_tar
from zipfile import ZipFile


# Verify the small package payload and the files needed to rebuild it.
distribution_directory = Path("dist")
wheel_path = next(distribution_directory.glob("*.whl"))
source_path = next(distribution_directory.glob("*.tar.gz"))

with ZipFile(wheel_path) as wheel:
    wheel_files = set(wheel.namelist())

assert "llm_scorekit/__init__.py" in wheel_files
assert "llm_scorekit/py.typed" in wheel_files
assert any(name.endswith(".dist-info/licenses/LICENSE") for name in wheel_files)

for name in wheel_files:
    root = PurePosixPath(name).parts[0]
    assert root == "llm_scorekit" or root.endswith(".dist-info"), name

with open_tar(source_path) as source:
    source_files = set()
    for member in source.getmembers():
        relative_path = member.name.partition("/")[2]
        source_files.add(relative_path)

required_source_files = {
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "MANIFEST.in",
    "constraints/studies.txt",
    "src/llm_scorekit/__init__.py",
    "src/llm_scorekit/py.typed",
}
assert required_source_files <= source_files

excluded_paths = {
    "data", "docs", "examples", "notebooks", "outputs", "__pycache__",
    "CODING_RULES.MD", "IMPLEMENTATION_PLAN.md",
}
for name in wheel_files | source_files:
    path_parts = set(PurePosixPath(name).parts)
    assert not path_parts.intersection(excluded_paths), name

print("Distribution contents verified.")
