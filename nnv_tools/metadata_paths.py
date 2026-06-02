from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_ROOT = REPO_ROOT / "metadata"
RESULTS_ROOT = REPO_ROOT / "results"


def setup_path(database: str) -> Path:
    return METADATA_ROOT / "setup" / f"{database}.json"


def functions_dir(database: str) -> Path:
    return METADATA_ROOT / "functions" / database


def filters_dir(database: str) -> Path:
    return METADATA_ROOT / "filters" / database


def generated_filters_dir(database: str) -> Path:
    return filters_dir(database) / "generated"


def benchmark_results_dir(database: str) -> Path:
    return RESULTS_ROOT / "benchmarks" / database


def models_dir(database: str) -> Path:
    return METADATA_ROOT / "models" / database


def compiled_models_dir(database: str) -> Path:
    return METADATA_ROOT / "compiled-models" / database
