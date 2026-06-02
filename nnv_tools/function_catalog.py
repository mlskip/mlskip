from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nnv_tools.metadata_paths import functions_dir


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    expression: str


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    description: str
    database: str
    table: str
    task_type: str
    target_expression: str
    features: list[FeatureSpec]
    num_classes: int | None = None

    @property
    def target_name(self) -> str:
        return self.name


def load_function_catalog(database: str, path: str | Path | None = None) -> list[FunctionSpec]:
    if path is not None:
        catalog_paths = [Path(path)]
    else:
        catalog_paths = sorted(functions_dir(database).glob("*.json"))
        if not catalog_paths:
            raise FileNotFoundError(
                f"No function metadata found for database '{database}' in "
                f"{functions_dir(database)}"
            )

    specs: list[FunctionSpec] = []
    for catalog_path in catalog_paths:
        raw_specs = json.loads(catalog_path.read_text())
        specs.extend(_parse_function_spec(database, item) for item in raw_specs)
    return specs


def get_function_specs(
    database: str,
    selected_names: list[str] | None,
    path: str | Path | None = None,
) -> list[FunctionSpec]:
    specs = load_function_catalog(database, path)
    if not selected_names:
        return specs

    requested = set(selected_names)
    selected = [spec for spec in specs if spec.name in requested]
    found = {spec.name for spec in selected}
    missing = requested - found
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Unknown function name(s): {missing_str}")
    return selected


def _parse_function_spec(database: str, raw: dict) -> FunctionSpec:
    features = [FeatureSpec(**feature) for feature in raw["features"]]
    return FunctionSpec(
        name=raw["name"],
        description=raw["description"],
        database=database,
        table=raw["table"],
        target_expression=raw["target_expression"],
        features=features,
        task_type=raw["task_type"],
        num_classes=raw.get("num_classes"),
    )
