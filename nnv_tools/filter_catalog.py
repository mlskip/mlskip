from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from nnv_tools.metadata_paths import filters_dir


@dataclass(frozen=True)
class FilterSpec:
    name: str
    description: str
    database: str
    table: str
    model_name: str
    sql_predicate: str
    filter_type: str
    predicate_lower: float | None = None
    predicate_upper: float | None = None
    target_class: int | None = None
    sampled_width: float | None = None
    sampled_start: float | None = None
    template_name: str | None = None
    block_metadata: dict | None = None


def load_filter_catalog(
    database: str,
    path: str | Path | Sequence[str | Path] | None = None,
) -> list[FilterSpec]:
    catalog_paths = _resolve_catalog_paths(database, path)

    specs: list[FilterSpec] = []
    for catalog_path in catalog_paths:
        raw_specs = json.loads(catalog_path.read_text())
        specs.extend(_parse_filter_spec(database, item) for item in raw_specs)
    return specs


def get_filter_specs(
    database: str,
    selected_names: list[str] | None,
    path: str | Path | Sequence[str | Path] | None = None,
) -> list[FilterSpec]:
    specs = load_filter_catalog(database, path)
    if not selected_names:
        return specs

    requested = set(selected_names)
    selected = [spec for spec in specs if spec.name in requested]
    found = {spec.name for spec in selected}
    missing = requested - found
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Unknown filter name(s): {missing_str}")
    return selected


def write_filter_specs(path: str | Path, specs: list[FilterSpec]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for index, spec in enumerate(specs, start=1):
        raw = asdict(spec)
        raw.pop("database", None)
        filtered = {key: value for key, value in raw.items() if value is not None}
        filtered["filter_id"] = index
        payload.append(filtered)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def _resolve_catalog_paths(
    database: str,
    path: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    if path is None:
        catalog_paths = sorted(filters_dir(database).glob("*.json"))
    elif isinstance(path, (str, Path)):
        raw_path = Path(path)
        catalog_paths = sorted(raw_path.glob("*.json")) if raw_path.is_dir() else [raw_path]
    else:
        catalog_paths = [Path(item) for item in path]

    if not catalog_paths:
        raise FileNotFoundError(
            f"No filter metadata found for database '{database}' in "
            f"{filters_dir(database)}"
        )
    return catalog_paths


def _parse_filter_spec(database: str, raw: dict) -> FilterSpec:
    return FilterSpec(
        name=raw["name"],
        description=raw["description"],
        database=database,
        table=raw["table"],
        model_name=raw["model_name"],
        sql_predicate=raw["sql_predicate"],
        filter_type=raw["filter_type"],
        predicate_lower=raw.get("predicate_lower"),
        predicate_upper=raw.get("predicate_upper"),
        target_class=raw.get("target_class"),
        sampled_width=raw.get("sampled_width"),
        sampled_start=raw.get("sampled_start"),
        template_name=raw.get("template_name"),
        block_metadata=raw.get("block_metadata"),
    )
