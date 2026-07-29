from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


METADATA_TYPE_ORDER = ["none", "minmax", "convex_hull", "grid", "bounded_convex_hull"]
METADATA_TYPE_LABELS = {
    "none": "none",
    "minmax": "min-max",
    "convex_hull": "convex-hull",
    "grid": "grid",
    "bounded_convex_hull": "bounded convex-hull",
}
BACKEND_ORDER = ["pytorch", "marabou", "geomcad", "batched_geomcad"]
BACKEND_LABELS = {"pytorch": "PyTorch", "marabou": "Marabou", "geomcad": "ML-QL", "batched_geomcad": "ML-QL (batched)"}
DEFAULT_RESULT_GLOB = "*/*/*/*/*.json"


@dataclass(frozen=True)
class BenchmarkPlotSelection:
    results_root: str
    benchmark_rows: int
    benchmark_block_size: int
    model_kind: str | None
    datasets: list[str]
    filter_template_names: list[str]
    available_filter_templates: list[str]
    selection_label: str
    selected_files: list[str]
    rows: list[dict[str, Any]]


def latest_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def block_size_from_path(path: Path) -> int | None:
    match = re.search(r"__bs(\d+)__", path.name)
    return int(match.group(1)) if match else None


def metadata_type_from_path(path: Path) -> str:
    match = re.search(r"__bm(.+?)__", path.name)
    if match:
        return match.group(1)

    payload = json.loads(path.read_text())
    params = payload.get("command", {}).get("parameters", {})
    value = params.get("block_metadata")
    return str(value) if value else "unknown"


def verifier_backend_from_path(path: Path) -> str:
    match = re.search(r"__vb(.+?)__", path.name)
    if match:
        return match.group(1)

    payload = json.loads(path.read_text())
    params = payload.get("command", {}).get("parameters", {})
    value = params.get("verifier_backend")
    return str(value) if value else "unknown"


def execution_mode_from_path(path: Path) -> str:
    match = re.search(r"__em(.+?)__", path.name)
    if match:
        return match.group(1)

    payload = json.loads(path.read_text())
    value = payload.get("benchmark_mode")
    if value:
        return str(value)

    params = payload.get("command", {}).get("parameters", {})
    measure_e2e = params.get("measure_e2e")
    if measure_e2e is True:
        return "e2e_per_block"
    if measure_e2e is False:
        return "verification_only"
    return "verification_only"


def file_benchmark_rows(path: Path) -> int | None:
    payload = json.loads(path.read_text())
    rows = payload.get("results", [])
    if not rows:
        return None
    value = rows[0].get("benchmark_rows")
    return int(value) if value is not None else None


def dataset_from_path(path: Path, results_root: Path) -> str:
    relative_parts = path.relative_to(results_root).parts
    if not relative_parts:
        raise ValueError(f"Unexpected benchmark result path: {path}")
    return relative_parts[0]


def filter_template_name_from_path(path: Path, results_root: Path) -> str:
    relative_parts = path.relative_to(results_root).parts
    if len(relative_parts) < 2:
        raise ValueError(f"Unexpected benchmark result path: {path}")
    return relative_parts[1]


def model_kind_from_path(path: Path, results_root: Path) -> str:
    relative_parts = path.relative_to(results_root).parts
    if len(relative_parts) < 3:
        raise ValueError(f"Unexpected benchmark result path: {path}")
    return relative_parts[2]


def metadata_type_label(metadata_type: str) -> str:
    return METADATA_TYPE_LABELS.get(metadata_type, metadata_type.replace("_", " "))


def backend_label(backend: str) -> str:
    return BACKEND_LABELS.get(backend, backend)


def series_label(backend: str, metadata_type: str) -> str:
    return f"{backend_label(backend)} ({metadata_type_label(metadata_type)})"


def ordered_series_keys(values: set[tuple[str, str]] | list[tuple[str, str]]) -> list[tuple[str, str]]:
    present = set(values)
    ordered = []
    for backend in BACKEND_ORDER:
        for metadata_type in METADATA_TYPE_ORDER:
            key = (backend, metadata_type)
            if key in present:
                ordered.append(key)
    extras = sorted(key for key in present if key not in ordered)
    return ordered + extras


def selection_label(
    datasets: list[str],
    filter_template_names: list[str],
    selected_filter_templates: list[str] | None,
) -> str:
    dataset_label = ", ".join(datasets)
    if selected_filter_templates is None:
        return f"{dataset_label} ({len(filter_template_names)} filter templates)"
    if len(selected_filter_templates) == 1:
        return selected_filter_templates[0]
    return f"{dataset_label} ({len(selected_filter_templates)} selected filter templates)"


def _as_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def load_rows(path: Path, results_root: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    block_size = block_size_from_path(path)
    metadata_type = metadata_type_from_path(path)
    backend = verifier_backend_from_path(path)
    dataset = dataset_from_path(path, results_root)
    filter_template_name = filter_template_name_from_path(path, results_root)
    model_kind = model_kind_from_path(path, results_root)

    rows = payload.get("results", [])
    if not rows:
        raise ValueError(f"No 'results' entries in {path}")

    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        selectivity = float(enriched.get("query_selectivity_pct", 0.0) or 0.0)
        skipped = float(enriched.get("skipped_blocks", 0.0) or 0.0)
        skippable = float(enriched.get("skippable_blocks", 0.0) or 0.0)
        total_blocks = float(enriched.get("total_blocks", 0.0) or 0.0)
        raw_skipping_ms = _as_float_or_nan(enriched.get("skipping_ms", float("nan")))
        calculated_pruning = 100.0 * skipped / skippable if skippable > 0 else float("nan")
        reported_pruning = float(enriched.get("pruning_effectiveness_pct", float("nan")))
        pruning = reported_pruning if reported_pruning == reported_pruning else calculated_pruning
        skipping_ms_per_block = (
            raw_skipping_ms / total_blocks
            if raw_skipping_ms == raw_skipping_ms and total_blocks > 0
            else float("nan")
        )

        enriched["_selectivity"] = selectivity
        enriched["_pruning"] = pruning
        enriched["_skipping_ms_per_block"] = skipping_ms_per_block
        enriched["_block_size"] = block_size
        enriched["_metadata_type"] = metadata_type
        enriched["_backend"] = backend
        enriched["_dataset"] = dataset
        enriched["_filter_template_name"] = filter_template_name
        enriched["_model_kind"] = model_kind
        enriched["_source_file"] = path.name
        enriched_rows.append(enriched)

    return enriched_rows


def collect_benchmark_plot_data(
    *,
    results_root: Path,
    benchmark_rows: int,
    benchmark_block_size: int,
    benchmark_datasets: list[str],
    model_kind: str | None = None,
    filter_template_names: list[str] | None = None,
    verifier_backends: list[str] | None = None,
    metadata_types: list[str] | None = None,
    execution_modes: list[str] | None = None,
    result_glob: str = DEFAULT_RESULT_GLOB,
) -> BenchmarkPlotSelection:
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    result_files = sorted(results_root.glob(result_glob))
    if not result_files:
        raise FileNotFoundError(
            f"No JSON files found under {results_root} (expected: {result_glob})."
        )

    requested_datasets = list(dict.fromkeys(benchmark_datasets))
    candidate_files = [
        path for path in result_files if dataset_from_path(path, results_root) in requested_datasets
    ]
    matching_files = [
        path
        for path in candidate_files
        if file_benchmark_rows(path) == benchmark_rows
        and block_size_from_path(path) == benchmark_block_size
        and (model_kind is None or model_kind_from_path(path, results_root) == model_kind)
        and (verifier_backends is None or verifier_backend_from_path(path) in verifier_backends)
        and (metadata_types is None or metadata_type_from_path(path) in metadata_types)
        and (execution_modes is None or execution_mode_from_path(path) in execution_modes)
    ]

    if not matching_files:
        raise FileNotFoundError(
            "Could not find benchmark JSON files for "
            f"datasets={requested_datasets!r}, benchmark_rows={benchmark_rows}, "
            f"block_size={benchmark_block_size}, and model_kind={model_kind!r}."
        )

    known_metadata_files = [
        path for path in matching_files if metadata_type_from_path(path) != "unknown"
    ]
    if known_metadata_files:
        matching_files = known_metadata_files

    available_filter_templates = sorted(
        {filter_template_name_from_path(path, results_root) for path in matching_files}
    )

    selected_filter_templates = None
    if filter_template_names is None:
        requested_filter_templates = available_filter_templates
    else:
        selected_filter_templates = list(dict.fromkeys(filter_template_names))
        missing_filter_templates = [
            name for name in selected_filter_templates if name not in available_filter_templates
        ]
        if missing_filter_templates:
            raise FileNotFoundError(
                "No benchmark JSON files found for "
                f"filter_template_names={missing_filter_templates!r} within "
                f"datasets={requested_datasets!r}, benchmark_rows={benchmark_rows}, "
                f"and block_size={benchmark_block_size}."
            )
        requested_filter_templates = selected_filter_templates

    selected_files: list[Path] = []
    for filter_template_name in requested_filter_templates:
        template_files = [
            path
            for path in matching_files
            if filter_template_name_from_path(path, results_root) == filter_template_name
        ]
        series_keys = ordered_series_keys(
            {
                (verifier_backend_from_path(path), metadata_type_from_path(path))
                for path in template_files
            }
        )
        for backend, metadata_type in series_keys:
            chosen = latest_file(
                [
                    path
                    for path in template_files
                    if verifier_backend_from_path(path) == backend
                    and metadata_type_from_path(path) == metadata_type
                ]
            )
            if chosen is not None:
                selected_files.append(chosen)

    if not selected_files:
        raise FileNotFoundError(
            "Could not choose benchmark files for backend/metadata comparison."
        )

    rows: list[dict[str, Any]] = []
    for path in selected_files:
        rows.extend(load_rows(path, results_root))

    return BenchmarkPlotSelection(
        results_root=str(results_root),
        benchmark_rows=benchmark_rows,
        benchmark_block_size=benchmark_block_size,
        model_kind=model_kind,
        datasets=requested_datasets,
        filter_template_names=requested_filter_templates,
        available_filter_templates=available_filter_templates,
        selection_label=selection_label(
            requested_datasets,
            requested_filter_templates,
            selected_filter_templates,
        ),
        selected_files=[str(path) for path in selected_files],
        rows=rows,
    )


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",")]
    return [item for item in values if item]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect latest benchmark rows for plotting notebooks."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results") / "benchmarks",
        help="Benchmark results root. Default: results/benchmarks",
    )
    parser.add_argument(
        "--benchmark-rows",
        type=int,
        required=True,
        help="Fixed benchmark row count to collect.",
    )
    parser.add_argument(
        "--benchmark-block-size",
        type=int,
        required=True,
        help="Fixed benchmark block size to collect.",
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset names, for example: tpch,tpcds",
    )
    parser.add_argument(
        "--model-kind",
        choices=["shallow", "deep"],
        help="Optional model kind to restrict the selection.",
    )
    parser.add_argument(
        "--filter-template-names",
        help="Optional comma-separated filter template names to restrict the selection.",
    )
    parser.add_argument(
        "--verifier-backends",
        help="Optional comma-separated verifier backends.",
    )
    parser.add_argument(
        "--metadata-types",
        help="Optional comma-separated metadata types.",
    )
    parser.add_argument(
        "--execution-modes",
        help="Optional comma-separated execution modes, for example: verification_only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. If omitted, print JSON to stdout.",
    )
    args = parser.parse_args()

    selection = collect_benchmark_plot_data(
        results_root=args.results_root.resolve(),
        benchmark_rows=int(args.benchmark_rows),
        benchmark_block_size=int(args.benchmark_block_size),
        benchmark_datasets=_parse_csv_list(args.datasets) or [],
        model_kind=args.model_kind,
        filter_template_names=_parse_csv_list(args.filter_template_names),
        verifier_backends=_parse_csv_list(args.verifier_backends),
        metadata_types=_parse_csv_list(args.metadata_types),
        execution_modes=_parse_csv_list(args.execution_modes),
    )
    payload = asdict(selection)
    output_text = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(output_text, end="")
        return
    args.output.write_text(output_text)


if __name__ == "__main__":
    main()
