from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import math
import json
import multiprocessing
import queue
import re
import sys
import time
import statistics
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import duckdb
import numpy as np

from nnv_tools.database_setup import load_database_setup
from nnv_tools.dataset_duckdb import (
    build_block_id_predicate,
    count_block_predicate_matches,
    count_rows_for_blocks,
    ensure_row_id_column,
    fetch_block_features,
    fetch_features_for_blocks,
    fetch_expression_range,
    list_block_ids,
    run_count_query,
)
from nnv_tools.filter_catalog import FilterSpec, get_filter_specs, write_filter_specs
from nnv_tools.function_catalog import FunctionSpec, get_function_specs
from nnv_tools.block_metadata import (
    BlockMetadata,
    BlockMetadataBundle,
    BlockMetadataKind,
    collect_block_metadata,
    grid_cell_rect,
)
from nnv_tools.block_verifier import (
    BlockSkipResult,
    BlockVerificationRequest,
    decide_block_skip,
    format_verifier_status,
    supported_verifier_backends,
)
from nnv_tools.geomcad_verify import decide_block_skips_batched as decide_geomcad_block_skips_batched
from nnv_tools.model_ground_truth import (
    ModelGroundTruthCache,
    ModelGroundTruthRequest,
    count_model_qualified_rows,
    ensure_model_ground_truth_caches,
)
from nnv_tools.metadata_paths import (
    benchmark_results_dir,
    filters_dir,
    generated_filters_dir,
    compiled_models_dir,
    models_dir,
)
from nnv_tools.model_runtime import model_onnx_path, predict_array_pytorch, predict_row


_GENERATED_FILTER_TOTAL_BUDGET = 150
_GENERATED_FILTER_MIN_PER_WIDTH = 5
_SELECTIVITY_BOX_EDGES = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class BenchmarkResult:
    filter_name: str
    filter_template_name: str
    verifier_backend: str
    model_kind: str
    block_metadata: str
    grid_depth: int | None
    baseline_count: int
    baseline_ms: float
    model_ground_truth_cache_path: str
    model_ground_truth_cache_key: str
    model_ground_truth_reused: bool
    udf_count: int | None
    udf_ms: float | None
    block_model_count: int
    e2e_execution_backend: str | None
    e2e_execution_mode: str | None
    e2e_count: int | None
    e2e_data_loading_ms: float | None
    e2e_inference_ms: float | None
    e2e_total_ms: float | None
    kept_blocks: int
    skipped_blocks: int
    timeout_blocks: int
    error_blocks: int
    total_blocks: int
    scanned_rows: int
    benchmark_rows: int
    matching_blocks: int
    skippable_blocks: int
    query_selectivity_pct: float | None
    pruning_effectiveness_pct: float | None
    metadata_collection_ms: float | None
    metadata_pair_count: int | None
    skipping_ms: float | None
    ground_truth_match_udf: bool | None
    ground_truth_match_block_model: bool
    ground_truth_match_e2e: bool | None


@dataclass(frozen=True)
class BlockEvaluation:
    block_id: int
    row_id_start: int
    row_id_end: int
    feature_bounds: dict[str, tuple[float, float]]
    block_metadata: BlockMetadata | None
    verifier_result: BlockSkipResult | None
    block_row_count: int | None
    matching_rows: int | None
    udf_count: int | None
    udf_ms: float | None

    @property
    def should_skip(self) -> bool | None:
        if self.verifier_result is None:
            return None
        return self.verifier_result.should_skip


@dataclass(frozen=True)
class BenchmarkJob:
    filter_id: int
    filter_spec: FilterSpec
    model_spec: FunctionSpec
    model_path: Path
    block_ids: list[int]
    excluded_training_blocks: int


@dataclass(frozen=True)
class SkipSummary:
    kept_blocks: list[int]
    timeout_blocks: int
    error_blocks: int
    skipping_ms: float


@dataclass(frozen=True)
class ProgressState:
    job_index: int
    total_jobs: int
    filter_id: int
    filter_name: str
    verifier_backend: str
    current: int
    total: int
    skipped: int
    kept: int
    block_id: int | None
    status: str
    done: bool


_LAST_PROGRESS_RENDERED_LINES = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark baseline SQL against model UDFs and optional verifier-driven block skipping."
    )
    parser.add_argument("--database", default="tpch")
    parser.add_argument(
        "--model-kind",
        choices=["shallow", "deep"],
        default="deep",
        help="Model architecture family to benchmark.",
    )
    parser.add_argument("--filter", action="append", dest="filters")
    parser.add_argument("--filters-path", type=Path)
    parser.add_argument(
        "--task-type",
        choices=["all", "regressor", "classifier"],
        default="all",
        help="Restrict benchmark filters to models of this task type.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        help=(
            "Write benchmark results under this directory; when exactly one filter result "
            "is produced, a .json file path is also allowed. Defaults to the standard "
            "benchmark results directory."
        ),
    )
    parser.add_argument(
        "--save-generated-filters",
        type=Path,
        help="Write the expanded benchmark filters to JSON so they can be reused for debugging.",
    )
    parser.add_argument(
        "--prepare-filters-only",
        action="store_true",
        help="Materialize expanded filters and exit without running the benchmark.",
    )
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--block-id", type=int)
    parser.add_argument(
        "--filter-id",
        type=int,
        help="Select one prepared filter by 1-based index, useful with --block-id debugging.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of benchmark queries to run in parallel after filter expansion.",
    )
    parser.add_argument(
        "--max-rows-total",
        type=int,
        help=(
            "Cap the benchmark by rows after filter expansion. The internal "
            "block budget is rounded up from this value and --block-size."
        ),
    )
    parser.add_argument(
        "--range-alpha",
        type=float,
        default=2.0,
        help="Width growth factor for sampled regressor ranges (alpha, alpha^2, ...).",
    )
    parser.add_argument(
        "--range-start-samples",
        type=int,
        default=10,
        help="Number of random start values to sample for each regressor range width.",
    )
    parser.add_argument(
        "--range-seed",
        type=int,
        default=0,
        help="Random seed for sampled regressor ranges.",
    )
    parser.add_argument(
        "--run-udf",
        action="store_true",
        help="Also run the model UDF benchmark in addition to block-skipping evaluation.",
    )
    parser.add_argument(
        "--measure-e2e",
        action="store_true",
        help=(
            "Measure end-to-end query time by loading kept blocks and running batched "
            "PyTorch inference to compute COUNT(*) over model matches."
        ),
    )
    parser.add_argument(
        "--disable-skipping",
        action="store_true",
        help="Skip the verifier-driven block-pruning phase.",
    )
    parser.add_argument(
        "--verifier-backend",
        choices=supported_verifier_backends(),
        default="marabou",
        help="Verifier backend to use for block pruning.",
    )
    parser.add_argument(
        "--verifier-timeout-seconds",
        type=float,
        dest="verifier_timeout_seconds",
        default=1.0,
        help=(
            "Per-block verifier timeout in seconds; fractional values are allowed, "
            "and timed out or unknown blocks are kept conservatively."
        ),
    )
    parser.add_argument(
        "--batched-geomcad",
        action="store_true",
        help=(
            "When using --verifier-backend=geomcad with min-max metadata, "
            "run a single batched GeomCAD query across all benchmark blocks."
        ),
    )
    parser.add_argument(
        "--block-metadata",
        choices=["minmax", "convex_hull", "grid", "bounded_convex_hull"],
        help=(
            "Per-block metadata to build before verification. If omitted, "
            "minmax is used."
        ),
    )
    parser.add_argument(
        "--grid-depth",
        type=int,
        help=(
            "Depth for grid-based 2D metadata; depth d means a 2^d by 2^d grid. "
            "Used by both grid and bounded_convex_hull metadata. Defaults to a "
            "filter JSON block_metadata.grid_depth when present, otherwise 4."
        ),
    )
    return parser


def run_benchmarks(args: argparse.Namespace) -> list[dict]:
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if args.verifier_timeout_seconds < 0:
        raise ValueError("--verifier-timeout-seconds must be non-negative.")
    if args.batched_geomcad and args.verifier_backend != "geomcad":
        raise ValueError("--batched-geomcad requires --verifier-backend=geomcad.")
    if args.range_alpha <= 1.0:
        raise ValueError("--range-alpha must be greater than 1.0.")
    if args.range_start_samples <= 0:
        raise ValueError("--range-start-samples must be positive.")
    if args.max_rows_total is not None and args.max_rows_total < 0:
        raise ValueError("--max-rows-total must be non-negative.")
    if args.filter_id is not None and args.filter_id <= 0:
        raise ValueError("--filter-id must be positive.")
    if args.grid_depth is not None and args.grid_depth < 0:
        raise ValueError("--grid-depth must be non-negative.")

    setup = load_database_setup(args.database)
    db_path = args.db_path if args.db_path is not None else setup.duckdb_file
    block_size = args.block_size
    if block_size <= 0:
        raise ValueError("--block-size must be positive.")
    max_blocks_total = _resolve_max_blocks_total(
        max_rows_total=args.max_rows_total,
        block_size=block_size,
    )
    args.effective_max_blocks_total = max_blocks_total
    args.effective_max_rows_total = (
        None if max_blocks_total is None else max_blocks_total * block_size
    )
    training_rows = setup.training_row_count
    training_blocks = setup.training_block_count(block_size)
    filters, resolved_filters_path = _resolve_filter_specs(
        args=args,
    )
    model_specs = {spec.name: spec for spec in get_function_specs(args.database, None, None)}
    filters = _filter_specs_by_task_type(filters, model_specs, args.task_type)
    total_loaded_filters = len(filters)
    selected_filter_index: int | None = None

    results: list[dict] = []
    print(
        f"[bench] Loading {len(filters)} filter specification(s) from "
        f"{_format_filter_source_label(resolved_filters_path or filters_dir(args.database))}"
    )
    if args.task_type != "all":
        print(f"[bench] Restricted filters to task_type='{args.task_type}'")
    if not filters:
        raise ValueError(f"No filters matched task_type='{args.task_type}'.")
    print(
        f"[bench] Not considering the first {training_rows} row(s) because they are reserved "
        f"for training; with block_size={block_size}, that means skipping the first "
        f"{training_blocks} block(s)"
    )
    if args.max_rows_total is not None:
        print(
            f"[bench] Benchmark row budget {args.max_rows_total} row(s) maps to "
            f"{max_blocks_total} block(s), covering up to "
            f"{args.effective_max_rows_total} row(s) at block_size={block_size}"
        )

    if (
        args.block_id is not None
        and args.filter_id is not None
        and not _filters_require_expansion(filters, model_specs)
    ):
        if args.filter_id > len(filters):
            raise ValueError(
                f"--filter-id={args.filter_id} is out of range; there are {len(filters)} loaded filter(s)."
            )
        selected_filter_index = args.filter_id
        filters = [filters[args.filter_id - 1]]

    jobs = _prepare_benchmark_jobs(
        args=args,
        db_path=db_path,
        block_size=block_size,
        training_blocks=training_blocks,
        max_blocks_total=max_blocks_total,
        filters=filters,
        model_specs=model_specs,
    )
    if args.filter_id is not None and args.block_id is None:
        if args.filter_id > len(jobs):
            raise ValueError(
                f"--filter-id={args.filter_id} is out of range; there are {len(jobs)} prepared filter(s)."
            )
        jobs = [jobs[args.filter_id - 1]]
    args.resolved_block_metadata_label = _results_block_metadata_label(
        cli_block_metadata=args.block_metadata,
        jobs=jobs,
        verifier_backend=args.verifier_backend,
    )
    args.resolved_verifier_backend = _results_verifier_backend_label(
        verifier_backend=args.verifier_backend,
        batched_geomcad=args.batched_geomcad,
    )
    expanded_filters = [job.filter_spec for job in jobs]
    if args.save_generated_filters is not None:
        write_filter_specs(args.save_generated_filters, expanded_filters)
        print(
            f"[bench] Wrote {len(expanded_filters)} expanded filter(s) to "
            f"{args.save_generated_filters}"
        )
    auto_generated_filters_anchor_path = _default_generated_filters_path(
        database=args.database,
        selected_filters=args.filters,
        range_alpha=args.range_alpha,
        range_start_samples=args.range_start_samples,
        range_seed=args.range_seed,
        task_type=args.task_type,
    )
    auto_generated_filter_paths = _default_generated_filters_paths(
        database=args.database,
        template_names=_selected_filter_template_names(expanded_filters, args.task_type),
        range_alpha=args.range_alpha,
        range_start_samples=args.range_start_samples,
        range_seed=args.range_seed,
        task_type=args.task_type,
    )
    if (
        args.filters_path is None
        and args.filter_id is None
        and auto_generated_filter_paths
        and not all(path.exists() for path in auto_generated_filter_paths.values())
    ):
        _write_grouped_filter_specs(auto_generated_filter_paths, expanded_filters)
        print(
            f"[bench] Cached {len(expanded_filters)} expanded filter(s) across "
            f"{len(auto_generated_filter_paths)} template file(s) in "
            f"{generated_filters_dir(args.database)}"
        )
    if args.prepare_filters_only:
        return _format_prepared_filters_for_display(expanded_filters)
    print(f"[bench] Prepared {len(jobs)} benchmark job(s) after filter expansion")
    if not args.disable_skipping and args.verifier_backend == "geomcad":
        if args.model_kind != "shallow":
            raise ValueError(
                "GeomCAD only supports shallow models. "
                f"Rerun with --model-kind shallow, got: {args.model_kind}."
            )
        jobs = _validate_geomcad_jobs(jobs=jobs, database=args.database)
        if not jobs:
            print(
                "[bench] Skipping benchmark run: no GeomCAD-compatible jobs remain "
                "after filtering missing compiled models"
            )
            return []

    model_ground_truth_cache_paths = _ground_truth_cache_paths_for_filters(
        database=args.database,
        jobs=jobs,
        resolved_filters_path=resolved_filters_path,
        saved_generated_filters_path=args.save_generated_filters,
        auto_generated_filter_paths=auto_generated_filter_paths,
        range_alpha=args.range_alpha,
        range_start_samples=args.range_start_samples,
        range_seed=args.range_seed,
        task_type=args.task_type,
    )
    model_ground_truth_caches = _prepare_model_ground_truth_caches(
        jobs=jobs,
        db_path=db_path,
        block_size=block_size,
        cache_paths_by_template=model_ground_truth_cache_paths,
    )
    metadata_bundles = _prepare_metadata_bundles(
        jobs=jobs,
        db_path=db_path,
        block_size=block_size,
        disable_skipping=args.disable_skipping,
        verifier_backend=args.verifier_backend,
        cli_block_metadata=args.block_metadata,
        grid_depth=args.grid_depth,
    )
    args._metadata_size_summary = _metadata_size_summary(metadata_bundles)

    # Inspect one block instead of running the full benchmark.
    if args.block_id is not None:
        return _inspect_block(
            args=args,
            db_path=db_path,
            block_size=block_size,
            training_blocks=training_blocks,
            jobs=jobs,
            total_filter_count=total_loaded_filters,
            selected_filter_index=selected_filter_index,
            model_ground_truth_caches=model_ground_truth_caches,
        )

    grouped_jobs = _group_jobs_by_template(jobs)
    args._results_paths_by_template = _results_paths(
        args,
        [
            {
                "filter_name": job.filter_spec.name,
                "filter_template_name": (job.filter_spec.template_name or job.filter_spec.name),
            }
            for job in jobs
        ],
    )
    if len(grouped_jobs) > 1:
        print(
            f"[bench] Running benchmark jobs grouped by filter template "
            f"({len(grouped_jobs)} template group(s))"
        )

    all_results: list[dict] = []
    metadata_size_summary_by_template: dict[str, dict[str, object]] = {}
    for group_index, (template_name, template_jobs) in enumerate(grouped_jobs, start=1):
        metadata_summary = _metadata_size_summary_for_jobs(
            template_jobs,
            metadata_bundles,
            cli_block_metadata=args.block_metadata,
            grid_depth=args.grid_depth,
        )
        if metadata_summary is not None:
            metadata_size_summary_by_template[template_name] = metadata_summary
        if len(grouped_jobs) > 1:
            print()
            print(
                f"[bench] Template group {group_index}/{len(grouped_jobs)}: "
                f"{template_name} ({len(template_jobs)} filter(s))"
            )
        if args.jobs == 1:
            group_results = _run_jobs_serially(
                jobs=template_jobs,
                db_path=db_path,
                block_size=block_size,
                run_udf=args.run_udf,
                measure_e2e=args.measure_e2e,
                disable_skipping=args.disable_skipping,
                verifier_backend=args.verifier_backend,
                batched_geomcad=args.batched_geomcad,
                verifier_timeout_seconds=args.verifier_timeout_seconds,
                cli_block_metadata=args.block_metadata,
                grid_depth=args.grid_depth,
                metadata_bundles=metadata_bundles,
                model_ground_truth_caches=model_ground_truth_caches,
            )
        else:
            print(f"[bench] Running up to {args.jobs} benchmark querie(s) in parallel")
            group_results = _run_jobs_in_parallel(
                jobs=template_jobs,
                db_path=db_path,
                block_size=block_size,
                run_udf=args.run_udf,
                measure_e2e=args.measure_e2e,
                disable_skipping=args.disable_skipping,
                verifier_backend=args.verifier_backend,
                batched_geomcad=args.batched_geomcad,
                verifier_timeout_seconds=args.verifier_timeout_seconds,
                cli_block_metadata=args.block_metadata,
                grid_depth=args.grid_depth,
                max_workers=args.jobs,
                metadata_bundles=metadata_bundles,
                model_ground_truth_caches=model_ground_truth_caches,
            )
        all_results.extend(group_results)
        args._metadata_size_summary_by_template = metadata_size_summary_by_template
        written_path = _write_results_for_template(args, template_name, group_results)
        if written_path is not None:
            written_paths = getattr(args, "_written_results_paths", [])
            written_paths.append(written_path)
            args._written_results_paths = written_paths
            print(
                f"[bench] Wrote {len(group_results)} result row(s) for template '"
                f"{template_name}' to {written_path}"
            )
    args._metadata_size_summary_by_template = metadata_size_summary_by_template
    return all_results


def _metadata_size_summary_for_jobs(
    jobs: list[BenchmarkJob],
    metadata_bundles: dict[tuple[str, tuple[int, ...], str, int], BlockMetadataBundle],
    *,
    cli_block_metadata: str | None,
    grid_depth: int | None,
) -> dict[str, object] | None:
    if not jobs or not metadata_bundles:
        return None

    relevant_bundles: dict[tuple[str, tuple[int, ...], str, int], BlockMetadataBundle] = {}
    for job in jobs:
        kind = _resolve_block_metadata_kind(cli_block_metadata, job.filter_spec)
        resolved_grid_depth = _resolve_grid_depth(grid_depth, job.filter_spec)
        cache_key = (
            job.model_spec.name,
            tuple(job.block_ids),
            kind,
            resolved_grid_depth,
        )
        bundle = metadata_bundles.get(cache_key)
        if bundle is not None:
            relevant_bundles[cache_key] = bundle
    return _metadata_size_summary(relevant_bundles)


def _inspect_block(
    *,
    args: argparse.Namespace,
    db_path: Path,
    block_size: int,
    training_blocks: int,
    jobs: list[BenchmarkJob],
    total_filter_count: int | None = None,
    selected_filter_index: int | None = None,
    model_ground_truth_caches: dict[int, ModelGroundTruthCache] | None = None,
) -> list[dict]:
    inspected_results: list[dict] = []
    block_id = int(args.block_id)
    row_start = block_id * block_size
    row_end = (block_id + 1) * block_size - 1

    print(
        f"[bench] Inspecting block_id={block_id} "
        f"(row_id range {row_start}-{row_end})"
    )
    if block_id < training_blocks:
        raise ValueError(
            f"Block {block_id} is inside the training prefix "
            f"(hidden blocks: 0-{training_blocks - 1})."
        )

    selected_jobs = jobs
    if args.filter_id is not None and selected_filter_index is None:
        if args.filter_id > len(jobs):
            raise ValueError(
                f"--filter-id={args.filter_id} is out of range; there are {len(jobs)} prepared filter(s)."
            )
        selected_jobs = [jobs[args.filter_id - 1]]

    for job_index, job in enumerate(selected_jobs):
        filter_spec = job.filter_spec
        model_spec = job.model_spec
        onnx_path = job.model_path
        if block_id not in job.block_ids:
            raise ValueError(
                f"Block {block_id} is not available for filter '{filter_spec.name}' "
                f"after excluding training blocks."
            )

        resolved_filter_index = (
            selected_filter_index
            if selected_filter_index is not None
            else args.filter_id
            if args.filter_id is not None
            else job_index + 1
        )
        total_filters = total_filter_count if total_filter_count is not None else len(jobs)
        print(
            f"[bench] Filter {resolved_filter_index}/{total_filters}: "
            f"'{filter_spec.name}' using model '{model_spec.name}'"
        )
        block_feature_con = None
        if args.verifier_backend == "pytorch":
            block_feature_con = duckdb.connect(str(db_path), read_only=True)
        try:
            evaluation = _evaluate_block(
                db_path=db_path,
                block_size=block_size,
                filter_spec=filter_spec,
                model_spec=model_spec,
                model_path=onnx_path,
                block_id=block_id,
                bounds=None,
                block_metadata=None,
                disable_skipping=args.disable_skipping,
                run_udf=args.run_udf,
                verifier_backend=args.verifier_backend,
                verifier_timeout_seconds=args.verifier_timeout_seconds,
                metadata_kind=_resolve_block_metadata_kind(args.block_metadata, filter_spec),
                grid_depth=_resolve_grid_depth(args.grid_depth, filter_spec),
                include_counts=True,
                verbose=True,
                model_ground_truth=(
                    None if model_ground_truth_caches is None else model_ground_truth_caches.get(job.filter_id)
                ),
                block_feature_con=block_feature_con,
            )
        finally:
            if block_feature_con is not None:
                block_feature_con.close()

        inspected = {
            "filter_name": filter_spec.name,
            "filter_template_name": (filter_spec.template_name or filter_spec.name),
            "table": filter_spec.table,
            "block_id": evaluation.block_id,
            "row_id_start": evaluation.row_id_start,
            "row_id_end": evaluation.row_id_end,
            "block_row_count": evaluation.block_row_count,
            "matching_rows": evaluation.matching_rows,
            "feature_bounds": evaluation.feature_bounds,
            "block_metadata": (
                None
                if evaluation.block_metadata is None
                else _metadata_summary(evaluation.block_metadata)
            ),
            "verifier_status": format_verifier_status(evaluation.verifier_result),
            "verifier_backend": (
                None
                if evaluation.verifier_result is None
                else evaluation.verifier_result.backend
            ),
            "verifier_ms": (
                None
                if evaluation.verifier_result is None
                else evaluation.verifier_result.elapsed_ms
            ),
            "should_skip": evaluation.should_skip,
            "udf_count": evaluation.udf_count,
            "udf_ms": evaluation.udf_ms,
        }
        inspected_results.append(inspected)
        print(
            f"[bench] Block {block_id} summary: rows={evaluation.block_row_count} "
            f"matching_rows={evaluation.matching_rows} "
            f"should_skip={evaluation.should_skip}"
        )

    return inspected_results


def _prepare_benchmark_jobs(
    *,
    args: argparse.Namespace,
    db_path: Path,
    block_size: int,
    training_blocks: int,
    max_blocks_total: int | None,
    filters: list[FilterSpec],
    model_specs: dict[str, FunctionSpec],
) -> list[BenchmarkJob]:
    direct_jobs: list[BenchmarkJob] = []
    regressor_templates: dict[tuple[str, str], BenchmarkJob] = {}
    classifier_templates: dict[tuple[str, str], BenchmarkJob] = {}

    for filter_spec in filters:
        model_spec = model_specs[filter_spec.model_name]
        onnx_path = model_onnx_path(
            args.database,
            args.model_kind,
            model_spec.task_type,
            model_spec.table,
            model_spec.name,
        )
        all_block_ids = list_block_ids(filter_spec.table, db_path, block_size)
        benchmark_block_ids = all_block_ids[training_blocks:]
        excluded_training_blocks = len(all_block_ids[:training_blocks])

        if (
            filter_spec.filter_type == "regressor_range"
            and model_spec.task_type == "regressor"
            and filter_spec.sampled_start is None
            and filter_spec.sampled_width is None
        ):
            key = (filter_spec.table, filter_spec.model_name)
            if key not in regressor_templates:
                regressor_templates[key] = BenchmarkJob(
                    filter_id=0,
                    filter_spec=filter_spec,
                    model_spec=model_spec,
                    model_path=onnx_path,
                    block_ids=benchmark_block_ids,
                    excluded_training_blocks=excluded_training_blocks,
                )
            continue
        if (
            filter_spec.filter_type == "classifier_class"
            and model_spec.task_type == "classifier"
            and not _is_generated_classifier_filter_name(
                filter_spec.name, filter_spec.model_name
            )
        ):
            key = (filter_spec.table, filter_spec.model_name)
            if key not in classifier_templates:
                classifier_templates[key] = BenchmarkJob(
                    filter_id=0,
                    filter_spec=filter_spec,
                    model_spec=model_spec,
                    model_path=onnx_path,
                    block_ids=benchmark_block_ids,
                    excluded_training_blocks=excluded_training_blocks,
                )
            continue

        direct_jobs.append(
            BenchmarkJob(
                filter_id=0,
                filter_spec=filter_spec,
                model_spec=model_spec,
                model_path=onnx_path,
                block_ids=benchmark_block_ids,
                excluded_training_blocks=excluded_training_blocks,
            )
        )

    generated_jobs: list[BenchmarkJob] = []
    for template_job in regressor_templates.values():
        sampled_filters = _sample_regressor_filters(
            template=template_job.filter_spec,
            model_spec=template_job.model_spec,
            db_path=db_path,
            block_size=block_size,
            benchmark_block_ids=template_job.block_ids,
            alpha=args.range_alpha,
            start_samples=args.range_start_samples,
        )
        print(
            f"[bench] Expanded regressor '{template_job.model_spec.name}' into "
            f"{len(sampled_filters)} sampled range filter(s)"
        )
        for sampled_filter in sampled_filters:
            generated_jobs.append(
                BenchmarkJob(
                    filter_id=0,
                    filter_spec=sampled_filter,
                    model_spec=template_job.model_spec,
                    model_path=template_job.model_path,
                    block_ids=list(template_job.block_ids),
                    excluded_training_blocks=template_job.excluded_training_blocks,
                )
            )
    for template_job in classifier_templates.values():
        sampled_filters = _sample_classifier_filters(
            template=template_job.filter_spec,
            model_spec=template_job.model_spec,
            db_path=db_path,
            block_size=block_size,
            benchmark_block_ids=template_job.block_ids,
        )
        print(
            f"[bench] Expanded classifier '{template_job.model_spec.name}' into "
            f"{len(sampled_filters)} class filter(s)"
        )
        for sampled_filter in sampled_filters:
            generated_jobs.append(
                BenchmarkJob(
                    filter_id=0,
                    filter_spec=sampled_filter,
                    model_spec=template_job.model_spec,
                    model_path=template_job.model_path,
                    block_ids=list(template_job.block_ids),
                    excluded_training_blocks=template_job.excluded_training_blocks,
                )
            )

    jobs = [
        BenchmarkJob(
            filter_id=index,
            filter_spec=job.filter_spec,
            model_spec=job.model_spec,
            model_path=job.model_path,
            block_ids=job.block_ids,
            excluded_training_blocks=job.excluded_training_blocks,
        )
        for index, job in enumerate(direct_jobs + generated_jobs, start=1)
    ]
    return _apply_block_budget(jobs, max_blocks_total)



def _validate_geomcad_jobs(*, jobs: list[BenchmarkJob], database: str) -> list[BenchmarkJob]:
    non_regressor_models = sorted({
        job.model_spec.name
        for job in jobs
        if job.model_spec.task_type != "regressor"
    })
    if non_regressor_models:
        joined = ", ".join(non_regressor_models)
        raise ValueError(
            "GeomCAD only supports regressor models. "
            "Rerun with --task-type regressor or restrict the selected filters. "
            f"Unsupported model(s): {joined}"
        )

    from nnv_tools.geomcad_verify import geomcad_compiled_model_path

    models_root = Path(__file__).resolve().parent / "metadata" / "models"
    compiled_root = Path(__file__).resolve().parent / "metadata" / "compiled-models"
    missing_by_model_path: dict[Path, str] = {}
    supported_model_paths: set[Path] = set()
    for job in jobs:
        if job.model_path in missing_by_model_path or job.model_path in supported_model_paths:
            continue
        try:
            geomcad_compiled_model_path(job.model_path)
        except FileNotFoundError:
            compiled_path = (compiled_root / job.model_path.relative_to(models_root)).with_suffix(
                ".geomcad.db"
            )
            missing_by_model_path[job.model_path] = f"{job.model_spec.name}: {compiled_path}"
        else:
            supported_model_paths.add(job.model_path)

    if not missing_by_model_path:
        return jobs

    filtered_jobs = [job for job in jobs if job.model_path not in missing_by_model_path]
    removed_jobs = len(jobs) - len(filtered_jobs)
    missing_artifacts = list(missing_by_model_path.values())
    preview = "\n".join(f"  - {item}" for item in missing_artifacts[:5])
    suffix = ""
    if len(missing_artifacts) > 5:
        suffix = f"\n  ... and {len(missing_artifacts) - 5} more"
    print(
        "[bench] Skipping GeomCAD jobs with missing compiled models under "
        f"{compiled_models_dir(database)}: removed {removed_jobs} job(s) across "
        f"{len(missing_artifacts)} model(s). Run `scripts/compile_geomcad_models.sh {database}` "
        "to include them.\n"
        f"{preview}{suffix}"
    )
    return filtered_jobs


def _filters_require_expansion(
    filters: list[FilterSpec],
    model_specs: dict[str, FunctionSpec],
) -> bool:
    for filter_spec in filters:
        model_spec = model_specs[filter_spec.model_name]
        if (
            filter_spec.filter_type == "regressor_range"
            and model_spec.task_type == "regressor"
            and filter_spec.sampled_start is None
            and filter_spec.sampled_width is None
        ):
            return True
        if (
            filter_spec.filter_type == "classifier_class"
            and model_spec.task_type == "classifier"
            and not _is_generated_classifier_filter_name(
                filter_spec.name, filter_spec.model_name
            )
        ):
            return True
    return False


def _inferred_task_type_for_filter(filter_spec: FilterSpec) -> str | None:
    if filter_spec.filter_type == "regressor_range":
        return "regressor"
    if filter_spec.filter_type == "classifier_class":
        return "classifier"
    return None



def _filter_specs_by_task_type(
    filters: list[FilterSpec],
    model_specs: dict[str, FunctionSpec],
    task_type: str,
) -> list[FilterSpec]:
    missing_model_filters = [
        filter_spec
        for filter_spec in filters
        if filter_spec.model_name not in model_specs
        and (
            task_type == "all"
            or _inferred_task_type_for_filter(filter_spec) in {None, task_type}
        )
    ]
    if missing_model_filters:
        missing_descriptions = ", ".join(
            f"{filter_spec.name} -> {filter_spec.model_name}"
            for filter_spec in missing_model_filters
        )
        raise ValueError(
            "Filter metadata references unknown model(s): "
            f"{missing_descriptions}. "
            "Your filter catalog and trained model/function metadata are out of sync; "
            "please retrain the models (for example `uv run python train.py --database "
            f"{filters[0].database}`) and regenerate any derived filter metadata before benchmarking."
        )
    known_filters = [
        filter_spec for filter_spec in filters if filter_spec.model_name in model_specs
    ]
    if task_type == "all":
        return known_filters
    return [
        filter_spec
        for filter_spec in known_filters
        if model_specs[filter_spec.model_name].task_type == task_type
    ]


def _resolve_filter_specs(
    *,
    args: argparse.Namespace,
) -> tuple[list[FilterSpec], Path | tuple[Path, ...] | None]:
    if args.filters_path is not None:
        return (
            get_filter_specs(args.database, args.filters, args.filters_path),
            Path(args.filters_path),
        )

    requested_filters = get_filter_specs(args.database, args.filters, None)
    auto_generated_filter_paths = _default_generated_filters_paths(
        database=args.database,
        template_names=_selected_filter_template_names(requested_filters, args.task_type),
        range_alpha=args.range_alpha,
        range_start_samples=args.range_start_samples,
        range_seed=args.range_seed,
        task_type=args.task_type,
    )
    if auto_generated_filter_paths and all(
        path.exists() for path in auto_generated_filter_paths.values()
    ):
        cached_paths = tuple(auto_generated_filter_paths.values())
        print(
            "[bench] Reusing cached expanded filters from "
            f"{_format_filter_source_label(cached_paths)}"
        )
        return get_filter_specs(args.database, None, list(cached_paths)), cached_paths

    return requested_filters, None


def _default_generated_filters_path(
    *,
    database: str,
    selected_filters: list[str] | None,
    range_alpha: float,
    range_start_samples: int,
    range_seed: int,
    task_type: str,
) -> Path:
    selected = sorted(selected_filters or [task_type if task_type != "all" else "all"])
    base_label = selected[0] if len(selected) == 1 else f"{selected[0]}_and_{len(selected) - 1}_more"
    digest_input = json.dumps(
        {
            "database": database,
            "selected_filters": selected,
            "task_type": task_type,
            "range_alpha": range_alpha,
            "range_start_samples": range_start_samples,
            "range_seed": range_seed,
            "sort_order": "query_selectivity_pct_v1",
            "sampling": "deterministic_v2_template_grouping",
            "classifier_sampling": "distinct_target_class_v2_template_grouping",
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
    safe_label = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in base_label
    ).strip("_")
    filename = (
        f"{safe_label or 'filters'}"
        f"__a{range_alpha:g}"
        f"__n{range_start_samples}"
        f"__s{range_seed}"
        f"__{digest}.json"
    )
    return generated_filters_dir(database) / filename


def _default_generated_filters_paths(
    *,
    database: str,
    template_names: list[str],
    range_alpha: float,
    range_start_samples: int,
    range_seed: int,
    task_type: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for template_name in sorted(set(template_names)):
        digest_input = json.dumps(
            {
                "database": database,
                "template_name": template_name,
                "task_type": task_type,
                "range_alpha": range_alpha,
                "range_start_samples": range_start_samples,
                "range_seed": range_seed,
                "sort_order": "query_selectivity_pct_v1",
                "sampling": "deterministic_v3_per_template",
                "classifier_sampling": "distinct_target_class_v2_template_grouping",
            },
            sort_keys=True,
        )
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
        safe_label = _sanitize_results_label(template_name) or "filters"
        filename = (
            f"{safe_label}"
            f"__a{range_alpha:g}"
            f"__n{range_start_samples}"
            f"__s{range_seed}"
            f"__{digest}.json"
        )
        paths[template_name] = generated_filters_dir(database) / filename
    return paths


def _selected_filter_template_names(
    filters: list[FilterSpec],
    task_type: str,
) -> list[str]:
    template_names: set[str] = set()
    for filter_spec in filters:
        inferred_task_type = _inferred_task_type_for_filter(filter_spec)
        if task_type != "all" and inferred_task_type not in {None, task_type}:
            continue
        template_names.add(filter_spec.template_name or filter_spec.name)
    return sorted(template_names)


def _write_grouped_filter_specs(
    paths_by_template: dict[str, Path],
    specs: list[FilterSpec],
) -> None:
    grouped_specs: dict[str, list[FilterSpec]] = {name: [] for name in paths_by_template}
    for spec in specs:
        template_name = spec.template_name or spec.name
        if template_name in grouped_specs:
            grouped_specs[template_name].append(spec)
    for template_name, output_path in paths_by_template.items():
        write_filter_specs(output_path, grouped_specs.get(template_name, []))


def _format_filter_source_label(
    source: Path | tuple[Path, ...],
) -> str:
    if isinstance(source, tuple):
        if len(source) == 1:
            return str(source[0])
        return f"{source[0].parent} ({len(source)} files)"
    return str(source)


def _ground_truth_cache_paths_for_filters(
    *,
    database: str,
    jobs: list[BenchmarkJob],
    resolved_filters_path: Path | tuple[Path, ...] | None,
    saved_generated_filters_path: Path | None,
    auto_generated_filter_paths: dict[str, Path],
    range_alpha: float,
    range_start_samples: int,
    range_seed: int,
    task_type: str,
) -> dict[str, Path]:
    template_names = sorted({job.filter_spec.template_name or job.filter_spec.name for job in jobs})
    if not template_names:
        return {}

    if auto_generated_filter_paths:
        json_paths = {
            template_name: auto_generated_filter_paths[template_name]
            for template_name in template_names
            if template_name in auto_generated_filter_paths
        }
    else:
        json_paths = _default_generated_filters_paths(
            database=database,
            template_names=template_names,
            range_alpha=range_alpha,
            range_start_samples=range_start_samples,
            range_seed=range_seed,
            task_type=task_type,
        )

    if saved_generated_filters_path is not None and len(template_names) == 1:
        template_name = template_names[0]
        json_paths[template_name] = saved_generated_filters_path
    elif isinstance(resolved_filters_path, Path) and len(template_names) == 1:
        template_name = template_names[0]
        json_paths[template_name] = resolved_filters_path

    return {
        template_name: path.with_name(f"{path.stem}__model_ground_truth.duckdb")
        for template_name, path in json_paths.items()
    }


def _prepare_model_ground_truth_caches(
    *,
    jobs: list[BenchmarkJob],
    db_path: Path,
    block_size: int,
    cache_paths_by_template: dict[str, Path],
) -> dict[int, ModelGroundTruthCache]:
    if not jobs:
        return {}

    grouped_requests: dict[str, list[ModelGroundTruthRequest]] = {}
    for job in jobs:
        template_name = job.filter_spec.template_name or job.filter_spec.name
        grouped_requests.setdefault(template_name, []).append(
            ModelGroundTruthRequest(
                request_id=job.filter_id,
                filter_spec=job.filter_spec,
                model_spec=job.model_spec,
                model_path=job.model_path,
                block_ids=job.block_ids,
            )
        )

    total_requests = sum(len(requests) for requests in grouped_requests.values())
    print(
        f"[bench] Preparing model ground truth for {total_requests} filter(s); "
        "filters sharing a model/block set reuse one prediction pass"
    )

    caches: dict[int, ModelGroundTruthCache] = {}
    cache_names: list[str] = []
    for template_name, requests in grouped_requests.items():
        cache_path = cache_paths_by_template.get(template_name)
        if cache_path is None:
            raise ValueError(
                f"Missing model ground truth cache path for filter template '{template_name}'."
            )
        grouped_caches = ensure_model_ground_truth_caches(
            db_path=db_path,
            requests=requests,
            block_size=block_size,
            cache_path=cache_path,
        )
        caches.update(grouped_caches)
        cache_names.append(cache_path.name)

    reused = sum(1 for cache in caches.values() if cache.reused)
    materialized = len(caches) - reused
    print(
        f"[bench] Model ground truth ready: reused={reused} "
        f"materialized={materialized} caches={', '.join(sorted(cache_names))}"
    )
    return caches


def _prepare_metadata_bundles(
    *,
    jobs: list[BenchmarkJob],
    db_path: Path,
    block_size: int,
    disable_skipping: bool,
    verifier_backend: str,
    cli_block_metadata: str | None,
    grid_depth: int | None,
) -> dict[tuple[str, tuple[int, ...], str, int], BlockMetadataBundle]:
    if disable_skipping or verifier_backend == "pytorch":
        return {}

    bundles: dict[tuple[str, tuple[int, ...], str, int], BlockMetadataBundle] = {}
    for job in jobs:
        kind = _resolve_block_metadata_kind(cli_block_metadata, job.filter_spec)
        resolved_grid_depth = _resolve_grid_depth(grid_depth, job.filter_spec)
        cache_key = (
            job.model_spec.name,
            tuple(job.block_ids),
            kind,
            resolved_grid_depth,
        )
        if cache_key in bundles:
            continue
        bundle = _collect_metadata_bundle(
            model_spec=job.model_spec,
            filter_spec=job.filter_spec,
            db_path=db_path,
            block_size=block_size,
            candidate_block_ids=job.block_ids,
            kind=kind,
            grid_depth=resolved_grid_depth,
        )
        bundles[cache_key] = bundle
    return bundles


def _format_prepared_filters_for_display(filters: list[FilterSpec]) -> list[dict]:
    return [
        {
            "filter_id": index,
            "name": spec.name,
            "table": spec.table,
            "model_name": spec.model_name,
            "sql_predicate": spec.sql_predicate,
            "predicate_lower": spec.predicate_lower,
            "predicate_upper": spec.predicate_upper,
            "sampled_start": spec.sampled_start,
            "sampled_width": spec.sampled_width,
            "template_name": spec.template_name or spec.name,
            "block_metadata": spec.block_metadata,
        }
        for index, spec in enumerate(filters, start=1)
    ]


def _group_jobs_by_template(jobs: list[BenchmarkJob]) -> list[tuple[str, list[BenchmarkJob]]]:
    grouped: dict[str, list[BenchmarkJob]] = {}
    ordered_names: list[str] = []
    for job in jobs:
        template_name = job.filter_spec.template_name or job.filter_spec.name
        if template_name not in grouped:
            grouped[template_name] = []
            ordered_names.append(template_name)
        grouped[template_name].append(job)
    return [(template_name, grouped[template_name]) for template_name in ordered_names]


def _run_jobs_serially(
    *,
    jobs: list[BenchmarkJob],
    db_path: Path,
    block_size: int,
    run_udf: bool,
    measure_e2e: bool,
    disable_skipping: bool,
    verifier_backend: str,
    batched_geomcad: bool,
    verifier_timeout_seconds: float,
    cli_block_metadata: str | None,
    grid_depth: int | None,
    metadata_bundles: dict[tuple[str, tuple[int, ...], str, int], BlockMetadataBundle],
    model_ground_truth_caches: dict[int, ModelGroundTruthCache],
) -> list[dict]:
    results: list[dict] = []
    total_jobs = len(jobs)
    for job_index, job in enumerate(jobs):
        kind = _resolve_block_metadata_kind(cli_block_metadata, job.filter_spec)
        resolved_grid_depth = _resolve_grid_depth(grid_depth, job.filter_spec)
        bundle = metadata_bundles.get(
            (
                job.model_spec.name,
                tuple(job.block_ids),
                kind,
                resolved_grid_depth,
            )
        )
        if job_index > 0:
            print()
        result = _run_benchmark_job(
            job=job,
            db_path=db_path,
            block_size=block_size,
            run_udf=run_udf,
            measure_e2e=measure_e2e,
            disable_skipping=disable_skipping,
            verifier_backend=verifier_backend,
            batched_geomcad=batched_geomcad,
            verifier_timeout_seconds=verifier_timeout_seconds,
            metadata_kind=kind,
            grid_depth=resolved_grid_depth,
            job_index=job_index,
            total_jobs=total_jobs,
            metadata_bundle=bundle,
            show_progress=True,
            progress_queue=None,
            model_ground_truth=model_ground_truth_caches[job.filter_id],
        )
        results.append(asdict(result))
    return results


def _run_jobs_in_parallel(
    *,
    jobs: list[BenchmarkJob],
    db_path: Path,
    block_size: int,
    run_udf: bool,
    measure_e2e: bool,
    disable_skipping: bool,
    verifier_backend: str,
    batched_geomcad: bool,
    verifier_timeout_seconds: float,
    cli_block_metadata: str | None,
    grid_depth: int | None,
    max_workers: int,
    metadata_bundles: dict[tuple[str, tuple[int, ...], str, int], BlockMetadataBundle],
    model_ground_truth_caches: dict[int, ModelGroundTruthCache],
) -> list[dict]:
    total_jobs = len(jobs)
    max_workers = min(max_workers, total_jobs) if total_jobs else max_workers
    completed_results: dict[int, dict] = {}
    progress_states: dict[int, ProgressState] = {}
    last_progress_render_at = 0.0
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for job_index, job in enumerate(jobs):
                kind = _resolve_block_metadata_kind(cli_block_metadata, job.filter_spec)
                resolved_grid_depth = _resolve_grid_depth(grid_depth, job.filter_spec)
                cache_key = (
                    job.model_spec.name,
                    tuple(job.block_ids),
                    kind,
                    resolved_grid_depth,
                )
                future_to_index[
                    executor.submit(
                        _run_benchmark_job_capture_logs,
                        job=job,
                        db_path=db_path,
                        block_size=block_size,
                        run_udf=run_udf,
                        measure_e2e=measure_e2e,
                        disable_skipping=disable_skipping,
                        verifier_backend=verifier_backend,
                        batched_geomcad=batched_geomcad,
                        verifier_timeout_seconds=verifier_timeout_seconds,
                        metadata_kind=kind,
                        grid_depth=resolved_grid_depth,
                        job_index=job_index,
                        total_jobs=total_jobs,
                        metadata_bundle=metadata_bundles.get(cache_key),
                        progress_queue=progress_queue,
                        model_ground_truth=model_ground_truth_caches[job.filter_id],
                    )
                ] = job_index
            pending = set(future_to_index)
            while pending:
                done, pending = wait(
                    pending,
                    timeout=0.5,
                    return_when=FIRST_COMPLETED,
                )
                _drain_progress_queue(progress_queue, progress_states)
                now = time.monotonic()
                if now - last_progress_render_at >= 1.0:
                    if _render_parallel_progress(progress_states):
                        last_progress_render_at = now
                for future in done:
                    job_index, result_dict, _logs = future.result()
                    _mark_progress_done(
                        progress_states,
                        job_index,
                        jobs[job_index],
                        result_dict,
                        total_jobs,
                    )
                    if _render_parallel_progress(progress_states, force=True):
                        last_progress_render_at = time.monotonic()
                    completed_results[job_index] = result_dict
            _drain_progress_queue(progress_queue, progress_states)
    _clear_parallel_progress_render()
    return [completed_results[index] for index in sorted(completed_results)]


def _apply_block_budget(
    jobs: list[BenchmarkJob],
    max_blocks_total: int | None,
) -> list[BenchmarkJob]:
    if max_blocks_total is None:
        return jobs
    if not jobs:
        return []

    budgeted_jobs: list[BenchmarkJob] = []
    for job in jobs:
        truncated_blocks = job.block_ids[:max_blocks_total]
        budgeted_jobs.append(
            BenchmarkJob(
                filter_id=job.filter_id,
                filter_spec=job.filter_spec,
                model_spec=job.model_spec,
                model_path=job.model_path,
                block_ids=truncated_blocks,
                excluded_training_blocks=job.excluded_training_blocks,
            )
        )
    return budgeted_jobs


def _resolve_max_blocks_total(
    *,
    max_rows_total: int | None,
    block_size: int,
) -> int | None:
    if max_rows_total is None:
        return None
    return (max_rows_total + block_size - 1) // block_size


def _run_benchmark_job_capture_logs(
    *,
    job: BenchmarkJob,
    db_path: Path,
    block_size: int,
    run_udf: bool,
    measure_e2e: bool,
    disable_skipping: bool,
    verifier_backend: str,
    batched_geomcad: bool,
    verifier_timeout_seconds: float,
    metadata_kind: str,
    grid_depth: int,
    job_index: int,
    total_jobs: int,
    metadata_bundle: BlockMetadataBundle | None,
    progress_queue: object | None,
    model_ground_truth: ModelGroundTruthCache,
) -> tuple[int, dict, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = _run_benchmark_job(
            job=job,
            db_path=db_path,
            block_size=block_size,
            run_udf=run_udf,
            measure_e2e=measure_e2e,
            disable_skipping=disable_skipping,
            verifier_backend=verifier_backend,
            batched_geomcad=batched_geomcad,
            verifier_timeout_seconds=verifier_timeout_seconds,
            metadata_kind=metadata_kind,
            grid_depth=grid_depth,
            job_index=job_index,
            total_jobs=total_jobs,
            metadata_bundle=metadata_bundle,
            show_progress=False,
            progress_queue=progress_queue,
            model_ground_truth=model_ground_truth,
        )
    return job_index, asdict(result), buffer.getvalue()


def _model_kind_for_job(job: BenchmarkJob) -> str:
    relative_path = job.model_path.relative_to(models_dir(job.model_spec.database))
    if not relative_path.parts:
        raise ValueError(f"Could not infer model kind from path: {job.model_path}")
    return relative_path.parts[0]


def _run_benchmark_job(
    *,
    job: BenchmarkJob,
    db_path: Path,
    block_size: int,
    run_udf: bool,
    measure_e2e: bool,
    disable_skipping: bool,
    verifier_backend: str,
    batched_geomcad: bool,
    verifier_timeout_seconds: float,
    metadata_kind: str,
    grid_depth: int,
    job_index: int,
    total_jobs: int,
    metadata_bundle: BlockMetadataBundle | None,
    show_progress: bool,
    progress_queue: object | None,
    model_ground_truth: ModelGroundTruthCache,
) -> BenchmarkResult:
    filter_spec = job.filter_spec
    model_spec = job.model_spec
    benchmark_block_ids = job.block_ids
    total_blocks = len(benchmark_block_ids)
    result_verifier_backend = _results_verifier_backend_label(
        verifier_backend=verifier_backend,
        batched_geomcad=batched_geomcad,
    )

    print(
        f"[bench] Filter {job_index + 1}/{total_jobs} in this run: "
        f"'{filter_spec.name}' using model '{model_spec.name}'"
    )
    print(f"[bench] Query predicate: {filter_spec.sql_predicate}")
    if filter_spec.sampled_width is not None and filter_spec.sampled_start is not None:
        print(
            "[bench] Query parameters: "
            f"start={filter_spec.sampled_start:.12g} "
            f"width={filter_spec.sampled_width:.12g} "
            f"end={filter_spec.predicate_upper:.12g}"
        )
    print(
        f"[bench] Benchmarking on {total_blocks} block(s) after excluding "
        f"{job.excluded_training_blocks} training block(s)"
    )

    benchmark_rows = count_rows_for_blocks(
        filter_spec.table,
        benchmark_block_ids,
        db_path,
        block_size,
    )
    baseline_count = model_ground_truth.qualified_count
    baseline_ms = model_ground_truth.materialize_ms
    matching_blocks = model_ground_truth.matching_blocks
    skippable_blocks = max(0, total_blocks - matching_blocks)
    query_selectivity_pct = (
        (baseline_count / benchmark_rows) * 100.0 if benchmark_rows else None
    )
    print(
        "[bench] Model selectivity="
        f"{_format_optional_pct(query_selectivity_pct)} "
        f"matching_blocks={matching_blocks}/{total_blocks} "
        f"qualified_rows={baseline_count}/{benchmark_rows}"
    )
    udf_count: int | None = None
    udf_ms: float | None = None
    if run_udf:
        udf_count, udf_ms = _run_udf_query(
            db_path=db_path,
            filter_spec=filter_spec,
            model_spec=model_spec,
            model_path=job.model_path,
            block_ids=benchmark_block_ids,
            block_size=block_size,
        )
        print(f"[bench] UDF count={udf_count} time_ms={udf_ms:.3f}")
    else:
        print("[bench] Skipping UDF benchmark; pass --run-udf to enable it")

    kept_blocks = benchmark_block_ids
    timeout_blocks = 0
    error_blocks = 0
    metadata_collection_ms: float | None = None
    metadata_pair_count: int | None = None
    skipping_ms: float | None = None
    result_metadata_label = _results_block_metadata_label(
        cli_block_metadata=metadata_kind,
        jobs=[job],
        verifier_backend=result_verifier_backend,
    )
    _publish_progress(
        progress_queue,
        ProgressState(
            job_index=job_index,
            total_jobs=total_jobs,
            filter_id=job.filter_id,
            filter_name=filter_spec.name,
            verifier_backend=_progress_backend_label(verifier_backend, result_metadata_label),
            current=0,
            total=total_blocks,
            skipped=0,
            kept=0,
            block_id=None,
            status="starting",
            done=False,
        ),
    )
    if not disable_skipping:
        metadata_by_block: dict[int, BlockMetadata] | None = None
        if verifier_backend != "pytorch":
            if metadata_bundle is None:
                metadata_bundle = _collect_metadata_bundle(
                    model_spec=model_spec,
                    filter_spec=filter_spec,
                    db_path=db_path,
                    block_size=block_size,
                    candidate_block_ids=benchmark_block_ids,
                    kind=metadata_kind,
                    grid_depth=grid_depth,
                )
            else:
                print(
                    f"[bench] Reusing {metadata_kind} metadata for "
                    f"{len(metadata_bundle.metadata_by_block)} block(s)"
                )
            metadata_collection_ms = metadata_bundle.collection_ms
            metadata_pair_count = _metadata_pair_count(metadata_bundle)
            metadata_by_block = metadata_bundle.metadata_by_block
        skip_summary = _kept_blocks_for_filter(
            filter_spec=filter_spec,
            model_spec=model_spec,
            db_path=db_path,
            block_size=block_size,
            model_path=job.model_path,
            candidate_block_ids=benchmark_block_ids,
            metadata_by_block=metadata_by_block,
            show_progress=show_progress,
            progress_callback=(
                None
                if progress_queue is None
                else lambda state: _publish_progress(progress_queue, state)
            ),
            job_index=job_index,
            total_jobs=total_jobs,
            filter_id=job.filter_id,
            verifier_backend=verifier_backend,
            batched_geomcad=batched_geomcad,
            verifier_timeout_seconds=verifier_timeout_seconds,
            metadata_kind=result_metadata_label,
            grid_depth=grid_depth,
        )
        kept_blocks = skip_summary.kept_blocks
        timeout_blocks = skip_summary.timeout_blocks
        error_blocks = skip_summary.error_blocks
        skipping_ms = skip_summary.skipping_ms
    scanned_rows = count_rows_for_blocks(
        filter_spec.table,
        kept_blocks,
        db_path,
        block_size,
    )
    block_model_count = count_model_qualified_rows(
        model_ground_truth.cache_path,
        model_ground_truth.cache_key,
        kept_blocks,
        block_size,
    )
    e2e_count: int | None = None
    e2e_data_loading_ms: float | None = None
    e2e_inference_ms: float | None = None
    e2e_total_ms: float | None = None
    if measure_e2e:
        e2e_count, e2e_data_loading_ms, e2e_inference_ms = _run_pytorch_e2e_query(
            db_path=db_path,
            filter_spec=filter_spec,
            model_spec=model_spec,
            model_path=job.model_path,
            block_ids=kept_blocks,
            block_size=block_size,
        )
        e2e_total_ms = float(e2e_data_loading_ms + e2e_inference_ms + (skipping_ms or 0.0))
    result = BenchmarkResult(
        filter_name=filter_spec.name,
        filter_template_name=(filter_spec.template_name or filter_spec.name),
        verifier_backend=result_verifier_backend,
        model_kind=_model_kind_for_job(job),
        block_metadata=result_metadata_label,
        grid_depth=(
            grid_depth
            if result_metadata_label in {"grid", "bounded_convex_hull"}
            else None
        ),
        baseline_count=baseline_count,
        baseline_ms=baseline_ms,
        model_ground_truth_cache_path=str(model_ground_truth.cache_path),
        model_ground_truth_cache_key=model_ground_truth.cache_key,
        model_ground_truth_reused=model_ground_truth.reused,
        udf_count=udf_count,
        udf_ms=udf_ms,
        block_model_count=block_model_count,
        e2e_execution_backend=("pytorch" if measure_e2e else None),
        e2e_execution_mode=("per_block" if measure_e2e else None),
        e2e_count=e2e_count,
        e2e_data_loading_ms=e2e_data_loading_ms,
        e2e_inference_ms=e2e_inference_ms,
        e2e_total_ms=e2e_total_ms,
        kept_blocks=len(kept_blocks),
        skipped_blocks=total_blocks - len(kept_blocks),
        timeout_blocks=timeout_blocks,
        error_blocks=error_blocks,
        total_blocks=total_blocks,
        scanned_rows=scanned_rows,
        benchmark_rows=benchmark_rows,
        matching_blocks=matching_blocks,
        skippable_blocks=skippable_blocks,
        query_selectivity_pct=query_selectivity_pct,
        pruning_effectiveness_pct=_compute_pruning_effectiveness_pct(
            skipped_blocks=total_blocks - len(kept_blocks),
            skippable_blocks=skippable_blocks,
        ),
        metadata_collection_ms=metadata_collection_ms,
        metadata_pair_count=metadata_pair_count,
        skipping_ms=skipping_ms,
        ground_truth_match_udf=(
            None if udf_count is None else udf_count == baseline_count
        ),
        ground_truth_match_block_model=block_model_count == baseline_count,
        ground_truth_match_e2e=(
            None if e2e_count is None else e2e_count == baseline_count
        ),
    )

    print(
        f"[bench] Block-pruned model count={block_model_count} "
        f"kept_blocks={len(kept_blocks)}/{total_blocks}"
    )
    if (
        e2e_count is not None
        and e2e_total_ms is not None
        and e2e_data_loading_ms is not None
        and e2e_inference_ms is not None
    ):
        print(
            f"[bench] E2E PyTorch count={e2e_count} "
            f"loading_ms={e2e_data_loading_ms:.3f} "
            f"inference_ms={e2e_inference_ms:.3f} total_ms={e2e_total_ms:.3f}"
        )
    print(
        "[bench] Pruning effectiveness="
        f"{_format_optional_pct(result.pruning_effectiveness_pct)} "
        f"skipped_blocks={result.skipped_blocks}/{result.skippable_blocks}"
    )
    if block_model_count != baseline_count:
        print(
            f"[warn] Block-pruned model result for '{filter_spec.name}' differs from baseline: "
            f"{block_model_count} vs {baseline_count}"
        )
    _publish_progress(
        progress_queue,
        ProgressState(
            job_index=job_index,
            total_jobs=total_jobs,
            filter_id=job.filter_id,
            filter_name=filter_spec.name,
            verifier_backend=_progress_backend_label(verifier_backend, result_metadata_label),
            current=total_blocks,
            total=total_blocks,
            skipped=result.skipped_blocks,
            kept=result.kept_blocks,
            block_id=benchmark_block_ids[-1] if benchmark_block_ids else None,
            status="done",
            done=True,
        ),
    )
    return result


def _publish_progress(progress_queue: object | None, state: ProgressState) -> None:
    if progress_queue is None:
        return
    progress_queue.put(asdict(state))


def _drain_progress_queue(
    progress_queue: object,
    progress_states: dict[int, ProgressState],
) -> None:
    while True:
        try:
            raw_state = progress_queue.get_nowait()
        except queue.Empty:
            break
        state = ProgressState(**raw_state)
        existing = progress_states.get(state.job_index)
        if existing is None:
            progress_states[state.job_index] = state
            continue
        if existing.done and not state.done:
            continue
        if (
            not state.done
            and state.current < existing.current
        ):
            continue
        progress_states[state.job_index] = state


def _mark_progress_done(
    progress_states: dict[int, ProgressState],
    job_index: int,
    job: BenchmarkJob,
    result_dict: dict,
    total_jobs: int,
) -> None:
    state = progress_states.get(job_index)
    total_blocks = len(job.block_ids)
    progress_states[job_index] = ProgressState(
        job_index=job_index,
        total_jobs=(state.total_jobs if state is not None else total_jobs),
        filter_id=job.filter_id,
        filter_name=job.filter_spec.name,
        verifier_backend=str(result_dict.get("verifier_backend", "verifier")),
        current=total_blocks,
        total=total_blocks,
        skipped=int(result_dict.get("skipped_blocks", 0)),
        kept=int(result_dict.get("kept_blocks", total_blocks)),
        block_id=(job.block_ids[-1] if job.block_ids else None),
        status="done",
        done=True,
    )


def _render_parallel_progress(
    progress_states: dict[int, ProgressState],
    *,
    force: bool = False,
) -> bool:
    global _LAST_PROGRESS_RENDERED_LINES
    active_states = [
        state for _, state in sorted(progress_states.items()) if not state.done
    ]
    if not active_states and not force:
        return False
    if not active_states and force:
        return False

    active_backends = sorted({state.verifier_backend for state in active_states})
    backend_label = ",".join(active_backends) if active_backends else "verifier"
    lines = [f"[{backend_label}] Active parallel filter(s): {len(active_states)}"]
    lines.extend(_format_parallel_progress_line(state) for state in active_states)
    if _supports_inplace_progress():
        _rewrite_progress_block(lines)
    else:
        if _LAST_PROGRESS_RENDERED_LINES:
            print()
        for line in lines:
            print(line)
    _LAST_PROGRESS_RENDERED_LINES = len(lines)
    return True


def _progress_backend_label(verifier_backend: str, metadata_kind: str) -> str:
    if metadata_kind == "none":
        return verifier_backend
    return f"{verifier_backend}+{metadata_kind}"


def _format_parallel_progress_line(state: ProgressState) -> str:
    total = max(state.total, 1)
    pct = (state.current / total) * 100.0
    block_desc = "n/a" if state.block_id is None else str(state.block_id)
    run_desc = f"{state.job_index + 1}/{state.total_jobs}"
    return (
        f"[{state.verifier_backend:<7}] "
        f"filter={state.job_index + 1:>3} "
        f"({run_desc:>7} in run) "
        f"{state.current:>3}/{state.total:<3} "
        f"({pct:>5.1f}%) "
        f"block={block_desc:>3} "
        f"status={state.status:<7} "
        f"skipped={state.skipped:>3} "
        f"kept={state.kept:>3}"
    )


def _supports_inplace_progress() -> bool:
    return sys.stdout.isatty()


def _rewrite_progress_block(lines: list[str]) -> None:
    global _LAST_PROGRESS_RENDERED_LINES
    if _LAST_PROGRESS_RENDERED_LINES:
        sys.stdout.write(f"\x1b[{_LAST_PROGRESS_RENDERED_LINES}F")
    for index, line in enumerate(lines):
        sys.stdout.write("\x1b[2K")
        sys.stdout.write(line)
        if index < len(lines) - 1:
            sys.stdout.write("\n")
    sys.stdout.write("\n")
    sys.stdout.flush()


def _clear_parallel_progress_render() -> None:
    global _LAST_PROGRESS_RENDERED_LINES
    if _LAST_PROGRESS_RENDERED_LINES == 0:
        return
    if _supports_inplace_progress():
        sys.stdout.write(f"\x1b[{_LAST_PROGRESS_RENDERED_LINES}F")
        for index in range(_LAST_PROGRESS_RENDERED_LINES):
            sys.stdout.write("\x1b[2K")
            if index < _LAST_PROGRESS_RENDERED_LINES - 1:
                sys.stdout.write("\n")
        sys.stdout.write(f"\x1b[{max(_LAST_PROGRESS_RENDERED_LINES - 1, 0)}F")
        sys.stdout.flush()
    _LAST_PROGRESS_RENDERED_LINES = 0


def _sample_regressor_filters(
    *,
    template: FilterSpec,
    model_spec: FunctionSpec,
    db_path: Path,
    block_size: int,
    benchmark_block_ids: list[int],
    alpha: float,
    start_samples: int,
) -> list[FilterSpec]:
    target_range = fetch_expression_range(
        template.table,
        model_spec.target_expression,
        db_path,
        block_ids=benchmark_block_ids,
        block_size=block_size,
    )
    if target_range is None:
        return []

    min_value, max_value = target_range
    span = max_value - min_value
    if span <= 0.0:
        return [
            _build_sampled_regressor_filter(
                template=template,
                target_expression=model_spec.target_expression,
                lower=min_value,
                upper=max_value,
                width=0.0,
                width_index=0,
                start_index=0,
            )
        ]

    widths = _build_range_widths(span, alpha)
    per_width_sample_budget = min(
        start_samples,
        max(_GENERATED_FILTER_MIN_PER_WIDTH, _GENERATED_FILTER_TOTAL_BUDGET // len(widths)),
    )
    sampled_filters: list[FilterSpec] = []
    for width_index, width in enumerate(widths, start=1):
        sample_count = 1 if width >= span else per_width_sample_budget
        max_start = max_value - width
        for start_index in range(sample_count):
            lower = _deterministic_range_start(
                min_value=min_value,
                max_start=max_start,
                sample_count=sample_count,
                start_index=start_index,
            )
            upper = lower + width
            sampled_filters.append(
                _build_sampled_regressor_filter(
                    template=template,
                    target_expression=model_spec.target_expression,
                    lower=lower,
                    upper=upper,
                    width=width,
                    width_index=width_index,
                    start_index=start_index,
                )
            )
    return _sort_filters_by_selectivity(
        filters=sampled_filters,
        db_path=db_path,
        block_size=block_size,
        benchmark_block_ids=benchmark_block_ids,
    )


def _deterministic_range_start(
    *,
    min_value: float,
    max_start: float,
    sample_count: int,
    start_index: int,
) -> float:
    if max_start <= min_value or sample_count <= 1:
        return float(min_value)
    step = (max_start - min_value) / float(sample_count - 1)
    lower = min_value + (step * start_index)
    return max(min_value, min(round(lower), max_start))


def _build_range_widths(span: float, alpha: float) -> list[float]:
    widths: list[float] = []
    width = alpha
    while width < span:
        widths.append(width)
        width *= alpha
    if not widths or abs(widths[-1] - span) > 1e-12:
        widths.append(span)
    return widths


def _build_sampled_regressor_filter(
    *,
    template: FilterSpec,
    target_expression: str,
    lower: float,
    upper: float,
    width: float,
    width_index: int,
    start_index: int,
) -> FilterSpec:
    lower_bound = float(lower)
    upper_bound = float(upper)
    sql_predicate = (
        f"CAST({target_expression} AS DOUBLE) BETWEEN "
        f"{lower_bound:.12g} AND {upper_bound:.12g}"
    )
    return FilterSpec(
        name=(
            f"{template.model_name}_w{width_index:02d}_s{start_index:02d}_"
            f"{lower_bound:.6g}_{upper_bound:.6g}"
        ),
        description=(
            f"Sampled regressor range for '{template.model_name}' with width={width:.6g} "
            f"and bounds [{lower_bound:.6g}, {upper_bound:.6g}]"
        ),
        database=template.database,
        table=template.table,
        model_name=template.model_name,
        sql_predicate=sql_predicate,
        filter_type=template.filter_type,
        predicate_lower=lower_bound,
        predicate_upper=upper_bound,
        target_class=None,
        sampled_width=float(width),
        sampled_start=lower_bound,
        template_name=(template.template_name or template.name),
        block_metadata=template.block_metadata,
    )


def _sample_classifier_filters(
    *,
    template: FilterSpec,
    model_spec: FunctionSpec,
    db_path: Path,
    block_size: int,
    benchmark_block_ids: list[int],
) -> list[FilterSpec]:
    target_classes = _fetch_target_classes(
        table=template.table,
        target_expression=model_spec.target_expression,
        db_path=db_path,
        block_size=block_size,
        benchmark_block_ids=benchmark_block_ids,
    )
    sampled_filters = [
        _build_sampled_classifier_filter(
            template=template,
            target_expression=model_spec.target_expression,
            target_class=target_class,
            class_index=class_index,
        )
        for class_index, target_class in enumerate(target_classes, start=1)
    ]
    return _sort_filters_by_selectivity(
        filters=sampled_filters,
        db_path=db_path,
        block_size=block_size,
        benchmark_block_ids=benchmark_block_ids,
    )


def _fetch_target_classes(
    *,
    table: str,
    target_expression: str,
    db_path: Path,
    block_size: int,
    benchmark_block_ids: list[int],
) -> list[int]:
    if not benchmark_block_ids:
        return []
    block_predicate = build_block_id_predicate(benchmark_block_ids, block_size)
    with duckdb.connect(str(db_path), read_only=True) as con:
        ensure_row_id_column(con, table)
        rows = con.execute(
            f"""
            SELECT DISTINCT CAST({target_expression} AS INTEGER) AS class_id
            FROM {table}
            WHERE {block_predicate}
            ORDER BY class_id
            """
        ).fetchall()
    return [int(row[0]) for row in rows if row[0] is not None]


def _build_sampled_classifier_filter(
    *,
    template: FilterSpec,
    target_expression: str,
    target_class: int,
    class_index: int,
) -> FilterSpec:
    sql_predicate = f"CAST({target_expression} AS INTEGER) = {target_class}"
    return FilterSpec(
        name=f"{template.model_name}_c{class_index:02d}_{target_class}",
        description=(
            f"Sampled classifier class for '{template.model_name}' with target_class={target_class}"
        ),
        database=template.database,
        table=template.table,
        model_name=template.model_name,
        sql_predicate=sql_predicate,
        filter_type=template.filter_type,
        predicate_lower=None,
        predicate_upper=None,
        target_class=int(target_class),
        sampled_width=None,
        sampled_start=None,
        template_name=(template.template_name or template.name),
        block_metadata=template.block_metadata,
    )


def _is_generated_classifier_filter_name(name: str, model_name: str) -> bool:
    return re.match(rf"^{re.escape(model_name)}_c\d{{2}}_-?\d+$", name) is not None


def _sort_filters_by_selectivity(
    *,
    filters: list[FilterSpec],
    db_path: Path,
    block_size: int,
    benchmark_block_ids: list[int],
) -> list[FilterSpec]:
    if len(filters) <= 1:
        return filters

    benchmark_rows = count_rows_for_blocks(
        filters[0].table,
        benchmark_block_ids,
        db_path,
        block_size,
    )
    ranked_filters: list[tuple[float | None, FilterSpec]] = []
    for filter_spec in filters:
        baseline_count, _ = run_count_query(
            filter_spec.table,
            filter_spec.sql_predicate,
            db_path,
            block_ids=benchmark_block_ids,
            block_size=block_size,
        )
        selectivity_pct = (
            (baseline_count / benchmark_rows) * 100.0 if benchmark_rows else None
        )
        ranked_filters.append((selectivity_pct, filter_spec))

    ranked_filters.sort(
        key=lambda item: (
            float("inf") if item[0] is None else item[0],
            item[1].sampled_width if item[1].sampled_width is not None else float("inf"),
            item[1].sampled_start if item[1].sampled_start is not None else float("inf"),
            item[1].name,
        )
    )
    return [filter_spec for _, filter_spec in ranked_filters]


def _compute_pruning_effectiveness_pct(
    *,
    skipped_blocks: int,
    skippable_blocks: int,
) -> float | None:
    if skippable_blocks <= 0:
        return 100.0
    return (skipped_blocks / skippable_blocks) * 100.0


def _format_optional_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def _resolve_block_metadata_kind(
    cli_block_metadata: str | None,
    filter_spec: FilterSpec,
) -> BlockMetadataKind:
    del filter_spec
    kind = cli_block_metadata or "minmax"
    return kind  # type: ignore[return-value]


def _prediction_qualifies(filter_spec: FilterSpec, predictions: np.ndarray) -> np.ndarray:
    if filter_spec.filter_type == "regressor_range":
        if filter_spec.predicate_lower is None or filter_spec.predicate_upper is None:
            raise ValueError(f"Filter '{filter_spec.name}' is missing regressor bounds.")
        return (predictions >= float(filter_spec.predicate_lower)) & (
            predictions <= float(filter_spec.predicate_upper)
        )
    if filter_spec.filter_type == "classifier_class":
        if filter_spec.target_class is None:
            raise ValueError(f"Filter '{filter_spec.name}' is missing target_class.")
        return predictions.astype(np.int64) == int(filter_spec.target_class)
    raise ValueError(f"Unsupported filter_type: {filter_spec.filter_type}")


def _feature_bounds_from_block_features(
    model_spec: FunctionSpec,
    block_features: np.ndarray,
) -> dict[str, tuple[float, float]]:
    if block_features.size == 0:
        return {feature.name: (math.nan, math.nan) for feature in model_spec.features}
    return {
        feature.name: (
            float(np.min(block_features[:, index])),
            float(np.max(block_features[:, index])),
        )
        for index, feature in enumerate(model_spec.features)
    }


def _decide_pytorch_block_skip(
    *,
    filter_spec: FilterSpec,
    model_spec: FunctionSpec,
    model_path: Path,
    block_id: int,
    block_features: np.ndarray,
) -> BlockSkipResult:
    started_at = time.perf_counter()
    predictions = predict_array_pytorch(model_spec, model_path, block_features)
    should_skip = not bool(np.any(_prediction_qualifies(filter_spec, predictions)))
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    return BlockSkipResult(
        backend="pytorch",
        block_id=block_id,
        predicate_lower=filter_spec.predicate_lower,
        predicate_upper=filter_spec.predicate_upper,
        target_class=filter_spec.target_class,
        status="unsat" if should_skip else "sat",
        should_skip=should_skip,
        elapsed_ms=float(elapsed_ms),
        summary=(
            "No batched PyTorch predictions satisfied the filter predicate."
            if should_skip
            else "At least one batched PyTorch prediction satisfied the filter predicate."
        ),
    )


def _resolve_grid_depth(
    cli_grid_depth: int | None,
    filter_spec: FilterSpec,
) -> int:
    if cli_grid_depth is not None:
        return int(cli_grid_depth)
    raw_metadata = filter_spec.block_metadata or {}
    raw_depth = raw_metadata.get("grid_depth", 4)
    depth = int(raw_depth)
    if depth < 0:
        raise ValueError(
            f"block_metadata.grid_depth for filter '{filter_spec.name}' must be non-negative."
        )
    return depth


def _metadata_pair_count(bundle: BlockMetadataBundle) -> int:
    for metadata in bundle.metadata_by_block.values():
        return len(metadata.pair_geometries)
    return 0


def _metadata_numeric_storage_bytes(metadata: BlockMetadata) -> int:
    # For convex_hull we report only the shape payload; bounded_convex_hull also includes input bounds.
    total_bytes = 0 if metadata.kind == "convex_hull" else len(metadata.input_bounds) * 2 * 8
    for geometry in metadata.pair_geometries:
        if geometry.hull is not None and metadata.kind != "bounded_convex_hull":
            total_bytes += len(geometry.hull) * 2 * 8
        if geometry.bounded_convex_hull is not None:
            total_bytes += _bounded_convex_hull_storage_bytes(geometry)
        if geometry.grid_cells is not None:
            if geometry.grid_depth is None:
                raise RuntimeError("grid_cells present without grid_depth")
            total_cells = 2 ** (2 * int(geometry.grid_depth))
            total_bytes += math.ceil(total_cells / 8)
    return int(total_bytes)


def _bounded_convex_hull_storage_bytes(geometry: PairGeometry) -> int:
    if geometry.bounded_convex_hull is None:
        return 0
    if geometry.grid_depth is None:
        raise RuntimeError("bounded_convex_hull present without grid_depth")
    bits_per_vertex = (2 * int(geometry.grid_depth)) + 2
    return math.ceil((len(geometry.bounded_convex_hull) * bits_per_vertex) / 8)


def _metadata_size_summary(
    bundles: dict[tuple[str, tuple[int, ...], str, int], BlockMetadataBundle],
) -> dict[str, object] | None:
    size_per_block_bytes: list[int] = []
    collection_ms_per_block: list[float] = []
    convex_hull_vertices_per_pair: list[int] = []
    bounded_convex_hull_vertices_per_pair: list[int] = []
    kinds: set[str] = set()
    grid_depths: set[int] = set()
    for bundle in bundles.values():
        for block_id, metadata in bundle.metadata_by_block.items():
            size_per_block_bytes.append(_metadata_numeric_storage_bytes(metadata))
            if block_id not in bundle.collection_ms_by_block:
                raise RuntimeError(
                    f"Missing metadata collection timing for block {block_id}."
                )
            collection_ms_per_block.append(
                float(bundle.collection_ms_by_block[block_id])
            )
            kinds.add(metadata.kind)
            for geometry in metadata.pair_geometries:
                if geometry.hull is not None:
                    convex_hull_vertices_per_pair.append(len(geometry.hull))
                if geometry.bounded_convex_hull is not None:
                    bounded_convex_hull_vertices_per_pair.append(
                        len(geometry.bounded_convex_hull)
                    )
                if geometry.grid_depth is not None:
                    grid_depths.add(int(geometry.grid_depth))
    if not size_per_block_bytes:
        return None
    kind_list = sorted(kinds)
    if len(kind_list) != 1:
        raise RuntimeError(
            "Expected exactly one metadata kind per benchmark run; "
            f"found {kind_list}."
        )
    if len(grid_depths) > 1:
        raise RuntimeError(
            "Expected exactly one grid depth per benchmark run; "
            f"found {sorted(grid_depths)}."
        )
    summary = {
        "kind": kind_list[0],
        "block_count": len(size_per_block_bytes),
        "avg_size_per_block_bytes": float(
            sum(size_per_block_bytes) / len(size_per_block_bytes)
        ),
        "median_size_per_block_bytes": float(statistics.median(size_per_block_bytes)),
        "max_size_per_block_bytes": int(max(size_per_block_bytes)),
        "total_collection_ms": float(sum(collection_ms_per_block)),
        "avg_collection_ms_per_block": float(
            sum(collection_ms_per_block) / len(collection_ms_per_block)
        ),
        "median_collection_ms_per_block": float(
            statistics.median(collection_ms_per_block)
        ),
        "max_collection_ms_per_block": float(max(collection_ms_per_block)),
    }
    vertex_counts = (
        bounded_convex_hull_vertices_per_pair
        if kind_list[0] == "bounded_convex_hull"
        else convex_hull_vertices_per_pair
    )
    if vertex_counts:
        summary["avg_convex_hull_vertices_per_pair"] = float(
            statistics.mean(vertex_counts)
        )
        summary["median_convex_hull_vertices_per_pair"] = float(
            statistics.median(vertex_counts)
        )
        summary["max_convex_hull_vertices_per_pair"] = int(
            max(vertex_counts)
        )
    if grid_depths:
        summary["grid_depth"] = next(iter(grid_depths))
    return summary


def _metadata_summary(metadata: BlockMetadata) -> dict:
    return {
        "kind": metadata.kind,
        "pair_count": len(metadata.pair_geometries),
        "pairs": [
            {
                "feature_x": geometry.feature_x,
                "feature_y": geometry.feature_y,
                "hull_vertices": (
                    len(geometry.hull) if geometry.hull is not None else None
                ),
                "grid_depth": geometry.grid_depth,
                "grid_cells": (
                    len(geometry.grid_cells)
                    if geometry.grid_cells is not None
                    else None
                ),
                "bounded_convex_hull_vertices": (
                    len(geometry.bounded_convex_hull)
                    if geometry.bounded_convex_hull is not None
                    else None
                ),
            }
            for geometry in metadata.pair_geometries
        ],
    }


def _decide_grid_block_skip(
    *,
    backend: str,
    request: BlockVerificationRequest,
    metadata: BlockMetadata,
) -> BlockSkipResult:
    if backend not in {"marabou", "geomcad"}:
        raise ValueError(
            "--block-metadata=grid currently requires --verifier-backend=marabou or geomcad."
        )
    started_at = time.perf_counter()
    solved_rectangles = 0
    timeout_seen = False
    first_depth = None
    for geometry in metadata.pair_geometries:
        if not geometry.grid_cells or geometry.grid_depth is None:
            continue
        first_depth = geometry.grid_depth if first_depth is None else first_depth
        all_rectangles_skip = True
        for cell_id in geometry.grid_cells:
            rect_bounds = dict(metadata.input_bounds)
            rect_bounds.update(
                grid_cell_rect(
                    cell_id,
                    input_bounds=metadata.input_bounds,
                    feature_x=geometry.feature_x,
                    feature_y=geometry.feature_y,
                    depth=geometry.grid_depth,
                )
            )
            rect_result = decide_block_skip(
                backend=backend,
                request=replace(
                    request,
                    input_bounds=rect_bounds,
                    pair_geometries=None,
                    verbose=False,
                ),
            )
            solved_rectangles += 1
            if rect_result.status == "timeout":
                timeout_seen = True
            if not rect_result.should_skip:
                all_rectangles_skip = False
                break
        if all_rectangles_skip and geometry.grid_cells:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            return BlockSkipResult(
                backend=f"{backend}+grid-depth{geometry.grid_depth}",
                block_id=request.block_id,
                predicate_lower=request.predicate_lower,
                predicate_upper=request.predicate_upper,
                target_class=request.target_class,
                status="unsat",
                should_skip=True,
                elapsed_ms=float(elapsed_ms),
                summary=(
                    "Every occupied 2D grid rectangle for at least one feature pair "
                    "was infeasible, so the block can be skipped."
                ),
            )
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    return BlockSkipResult(
        backend=f"{backend}+grid-depth{first_depth if first_depth is not None else 'n/a'}",
        block_id=request.block_id,
        predicate_lower=request.predicate_lower,
        predicate_upper=request.predicate_upper,
        target_class=request.target_class,
        status="timeout" if timeout_seen else "sat",
        should_skip=False,
        elapsed_ms=float(elapsed_ms),
        summary=(
            f"Occupied-grid metadata did not prove the block skippable "
            f"after {solved_rectangles} rectangle check(s)."
        ),
    )


def _evaluate_block(
    *,
    db_path: Path,
    block_size: int,
    filter_spec: FilterSpec,
    model_spec: FunctionSpec,
    model_path: Path,
    block_id: int,
    bounds: dict[str, tuple[float, float]] | None,
    block_metadata: BlockMetadata | None,
    disable_skipping: bool,
    run_udf: bool,
    verifier_backend: str,
    verifier_timeout_seconds: float,
    metadata_kind: str,
    grid_depth: int,
    include_counts: bool,
    verbose: bool,
    model_ground_truth: ModelGroundTruthCache | None = None,
    block_feature_con: duckdb.DuckDBPyConnection | None = None,
) -> BlockEvaluation:
    row_id_start = block_id * block_size
    row_id_end = (block_id + 1) * block_size - 1
    effective_metadata = block_metadata
    effective_bounds = bounds
    block_features: np.ndarray | None = None
    if verifier_backend == "pytorch":
        block_features = fetch_block_features(
            model_spec,
            block_id,
            db_path,
            block_size,
            con=block_feature_con,
        )
        if effective_bounds is None:
            effective_bounds = _feature_bounds_from_block_features(model_spec, block_features)
        if effective_metadata is None:
            effective_metadata = BlockMetadata(
                kind="minmax",
                input_bounds=effective_bounds,
                pair_geometries=[],
            )
    elif effective_metadata is None:
        if effective_bounds is None or metadata_kind != "minmax":
            bundle = collect_block_metadata(
                spec=model_spec,
                block_ids=[block_id],
                db_path=db_path,
                block_size=block_size,
                kind=metadata_kind,  # type: ignore[arg-type]
                grid_depth=grid_depth,
            )
            effective_metadata = bundle.metadata_by_block[block_id]
            effective_bounds = effective_metadata.input_bounds
        else:
            effective_metadata = BlockMetadata(
                kind="minmax",
                input_bounds=effective_bounds,
                pair_geometries=[],
            )
    if effective_metadata is None or effective_bounds is None:
        raise RuntimeError(f"Missing block metadata or bounds for block {block_id}.")
    effective_bounds = effective_metadata.input_bounds

    block_row_count: int | None = None
    matching_rows: int | None = None
    if include_counts:
        if model_ground_truth is None:
            matching_rows = count_block_predicate_matches(
                filter_spec,
                block_id,
                db_path,
                block_size,
            )
        else:
            matching_rows = count_model_qualified_rows(
                model_ground_truth.cache_path,
                model_ground_truth.cache_key,
                [block_id],
                block_size,
            )
        block_row_count = count_rows_for_blocks(
            filter_spec.table,
            [block_id],
            db_path,
            block_size,
        )

    verifier_result: BlockSkipResult | None = None
    if disable_skipping:
        if verbose:
            print("[bench] Skipping verifier inspection because --disable-skipping was passed")
    else:
        request = BlockVerificationRequest(
            model_path=model_path,
            spec=model_spec,
            input_bounds=effective_bounds,
            block_id=block_id,
            pair_geometries=(
                effective_metadata.pair_geometries
                if metadata_kind in {"convex_hull", "bounded_convex_hull"}
                else None
            ),
            predicate_lower=filter_spec.predicate_lower,
            predicate_upper=filter_spec.predicate_upper,
            target_class=filter_spec.target_class,
            timeout_seconds=verifier_timeout_seconds,
            verbose=verbose,
        )
        if verifier_backend == "pytorch":
            verifier_result = _decide_pytorch_block_skip(
                filter_spec=filter_spec,
                model_spec=model_spec,
                model_path=model_path,
                block_id=block_id,
                block_features=(
                    block_features
                    if block_features is not None
                    else fetch_block_features(
                        model_spec,
                        block_id,
                        db_path,
                        block_size,
                        con=block_feature_con,
                    )
                ),
            )
        elif metadata_kind == "grid":
            verifier_result = _decide_grid_block_skip(
                backend=verifier_backend,
                request=request,
                metadata=effective_metadata,
            )
        else:
            verifier_result = decide_block_skip(
                backend=verifier_backend,
                request=request,
            )
        verifier_result = replace(
            verifier_result,
            backend=_progress_backend_label(verifier_backend, metadata_kind),
        )

    udf_count: int | None = None
    udf_ms: float | None = None
    if run_udf:
        udf_count, udf_ms = _run_udf_query(
            db_path=db_path,
            filter_spec=filter_spec,
            model_spec=model_spec,
            model_path=model_path,
            block_ids=[block_id],
            block_size=block_size,
        )

    return BlockEvaluation(
        block_id=block_id,
        row_id_start=row_id_start,
        row_id_end=row_id_end,
        feature_bounds=effective_bounds,
        block_metadata=effective_metadata,
        verifier_result=verifier_result,
        block_row_count=block_row_count,
        matching_rows=matching_rows,
        udf_count=udf_count,
        udf_ms=udf_ms,
    )


def _run_udf_query(
    *,
    db_path: Path,
    filter_spec: FilterSpec,
    model_spec: FunctionSpec,
    model_path: Path,
    block_ids: list[int],
    block_size: int,
) -> tuple[int, float]:
    udf_name = f"predict_{model_spec.name}"
    feature_expressions = [feature.expression for feature in model_spec.features]
    udf_predicate = _build_udf_predicate(filter_spec, udf_name, feature_expressions)
    if not block_ids:
        return 0, 0.0
    block_predicate = build_block_id_predicate(block_ids, block_size)

    with duckdb.connect(str(db_path), read_only=True) as con:
        ensure_row_id_column(con, filter_spec.table)
        con.create_function(
            udf_name,
            lambda *args: predict_row(model_spec, model_path, list(args)),
            return_type="INTEGER" if model_spec.task_type == "classifier" else "DOUBLE",
        )
        start = time.perf_counter()
        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {filter_spec.table}
            WHERE {block_predicate}
              AND ({udf_predicate})
            """
        ).fetchone()[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    return int(count), float(elapsed_ms)


def _run_pytorch_e2e_query(
    *,
    db_path: Path,
    filter_spec: FilterSpec,
    model_spec: FunctionSpec,
    model_path: Path,
    block_ids: list[int],
    block_size: int,
) -> tuple[int, float, float]:
    if not block_ids:
        return 0, 0.0, 0.0

    data_loading_ms = 0.0
    inference_ms = 0.0
    total_matches = 0

    with duckdb.connect(str(db_path), read_only=True) as con:
        for block_id in block_ids:
            load_started_at = time.perf_counter()
            feature_matrix = fetch_block_features(
                model_spec,
                block_id,
                db_path,
                block_size,
                con=con,
            )
            data_loading_ms += (time.perf_counter() - load_started_at) * 1000.0

            inference_started_at = time.perf_counter()
            predictions = predict_array_pytorch(model_spec, model_path, feature_matrix)
            qualifies = _prediction_qualifies(filter_spec, predictions)
            inference_ms += (time.perf_counter() - inference_started_at) * 1000.0
            total_matches += int(np.count_nonzero(qualifies))

    return (
        total_matches,
        float(data_loading_ms),
        float(inference_ms),
    )


def _build_udf_predicate(
    filter_spec: FilterSpec,
    udf_name: str,
    feature_expressions: list[str],
) -> str:
    args = ", ".join(feature_expressions)
    if filter_spec.filter_type == "regressor_range":
        return (
            f"{udf_name}({args}) BETWEEN "
            f"{filter_spec.predicate_lower} AND {filter_spec.predicate_upper}"
        )
    if filter_spec.filter_type == "classifier_class":
        return f"{udf_name}({args}) = {filter_spec.target_class}"
    raise ValueError(f"Unsupported filter_type: {filter_spec.filter_type}")


def _kept_blocks_for_filter(
    *,
    filter_spec: FilterSpec,
    model_spec: FunctionSpec,
    db_path: Path,
    block_size: int,
    model_path: Path,
    candidate_block_ids: list[int],
    metadata_by_block: dict[int, BlockMetadata] | None,
    show_progress: bool,
    progress_callback: Callable[[ProgressState], None] | None,
    job_index: int,
    total_jobs: int,
    filter_id: int,
    verifier_backend: str,
    batched_geomcad: bool,
    verifier_timeout_seconds: float,
    metadata_kind: str,
    grid_depth: int,
) -> SkipSummary:
    total_blocks = len(candidate_block_ids)
    skipped_blocks = 0
    timeout_blocks = 0
    error_blocks = 0
    kept_blocks: list[int] = []
    skipping_ms = 0.0
    block_feature_con = None
    if verifier_backend == "pytorch":
        block_feature_con = duckdb.connect(str(db_path), read_only=True)
    try:
        if batched_geomcad:
            if verifier_backend != "geomcad":
                raise ValueError("Batched GeomCAD requires verifier_backend='geomcad'.")
            if metadata_kind != "minmax":
                raise ValueError("Batched GeomCAD currently supports only min-max metadata.")
            if metadata_by_block is None:
                raise RuntimeError("Batched GeomCAD requires precomputed min-max metadata.")
            requests = [
                BlockVerificationRequest(
                    model_path=model_path,
                    spec=model_spec,
                    input_bounds=metadata_by_block[block_id].input_bounds,
                    block_id=block_id,
                    predicate_lower=filter_spec.predicate_lower,
                    predicate_upper=filter_spec.predicate_upper,
                    target_class=filter_spec.target_class,
                    timeout_seconds=verifier_timeout_seconds,
                    verbose=False,
                )
                for block_id in candidate_block_ids
            ]
            batched_results = decide_geomcad_block_skips_batched(requests)
            progress_backend = _progress_backend_label("batched_geomcad", metadata_kind)
            for index, result in enumerate(batched_results, start=1):
                if result.status == "timeout":
                    timeout_blocks += 1
                elif result.status == "solver_error":
                    error_blocks += 1
                if result.should_skip:
                    skipped_blocks += 1
                else:
                    kept_blocks.append(result.block_id)
                if show_progress:
                    _print_verifier_progress(
                        current=index,
                        total=total_blocks,
                        block_id=result.block_id,
                        skipped=skipped_blocks,
                        kept=len(kept_blocks),
                        status=format_verifier_status(result) or result.status,
                        backend=progress_backend,
                    )
                if progress_callback is not None:
                    progress_callback(
                        ProgressState(
                            job_index=job_index,
                            total_jobs=total_jobs,
                            filter_id=filter_id,
                            filter_name=filter_spec.name,
                            verifier_backend=progress_backend,
                            current=index,
                            total=total_blocks,
                            skipped=skipped_blocks,
                            kept=len(kept_blocks),
                            block_id=result.block_id,
                            status=(format_verifier_status(result) or result.status),
                            done=False,
                        )
                    )
            skipping_ms = float(sum(result.elapsed_ms for result in batched_results))
            return SkipSummary(
                kept_blocks=kept_blocks,
                timeout_blocks=timeout_blocks,
                error_blocks=error_blocks,
                skipping_ms=skipping_ms,
            )
        for index, block_id in enumerate(candidate_block_ids, start=1):
            # Reuse precomputed bounds in the full benchmark path.
            evaluation = _evaluate_block(
                db_path=db_path,
                block_size=block_size,
                filter_spec=filter_spec,
                model_spec=model_spec,
                model_path=model_path,
                block_id=block_id,
                bounds=(None if metadata_by_block is None else metadata_by_block[block_id].input_bounds),
                block_metadata=(None if metadata_by_block is None else metadata_by_block[block_id]),
                disable_skipping=False,
                run_udf=False,
                verifier_backend=verifier_backend,
                verifier_timeout_seconds=verifier_timeout_seconds,
                metadata_kind=metadata_kind,
                grid_depth=grid_depth,
                include_counts=False,
                verbose=False,
                block_feature_con=block_feature_con,
            )
            result = evaluation.verifier_result
            if result is None:
                raise RuntimeError("Expected a verifier result during block-pruning evaluation.")
            skipping_ms += result.elapsed_ms
            if result.status == "timeout":
                timeout_blocks += 1
            elif result.status == "solver_error":
                error_blocks += 1
            if result.should_skip:
                skipped_blocks += 1
            else:
                kept_blocks.append(block_id)
            if show_progress:
                _print_verifier_progress(
                    current=index,
                    total=total_blocks,
                    block_id=block_id,
                    skipped=skipped_blocks,
                    kept=len(kept_blocks),
                    status=format_verifier_status(result) or result.status,
                    backend=result.backend,
                )
            if progress_callback is not None:
                progress_callback(
                    ProgressState(
                        job_index=job_index,
                        total_jobs=total_jobs,
                        filter_id=filter_id,
                        filter_name=filter_spec.name,
                        verifier_backend=result.backend,
                        current=index,
                        total=total_blocks,
                        skipped=skipped_blocks,
                        kept=len(kept_blocks),
                        block_id=block_id,
                        status=(format_verifier_status(result) or result.status),
                        done=False,
                    )
                )
    finally:
        if block_feature_con is not None:
            block_feature_con.close()
    if total_blocks and show_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return SkipSummary(
        kept_blocks=kept_blocks,
        timeout_blocks=timeout_blocks,
        error_blocks=error_blocks,
        skipping_ms=float(skipping_ms),
    )


def _print_verifier_progress(
    *,
    current: int,
    total: int,
    block_id: int,
    skipped: int,
    kept: int,
    status: str,
    backend: str,
) -> None:
    if total <= 0:
        return
    bar_width = max(10, min(40, total))
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    message = (
        f"\r[{backend}] [{bar}] {current}/{total} "
        f"block={block_id} status={status} skipped={skipped} kept={kept}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def _collect_metadata_bundle(
    *,
    model_spec: FunctionSpec,
    filter_spec: FilterSpec,
    db_path: Path,
    block_size: int,
    candidate_block_ids: list[int],
    kind: str,
    grid_depth: int,
) -> BlockMetadataBundle:
    if not candidate_block_ids:
        print("[bench] No benchmark blocks remain after excluding training rows")
        return BlockMetadataBundle(
            metadata_by_block={},
            collection_ms=0.0,
            collection_ms_by_block={},
        )

    pair_count = (len(model_spec.features) * (len(model_spec.features) - 1)) // 2
    print(
        f"[bench] Collecting {kind} block metadata for filter_template="
        f"{filter_spec.template_name or filter_spec.name} "
        f"model={model_spec.name} on {len(candidate_block_ids)} benchmark block(s) "
        f"before verifier checks"
    )
    if kind != "minmax" and pair_count:
        print(
            f"[bench] Building 2D metadata for filter_template="
            f"{filter_spec.template_name or filter_spec.name} "
            f"model={model_spec.name}: {pair_count} feature pair(s) per block"
        )
    bundle = collect_block_metadata(
        spec=model_spec,
        block_ids=candidate_block_ids,
        db_path=db_path,
        block_size=block_size,
        kind=kind,  # type: ignore[arg-type]
        grid_depth=grid_depth,
    )
    print(
        f"[bench] Collected {kind} block metadata for filter_template="
        f"{filter_spec.template_name or filter_spec.name} "
        f"model={model_spec.name}: {len(bundle.metadata_by_block)} block(s) in "
        f"{bundle.collection_ms:.3f} ms"
    )
    return bundle


def _format_results_for_display(
    results: list[dict],
    *,
    sql_preview_chars: int = 100,
) -> list[dict]:
    del sql_preview_chars
    return [dict(result) for result in results]


def _sanitize_results_label(label: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in label
    ).strip("_")



def _result_filename(*, args: argparse.Namespace, base_label: str, timestamp: str, digest: str) -> str:
    safe_label = _sanitize_results_label(base_label)
    execution_mode = _result_execution_mode_label(args)
    return (
        f"{safe_label or 'filters'}"
        f"__bs{args.block_size}"
        f"__vb{getattr(args, 'resolved_verifier_backend', args.verifier_backend)}"
        f"__bm{getattr(args, 'resolved_block_metadata_label', args.block_metadata or 'minmax')}"
        f"__em{execution_mode}"
        f"__mr{args.max_rows_total if args.max_rows_total is not None else 'all'}"
        f"__tt{args.task_type}"
        f"__a{args.range_alpha:g}"
        f"__n{args.range_start_samples}"
        f"__s{args.range_seed}"
        f"__ts{timestamp}"
        f"__{digest}.json"
    )



def _results_paths(
    args: argparse.Namespace,
    results: list[dict],
) -> dict[str, Path]:
    template_names = sorted(
        {
            str(result.get("filter_template_name") or result["filter_name"])
            for result in results
            if isinstance(result, dict) and result.get("filter_name")
        }
    )
    if not template_names:
        return {}

    cached_paths = getattr(args, "_results_paths_by_template", None)
    if isinstance(cached_paths, dict):
        return {
            template_name: cached_paths[template_name]
            for template_name in template_names
            if template_name in cached_paths
        }

    if args.results_path is not None and args.results_path.suffix == ".json":
        if len(template_names) != 1:
            raise ValueError(
                "--results-path may only point to a .json file when exactly one filter "
                "template result is produced; otherwise pass a directory path or omit --results-path."
            )
        return {template_names[0]: args.results_path}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selected_filters = sorted(args.filters or [])
    common_payload = {
        "database": args.database,
        "selected_filters": selected_filters,
        "model_kind": args.model_kind,
        "block_size": args.block_size,
        "range_alpha": args.range_alpha,
        "range_start_samples": args.range_start_samples,
        "range_seed": args.range_seed,
        "task_type": args.task_type,
        "max_rows_total": args.max_rows_total,
        "run_udf": args.run_udf,
        "measure_e2e": args.measure_e2e,
        "disable_skipping": args.disable_skipping,
        "jobs": args.jobs,
        "verifier_backend": getattr(args, "resolved_verifier_backend", args.verifier_backend),
        "requested_verifier_backend": args.verifier_backend,
        "batched_geomcad": args.batched_geomcad,
        "verifier_timeout_seconds": args.verifier_timeout_seconds,
        "block_metadata": args.block_metadata,
        "grid_depth": args.grid_depth,
    }

    root = args.results_path or benchmark_results_dir(args.database)
    paths: dict[str, Path] = {}
    for template_name in template_names:
        digest_input = json.dumps(
            {
                **common_payload,
                "selected_filters": selected_filters or [template_name],
                "result_filter_template_name": template_name,
            },
            sort_keys=True,
        )
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
        safe_label = _sanitize_results_label(template_name) or "filters"
        metadata_label = getattr(args, "resolved_block_metadata_label", args.block_metadata or "minmax")
        paths[template_name] = root / safe_label / args.model_kind / metadata_label / _result_filename(
            args=args,
            base_label=template_name,
            timestamp=timestamp,
            digest=digest,
        )
    args._results_paths_by_template = dict(paths)
    return paths


def _results_block_metadata_label(
    *,
    cli_block_metadata: str | None,
    jobs: list[BenchmarkJob],
    verifier_backend: str,
) -> str:
    del jobs
    if verifier_backend == "pytorch":
        return "none"
    return cli_block_metadata or "minmax"


def _results_verifier_backend_label(
    *,
    verifier_backend: str,
    batched_geomcad: bool,
) -> str:
    if verifier_backend == "geomcad" and batched_geomcad:
        return "batched_geomcad"
    return verifier_backend


def _result_execution_mode_label(args: argparse.Namespace) -> str:
    if args.measure_e2e:
        return "e2e_per_block"
    return "verification_only"


def _pruning_effectiveness_for_result(result: dict) -> float | None:
    reported = result.get("pruning_effectiveness_pct")
    if reported is not None:
        try:
            value = float(reported)
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            return value

    skipped = float(result.get("skipped_blocks", 0.0) or 0.0)
    skippable = float(result.get("skippable_blocks", 0.0) or 0.0)
    if skippable <= 0.0:
        return None
    return 100.0 * skipped / skippable


def _selectivity_bin_label(value: float, edges: tuple[float, ...] = _SELECTIVITY_BOX_EDGES) -> str | None:
    if value == 0.0:
        return "= 0"
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if hi <= 0.0:
            continue
        lower_ok = value > lo if lo == 0.0 else value >= lo
        upper_ok = value <= hi if index == len(edges) - 2 else value < hi
        if lower_ok and upper_ok:
            return f"{lo:g}-{hi:g}"
    return None


def _pruning_performance_summary(results: list[dict]) -> list[dict]:
    grouped: dict[str, list[float]] = {}
    for result in results:
        raw_selectivity = result.get("query_selectivity_pct")
        if raw_selectivity is None:
            continue
        try:
            selectivity = float(raw_selectivity)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(selectivity):
            continue

        pruning = _pruning_effectiveness_for_result(result)
        if pruning is None or not math.isfinite(pruning):
            continue

        label = _selectivity_bin_label(selectivity)
        if label is None:
            continue
        grouped.setdefault(label, []).append(float(pruning))

    ordered_labels: list[str] = []
    if "= 0" in grouped:
        ordered_labels.append("= 0")
    for lo, hi in zip(_SELECTIVITY_BOX_EDGES[:-1], _SELECTIVITY_BOX_EDGES[1:], strict=True):
        if hi <= 0.0:
            continue
        label = f"{lo:g}-{hi:g}"
        if label in grouped:
            ordered_labels.append(label)

    summary: list[dict] = []
    for label in ordered_labels:
        values = grouped[label]
        summary.append(
            {
                "selectivity_range": label,
                "count": len(values),
                "avg_pruning_effectiveness_pct": float(statistics.mean(values)),
                "median_pruning_effectiveness_pct": float(statistics.median(values)),
            }
        )
    return summary


def _write_results(
    path: Path,
    args: argparse.Namespace,
    results: list[dict],
    *,
    metadata_size_summary: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_results_payload(
        args,
        results,
        metadata_size_summary=metadata_size_summary,
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_results_for_template(
    args: argparse.Namespace,
    template_name: str,
    results: list[dict],
) -> Path | None:
    paths_by_filter = _results_paths(args, results)
    path = paths_by_filter.get(template_name)
    if path is None:
        return None
    metadata_by_template = getattr(args, "_metadata_size_summary_by_template", {})
    filter_results = [
        result
        for result in results
        if isinstance(result, dict)
        and (result.get("filter_template_name") or result.get("filter_name")) == template_name
    ]
    _write_results(
        path,
        args,
        filter_results,
        metadata_size_summary=metadata_by_template.get(
            template_name,
            getattr(args, "_metadata_size_summary", None),
        ),
    )
    return path


def _write_results_by_filter(args: argparse.Namespace, results: list[dict]) -> list[Path]:
    paths_by_filter = _results_paths(args, results)
    metadata_by_template = getattr(args, "_metadata_size_summary_by_template", {})
    written_paths: list[Path] = []
    for template_name, path in paths_by_filter.items():
        filter_results = [
            result
            for result in results
            if isinstance(result, dict)
            and (result.get("filter_template_name") or result.get("filter_name")) == template_name
        ]
        _write_results(
            path,
            args,
            filter_results,
            metadata_size_summary=metadata_by_template.get(
                template_name,
                getattr(args, "_metadata_size_summary", None),
            ),
        )
        written_paths.append(path)
    return written_paths


def _build_results_payload(
    args: argparse.Namespace,
    results: list[dict],
    *,
    metadata_size_summary: dict[str, object] | None = None,
) -> dict:
    if metadata_size_summary is None:
        metadata_size_summary = getattr(args, "_metadata_size_summary", None)
    pruning_performance_summary = _pruning_performance_summary(results)
    payload: dict[str, object] = {}
    payload["benchmark_mode"] = _result_execution_mode_label(args)
    if metadata_size_summary is not None:
        payload["metadata_size_summary"] = metadata_size_summary
    if pruning_performance_summary:
        payload["pruning_performance_by_selectivity_group"] = pruning_performance_summary
    payload["command"] = {
        "argv": list(sys.argv),
        "command": " ".join(_shell_quote(arg) for arg in sys.argv),
        "parameters": _serialize_args(args),
    }
    payload["results"] = results
    return payload


def _serialize_args(args: argparse.Namespace) -> dict:
    serialized: dict[str, object] = {}
    for key, value in vars(args).items():
        if key.startswith("_"):
            continue
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, list):
            serialized[key] = [
                str(item) if isinstance(item, Path) else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


def _shell_quote(value: str) -> str:
    if value == "":
        return "''"
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/:=")
    if all(char in safe_chars for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_benchmarks(args)
    if args.prepare_filters_only:
        display_results = _format_prepared_filters_for_display(results)
        print(json.dumps(display_results, indent=2))
        return
    written_paths = getattr(args, "_written_results_paths", None)
    if written_paths:
        written_paths = list(dict.fromkeys(written_paths))
    else:
        written_paths = _write_results_by_filter(args, results)
    print(
        f"[bench] Wrote {len(results)} result row(s) across "
        f"{len(written_paths)} per-filter result file(s)"
    )


if __name__ == "__main__":
    main()
