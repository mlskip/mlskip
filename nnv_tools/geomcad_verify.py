"""
The GeomCAD verifier requires each ONNX model to be precompiled into a DuckDB
artifact stored under metadata/compiled-models/.

Typical workflow:

1. Train the models, for example:

  uv run python train.py --database tpch --model-kind shallow --force-retrain

2. Compile the shallow regressor ONNX models into GeomCAD databases:

  scripts/compile_geomcad_models.sh tpch

3. Benchmark with GeomCAD:

  uv run python bench.py --database tpch --model-kind shallow --block-size 1000 --verifier-backend geomcad
"""

from __future__ import annotations

import atexit
import multiprocessing
import queue
import time
import traceback
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import duckdb as db

from nnv_tools.block_verifier import BlockSkipResult, BlockVerificationRequest
from nnv_tools.function_catalog import FeatureSpec, FunctionSpec
from nnv_tools.metadata_paths import METADATA_ROOT, REPO_ROOT


_QUERIES_DIR = REPO_ROOT / "geometrical-cad" / "queries"
_GEOMCAD_QUERY_2IN = (_QUERIES_DIR / "geomcad.sql").read_text()
_GEOMCAD_QUERY_3IN = (_QUERIES_DIR / "geomcad_3inputs.sql").read_text()
_GEOMCAD_QUERY_4IN = (_QUERIES_DIR / "geomcad_4inputs.sql").read_text()
_MODELS_ROOT = METADATA_ROOT / "models"
_COMPILED_MODELS_ROOT = METADATA_ROOT / "compiled-models"


@dataclass(frozen=True)
class _GeomcadSessionKey:
    compiled_model_path: str
    spec_name: str
    feature_names: tuple[str, ...]


@dataclass
class _PersistentGeomcadWorker:
    key: _GeomcadSessionKey
    ctx: multiprocessing.context.BaseContext
    request_queue: object
    response_queue: object
    process: multiprocessing.Process


_ACTIVE_WORKER: _PersistentGeomcadWorker | None = None


def decide_block_skip(request: BlockVerificationRequest) -> BlockSkipResult:
    if request.spec.task_type == "classifier":
        raise ValueError("geomcad verifier does not support classifier verification.")
    if request.predicate_lower is None or request.predicate_upper is None:
        raise ValueError("geomcad verifier requires predicate bounds.")

    target_desc = f"range=[{request.predicate_lower}, {request.predicate_upper}]"
    if request.verbose:
        print(f"[geomcad] Checking block_id={request.block_id} for {target_desc}")

    result = _decide_block_skip(request)

    if request.verbose:
        print(
            f"[geomcad] Result for block {request.block_id}: "
            f"should_skip={result.should_skip} ({format_verifier_status(result)})"
        )
    return result


def _decide_block_skip(request: BlockVerificationRequest) -> BlockSkipResult:
    worker = _get_or_create_worker(request)
    started_at = time.perf_counter()
    worker.request_queue.put({"kind": "solve", "request": asdict(request)})

    try:
        if request.timeout_seconds > 0:
            payload = worker.response_queue.get(timeout=request.timeout_seconds)
        else:
            payload = worker.response_queue.get()
    except queue.Empty:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        _invalidate_active_worker()
        return _timeout_result(request, elapsed_ms)

    if payload["kind"] == "result":
        return _block_skip_result_from_payload(payload["value"])

    _invalidate_active_worker()
    raise RuntimeError(
        "GeomCAD verification failed in a worker process:\n"
        f"{payload['type']}: {payload['message']}\n"
        f"{payload['traceback']}"
    )



def decide_block_skips_batched(requests: list[BlockVerificationRequest]) -> list[BlockSkipResult]:
    if not requests:
        return []

    first = requests[0]
    if first.spec.task_type == "classifier":
        raise ValueError("geomcad verifier does not support classifier verification.")

    worker_key = _worker_key(first)
    for request in requests:
        if request.spec.task_type == "classifier":
            raise ValueError("geomcad verifier does not support classifier verification.")
        if request.predicate_lower is None or request.predicate_upper is None:
            raise ValueError("geomcad verifier requires predicate bounds.")
        if _worker_key(request) != worker_key:
            raise ValueError(
                "Batched GeomCAD requests must share the same compiled model and feature schema."
            )

    batched_query = _batched_query_for_requests(requests)
    started_at = time.perf_counter()
    with db.connect(worker_key.compiled_model_path, read_only=True) as con:
        rows = con.execute(batched_query).fetchall()
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0

    reachable_by_block = {int(block_id): bool(f_reachable) for block_id, f_reachable in rows}
    per_block_ms = float(elapsed_ms) / len(requests)
    results: list[BlockSkipResult] = []
    for request in requests:
        f_reachable = reachable_by_block.get(int(request.block_id), False)
        if f_reachable:
            status = "sat"
            should_skip = False
            summary = (
                "Geomcad proved that some outputs in the block are within the target range, "
                "so the block should be kept."
            )
        else:
            status = "unsat"
            should_skip = True
            summary = (
                "Geomcad found no feasible output for the requested predicate, "
                "so the block can be skipped."
            )
        results.append(
            BlockSkipResult(
                backend="geomcad",
                block_id=request.block_id,
                predicate_lower=(
                    float(request.predicate_lower)
                    if request.predicate_lower is not None
                    else None
                ),
                predicate_upper=(
                    float(request.predicate_upper)
                    if request.predicate_upper is not None
                    else None
                ),
                target_class=request.target_class,
                status=status,
                should_skip=should_skip,
                elapsed_ms=per_block_ms,
                summary=summary,
                setup_ms=0.0,
                solve_ms=per_block_ms,
            )
        )
    return results



def _batched_values_clause(requests: list[BlockVerificationRequest]) -> str:
    features = requests[0].spec.features
    rows: list[str] = []
    for request in requests:
        if len(features) == 2:
            axis_names = ("x", "y")
        else:
            axis_names = tuple(f"x{i}" for i in range(1, len(features) + 1))
        values = [str(int(request.block_id))]
        for axis_name, feature in zip(axis_names, features, strict=True):
            min_bound, max_bound = request.input_bounds[feature.name]
            values.append(_sql_float(min_bound))
            values.append(_sql_float(max_bound))
        values.append(_sql_float(request.predicate_lower))
        values.append(_sql_float(request.predicate_upper))
        rows.append(f"({', '.join(values)})")
    if len(features) == 2:
        columns = "block_id, x_min, x_max, y_min, y_max, f_min, f_max"
    else:
        bound_columns = []
        for i in range(1, len(features) + 1):
            bound_columns.extend([f"x{i}_min", f"x{i}_max"])
        columns = ", ".join(["block_id", *bound_columns, "f_min", "f_max"])
    row_values = ",\n        ".join(rows)
    return (
        "bounds AS (\n"
        "    SELECT * FROM (VALUES\n"
        f"        {row_values}\n"
        f"    ) AS input_bounds({columns})\n"
        "),\n"
    )


def _sql_float(value: float | None) -> str:
    if value is None:
        raise ValueError("GeomCAD SQL generation requires finite numeric bounds.")
    return format(float(value), ".17g") + "::DOUBLE"


def _inline_query_for_request(request: BlockVerificationRequest) -> str:
    query_sql = _query_for_request(request)
    query_args = _query_args_for_request(request)
    for key, value in sorted(query_args.items(), key=lambda item: len(item[0]), reverse=True):
        query_sql = query_sql.replace(f"${key}", _sql_float(value))
    return query_sql


def _batched_geomcad_query_2in(requests: list[BlockVerificationRequest]) -> str:
    bounds_cte = _batched_values_clause(requests)
    return f"""
WITH
{bounds_cte}
matching_dim1_cells AS (
    SELECT
        b.block_id,
        d1.id,
        b.y_min,
        b.y_max,
        GREATEST(d1.a0_lower, b.x_min) AS x_left,
        LEAST(d1.a0_upper, b.x_max) AS x_right
    FROM Cell_Dim1 d1
    JOIN bounds b
      ON  d1.a0_upper >= b.x_min AND d1.a0_lower <= b.x_max
),
matching_dim2_cells AS (
    SELECT
        d1.block_id,
        d1.x_left,
        d1.x_right,
        d2.a0_lower + d2.a1_lower * d1.x_left AS y_bottom_left,
        d2.a0_lower + d2.a1_lower * d1.x_right AS y_bottom_right,
        d2.a0_upper + d2.a1_upper * d1.x_left AS y_top_left,
        d2.a0_upper + d2.a1_upper * d1.x_right AS y_top_right,
        d2.component_function_id,
        d1.y_min,
        d1.y_max
    FROM Cell_Dim2 d2
    JOIN matching_dim1_cells d1 ON d2.parent_cell_id = d1.id
    WHERE GREATEST(y_top_left, y_top_right) >= d1.y_min
        AND LEAST(y_bottom_left, y_bottom_right) <= d1.y_max
),
vertices_in_bounds AS (
    SELECT
        block_id,
        x_left AS x1,
        GREATEST(y_bottom_left, y_min) AS y1,
        x_left AS x2,
        LEAST(y_top_left, y_max) AS y2,
        x_right AS x3,
        GREATEST(y_bottom_right, y_min) AS y3,
        x_right AS x4,
        LEAST(y_top_right, y_max) AS y4,
        component_function_id
    FROM matching_dim2_cells
),
evaluated_polytopes AS (
    SELECT
        v.block_id,
        a0 + a1*x1 + a2*y1 AS v1,
        a0 + a1*x2 + a2*y2 AS v2,
        a0 + a1*x3 + a2*y3 AS v3,
        a0 + a1*x4 + a2*y4 AS v4
    FROM vertices_in_bounds v
    JOIN ComponentFunction c ON v.component_function_id = c.id
)
SELECT
    b.block_id,
    COALESCE(
        BOOL_OR(
            GREATEST(ep.v1, ep.v2, ep.v3, ep.v4) >= b.f_min
            AND LEAST(ep.v1, ep.v2, ep.v3, ep.v4) <= b.f_max
        ),
        FALSE
    ) AS f_reachable
FROM bounds b
LEFT JOIN evaluated_polytopes ep ON ep.block_id = b.block_id
GROUP BY b.block_id, b.f_min, b.f_max
ORDER BY b.block_id
"""


def _batched_geomcad_query_3in(requests: list[BlockVerificationRequest]) -> str:
    bounds_cte = _batched_values_clause(requests)
    return f"""
WITH
{bounds_cte}
matching_dim1_cells AS (
    SELECT
        b.block_id,
        d1.id,
        b.x2_min,
        b.x2_max,
        b.x3_min,
        b.x3_max,
        GREATEST(d1.a0_lower, b.x1_min) AS x1_left,
        LEAST(d1.a0_upper, b.x1_max) AS x1_right
    FROM Cell_Dim1 d1
    JOIN bounds b
      ON d1.a0_upper >= b.x1_min AND d1.a0_lower <= b.x1_max
),
matching_dim2_cells AS (
    SELECT
        d1.block_id,
        d2.id AS dim2_id,
        d1.x1_left,
        d1.x1_right,
        d1.x2_min,
        d1.x2_max,
        d1.x3_min,
        d1.x3_max,
        d2.a0_lower + d2.a1_lower * d1.x1_left AS x2_bottom_left,
        d2.a0_lower + d2.a1_lower * d1.x1_right AS x2_bottom_right,
        d2.a0_upper + d2.a1_upper * d1.x1_left AS x2_top_left,
        d2.a0_upper + d2.a1_upper * d1.x1_right AS x2_top_right
    FROM Cell_Dim2 d2
    JOIN matching_dim1_cells d1 ON d2.parent_cell_id = d1.id
    WHERE
        GREATEST(x2_top_left, x2_top_right) >= d1.x2_min
        AND LEAST(x2_bottom_left, x2_bottom_right) <= d1.x2_max
),
matching_dim2_cells_bounded AS (
    SELECT
        block_id,
        dim2_id,
        x1_left,
        x1_right,
        x3_min,
        x3_max,
        GREATEST(x2_bottom_left, x2_min) AS x2_bottom_left,
        LEAST(x2_top_left, x2_max) AS x2_top_left,
        GREATEST(x2_bottom_right, x2_min) AS x2_bottom_right,
        LEAST(x2_top_right, x2_max) AS x2_top_right
    FROM matching_dim2_cells
),
matching_dim3_cells AS (
    SELECT
        d2.block_id,
        d2.x1_left,
        d2.x1_right,
        d2.x2_bottom_left,
        d2.x2_bottom_right,
        d2.x2_top_left,
        d2.x2_top_right,
        d2.x3_min,
        d2.x3_max,
        d3.a0_lower + d3.a1_lower * d2.x1_left + d3.a2_lower * d2.x2_bottom_left AS x3_lower_bottom_left,
        d3.a0_lower + d3.a1_lower * d2.x1_left + d3.a2_lower * d2.x2_top_left AS x3_lower_top_left,
        d3.a0_lower + d3.a1_lower * d2.x1_right + d3.a2_lower * d2.x2_bottom_right AS x3_lower_bottom_right,
        d3.a0_lower + d3.a1_lower * d2.x1_right + d3.a2_lower * d2.x2_top_right AS x3_lower_top_right,
        d3.a0_upper + d3.a1_upper * d2.x1_left + d3.a2_upper * d2.x2_bottom_left AS x3_upper_bottom_left,
        d3.a0_upper + d3.a1_upper * d2.x1_left + d3.a2_upper * d2.x2_top_left AS x3_upper_top_left,
        d3.a0_upper + d3.a1_upper * d2.x1_right + d3.a2_upper * d2.x2_bottom_right AS x3_upper_bottom_right,
        d3.a0_upper + d3.a1_upper * d2.x1_right + d3.a2_upper * d2.x2_top_right AS x3_upper_top_right,
        d3.component_function_id
    FROM Cell_Dim3 d3
    JOIN matching_dim2_cells_bounded d2 ON d3.parent_cell_id = d2.dim2_id
    WHERE
        GREATEST(x3_upper_bottom_left, x3_upper_top_left, x3_upper_bottom_right, x3_upper_top_right) >= d2.x3_min
        AND LEAST(x3_lower_bottom_left, x3_lower_top_left, x3_lower_bottom_right, x3_lower_top_right) <= d2.x3_max
),
vertices_in_bounds AS (
    SELECT
        block_id,
        x1_left AS x1,
        x2_bottom_left AS y1,
        GREATEST(LEAST(x3_lower_bottom_left, x3_max), x3_min) AS z1,
        x1_left AS x2,
        x2_top_left AS y2,
        GREATEST(LEAST(x3_lower_top_left, x3_max), x3_min) AS z2,
        x1_right AS x3,
        x2_bottom_right AS y3,
        GREATEST(LEAST(x3_lower_bottom_right, x3_max), x3_min) AS z3,
        x1_right AS x4,
        x2_top_right AS y4,
        GREATEST(LEAST(x3_lower_top_right, x3_max), x3_min) AS z4,
        x1_left AS x5,
        x2_bottom_left AS y5,
        GREATEST(LEAST(x3_upper_bottom_left, x3_max), x3_min) AS z5,
        x1_left AS x6,
        x2_top_left AS y6,
        GREATEST(LEAST(x3_upper_top_left, x3_max), x3_min) AS z6,
        x1_right AS x7,
        x2_bottom_right AS y7,
        GREATEST(LEAST(x3_upper_bottom_right, x3_max), x3_min) AS z7,
        x1_right AS x8,
        x2_top_right AS y8,
        GREATEST(LEAST(x3_upper_top_right, x3_max), x3_min) AS z8,
        component_function_id
    FROM matching_dim3_cells
),
evaluated_polytopes AS (
    SELECT
        v.block_id,
        a0 + a1*x1 + a2*y1 + a3*z1 AS v1,
        a0 + a1*x2 + a2*y2 + a3*z2 AS v2,
        a0 + a1*x3 + a2*y3 + a3*z3 AS v3,
        a0 + a1*x4 + a2*y4 + a3*z4 AS v4,
        a0 + a1*x5 + a2*y5 + a3*z5 AS v5,
        a0 + a1*x6 + a2*y6 + a3*z6 AS v6,
        a0 + a1*x7 + a2*y7 + a3*z7 AS v7,
        a0 + a1*x8 + a2*y8 + a3*z8 AS v8
    FROM vertices_in_bounds v
    JOIN ComponentFunction c ON v.component_function_id = c.id
)
SELECT
    b.block_id,
    COALESCE(
        BOOL_OR(
            GREATEST(ep.v1, ep.v2, ep.v3, ep.v4, ep.v5, ep.v6, ep.v7, ep.v8) >= b.f_min
            AND LEAST(ep.v1, ep.v2, ep.v3, ep.v4, ep.v5, ep.v6, ep.v7, ep.v8) <= b.f_max
        ),
        FALSE
    ) AS f_reachable
FROM bounds b
LEFT JOIN evaluated_polytopes ep ON ep.block_id = b.block_id
GROUP BY b.block_id, b.f_min, b.f_max
ORDER BY b.block_id
"""


def _batched_geomcad_query_4in(requests: list[BlockVerificationRequest]) -> str:
    bounds_cte = _batched_values_clause(requests)
    return f"""
WITH
{bounds_cte}
matching_dim1_cells AS (
    SELECT
        b.block_id,
        d1.id,
        b.x2_min,
        b.x2_max,
        b.x3_min,
        b.x3_max,
        b.x4_min,
        b.x4_max,
        GREATEST(d1.a0_lower, b.x1_min) AS x1_left,
        LEAST(d1.a0_upper, b.x1_max) AS x1_right
    FROM Cell_Dim1 d1
    JOIN bounds b
      ON d1.a0_upper >= b.x1_min AND d1.a0_lower <= b.x1_max
),
matching_dim2_cells AS (
    SELECT
        d1.block_id,
        d2.id AS dim2_id,
        d1.x1_left,
        d1.x1_right,
        d1.x2_min,
        d1.x2_max,
        d1.x3_min,
        d1.x3_max,
        d1.x4_min,
        d1.x4_max,
        d2.a0_lower + d2.a1_lower * d1.x1_left AS x2_bottom_left,
        d2.a0_lower + d2.a1_lower * d1.x1_right AS x2_bottom_right,
        d2.a0_upper + d2.a1_upper * d1.x1_left AS x2_top_left,
        d2.a0_upper + d2.a1_upper * d1.x1_right AS x2_top_right
    FROM Cell_Dim2 d2
    JOIN matching_dim1_cells d1 ON d2.parent_cell_id = d1.id
    WHERE
        GREATEST(x2_top_left, x2_top_right) >= d1.x2_min
        AND LEAST(x2_bottom_left, x2_bottom_right) <= d1.x2_max
),
matching_dim2_cells_bounded AS (
    SELECT
        block_id,
        dim2_id,
        x1_left,
        x1_right,
        x3_min,
        x3_max,
        x4_min,
        x4_max,
        GREATEST(x2_bottom_left, x2_min) AS x2_bottom_left,
        LEAST(x2_top_left, x2_max) AS x2_top_left,
        GREATEST(x2_bottom_right, x2_min) AS x2_bottom_right,
        LEAST(x2_top_right, x2_max) AS x2_top_right
    FROM matching_dim2_cells
),
matching_dim3_cells AS (
    SELECT
        d2.block_id,
        d3.id AS dim3_id,
        d2.x1_left,
        d2.x1_right,
        d2.x2_bottom_left,
        d2.x2_bottom_right,
        d2.x2_top_left,
        d2.x2_top_right,
        d2.x3_min,
        d2.x3_max,
        d2.x4_min,
        d2.x4_max,
        d3.a0_lower + d3.a1_lower * d2.x1_left + d3.a2_lower * d2.x2_bottom_left AS x3_lower_bottom_left,
        d3.a0_lower + d3.a1_lower * d2.x1_left + d3.a2_lower * d2.x2_top_left AS x3_lower_top_left,
        d3.a0_lower + d3.a1_lower * d2.x1_right + d3.a2_lower * d2.x2_bottom_right AS x3_lower_bottom_right,
        d3.a0_lower + d3.a1_lower * d2.x1_right + d3.a2_lower * d2.x2_top_right AS x3_lower_top_right,
        d3.a0_upper + d3.a1_upper * d2.x1_left + d3.a2_upper * d2.x2_bottom_left AS x3_upper_bottom_left,
        d3.a0_upper + d3.a1_upper * d2.x1_left + d3.a2_upper * d2.x2_top_left AS x3_upper_top_left,
        d3.a0_upper + d3.a1_upper * d2.x1_right + d3.a2_upper * d2.x2_bottom_right AS x3_upper_bottom_right,
        d3.a0_upper + d3.a1_upper * d2.x1_right + d3.a2_upper * d2.x2_top_right AS x3_upper_top_right
    FROM Cell_Dim3 d3
    JOIN matching_dim2_cells_bounded d2 ON d3.parent_cell_id = d2.dim2_id
    WHERE
        GREATEST(x3_upper_bottom_left, x3_upper_top_left, x3_upper_bottom_right, x3_upper_top_right) >= d2.x3_min
        AND LEAST(x3_lower_bottom_left, x3_lower_top_left, x3_lower_bottom_right, x3_lower_top_right) <= d2.x3_max
),
matching_dim3_cells_bounded AS (
    SELECT
        block_id,
        dim3_id,
        x1_left,
        x1_right,
        x2_bottom_left,
        x2_top_left,
        x2_bottom_right,
        x2_top_right,
        GREATEST(LEAST(x3_lower_bottom_left, x3_max), x3_min) AS x3_lower_bottom_left,
        GREATEST(LEAST(x3_lower_top_left, x3_max), x3_min) AS x3_lower_top_left,
        GREATEST(LEAST(x3_lower_bottom_right, x3_max), x3_min) AS x3_lower_bottom_right,
        GREATEST(LEAST(x3_lower_top_right, x3_max), x3_min) AS x3_lower_top_right,
        GREATEST(LEAST(x3_upper_bottom_left, x3_max), x3_min) AS x3_upper_bottom_left,
        GREATEST(LEAST(x3_upper_top_left, x3_max), x3_min) AS x3_upper_top_left,
        GREATEST(LEAST(x3_upper_bottom_right, x3_max), x3_min) AS x3_upper_bottom_right,
        GREATEST(LEAST(x3_upper_top_right, x3_max), x3_min) AS x3_upper_top_right,
        x4_min,
        x4_max
    FROM matching_dim3_cells
),
matching_dim4_cells AS (
    SELECT
        d3.block_id,
        d3.x1_left,
        d3.x1_right,
        d3.x2_bottom_left,
        d3.x2_bottom_right,
        d3.x2_top_left,
        d3.x2_top_right,
        d3.x3_lower_bottom_left,
        d3.x3_lower_top_left,
        d3.x3_lower_bottom_right,
        d3.x3_lower_top_right,
        d3.x3_upper_bottom_left,
        d3.x3_upper_top_left,
        d3.x3_upper_bottom_right,
        d3.x3_upper_top_right,
        d4.a0_lower + d4.a1_lower * d3.x1_left + d4.a2_lower * d3.x2_bottom_left + d4.a3_lower * d3.x3_lower_bottom_left AS x4_lower_lb_lower,
        d4.a0_lower + d4.a1_lower * d3.x1_left + d4.a2_lower * d3.x2_bottom_left + d4.a3_lower * d3.x3_upper_bottom_left AS x4_lower_lb_upper,
        d4.a0_lower + d4.a1_lower * d3.x1_left + d4.a2_lower * d3.x2_top_left + d4.a3_lower * d3.x3_lower_top_left AS x4_lower_lt_lower,
        d4.a0_lower + d4.a1_lower * d3.x1_left + d4.a2_lower * d3.x2_top_left + d4.a3_lower * d3.x3_upper_top_left AS x4_lower_lt_upper,
        d4.a0_lower + d4.a1_lower * d3.x1_right + d4.a2_lower * d3.x2_bottom_right + d4.a3_lower * d3.x3_lower_bottom_right AS x4_lower_rb_lower,
        d4.a0_lower + d4.a1_lower * d3.x1_right + d4.a2_lower * d3.x2_bottom_right + d4.a3_lower * d3.x3_upper_bottom_right AS x4_lower_rb_upper,
        d4.a0_lower + d4.a1_lower * d3.x1_right + d4.a2_lower * d3.x2_top_right + d4.a3_lower * d3.x3_lower_top_right AS x4_lower_rt_lower,
        d4.a0_lower + d4.a1_lower * d3.x1_right + d4.a2_lower * d3.x2_top_right + d4.a3_lower * d3.x3_upper_top_right AS x4_lower_rt_upper,
        d4.a0_upper + d4.a1_upper * d3.x1_left + d4.a2_upper * d3.x2_bottom_left + d4.a3_upper * d3.x3_lower_bottom_left AS x4_upper_lb_lower,
        d4.a0_upper + d4.a1_upper * d3.x1_left + d4.a2_upper * d3.x2_bottom_left + d4.a3_upper * d3.x3_upper_bottom_left AS x4_upper_lb_upper,
        d4.a0_upper + d4.a1_upper * d3.x1_left + d4.a2_upper * d3.x2_top_left + d4.a3_upper * d3.x3_lower_top_left AS x4_upper_lt_lower,
        d4.a0_upper + d4.a1_upper * d3.x1_left + d4.a2_upper * d3.x2_top_left + d4.a3_upper * d3.x3_upper_top_left AS x4_upper_lt_upper,
        d4.a0_upper + d4.a1_upper * d3.x1_right + d4.a2_upper * d3.x2_bottom_right + d4.a3_upper * d3.x3_lower_bottom_right AS x4_upper_rb_lower,
        d4.a0_upper + d4.a1_upper * d3.x1_right + d4.a2_upper * d3.x2_bottom_right + d4.a3_upper * d3.x3_upper_bottom_right AS x4_upper_rb_upper,
        d4.a0_upper + d4.a1_upper * d3.x1_right + d4.a2_upper * d3.x2_top_right + d4.a3_upper * d3.x3_lower_top_right AS x4_upper_rt_lower,
        d4.a0_upper + d4.a1_upper * d3.x1_right + d4.a2_upper * d3.x2_top_right + d4.a3_upper * d3.x3_upper_top_right AS x4_upper_rt_upper,
        d3.x4_min,
        d3.x4_max,
        d4.component_function_id
    FROM Cell_Dim4 d4
    JOIN matching_dim3_cells_bounded d3 ON d4.parent_cell_id = d3.dim3_id
    WHERE
        GREATEST(
            x4_upper_lb_lower, x4_upper_lb_upper,
            x4_upper_lt_lower, x4_upper_lt_upper,
            x4_upper_rb_lower, x4_upper_rb_upper,
            x4_upper_rt_lower, x4_upper_rt_upper
        ) >= d3.x4_min
        AND LEAST(
            x4_lower_lb_lower, x4_lower_lb_upper,
            x4_lower_lt_lower, x4_lower_lt_upper,
            x4_lower_rb_lower, x4_lower_rb_upper,
            x4_lower_rt_lower, x4_lower_rt_upper
        ) <= d3.x4_max
),
vertices_in_bounds AS (
    SELECT
        block_id,
        x1_left AS x1,
        x2_bottom_left AS y1,
        x3_lower_bottom_left AS z1,
        GREATEST(LEAST(x4_lower_lb_lower, x4_max), x4_min) AS w1,
        x1_left AS x2,
        x2_bottom_left AS y2,
        x3_upper_bottom_left AS z2,
        GREATEST(LEAST(x4_lower_lb_upper, x4_max), x4_min) AS w2,
        x1_left AS x3,
        x2_top_left AS y3,
        x3_lower_top_left AS z3,
        GREATEST(LEAST(x4_lower_lt_lower, x4_max), x4_min) AS w3,
        x1_left AS x4,
        x2_top_left AS y4,
        x3_upper_top_left AS z4,
        GREATEST(LEAST(x4_lower_lt_upper, x4_max), x4_min) AS w4,
        x1_right AS x5,
        x2_bottom_right AS y5,
        x3_lower_bottom_right AS z5,
        GREATEST(LEAST(x4_lower_rb_lower, x4_max), x4_min) AS w5,
        x1_right AS x6,
        x2_bottom_right AS y6,
        x3_upper_bottom_right AS z6,
        GREATEST(LEAST(x4_lower_rb_upper, x4_max), x4_min) AS w6,
        x1_right AS x7,
        x2_top_right AS y7,
        x3_lower_top_right AS z7,
        GREATEST(LEAST(x4_lower_rt_lower, x4_max), x4_min) AS w7,
        x1_right AS x8,
        x2_top_right AS y8,
        x3_upper_top_right AS z8,
        GREATEST(LEAST(x4_lower_rt_upper, x4_max), x4_min) AS w8,
        x1_left AS x9,
        x2_bottom_left AS y9,
        x3_lower_bottom_left AS z9,
        GREATEST(LEAST(x4_upper_lb_lower, x4_max), x4_min) AS w9,
        x1_left AS x10,
        x2_bottom_left AS y10,
        x3_upper_bottom_left AS z10,
        GREATEST(LEAST(x4_upper_lb_upper, x4_max), x4_min) AS w10,
        x1_left AS x11,
        x2_top_left AS y11,
        x3_lower_top_left AS z11,
        GREATEST(LEAST(x4_upper_lt_lower, x4_max), x4_min) AS w11,
        x1_left AS x12,
        x2_top_left AS y12,
        x3_upper_top_left AS z12,
        GREATEST(LEAST(x4_upper_lt_upper, x4_max), x4_min) AS w12,
        x1_right AS x13,
        x2_bottom_right AS y13,
        x3_lower_bottom_right AS z13,
        GREATEST(LEAST(x4_upper_rb_lower, x4_max), x4_min) AS w13,
        x1_right AS x14,
        x2_bottom_right AS y14,
        x3_upper_bottom_right AS z14,
        GREATEST(LEAST(x4_upper_rb_upper, x4_max), x4_min) AS w14,
        x1_right AS x15,
        x2_top_right AS y15,
        x3_lower_top_right AS z15,
        GREATEST(LEAST(x4_upper_rt_lower, x4_max), x4_min) AS w15,
        x1_right AS x16,
        x2_top_right AS y16,
        x3_upper_top_right AS z16,
        GREATEST(LEAST(x4_upper_rt_upper, x4_max), x4_min) AS w16,
        component_function_id
    FROM matching_dim4_cells
),
evaluated_polytopes AS (
    SELECT
        v.block_id,
        a0 + a1*x1 + a2*y1 + a3*z1 + a4*w1 AS v1,
        a0 + a1*x2 + a2*y2 + a3*z2 + a4*w2 AS v2,
        a0 + a1*x3 + a2*y3 + a3*z3 + a4*w3 AS v3,
        a0 + a1*x4 + a2*y4 + a3*z4 + a4*w4 AS v4,
        a0 + a1*x5 + a2*y5 + a3*z5 + a4*w5 AS v5,
        a0 + a1*x6 + a2*y6 + a3*z6 + a4*w6 AS v6,
        a0 + a1*x7 + a2*y7 + a3*z7 + a4*w7 AS v7,
        a0 + a1*x8 + a2*y8 + a3*z8 + a4*w8 AS v8,
        a0 + a1*x9 + a2*y9 + a3*z9 + a4*w9 AS v9,
        a0 + a1*x10 + a2*y10 + a3*z10 + a4*w10 AS v10,
        a0 + a1*x11 + a2*y11 + a3*z11 + a4*w11 AS v11,
        a0 + a1*x12 + a2*y12 + a3*z12 + a4*w12 AS v12,
        a0 + a1*x13 + a2*y13 + a3*z13 + a4*w13 AS v13,
        a0 + a1*x14 + a2*y14 + a3*z14 + a4*w14 AS v14,
        a0 + a1*x15 + a2*y15 + a3*z15 + a4*w15 AS v15,
        a0 + a1*x16 + a2*y16 + a3*z16 + a4*w16 AS v16
    FROM vertices_in_bounds v
    JOIN ComponentFunction c ON v.component_function_id = c.id
)
SELECT
    b.block_id,
    COALESCE(
        BOOL_OR(
            GREATEST(ep.v1, ep.v2, ep.v3, ep.v4, ep.v5, ep.v6, ep.v7, ep.v8, ep.v9, ep.v10, ep.v11, ep.v12, ep.v13, ep.v14, ep.v15, ep.v16) >= b.f_min
            AND LEAST(ep.v1, ep.v2, ep.v3, ep.v4, ep.v5, ep.v6, ep.v7, ep.v8, ep.v9, ep.v10, ep.v11, ep.v12, ep.v13, ep.v14, ep.v15, ep.v16) <= b.f_max
        ),
        FALSE
    ) AS f_reachable
FROM bounds b
LEFT JOIN evaluated_polytopes ep ON ep.block_id = b.block_id
GROUP BY b.block_id, b.f_min, b.f_max
ORDER BY b.block_id
"""


def _get_or_create_worker(request: BlockVerificationRequest) -> _PersistentGeomcadWorker:
    global _ACTIVE_WORKER

    key = _worker_key(request)
    if _ACTIVE_WORKER is not None and _ACTIVE_WORKER.key == key and _ACTIVE_WORKER.process.is_alive():
        return _ACTIVE_WORKER

    _invalidate_active_worker()

    ctx = multiprocessing.get_context(_worker_start_method())
    request_queue = ctx.Queue(maxsize=1)
    response_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_persistent_worker_entrypoint,
        args=(request_queue, response_queue, key.compiled_model_path),
        daemon=True,
    )
    process.start()
    _ACTIVE_WORKER = _PersistentGeomcadWorker(
        key=key,
        ctx=ctx,
        request_queue=request_queue,
        response_queue=response_queue,
        process=process,
    )
    return _ACTIVE_WORKER


def _worker_key(request: BlockVerificationRequest) -> _GeomcadSessionKey:
    return _GeomcadSessionKey(
        compiled_model_path=str(geomcad_compiled_model_path(request.model_path)),
        spec_name=request.spec.name,
        feature_names=tuple(feature.name for feature in request.spec.features),
    )


def _persistent_worker_entrypoint(request_queue, response_queue, db_path: str) -> None:
    try:
        con = db.connect(db_path, read_only=True)
        while True:
            message = request_queue.get()
            if message["kind"] == "close":
                break
            request = _request_from_payload(message["request"])
            result = _solve_with_connection(con=con, request=request)
            response_queue.put({"kind": "result", "value": asdict(result)})
    except Exception as exc:  # pragma: no cover
        response_queue.put(
            {
                "kind": "error",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _solve_with_connection(
    *,
    con: db.DuckDBPyConnection,
    request: BlockVerificationRequest,
) -> BlockSkipResult:
    geomcad_query = _query_for_request(request)
    query_args = _query_args_for_request(request)

    solve_started_at = time.perf_counter()
    (f_reachable,) = con.execute(geomcad_query, query_args).fetchone()
    solve_ms = (time.perf_counter() - solve_started_at) * 1000.0

    if f_reachable:
        status = "sat"
        should_skip = False
        summary = (
            "Geomcad proved that some outputs in the block are within the target range, "
            "so the block should be kept."
        )
    else:
        status = "unsat"
        should_skip = True
        summary = (
            "Geomcad found no feasible output for the requested predicate, "
            "so the block can be skipped."
        )

    return BlockSkipResult(
        backend="geomcad",
        block_id=request.block_id,
        predicate_lower=(
            float(request.predicate_lower)
            if request.predicate_lower is not None
            else None
        ),
        predicate_upper=(
            float(request.predicate_upper)
            if request.predicate_upper is not None
            else None
        ),
        target_class=request.target_class,
        status=status,
        should_skip=should_skip,
        elapsed_ms=float(solve_ms),
        summary=summary,
        setup_ms=0.0,
        solve_ms=float(solve_ms),
    )


def _query_for_request(request: BlockVerificationRequest) -> str:
    match len(request.spec.features):
        case 2:
            return _GEOMCAD_QUERY_2IN
        case 3:
            return _GEOMCAD_QUERY_3IN
        case 4:
            return _GEOMCAD_QUERY_4IN
        case _:
            raise ValueError(
                "geomcad verifier requires exactly 2, 3, or 4 input features, "
                f"got {len(request.spec.features)}."
            )


def _batched_query_for_requests(requests: list[BlockVerificationRequest]) -> str:
    match len(requests[0].spec.features):
        case 2:
            return _batched_geomcad_query_2in(requests)
        case 3:
            return _batched_geomcad_query_3in(requests)
        case 4:
            return _batched_geomcad_query_4in(requests)
        case _:
            raise ValueError(
                "geomcad verifier requires exactly 2, 3, or 4 input features, "
                f"got {len(requests[0].spec.features)}."
            )

def _query_args_for_request(request: BlockVerificationRequest) -> dict[str, float]:
    query_args = {
        "f_min": float(request.predicate_lower),
        "f_max": float(request.predicate_upper),
    }
    features = request.spec.features
    if len(features) == 2:
        axis_names = ("x", "y")
        for axis_name, feature in zip(axis_names, features, strict=True):
            min_bound, max_bound = request.input_bounds[feature.name]
            query_args[f"{axis_name}_min"] = float(min_bound)
            query_args[f"{axis_name}_max"] = float(max_bound)
    else:
        for i, feature in enumerate(features, start=1):
            min_bound, max_bound = request.input_bounds[feature.name]
            query_args[f"x{i}_min"] = float(min_bound)
            query_args[f"x{i}_max"] = float(max_bound)
    return query_args


def _request_from_payload(payload: dict) -> BlockVerificationRequest:
    model_path = payload["model_path"]
    if isinstance(model_path, dict):
        model_path = model_path.get("path", model_path)
    raw_spec = payload["spec"]
    spec = FunctionSpec(
        name=raw_spec["name"],
        description=raw_spec["description"],
        database=raw_spec["database"],
        table=raw_spec["table"],
        task_type=raw_spec["task_type"],
        target_expression=raw_spec["target_expression"],
        features=[FeatureSpec(**feature) for feature in raw_spec["features"]],
        num_classes=raw_spec.get("num_classes"),
    )
    return BlockVerificationRequest(
        model_path=model_path,
        spec=spec,
        input_bounds={
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in payload["input_bounds"].items()
        },
        block_id=int(payload["block_id"]),
        pair_geometries=None,
        predicate_lower=payload["predicate_lower"],
        predicate_upper=payload["predicate_upper"],
        target_class=payload["target_class"],
        timeout_seconds=float(payload["timeout_seconds"]),
        verbose=bool(payload["verbose"]),
    )


def _block_skip_result_from_payload(payload: dict) -> BlockSkipResult:
    return BlockSkipResult(
        backend=payload["backend"],
        block_id=int(payload["block_id"]),
        predicate_lower=payload["predicate_lower"],
        predicate_upper=payload["predicate_upper"],
        target_class=payload["target_class"],
        status=payload["status"],
        should_skip=bool(payload["should_skip"]),
        elapsed_ms=float(payload["elapsed_ms"]),
        summary=payload["summary"],
        setup_ms=(
            float(payload["setup_ms"]) if payload.get("setup_ms") is not None else None
        ),
        solve_ms=(
            float(payload["solve_ms"]) if payload.get("solve_ms") is not None else None
        ),
    )


def _timeout_result(request: BlockVerificationRequest, elapsed_ms: float) -> BlockSkipResult:
    return BlockSkipResult(
        backend="geomcad",
        block_id=request.block_id,
        predicate_lower=(
            float(request.predicate_lower)
            if request.predicate_lower is not None
            else None
        ),
        predicate_upper=(
            float(request.predicate_upper)
            if request.predicate_upper is not None
            else None
        ),
        target_class=request.target_class,
        status="timeout",
        should_skip=False,
        elapsed_ms=float(elapsed_ms),
        summary="GeomCAD timed out before it could prove the block skippable.",
        setup_ms=None,
        solve_ms=None,
    )


def _invalidate_active_worker() -> None:
    global _ACTIVE_WORKER
    if _ACTIVE_WORKER is None:
        return
    _stop_worker(_ACTIVE_WORKER)
    _ACTIVE_WORKER = None


def _stop_worker(worker: _PersistentGeomcadWorker) -> None:
    if worker.process.is_alive():
        try:
            worker.request_queue.put_nowait({"kind": "close"})
        except Exception:
            pass
        worker.process.join(timeout=_shutdown_grace_seconds(0.05))
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout=_shutdown_grace_seconds(0.05))
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join(timeout=_shutdown_grace_seconds(0.05))
    _close_queue(worker.request_queue)
    _close_queue(worker.response_queue)


def _shutdown_grace_seconds(timeout_seconds: float) -> float:
    return min(0.05, max(0.005, timeout_seconds / 2.0))


def _worker_start_method() -> str:
    available_methods = multiprocessing.get_all_start_methods()
    if "fork" in available_methods:
        return "fork"
    return "spawn"


def _close_queue(work_queue) -> None:
    work_queue.close()
    work_queue.join_thread()


def geomcad_compiled_model_path(onnx_path: str | Path) -> Path:
    onnx_path = Path(onnx_path)
    try:
        relative_path = onnx_path.relative_to(_MODELS_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Expected model path under {_MODELS_ROOT}, got {onnx_path}."
        ) from exc
    compiled_path = (_COMPILED_MODELS_ROOT / relative_path).with_suffix(".geomcad.db")
    if not compiled_path.exists():
        raise FileNotFoundError(
            "geomcad requires a precompiled model database under metadata/compiled-models. "
            f"Missing artifact: {compiled_path}. "
            "Run scripts/compile_geomcad_models.sh before benchmarking."
        )
    return compiled_path


atexit.register(_invalidate_active_worker)
