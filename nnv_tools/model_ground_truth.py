from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd

from nnv_tools.dataset_duckdb import (
    ROW_ID_COLUMN,
    build_block_id_predicate,
    ensure_row_id_column,
)
from nnv_tools.filter_catalog import FilterSpec
from nnv_tools.function_catalog import FunctionSpec
from nnv_tools.model_runtime import predict_array


@dataclass(frozen=True)
class ModelGroundTruthRequest:
    request_id: int
    filter_spec: FilterSpec
    model_spec: FunctionSpec
    model_path: Path
    block_ids: list[int]


@dataclass(frozen=True)
class ModelGroundTruthCache:
    cache_path: Path
    cache_key: str
    row_count: int
    qualified_count: int
    matching_blocks: int
    materialize_ms: float
    reused: bool


def model_ground_truth_cache_path(db_path: str | Path) -> Path:
    source = Path(db_path)
    return source.with_name(f"{source.stem}__model_ground_truth.duckdb")


def ensure_model_ground_truth_cache(
    *,
    db_path: str | Path,
    filter_spec: FilterSpec,
    model_spec: FunctionSpec,
    model_path: str | Path,
    block_ids: list[int],
    block_size: int,
    cache_path: str | Path | None = None,
) -> ModelGroundTruthCache:
    request = ModelGroundTruthRequest(
        request_id=0,
        filter_spec=filter_spec,
        model_spec=model_spec,
        model_path=Path(model_path),
        block_ids=block_ids,
    )
    return ensure_model_ground_truth_caches(
        db_path=db_path,
        requests=[request],
        block_size=block_size,
        cache_path=cache_path,
    )[0]


def ensure_model_ground_truth_caches(
    *,
    db_path: str | Path,
    requests: list[ModelGroundTruthRequest],
    block_size: int,
    cache_path: str | Path | None = None,
) -> dict[int, ModelGroundTruthCache]:
    db_path = Path(db_path)
    cache_path = Path(cache_path) if cache_path is not None else model_ground_truth_cache_path(db_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[int, ModelGroundTruthCache] = {}
    missing: list[tuple[ModelGroundTruthRequest, str]] = []

    with duckdb.connect(str(cache_path)) as cache_con:
        _ensure_cache_schema(cache_con)
        for request in requests:
            cache_key = _cache_key(
                request.filter_spec,
                request.model_spec,
                request.model_path,
                request.block_ids,
                block_size,
            )
            if not request.block_ids:
                results[request.request_id] = ModelGroundTruthCache(
                    cache_path=cache_path,
                    cache_key=cache_key,
                    row_count=0,
                    qualified_count=0,
                    matching_blocks=0,
                    materialize_ms=0.0,
                    reused=True,
                )
                continue

            cached = _fetch_cached_summary(cache_con, cache_key)
            if cached is None:
                missing.append((request, cache_key))
                continue

            row_count, qualified_count, matching_blocks = cached
            results[request.request_id] = ModelGroundTruthCache(
                cache_path=cache_path,
                cache_key=cache_key,
                row_count=row_count,
                qualified_count=qualified_count,
                matching_blocks=matching_blocks,
                materialize_ms=0.0,
                reused=True,
            )

    if not missing:
        return results

    grouped: dict[tuple, list[tuple[ModelGroundTruthRequest, str]]] = {}
    for item in missing:
        request, _cache_key_value = item
        grouped.setdefault(_prediction_group_key(request, block_size), []).append(item)

    with duckdb.connect(str(cache_path)) as cache_con:
        _ensure_cache_schema(cache_con)
        for group_items in grouped.values():
            group_started_at = perf_counter()
            first_request = group_items[0][0]
            frame, predictions = _load_predictions(
                db_path=db_path,
                request=first_request,
                block_size=block_size,
            )
            row_count = int(len(frame))

            for request, cache_key in group_items:
                qualifies = _qualifies(request.filter_spec, request.model_spec, predictions)
                qualified_count = int(np.count_nonzero(qualifies))
                matching_blocks = int(frame.loc[qualifies, "block_id"].nunique()) if row_count else 0

                cache_con.execute("DELETE FROM filter_qualifications WHERE cache_key = ?", [cache_key])
                cache_con.execute("DELETE FROM filter_metadata WHERE cache_key = ?", [cache_key])
                if row_count:
                    qualification_frame = pd.DataFrame(
                        {
                            "cache_key": cache_key,
                            "row_id": frame["row_id"].to_numpy(dtype=np.int64),
                            "qualifies": qualifies.astype(bool),
                        }
                    )
                    cache_con.register("qualification_frame", qualification_frame)
                    cache_con.execute(
                        """
                        INSERT INTO filter_qualifications(cache_key, row_id, qualifies)
                        SELECT cache_key, row_id, qualifies
                        FROM qualification_frame
                        """
                    )
                    cache_con.unregister("qualification_frame")

                cache_con.execute(
                    """
                    INSERT INTO filter_metadata(
                        cache_key, created_at, database_name, table_name, model_name, filter_name,
                        model_path, model_mtime_ns, block_size, block_ids_json, filter_signature,
                        row_count, qualified_count, matching_blocks
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        cache_key,
                        datetime.now(timezone.utc).isoformat(),
                        request.filter_spec.database,
                        request.filter_spec.table,
                        request.model_spec.name,
                        request.filter_spec.name,
                        str(request.model_path),
                        _model_mtime_ns(request.model_path),
                        int(block_size),
                        json.dumps(sorted(int(block_id) for block_id in request.block_ids)),
                        _filter_signature(request.filter_spec),
                        row_count,
                        qualified_count,
                        matching_blocks,
                    ],
                )

                results[request.request_id] = ModelGroundTruthCache(
                    cache_path=cache_path,
                    cache_key=cache_key,
                    row_count=row_count,
                    qualified_count=qualified_count,
                    matching_blocks=matching_blocks,
                    materialize_ms=(perf_counter() - group_started_at) * 1000.0,
                    reused=False,
                )

    return results


def count_model_qualified_rows(
    cache_path: str | Path,
    cache_key: str,
    block_ids: list[int] | None = None,
    block_size: int | None = None,
) -> int:
    with duckdb.connect(str(cache_path), read_only=True) as con:
        if block_ids is None:
            return int(con.execute(
                "SELECT COUNT(*) FROM filter_qualifications WHERE cache_key = ? AND qualifies",
                [cache_key],
            ).fetchone()[0])
        if not block_ids:
            return 0
        if block_size is None:
            raise ValueError("block_size is required when block_ids are provided.")
        return int(con.execute(
            f"""
            SELECT COUNT(*)
            FROM filter_qualifications
            WHERE cache_key = ?
              AND qualifies
              AND CAST(FLOOR(row_id / {int(block_size)}) AS BIGINT) IN ({_int_list(block_ids)})
            """,
            [cache_key],
        ).fetchone()[0])


def count_model_matching_blocks(
    cache_path: str | Path,
    cache_key: str,
    block_ids: list[int],
    block_size: int,
) -> int:
    if not block_ids:
        return 0
    with duckdb.connect(str(cache_path), read_only=True) as con:
        return int(con.execute(
            f"""
            SELECT COUNT(
                DISTINCT CAST(FLOOR(row_id / {int(block_size)}) AS BIGINT)
            )
            FROM filter_qualifications
            WHERE cache_key = ?
              AND qualifies
              AND CAST(FLOOR(row_id / {int(block_size)}) AS BIGINT) IN ({_int_list(block_ids)})
            """,
            [cache_key],
        ).fetchone()[0])


def _load_predictions(
    *,
    db_path: Path,
    request: ModelGroundTruthRequest,
    block_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    block_predicate = build_block_id_predicate(request.block_ids, block_size)
    select_items = [
        f"{feature.expression} AS feature_{index}"
        for index, feature in enumerate(request.model_spec.features)
    ]
    block_expression = f"CAST(FLOOR({ROW_ID_COLUMN} / {int(block_size)}) AS BIGINT)"
    with duckdb.connect(str(db_path), read_only=True) as source_con:
        ensure_row_id_column(source_con, request.filter_spec.table)
        frame = source_con.execute(
            f"""
            SELECT
                {ROW_ID_COLUMN} AS row_id,
                {block_expression} AS block_id,
                {", ".join(select_items)}
            FROM {request.filter_spec.table}
            WHERE {block_predicate}
            ORDER BY {ROW_ID_COLUMN}
            """
        ).fetch_df()

    if frame.empty:
        return frame, np.asarray([], dtype=np.float64)

    feature_columns = [f"feature_{index}" for index in range(len(request.model_spec.features))]
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    predictions = predict_array(request.model_spec, request.model_path, features)
    return frame[["row_id", "block_id"]].copy(), predictions


def _qualifies(filter_spec: FilterSpec, model_spec: FunctionSpec, predictions: np.ndarray) -> np.ndarray:
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


def _ensure_cache_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS filter_metadata (
            cache_key TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            database_name TEXT NOT NULL,
            table_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            filter_name TEXT NOT NULL,
            model_path TEXT NOT NULL,
            model_mtime_ns BIGINT,
            block_size INTEGER NOT NULL,
            block_ids_json TEXT NOT NULL,
            filter_signature TEXT NOT NULL,
            row_count BIGINT NOT NULL,
            qualified_count BIGINT NOT NULL,
            matching_blocks BIGINT NOT NULL
        )
        """
    )
    metadata_columns = {
        str(row[1]) for row in con.execute("PRAGMA table_info('filter_metadata')").fetchall()
    }
    if "inference_ms" in metadata_columns:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS filter_metadata_v2 (
                cache_key TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                database_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                filter_name TEXT NOT NULL,
                model_path TEXT NOT NULL,
                model_mtime_ns BIGINT,
                block_size INTEGER NOT NULL,
                block_ids_json TEXT NOT NULL,
                filter_signature TEXT NOT NULL,
                row_count BIGINT NOT NULL,
                qualified_count BIGINT NOT NULL,
                matching_blocks BIGINT NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO filter_metadata_v2(
                cache_key, created_at, database_name, table_name, model_name, filter_name,
                model_path, model_mtime_ns, block_size, block_ids_json, filter_signature,
                row_count, qualified_count, matching_blocks
            )
            SELECT
                cache_key, created_at, database_name, table_name, model_name, filter_name,
                model_path, model_mtime_ns, block_size, block_ids_json, filter_signature,
                row_count, qualified_count, matching_blocks
            FROM filter_metadata
            """
        )
        con.execute("DROP TABLE filter_metadata")
        con.execute("ALTER TABLE filter_metadata_v2 RENAME TO filter_metadata")

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS filter_qualifications (
            cache_key TEXT NOT NULL,
            row_id BIGINT NOT NULL,
            qualifies BOOLEAN NOT NULL
        )
        """
    )
    qualification_columns = {
        str(row[1]) for row in con.execute("PRAGMA table_info('filter_qualifications')").fetchall()
    }
    if "block_id" in qualification_columns:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS filter_qualifications_v2 (
                cache_key TEXT NOT NULL,
                row_id BIGINT NOT NULL,
                qualifies BOOLEAN NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO filter_qualifications_v2(cache_key, row_id, qualifies)
            SELECT cache_key, row_id, qualifies
            FROM filter_qualifications
            """
        )
        con.execute("DROP TABLE filter_qualifications")
        con.execute("ALTER TABLE filter_qualifications_v2 RENAME TO filter_qualifications")


def _fetch_cached_summary(
    con: duckdb.DuckDBPyConnection,
    cache_key: str,
) -> tuple[int, int, int] | None:
    row = con.execute(
        """
        SELECT row_count, qualified_count, matching_blocks
        FROM filter_metadata
        WHERE cache_key = ?
        """,
        [cache_key],
    ).fetchone()
    if row is None:
        return None
    actual_count = con.execute(
        "SELECT COUNT(*) FROM filter_qualifications WHERE cache_key = ?",
        [cache_key],
    ).fetchone()[0]
    if int(actual_count) != int(row[0]):
        return None
    return int(row[0]), int(row[1]), int(row[2])


def _prediction_group_key(request: ModelGroundTruthRequest, block_size: int) -> tuple:
    return (
        request.filter_spec.table,
        request.model_spec.name,
        str(request.model_path),
        _model_mtime_ns(request.model_path),
        int(block_size),
        tuple(sorted(int(block_id) for block_id in request.block_ids)),
    )


def _cache_key(
    filter_spec: FilterSpec,
    model_spec: FunctionSpec,
    model_path: Path,
    block_ids: list[int],
    block_size: int,
) -> str:
    model_path = Path(model_path)
    payload = {
        "database": filter_spec.database,
        "table": filter_spec.table,
        "model": model_spec.name,
        "model_path": str(model_path),
        "model_mtime_ns": _model_mtime_ns(model_path),
        "block_size": int(block_size),
        "block_ids": sorted(int(block_id) for block_id in block_ids),
        "filter_signature": _filter_signature(filter_spec),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:24]


def _filter_signature(filter_spec: FilterSpec) -> str:
    payload = {
        "name": filter_spec.name,
        "database": filter_spec.database,
        "table": filter_spec.table,
        "filter_type": filter_spec.filter_type,
        "target_class": filter_spec.target_class,
        "predicate_lower": filter_spec.predicate_lower,
        "predicate_upper": filter_spec.predicate_upper,
        "sql_predicate": filter_spec.sql_predicate,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _model_mtime_ns(model_path: Path) -> int | None:
    try:
        return model_path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _int_list(values: list[int]) -> str:
    if not values:
        raise ValueError("Expected at least one integer value.")
    return ", ".join(str(int(value)) for value in values)
