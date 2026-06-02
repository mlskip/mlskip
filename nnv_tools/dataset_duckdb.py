from __future__ import annotations

from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd

from nnv_tools.filter_catalog import FilterSpec
from nnv_tools.function_catalog import FunctionSpec

ROW_ID_COLUMN = "row_id"


def _connect_read_only(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


def fetch_function_dataset(
    spec: FunctionSpec,
    sample_size: int,
    db_path: str | Path,
    block_size: int = 1000,
    block_ids: list[int] | None = None,
) -> pd.DataFrame:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[duckdb] Opening {db_path} for database '{spec.database}' with block_size={block_size}"
    )
    with duckdb.connect(str(db_path)) as con:
        ensure_dataset_loaded(con, spec)
        if block_ids is None:
            query = build_dataset_query(spec, sample_size)
        else:
            ensure_row_id_column(con, spec.table)
            query = build_block_dataset_query(spec, sample_size, block_ids, block_size)
        dataframe = con.execute(query).fetch_df()
    print(
        f"[duckdb] Loaded {len(dataframe)} rows for function '{spec.name}' "
        f"with features {[feature.name for feature in spec.features]}"
    )
    return dataframe


def ensure_dataset_loaded(
    con: duckdb.DuckDBPyConnection,
    spec: FunctionSpec,
) -> None:
    if _has_table(con, spec.table):
        return
    raise FileNotFoundError(
        f"Table '{spec.table}' is missing in the DuckDB cache for database "
        f"'{spec.database}'. Run `scripts/setup_database.sh {spec.database}` first."
    )


def build_dataset_query(spec: FunctionSpec, sample_size: int) -> str:
    feature_columns = [
        f"{feature.expression} AS {feature.name}" for feature in spec.features
    ]
    target_cast = "INTEGER" if spec.task_type == "classifier" else "DOUBLE"
    select_list = ",\n        ".join(
        [
            *feature_columns,
            f"CAST({spec.target_expression} AS {target_cast}) AS {spec.target_name}",
        ]
    )
    output_columns = ", ".join(
        _quote_identifier(name) for name in _dataset_column_names(spec)
    )
    valid_predicate = _valid_dataset_predicate(spec)
    return f"""
        WITH dataset AS (
            SELECT
                {select_list}
            FROM {spec.table}
        )
        SELECT
            {output_columns}
        FROM dataset
        WHERE {valid_predicate}
        LIMIT {int(sample_size)}
    """


def build_block_dataset_query(
    spec: FunctionSpec,
    sample_size: int,
    block_ids: list[int],
    block_size: int,
) -> str:
    block_expression = _block_id_expression(block_size)
    feature_columns = [
        f"{feature.expression} AS {feature.name}" for feature in spec.features
    ]
    target_cast = "INTEGER" if spec.task_type == "classifier" else "DOUBLE"
    select_list = ",\n        ".join(
        [
            *feature_columns,
            f"CAST({spec.target_expression} AS {target_cast}) AS {spec.target_name}",
            f"{block_expression} AS __block_id",
            f"{ROW_ID_COLUMN} AS __row_id",
        ]
    )
    block_predicate = build_block_id_predicate(block_ids, block_size)
    output_columns = ", ".join(
        _quote_identifier(name) for name in _dataset_column_names(spec)
    )
    valid_predicate = _valid_dataset_predicate(spec)
    return f"""
        WITH dataset AS (
            SELECT
                {select_list}
            FROM {spec.table}
            WHERE {block_predicate}
        )
        SELECT
            {output_columns}
        FROM dataset
        WHERE {valid_predicate}
        ORDER BY __block_id, __row_id
        LIMIT {int(sample_size)}
    """


def _dataset_column_names(spec: FunctionSpec) -> list[str]:
    return [feature.name for feature in spec.features] + [spec.target_name]


def _valid_dataset_predicate(spec: FunctionSpec) -> str:
    clauses = [
        clause
        for name in _dataset_column_names(spec)
        for clause in (
            f"{_quote_identifier(name)} IS NOT NULL",
            f"isfinite(CAST({_quote_identifier(name)} AS DOUBLE))",
        )
    ]
    if spec.task_type == "classifier":
        target = _quote_identifier(spec.target_name)
        clauses.append(f"{target} >= 0")
        if spec.num_classes is not None:
            clauses.append(f"{target} < {int(spec.num_classes)}")
    return " AND ".join(clauses)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def fetch_block_bounds(
    spec: FunctionSpec,
    block_id: int,
    db_path: str | Path,
    block_size: int = 1000,
) -> dict[str, tuple[float, float]]:
    bounds_by_block = fetch_blocks_bounds(
        spec,
        [block_id],
        db_path,
        block_size,
        verbose=True,
    )
    if block_id not in bounds_by_block:
        raise ValueError(f"Block {block_id} does not exist in {spec.table}.")
    return bounds_by_block[block_id]


def fetch_blocks_bounds(
    spec: FunctionSpec,
    block_ids: list[int],
    db_path: str | Path,
    block_size: int = 1000,
    verbose: bool = False,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[int, dict[str, tuple[float, float]]]:
    if not block_ids:
        return {}
    if verbose:
        print(
            f"[duckdb] Collecting feature bounds for {len(block_ids)} block(s) "
            f"from table '{spec.table}'"
        )
    owns_connection = con is None
    if con is None:
        con = _connect_read_only(db_path)
    try:
        ensure_dataset_loaded(con, spec)
        ensure_row_id_column(con, spec.table)
        block_expression = _block_id_expression(block_size)
        block_predicate = build_block_id_predicate(block_ids, block_size)
        select_items = []
        for feature in spec.features:
            select_items.append(f"MIN({feature.expression}) AS min_{feature.name}")
            select_items.append(f"MAX({feature.expression}) AS max_{feature.name}")
        rows = con.execute(
            f"""
            SELECT
                {block_expression} AS block_id,
                {", ".join(select_items)}
            FROM {spec.table}
            WHERE {block_predicate}
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchall()
    finally:
        if owns_connection:
            con.close()

    bounds_by_block: dict[int, dict[str, tuple[float, float]]] = {}
    for row in rows:
        current_block_id = int(row[0])
        bounds: dict[str, tuple[float, float]] = {}
        for index, feature in enumerate(spec.features):
            bounds[feature.name] = (
                float(row[index * 2 + 1]),
                float(row[index * 2 + 2]),
            )
        bounds_by_block[current_block_id] = bounds
        if verbose:
            print(f"[duckdb] Block {current_block_id} bounds: {bounds}")
    return bounds_by_block


def count_block_predicate_matches(
    filter_spec: FilterSpec,
    block_id: int,
    db_path: str | Path,
    block_size: int = 1000,
) -> int:
    print(
        f"[duckdb] Counting rows in block_id={block_id} for filter '{filter_spec.name}'"
    )
    with _connect_read_only(db_path) as con:
        ensure_dataset_loaded(con, _filter_to_function_stub(filter_spec))
        ensure_row_id_column(con, filter_spec.table)
        block_predicate = _single_block_predicate(block_id, block_size)
        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {filter_spec.table}
            WHERE {block_predicate}
              AND ({filter_spec.sql_predicate})
            """
        ).fetchone()[0]
    print(f"[duckdb] Block {block_id} has {int(count)} matching row(s)")
    return int(count)


def run_count_query(
    table: str,
    predicate_sql: str,
    db_path: str | Path,
    block_ids: list[int] | None = None,
    block_size: int | None = None,
) -> tuple[int, float]:
    with _connect_read_only(db_path) as con:
        start = perf_counter()
        if block_ids is None:
            result = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {predicate_sql}"
            ).fetchone()[0]
        else:
            if block_size is None:
                raise ValueError("block_size is required when block_ids are provided.")
            ensure_row_id_column(con, table)
            block_predicate = build_block_id_predicate(block_ids, block_size)
            result = con.execute(
                f"""
                SELECT COUNT(*)
                FROM {table}
                WHERE {block_predicate}
                  AND ({predicate_sql})
                """
            ).fetchone()[0]
        elapsed_ms = (perf_counter() - start) * 1000.0
    return int(result), float(elapsed_ms)


def fetch_block_features(
    spec: FunctionSpec,
    block_id: int,
    db_path: str | Path,
    block_size: int = 1000,
    con: duckdb.DuckDBPyConnection | None = None,
) -> np.ndarray:
    owns_connection = con is None
    if con is None:
        con = _connect_read_only(db_path)
    try:
        ensure_dataset_loaded(con, spec)
        ensure_row_id_column(con, spec.table)
        rows = con.execute(
            f"""
            SELECT
                {", ".join(feature.expression for feature in spec.features)}
            FROM {spec.table}
            WHERE {_single_block_predicate(block_id, block_size)}
            ORDER BY {ROW_ID_COLUMN}
            """
        ).fetchall()
    finally:
        if owns_connection:
            con.close()
    if not rows:
        return np.empty((0, len(spec.features)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def fetch_features_for_blocks(
    spec: FunctionSpec,
    block_ids: list[int],
    db_path: str | Path,
    block_size: int = 1000,
    con: duckdb.DuckDBPyConnection | None = None,
) -> np.ndarray:
    if not block_ids:
        return np.empty((0, len(spec.features)), dtype=np.float32)

    owns_connection = con is None
    if con is None:
        con = _connect_read_only(db_path)
    try:
        ensure_dataset_loaded(con, spec)
        ensure_row_id_column(con, spec.table)
        rows = con.execute(
            f"""
            SELECT
                {", ".join(feature.expression for feature in spec.features)}
            FROM {spec.table}
            WHERE {build_block_id_predicate(block_ids, block_size)}
            ORDER BY {ROW_ID_COLUMN}
            """
        ).fetchall()
    finally:
        if owns_connection:
            con.close()
    if not rows:
        return np.empty((0, len(spec.features)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def count_matching_blocks(
    table: str,
    predicate_sql: str,
    db_path: str | Path,
    block_ids: list[int],
    block_size: int,
) -> int:
    if not block_ids:
        return 0

    block_expression = _block_id_expression(block_size)
    block_predicate = build_block_id_predicate(block_ids, block_size)
    with _connect_read_only(db_path) as con:
        ensure_dataset_loaded(con, _table_stub(table))
        ensure_row_id_column(con, table)
        result = con.execute(
            f"""
            SELECT COUNT(DISTINCT {block_expression})
            FROM {table}
            WHERE {block_predicate}
              AND ({predicate_sql})
            """
        ).fetchone()[0]
    return int(result)


def run_block_filtered_query(
    table: str,
    predicate_sql: str,
    block_ids: list[int],
    db_path: str | Path,
    block_size: int,
) -> tuple[int, float, str]:
    if not block_ids:
        return 0, 0.0, "FALSE"

    in_list = build_block_id_predicate(block_ids, block_size)
    with _connect_read_only(db_path) as con:
        ensure_row_id_column(con, table)
        start = perf_counter()
        result = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {table}
            WHERE {in_list}
              AND ({predicate_sql})
            """
        ).fetchone()[0]
        elapsed_ms = (perf_counter() - start) * 1000.0
    return int(result), float(elapsed_ms), in_list


def fetch_expression_range(
    table: str,
    expression: str,
    db_path: str | Path,
    block_ids: list[int] | None = None,
    block_size: int | None = None,
) -> tuple[float, float] | None:
    with _connect_read_only(db_path) as con:
        ensure_dataset_loaded(con, _table_stub(table))
        ensure_row_id_column(con, table)
        where_clause = ""
        if block_ids is not None:
            if block_size is None:
                raise ValueError("block_size is required when block_ids are provided.")
            where_clause = f"WHERE {build_block_id_predicate(block_ids, block_size)}"
        row = con.execute(
            f"""
            SELECT
                MIN(CAST({expression} AS DOUBLE)),
                MAX(CAST({expression} AS DOUBLE))
            FROM {table}
            {where_clause}
            """
        ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        return None
    return float(row[0]), float(row[1])


def list_block_ids(table: str, db_path: str | Path, block_size: int) -> list[int]:
    with _connect_read_only(db_path) as con:
        ensure_dataset_loaded(con, _table_stub(table))
        ensure_row_id_column(con, table)
        block_expression = _block_id_expression(block_size)
        rows = con.execute(
            f"""
            SELECT DISTINCT {block_expression} AS block_id
            FROM {table}
            ORDER BY block_id
            """
        ).fetchall()
    return [int(row[0]) for row in rows]


def count_rows_for_blocks(
    table: str,
    block_ids: list[int],
    db_path: str | Path,
    block_size: int,
) -> int:
    if not block_ids:
        return 0
    block_predicate = build_block_id_predicate(block_ids, block_size)
    with _connect_read_only(db_path) as con:
        ensure_row_id_column(con, table)
        return int(
            con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {block_predicate}"
            ).fetchone()[0]
        )


def _has_table(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    query = """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """
    count = con.execute(query, [table_name]).fetchone()[0]
    return bool(count)


def ensure_row_id_column(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
) -> None:
    column_names = {
        row[0]
        for row in con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchall()
    }
    if ROW_ID_COLUMN in column_names:
        min_row_id = con.execute(
            f"SELECT MIN({ROW_ID_COLUMN}) FROM {table_name}"
        ).fetchone()[0]
        if min_row_id != 0:
            raise AssertionError(
                f"Expected 0-based row_id in table '{table_name}', found min(row_id)={min_row_id}. "
                "Rebuild the DuckDB cache with setup_database.sh."
            )
        return

    print(f"[duckdb] Rebuilding table {table_name} to add {ROW_ID_COLUMN}")

    temp_table_name = f"{table_name}__with_{ROW_ID_COLUMN}"
    con.execute(
        f"""
        CREATE TABLE {temp_table_name} AS
        SELECT
            {table_name}.*,
            CAST(ROW_NUMBER() OVER () - 1 AS BIGINT) AS {ROW_ID_COLUMN}
        FROM {table_name}
        """
    )
    con.execute(f"DROP TABLE {table_name}")
    con.execute(f"ALTER TABLE {temp_table_name} RENAME TO {table_name}")


def _block_id_expression(block_size: int) -> str:
    return f"CAST(FLOOR({ROW_ID_COLUMN} / {int(block_size)}) AS BIGINT)"


def _single_block_predicate(block_id: int, block_size: int) -> str:
    start_row = block_id * int(block_size)
    end_row = (block_id + 1) * int(block_size) - 1
    return f"{ROW_ID_COLUMN} BETWEEN {start_row} AND {end_row}"


def build_block_id_predicate(block_ids: list[int], block_size: int) -> str:
    if not block_ids:
        return "FALSE"
    block_list = ", ".join(str(block_id) for block_id in sorted(block_ids))
    return f"{_block_id_expression(block_size)} IN ({block_list})"


def _filter_to_function_stub(filter_spec: FilterSpec) -> FunctionSpec:
    return FunctionSpec(
        name=filter_spec.name,
        description=filter_spec.description,
        database=filter_spec.database,
        table=filter_spec.table,
        task_type="regressor",
        target_expression="",
        features=[],
        num_classes=None,
    )


def _table_stub(table: str) -> FunctionSpec:
    return FunctionSpec(
        name=table,
        description=table,
        database="unknown",
        table=table,
        task_type="regressor",
        target_expression="",
        features=[],
        num_classes=None,
    )
