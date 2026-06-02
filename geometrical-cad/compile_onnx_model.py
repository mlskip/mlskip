"""
Compiles a neural network to its geometrical representation and stores it in a
DuckDB database. Dynamically handles any number of inputs and hidden
layers/width, but assumes 1 output and ReLU activations.

For example, to compile model from the benchmarking code:

  uv run compile_onnx_model.py --onnx-file
  ../metadata/models/tpch/regressor/lineitem/discounted_price/discounted_price.onnx
  --duckdb-file db/test.db
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb as db
import geometrical.network as network
import geometrical.pycad as pycad
import numpy as np
import pandas as pd
from geometrical.geometrical import construct_network_pwl


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compile a neural network to its geometrical representation."
    )
    parser.add_argument("--duckdb-file", type=Path, default="db/model.db")
    parser.add_argument("--onnx-file", type=Path)
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--max-columns", type=int, default=2)

    return parser


def cell_depth(cell):
    depth = 1
    c = cell
    while c.parent is not None:
        depth += 1
        c = c.parent
    return depth


def setup_cad_based(con, pwl):
    leaf_cells = list(pycad.merge_vertically_adjacent_leaf_cells(pwl.cad))
    if not leaf_cells:
        return
    n_dims = cell_depth(leaf_cells[0])

    for k in range(1, n_dims):
        con.sql(f"DROP TABLE IF EXISTS Cell_Dim{k}")

        cols = ["id INT64 PRIMARY KEY"]
        if k > 1:
            cols.append("parent_cell_id INT64")
        for i in range(k):
            cols.append(f"a{i}_lower DOUBLE")
        for i in range(k):
            cols.append(f"a{i}_upper DOUBLE")
        if k == (n_dims - 1):
            cols.append("component_function_id INT64")

        cols_str = ",\n    ".join(cols)
        con.sql(
            f"""
            CREATE TABLE Cell_Dim{k}(
                {cols_str}
            )
            """
        )

    con.sql("DROP TABLE IF EXISTS ComponentFunction")
    comp_cols = ["id INT64 PRIMARY KEY"] + [f"a{i} DOUBLE" for i in range(n_dims)]
    comp_cols_str = ",\n    ".join(comp_cols)
    con.sql(
        f"""
        CREATE TABLE ComponentFunction(
            {comp_cols_str}
        )
        """
    )

    component_functions = {}
    cells = [{} for _ in range(1, n_dims)]

    for cell in leaf_cells:
        ancestors = [cell]

        # Exclude ourselves and the root cell.
        for _ in range(n_dims - 2):
            ancestors.insert(0, ancestors[0].parent)

        for k in range(1, n_dims):
            ancestor = ancestors[k - 1]
            ancestor_id = id(ancestor)

            if ancestor_id in cells[k - 1]:
                continue

            lb = ancestor.lower_bound / -ancestor.lower_bound[-1]
            ub = ancestor.upper_bound / -ancestor.upper_bound[-1]

            row = [ancestor_id]
            if k > 1:
                row.append(id(ancestors[k - 2]))
            for i in range(k):
                row.append(lb[i].item() if not np.isinf(lb[i]) else "-infinity")
            for i in range(k):
                row.append(ub[i].item() if not np.isinf(ub[i]) else "+infinity")
            if k == (n_dims - 1):
                component_fn_key = str(cell.component_function.tolist()[:-1])
                if component_fn_key not in component_functions:
                    component_functions[component_fn_key] = [
                        id(cell.component_function)
                    ] + cell.component_function.tolist()[:-1]
                row.append(component_functions[component_fn_key][0])

            cells[k - 1][ancestor_id] = row

    comp_cols_df = ["id"] + [f"a{i}" for i in range(n_dims)]
    df = pd.DataFrame(component_functions.values(), columns=comp_cols_df)
    con.sql("INSERT INTO ComponentFunction SELECT * FROM df")

    for k in range(1, n_dims):
        cols = ["id"]
        if k > 1:
            cols.append("parent_cell")
        for i in range(k):
            cols.append(f"a{i}_lower")
        for i in range(k):
            cols.append(f"a{i}_upper")
        if k == (n_dims - 1):
            cols.append("component_function_id")

        df = pd.DataFrame(cells[k - 1].values(), columns=cols)
        con.sql(f"INSERT INTO Cell_Dim{k} SELECT * FROM df")


def _default_metrics_path(duckdb_path: Path) -> Path:
    return duckdb_path.with_suffix(".json")


def _write_metrics(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def _format_ms(duration_ms: float) -> str:
    if duration_ms >= 1000.0:
        return f"{duration_ms / 1000.0:.2f}s"
    return f"{duration_ms:.1f}ms"


def _metadata_path_for_onnx(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(".metadata.json")


def _feature_count_for_model(onnx_path: Path) -> int | None:
    metadata_path = _metadata_path_for_onnx(onnx_path)
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    return len(metadata.get("features", []))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    onnx_path = Path(args.onnx_file)
    duckdb_path = Path(args.duckdb_file)
    metrics_path = (
        Path(args.metrics_file)
        if args.metrics_file is not None
        else _default_metrics_path(duckdb_path)
    )

    if args.max_columns < 1:
        raise ValueError(f"--max-columns must be positive, got: {args.max_columns}")

    feature_count = _feature_count_for_model(onnx_path)
    if feature_count is not None and feature_count > args.max_columns:
        print(
            f"[geomcad] Skipping {onnx_path}: {feature_count} feature(s) exceeds "
            f"--max-columns {args.max_columns}"
        )
        raise SystemExit(0)

    print(f"[geomcad] Loading model from {onnx_path}")
    load_started_at = time.perf_counter()
    nn = network.onnx_model_to_graph(onnx_path)
    onnx_load_ms = (time.perf_counter() - load_started_at) * 1000.0
    print(f"[geomcad] Loaded ONNX graph in {_format_ms(onnx_load_ms)}")

    print("[geomcad] Creating geometrical representation")
    compile_started_at = time.perf_counter()
    pwl = construct_network_pwl(nn)
    geomcad_compile_ms = (time.perf_counter() - compile_started_at) * 1000.0
    print(
        f"[geomcad] Constructed geometrical representation in "
        f"{_format_ms(geomcad_compile_ms)}"
    )

    print(f"[geomcad] Writing DuckDB artifact to {duckdb_path}")
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    write_started_at = time.perf_counter()
    with db.connect(duckdb_path) as con:
        setup_cad_based(con, pwl)
    duckdb_write_ms = (time.perf_counter() - write_started_at) * 1000.0
    print(f"[geomcad] Wrote DuckDB artifact in {_format_ms(duckdb_write_ms)}")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "onnx_file": str(onnx_path),
        "duckdb_file": str(duckdb_path),
        "metrics_file": str(metrics_path),
        "onnx_load_ms": onnx_load_ms,
        "geomcad_compile_ms": geomcad_compile_ms,
        "duckdb_write_ms": duckdb_write_ms,
        "total_ms": onnx_load_ms + geomcad_compile_ms + duckdb_write_ms,
    }
    _write_metrics(metrics_path, metrics)
    print(f"[geomcad] Wrote metrics to {metrics_path}")
    print(f"[geomcad] Total compile time: {_format_ms(metrics['total_ms'])}")
    print("[geomcad] Done!")
