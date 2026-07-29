from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import duckdb

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nnv_tools.block_metadata import BlockMetadata
from nnv_tools.block_metadata import PairGeometry
from nnv_tools.block_metadata import BlockMetadataBundle
from nnv_tools.filter_catalog import FilterSpec
from nnv_tools.function_catalog import FeatureSpec, FunctionSpec
from nnv_tools.model_ground_truth import count_model_qualified_rows_by_block


_SPEC = importlib.util.spec_from_file_location(
    "bench_module",
    Path(__file__).resolve().parents[1] / "bench.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load bench.py for testing")
_BENCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BENCH
_SPEC.loader.exec_module(_BENCH)
_metadata_size_summary = _BENCH._metadata_size_summary
_build_results_payload = _BENCH._build_results_payload
_resolve_export_grid_depth = _BENCH._resolve_export_grid_depth


def test_metadata_size_summary_includes_collection_timing() -> None:
    metadata_block_a = BlockMetadata(
        kind="minmax",
        input_bounds={"x": (0.0, 1.0), "y": (1.0, 2.0)},
        pair_geometries=[
            PairGeometry(
                feature_x="x",
                feature_y="y",
                hull=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
                bounded_convex_hull=[(0.0, 0.0), (1.0, 0.0)],
                grid_depth=4,
            )
        ],
    )
    metadata_block_b = BlockMetadata(
        kind="minmax",
        input_bounds={"x": (2.0, 3.0), "y": (4.0, 5.0)},
        pair_geometries=[
            PairGeometry(
                feature_x="x",
                feature_y="y",
                hull=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                bounded_convex_hull=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
                grid_depth=4,
            )
        ],
    )
    summary = _metadata_size_summary(
        {
            ("model_a", (0, 1), "minmax", 0): BlockMetadataBundle(
                metadata_by_block={0: metadata_block_a},
                collection_ms=12.5,
                collection_ms_by_block={0: 12.5},
            ),
            ("model_b", (2,), "minmax", 0): BlockMetadataBundle(
                metadata_by_block={2: metadata_block_b},
                collection_ms=7.5,
                collection_ms_by_block={2: 7.5},
            ),
        }
    )

    assert summary is not None
    assert summary["kind"] == "minmax"
    assert summary["block_count"] == 2
    assert summary["avg_size_per_block_bytes"] == 91.5
    assert summary["median_size_per_block_bytes"] == 91.5
    assert summary["max_size_per_block_bytes"] == 100
    assert summary["total_collection_ms"] == 20.0
    assert summary["avg_collection_ms_per_block"] == 10.0
    assert summary["median_collection_ms_per_block"] == 10.0
    assert summary["max_collection_ms_per_block"] == 12.5
    assert summary["avg_convex_hull_vertices_per_pair"] == 3.5
    assert summary["median_convex_hull_vertices_per_pair"] == 3.5
    assert summary["max_convex_hull_vertices_per_pair"] == 4
    assert summary["grid_depth"] == 4


def test_build_results_payload_includes_pruning_summary_by_selectivity_group() -> None:
    class _Args:
        pass

    args = _Args()
    args._metadata_size_summary = None
    args.measure_e2e = False

    payload = _build_results_payload(
        args,
        [
            {
                "query_selectivity_pct": 0.0,
                "pruning_effectiveness_pct": 10.0,
                "skipped_blocks": 1,
                "skippable_blocks": 10,
            },
            {
                "query_selectivity_pct": 0.0,
                "pruning_effectiveness_pct": 30.0,
                "skipped_blocks": 3,
                "skippable_blocks": 10,
            },
            {
                "query_selectivity_pct": 0.005,
                "pruning_effectiveness_pct": 50.0,
                "skipped_blocks": 5,
                "skippable_blocks": 10,
            },
            {
                "query_selectivity_pct": 0.005,
                "pruning_effectiveness_pct": None,
                "skipped_blocks": 2,
                "skippable_blocks": 4,
            },
            {
                "query_selectivity_pct": None,
                "pruning_effectiveness_pct": 99.0,
                "skipped_blocks": 99,
                "skippable_blocks": 100,
            },
        ],
    )

    assert payload["pruning_performance_by_selectivity_group"] == [
        {
            "selectivity_range": "= 0",
            "count": 2,
            "avg_pruning_effectiveness_pct": 20.0,
            "median_pruning_effectiveness_pct": 20.0,
        },
        {
            "selectivity_range": "0.001-0.01",
            "count": 2,
            "avg_pruning_effectiveness_pct": 50.0,
            "median_pruning_effectiveness_pct": 50.0,
        },
    ]


def test_write_results_by_filter_uses_template_local_metadata_summary(tmp_path: Path) -> None:
    class _Args:
        pass

    args = _Args()
    args.results_path = tmp_path
    args.database = "tpch"
    args.filters = None
    args.model_kind = "shallow"
    args.block_size = 1000
    args.range_alpha = 2.0
    args.range_start_samples = 10
    args.range_seed = 0
    args.task_type = "regressor"
    args.max_rows_total = 100000
    args.run_udf = False
    args.measure_e2e = False
    args.disable_skipping = False
    args.jobs = 20
    args.verifier_backend = "marabou"
    args.batched_geomcad = False
    args.verifier_timeout_seconds = 1.0
    args.block_metadata = "minmax"
    args.grid_depth = None
    args.resolved_block_metadata_label = "minmax"
    args._metadata_size_summary = {"block_count": 200, "kind": "minmax"}
    args._metadata_size_summary_by_template = {
        "charge": {"block_count": 100, "kind": "minmax"},
        "discounted_price": {"block_count": 100, "kind": "minmax"},
    }

    results = [
        {
            "filter_name": "charge_filter",
            "filter_template_name": "charge",
            "skipped_blocks": 0,
            "skippable_blocks": 1,
        },
        {
            "filter_name": "discounted_price_filter",
            "filter_template_name": "discounted_price",
            "skipped_blocks": 0,
            "skippable_blocks": 1,
        },
    ]

    written_paths = _BENCH._write_results_by_filter(args, results)

    assert len(written_paths) == 2
    payloads = {path.parent.parent.parent.name: json.loads(path.read_text()) for path in written_paths}
    assert payloads["charge"]["metadata_size_summary"]["block_count"] == 100
    assert payloads["discounted_price"]["metadata_size_summary"]["block_count"] == 100




def test_bounded_convex_hull_size_summary_excludes_exact_hull_bytes() -> None:
    metadata = BlockMetadata(
        kind="bounded_convex_hull",
        input_bounds={"x": (0.0, 1.0), "y": (1.0, 2.0)},
        pair_geometries=[
            PairGeometry(
                feature_x="x",
                feature_y="y",
                hull=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                bounded_convex_hull=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
                grid_depth=4,
            )
        ],
    )
    summary = _metadata_size_summary(
        {
            ("model_a", (0,), "bounded_convex_hull", 4): BlockMetadataBundle(
                metadata_by_block={0: metadata},
                collection_ms=12.5,
                collection_ms_by_block={0: 12.5},
            )
        }
    )

    assert summary is not None
    assert summary["kind"] == "bounded_convex_hull"
    assert summary["avg_size_per_block_bytes"] == 36.0
    assert summary["max_size_per_block_bytes"] == 36
    assert summary["avg_convex_hull_vertices_per_pair"] == 3.0
    assert summary["median_convex_hull_vertices_per_pair"] == 3.0
    assert summary["max_convex_hull_vertices_per_pair"] == 3
    assert summary["grid_depth"] == 4


def test_resolve_export_grid_depth_uses_block_metadata_geometry() -> None:
    blocks = [
        {
            "block_id": 2,
            "metadata": {
                "kind": "bounded_convex_hull",
                "pair_geometries": [
                    {
                        "feature_x": "extendedprice",
                        "feature_y": "discount",
                        "grid_depth": 4,
                        "bounded_convex_hull": [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0],
                        ],
                    }
                ],
            },
        }
    ]

    assert _resolve_export_grid_depth(blocks, "bounded_convex_hull") == 4


def test_write_export_payload_splits_metadata_from_filter_ground_truth(tmp_path: Path) -> None:
    class _Args:
        pass

    args = _Args()
    args.export = tmp_path
    args.database = "tpch"
    args.model_kind = "shallow"
    args.block_size = 1000

    model_spec = FunctionSpec(
        name="charge",
        description="charge",
        database="tpch",
        table="lineitem",
        task_type="regressor",
        target_expression="x",
        features=[FeatureSpec(name="x", expression="x")],
    )
    filter_a = FilterSpec(
        name="charge_low",
        description="low",
        database="tpch",
        table="lineitem",
        model_name="charge",
        sql_predicate="charge BETWEEN 0 AND 1",
        filter_type="regressor_range",
        predicate_lower=0.0,
        predicate_upper=1.0,
        template_name="charge",
    )
    filter_b = FilterSpec(
        name="charge_high",
        description="high",
        database="tpch",
        table="lineitem",
        model_name="charge",
        sql_predicate="charge BETWEEN 10 AND 11",
        filter_type="regressor_range",
        predicate_lower=10.0,
        predicate_upper=11.0,
        template_name="charge",
    )
    jobs = [
        _BENCH.BenchmarkJob(
            filter_id=1,
            filter_spec=filter_a,
            model_spec=model_spec,
            model_path=Path("/tmp/charge.onnx"),
            block_ids=[0],
            excluded_training_blocks=0,
        ),
        _BENCH.BenchmarkJob(
            filter_id=2,
            filter_spec=filter_b,
            model_spec=model_spec,
            model_path=Path("/tmp/charge.onnx"),
            block_ids=[0],
            excluded_training_blocks=0,
        ),
    ]

    export_dir = _BENCH._write_export_payload(
        args,
        jobs,
        [],
        [
            {
                "block_id": 0,
                "_export_table": "lineitem",
                "_export_model_name": "charge",
                "row_id_start": 0,
                "row_id_end": 999,
                "row_count": 1000,
                "metadata": {"kind": "minmax", "input_bounds": {"x": [0.0, 1.0]}, "pair_geometries": []},
            },
            {
                "table": "lineitem",
                "model_name": "charge",
                "model_task_type": "regressor",
                "filter_name": "charge_low",
                "filter_template_name": "charge",
                "filter_type": "regressor_range",
                "sql_predicate": "charge BETWEEN 0 AND 1",
                "feature_columns": ["x"],
                "block_id": 0,
                "row_id_start": 0,
                "row_id_end": 999,
                "block_row_count": 1000,
                "matching_rows": 7,
            },
            {
                "table": "lineitem",
                "model_name": "charge",
                "model_task_type": "regressor",
                "filter_name": "charge_high",
                "filter_template_name": "charge",
                "filter_type": "regressor_range",
                "sql_predicate": "charge BETWEEN 10 AND 11",
                "feature_columns": ["x"],
                "block_id": 0,
                "row_id_start": 0,
                "row_id_end": 999,
                "block_row_count": 1000,
                "matching_rows": 3,
            },
        ],
    )

    assert export_dir == tmp_path / "tpch" / "shallow" / "bs1000"
    metadata_path = export_dir / "lineitem" / "charge" / "minmax-metadata.json"
    low_path = export_dir / "lineitem" / "charge" / "filters" / "charge_low" / "ground-truth.json"
    high_path = export_dir / "lineitem" / "charge" / "filters" / "charge_high" / "ground-truth.json"
    assert metadata_path.exists()
    assert low_path.exists()
    assert high_path.exists()

    metadata_payload = json.loads(metadata_path.read_text())
    low_payload = json.loads(low_path.read_text())
    high_payload = json.loads(high_path.read_text())
    assert len(metadata_payload["blocks"]) == 1
    assert metadata_payload["blocks"][0]["metadata"]["kind"] == "minmax"
    assert "metadata_file" not in low_payload
    assert "metadata_file" not in high_payload
    assert "metadata_kind" not in low_payload
    assert "metadata_kind" not in high_payload
    assert low_payload["filter"]["name"] == "charge_low"
    assert high_payload["filter"]["name"] == "charge_high"
    assert low_payload["blocks"][0]["matching_rows"] == 7
    assert high_payload["blocks"][0]["matching_rows"] == 3
    assert low_payload["ground_truth_summary"]["matching_rows"] == 7
    assert high_payload["ground_truth_summary"]["matching_rows"] == 3


def test_export_ground_truth_matches_existing_cache_key(tmp_path: Path) -> None:
    path = tmp_path / "ground-truth.json"
    assert not _BENCH._export_ground_truth_matches_cache(path, "cache-a")

    path.write_text(json.dumps({"model_ground_truth_cache_key": "cache-a"}) + "\n")
    assert _BENCH._export_ground_truth_matches_cache(path, "cache-a")
    assert not _BENCH._export_ground_truth_matches_cache(path, "cache-b")


def test_count_model_qualified_rows_by_block_reads_duckdb_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "ground_truth.duckdb"
    with duckdb.connect(str(cache_path)) as con:
        con.execute(
            """
            CREATE TABLE filter_qualifications (
                cache_key TEXT NOT NULL,
                row_id BIGINT NOT NULL,
                qualifies BOOLEAN NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO filter_qualifications VALUES
                ('cache-a', 0, true),
                ('cache-a', 1, false),
                ('cache-a', 2, true),
                ('cache-a', 5, true),
                ('cache-a', 7, false),
                ('cache-a', 10, true),
                ('cache-b', 0, true)
            """
        )

    assert count_model_qualified_rows_by_block(
        cache_path,
        "cache-a",
        [0, 1, 2, 3],
        5,
    ) == {
        0: 2,
        1: 1,
        2: 1,
        3: 0,
    }


def test_count_rows_by_block_reads_source_duckdb_once(tmp_path: Path) -> None:
    db_path = tmp_path / "source.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE TABLE lineitem(row_id BIGINT NOT NULL, value INTEGER)")
        con.execute(
            """
            INSERT INTO lineitem VALUES
                (0, 1),
                (1, 2),
                (2, 3),
                (5, 4),
                (6, 5),
                (10, 6)
            """
        )

    assert _BENCH._count_rows_by_block(
        "lineitem",
        [0, 1, 2, 3],
        db_path,
        5,
    ) == {
        0: 3,
        1: 2,
        2: 1,
        3: 0,
    }
