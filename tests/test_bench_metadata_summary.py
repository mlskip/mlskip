from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nnv_tools.block_metadata import BlockMetadata
from nnv_tools.block_metadata import PairGeometry
from nnv_tools.block_metadata import BlockMetadataBundle


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
    args.disable_skipping = False
    args.jobs = 20
    args.verifier_backend = "marabou"
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
