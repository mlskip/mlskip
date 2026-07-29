from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench
from nnv_tools.block_metadata import PairGeometry
from nnv_tools.filter_catalog import FilterSpec, get_filter_specs, write_filter_specs
from nnv_tools.function_catalog import FunctionSpec


def test_get_filter_specs_loads_multiple_generated_files(tmp_path: Path) -> None:
    first = FilterSpec(
        name="f1",
        description="first",
        database="tpch",
        table="lineitem",
        model_name="m1",
        sql_predicate="x > 1",
        filter_type="regressor_range",
        template_name="template_a",
    )
    second = FilterSpec(
        name="f2",
        description="second",
        database="tpch",
        table="orders",
        model_name="m2",
        sql_predicate="y > 2",
        filter_type="classifier_class",
        template_name="template_b",
    )
    path_a = tmp_path / "template_a.json"
    path_b = tmp_path / "template_b.json"

    write_filter_specs(path_a, [first])
    write_filter_specs(path_b, [second])

    loaded = get_filter_specs("tpch", None, [path_a, path_b])

    assert [spec.name for spec in loaded] == ["f1", "f2"]


def test_default_generated_filter_paths_are_grouped_per_template() -> None:
    paths = bench._default_generated_filters_paths(
        database="tpch",
        template_names=["charge band", "discounted_price_band"],
        range_alpha=2.0,
        range_start_samples=10,
        range_seed=0,
        task_type="regressor",
    )

    assert set(paths) == {"charge band", "discounted_price_band"}
    assert paths["charge band"].name.startswith("charge_band__")
    assert paths["discounted_price_band"].name.startswith("discounted_price_band__")


def test_cleanup_stale_generated_filter_specs_removes_old_hash_siblings(tmp_path: Path) -> None:
    current = tmp_path / "discounted_price__a2__n10__s0__current.json"
    stale = tmp_path / "discounted_price__a2__n10__s0__stale.json"
    stale_cache = tmp_path / "discounted_price__a2__n10__s0__stale__model_ground_truth.duckdb"
    orphan_cache = tmp_path / "discounted_price__a2__n10__s0__orphan__model_ground_truth.duckdb"
    unrelated = tmp_path / "charge__a2__n10__s0__stale.json"
    current.write_text("[]\n")
    stale.write_text("[]\n")
    stale_cache.write_text("cache")
    orphan_cache.write_text("cache")
    unrelated.write_text("[]\n")

    bench._cleanup_stale_generated_filter_specs([current])

    assert current.exists()
    assert not stale.exists()
    assert not stale_cache.exists()
    assert not orphan_cache.exists()
    assert unrelated.exists()


def test_resolve_filter_specs_reuses_generated_filters_for_export(tmp_path: Path) -> None:
    requested = FilterSpec(
        name="discounted_price",
        description="template",
        database="tpch",
        table="lineitem",
        model_name="discounted_price",
        sql_predicate="TRUE",
        filter_type="regressor_range",
        template_name="discounted_price",
    )
    generated = FilterSpec(
        name="discounted_price_w01_s00_0_2",
        description="generated",
        database="tpch",
        table="lineitem",
        model_name="discounted_price",
        sql_predicate="x BETWEEN 0 AND 2",
        filter_type="regressor_range",
        sampled_width=2.0,
        sampled_start=0.0,
        template_name="discounted_price",
    )
    generated_path = tmp_path / "discounted_price__a2__n10__s0__hash.json"
    write_filter_specs(generated_path, [generated])

    class _Args:
        pass

    args = _Args()
    args.database = "tpch"
    args.filters = None
    args.filters_path = None
    args.export = tmp_path
    args.task_type = "regressor"
    args.range_alpha = 2.0
    args.range_start_samples = 10
    args.range_seed = 0

    original_get_filter_specs = bench.get_filter_specs
    original_default_paths = bench._default_generated_filters_paths
    try:
        def fake_get_filter_specs(database, selected_names, path=None):
            if path is None:
                return [requested]
            return original_get_filter_specs(database, selected_names, path)

        bench.get_filter_specs = fake_get_filter_specs
        bench._default_generated_filters_paths = lambda **kwargs: {
            "discounted_price": generated_path
        }

        resolved, source = bench._resolve_filter_specs(args=args)
    finally:
        bench.get_filter_specs = original_get_filter_specs
        bench._default_generated_filters_paths = original_default_paths

    assert source == (generated_path,)
    assert [spec.name for spec in resolved] == ["discounted_price_w01_s00_0_2"]


def test_regressor_sampling_respects_total_budget_and_per_width_floor() -> None:
    template = FilterSpec(
        name="template",
        description="template",
        database="tpch",
        table="lineitem",
        model_name="model",
        sql_predicate="TRUE",
        filter_type="regressor_range",
    )
    model_spec = FunctionSpec(
        name="model",
        description="model",
        database="tpch",
        table="lineitem",
        task_type="regressor",
        target_expression="target",
        features=[],
    )

    original_fetch = bench.fetch_expression_range
    original_sort = bench._sort_filters_by_selectivity
    try:
        bench.fetch_expression_range = lambda *args, **kwargs: (0.0, 1000.0)
        bench._sort_filters_by_selectivity = lambda **kwargs: kwargs["filters"]

        sampled = bench._sample_regressor_filters(
            template=template,
            model_spec=model_spec,
            db_path=Path("/tmp/unused.duckdb"),
            block_size=1000,
            benchmark_block_ids=[1, 2, 3],
            alpha=2.0,
            start_samples=1000,
        )
    finally:
        bench.fetch_expression_range = original_fetch
        bench._sort_filters_by_selectivity = original_sort

    counts_by_width = Counter(spec.sampled_width for spec in sampled)
    widths = bench._build_range_widths(1000.0, 2.0)

    assert len(sampled) <= bench._GENERATED_FILTER_TOTAL_BUDGET
    for width in widths[:-1]:
        assert counts_by_width[width] >= bench._GENERATED_FILTER_MIN_PER_WIDTH
    assert counts_by_width[widths[-1]] == 1





def test_sort_filters_by_selectivity_uses_count_query() -> None:
    first = FilterSpec(
        name="f_low",
        description="low",
        database="tpch",
        table="lineitem",
        model_name="model",
        sql_predicate="x > 1",
        filter_type="regressor_range",
        sampled_width=10.0,
        sampled_start=0.0,
    )
    second = FilterSpec(
        name="f_high",
        description="high",
        database="tpch",
        table="lineitem",
        model_name="model",
        sql_predicate="x > 2",
        filter_type="regressor_range",
        sampled_width=20.0,
        sampled_start=5.0,
    )

    original_count_rows = bench.count_rows_for_blocks
    original_run_count_query = bench.run_count_query
    try:
        bench.count_rows_for_blocks = lambda *args, **kwargs: 100
        bench.run_count_query = (
            lambda table, predicate_sql, db_path, block_ids=None, block_size=None:
            (10 if predicate_sql == "x > 1" else 30, 0.0)
        )

        ranked = bench._sort_filters_by_selectivity(
            filters=[second, first],
            db_path=Path("/tmp/unused.duckdb"),
            block_size=1000,
            benchmark_block_ids=[1, 2, 3],
        )
    finally:
        bench.count_rows_for_blocks = original_count_rows
        bench.run_count_query = original_run_count_query

    assert [spec.name for spec in ranked] == ["f_low", "f_high"]

def test_bounded_convex_hull_size_uses_cell_and_corner_encoding() -> None:
    geometry = PairGeometry(
        feature_x="x",
        feature_y="y",
        grid_depth=4,
        bounded_convex_hull=[(0.0, 0.0)] * 5,
    )

    assert bench._bounded_convex_hull_storage_bytes(geometry) == 7
