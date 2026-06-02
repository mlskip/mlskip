from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench
from nnv_tools.database_setup import load_database_setup
from nnv_tools.filter_catalog import FilterSpec
from nnv_tools.function_catalog import FeatureSpec
from nnv_tools.function_catalog import FunctionSpec
from nnv_tools.dataset_duckdb import fetch_features_for_blocks
from nnv_tools.model_runtime import _pytorch_model_for_path
from nnv_tools.model_runtime import predict_array
from nnv_tools.model_runtime import predict_array_pytorch


def _regressor_spec() -> FunctionSpec:
    return FunctionSpec(
        name="discounted_price",
        description="discounted price",
        database="tpch",
        table="lineitem",
        task_type="regressor",
        target_expression="0.0",
        features=[
            FeatureSpec(name="quantity", expression="CAST(l_quantity AS DOUBLE)"),
            FeatureSpec(name="discount", expression="CAST(l_discount AS DOUBLE)"),
        ],
    )


def _classifier_spec() -> FunctionSpec:
    return FunctionSpec(
        name="ship_mode_classifier",
        description="ship mode classifier",
        database="tpch",
        table="lineitem",
        task_type="classifier",
        target_expression="0",
        features=[
            FeatureSpec(name="quantity", expression="CAST(l_quantity AS DOUBLE)"),
            FeatureSpec(name="discount", expression="CAST(l_discount AS DOUBLE)"),
            FeatureSpec(name="tax", expression="CAST(l_tax AS DOUBLE)"),
            FeatureSpec(name="shipdate", expression="CAST(epoch(l_shipdate) AS DOUBLE)"),
            FeatureSpec(name="receiptdate", expression="CAST(epoch(l_receiptdate) AS DOUBLE)"),
        ],
        num_classes=4,
    )


def test_predict_array_pytorch_matches_onnx_regressor() -> None:
    spec = _regressor_spec()
    model_path = Path('metadata/models/tpch/deep/regressor/lineitem/discounted_price/discounted_price.onnx')
    values = np.asarray([[1.0, 0.0], [20.0, 0.05], [50.0, 0.1]], dtype=np.float32)

    expected = predict_array(spec, model_path, values)
    actual = predict_array_pytorch(spec, model_path, values)

    assert np.allclose(actual, expected)


def test_predict_array_pytorch_matches_onnx_classifier() -> None:
    spec = _classifier_spec()
    model_path = Path('metadata/models/tpch/deep/classifier/lineitem/ship_mode_classifier/ship_mode_classifier.onnx')
    values = np.asarray(
        [
            [1.0, 0.0, 0.0, 695520000.0, 697334400.0],
            [20.0, 0.05, 0.02, 800000000.0, 800086400.0],
            [50.0, 0.1, 0.08, 911952000.0, 914544000.0],
        ],
        dtype=np.float32,
    )

    expected = predict_array(spec, model_path, values)
    actual = predict_array_pytorch(spec, model_path, values)

    assert np.array_equal(actual, expected)


def test_prediction_qualifies_matches_filter_types() -> None:
    regressor_filter = FilterSpec(
        name="r",
        description="r",
        database="tpch",
        table="lineitem",
        model_name="discounted_price",
        sql_predicate="TRUE",
        filter_type="regressor_range",
        predicate_lower=1.0,
        predicate_upper=2.0,
    )
    classifier_filter = FilterSpec(
        name="c",
        description="c",
        database="tpch",
        table="lineitem",
        model_name="ship_mode_classifier",
        sql_predicate="TRUE",
        filter_type="classifier_class",
        target_class=2,
    )

    assert np.array_equal(
        bench._prediction_qualifies(regressor_filter, np.asarray([0.5, 1.0, 1.5, 3.0])),
        np.asarray([False, True, True, False]),
    )
    assert np.array_equal(
        bench._prediction_qualifies(classifier_filter, np.asarray([1, 2, 3])),
        np.asarray([False, True, False]),
    )


def test_results_block_metadata_label_uses_none_for_pytorch() -> None:
    assert bench._results_block_metadata_label(
        cli_block_metadata=None,
        jobs=[],
        verifier_backend="pytorch",
    ) == "none"


def test_results_verifier_backend_label_uses_batched_geomcad_name() -> None:
    assert bench._results_verifier_backend_label(
        verifier_backend="geomcad",
        batched_geomcad=True,
    ) == "batched_geomcad"


def test_run_pytorch_e2e_query_matches_direct_prediction_count() -> None:
    spec = _regressor_spec()
    model_path = Path("metadata/models/tpch/deep/regressor/lineitem/discounted_price/discounted_price.onnx")
    filter_spec = FilterSpec(
        name="discount_band",
        description="discounted price range",
        database="tpch",
        table="lineitem",
        model_name="discounted_price",
        sql_predicate="TRUE",
        filter_type="regressor_range",
        predicate_lower=100.0,
        predicate_upper=1000.0,
    )
    block_ids = [75, 76]
    block_size = 1000
    db_path = load_database_setup("tpch").duckdb_file
    if not db_path.exists():
        pytest.skip(f"TPC-H DuckDB cache is not available at {db_path}")

    count, data_loading_ms, inference_ms = bench._run_pytorch_e2e_query(
        db_path=db_path,
        filter_spec=filter_spec,
        model_spec=spec,
        model_path=model_path,
        block_ids=block_ids,
        block_size=block_size,
    )

    features = fetch_features_for_blocks(
        spec,
        block_ids,
        db_path,
        block_size,
    )
    predictions = predict_array_pytorch(spec, model_path, features)
    expected = int(np.count_nonzero(bench._prediction_qualifies(filter_spec, predictions)))

    assert count == expected
    assert data_loading_ms >= 0.0
    assert inference_ms >= 0.0


def test_run_pytorch_e2e_query_sums_per_block_predictions() -> None:
    spec = _regressor_spec()
    model_path = Path("metadata/models/tpch/deep/regressor/lineitem/discounted_price/discounted_price.onnx")
    filter_spec = FilterSpec(
        name="discount_band",
        description="discounted price range",
        database="tpch",
        table="lineitem",
        model_name="discounted_price",
        sql_predicate="TRUE",
        filter_type="regressor_range",
        predicate_lower=100.0,
        predicate_upper=1000.0,
    )
    block_ids = [75, 76]
    block_size = 1000
    db_path = load_database_setup("tpch").duckdb_file
    if not db_path.exists():
        pytest.skip(f"TPC-H DuckDB cache is not available at {db_path}")

    count, _, _ = bench._run_pytorch_e2e_query(
        db_path=db_path,
        filter_spec=filter_spec,
        model_spec=spec,
        model_path=model_path,
        block_ids=block_ids,
        block_size=block_size,
    )

    expected = 0
    for block_id in block_ids:
        features = bench.fetch_block_features(spec, block_id, db_path, block_size)
        predictions = predict_array_pytorch(spec, model_path, features)
        expected += int(np.count_nonzero(bench._prediction_qualifies(filter_spec, predictions)))

    assert count == expected


def test_pytorch_model_is_pinned_to_cpu() -> None:
    model_path = Path("metadata/models/tpch/deep/regressor/lineitem/discounted_price/discounted_price.onnx")
    model = _pytorch_model_for_path(str(model_path))
    assert next(model.parameters()).device.type == "cpu"
