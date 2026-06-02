from __future__ import annotations

import sys
import queue
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnv_tools.block_verifier import BlockVerificationRequest
from nnv_tools.block_verifier import format_verifier_status
from nnv_tools import marabou_verify
from nnv_tools.function_catalog import FeatureSpec
from nnv_tools.function_catalog import FunctionSpec


def _sample_request() -> BlockVerificationRequest:
    return BlockVerificationRequest(
        model_path=Path("model.onnx"),
        spec=FunctionSpec(
            name="demo",
            description="demo spec",
            database="tpch",
            table="lineitem",
            task_type="regressor",
            target_expression="y",
            features=[
                FeatureSpec(name="x0", expression="x0"),
                FeatureSpec(name="x1", expression="x1"),
            ],
        ),
        input_bounds={"x0": (0.0, 1.0), "x1": (2.0, 3.0)},
        block_id=7,
        predicate_lower=10.0,
        predicate_upper=11.0,
        timeout_seconds=0.5,
        verbose=False,
    )


def test_should_use_native_timeout_only_for_non_fractional_values() -> None:
    assert marabou_verify._should_use_native_timeout(0.0)
    assert marabou_verify._should_use_native_timeout(1.0)
    assert marabou_verify._should_use_native_timeout(30.0)
    assert not marabou_verify._should_use_native_timeout(0.5)


def test_request_payload_roundtrip_rebuilds_function_spec() -> None:
    request = _sample_request()
    rebuilt = marabou_verify._request_from_payload(asdict(request))

    assert rebuilt.model_path == Path("model.onnx")
    assert rebuilt.spec.name == request.spec.name
    assert rebuilt.spec.features[0].name == "x0"
    assert rebuilt.input_bounds == request.input_bounds
    assert rebuilt.timeout_seconds == 0.5


def test_timeout_result_is_conservative() -> None:
    request = _sample_request()
    result = marabou_verify._timeout_result(
        request,
        elapsed_ms=123.0,
    )

    assert result.status == "timeout"
    assert not result.should_skip
    assert result.elapsed_ms == 123.0
    assert result.setup_ms is None
    assert result.solve_ms is None
    assert format_verifier_status(result) == "TIMEOUT"



def test_solver_error_result_is_conservative() -> None:
    request = _sample_request()
    result = marabou_verify._solver_error_result(
        request,
        elapsed_ms=45.0,
        error_type="MalformedBasisException",
        error_message="bad basis",
    )

    assert result.status == "solver_error"
    assert result.status_detail == "MalformedBasisException"
    assert not result.should_skip
    assert result.elapsed_ms == 45.0
    assert "MalformedBasisException" in result.summary
    assert format_verifier_status(result) == "ERROR (MalformedBasisException)"


class _FakeResponseQueue:
    def __init__(self, payload):
        self._payload = payload

    def get(self, timeout):
        return self._payload


class _FakeRequestQueue:
    def put(self, payload):
        self.payload = payload


class _FakeWorker:
    def __init__(self, payload):
        self.request_queue = _FakeRequestQueue()
        self.response_queue = _FakeResponseQueue(payload)


def test_worker_error_returns_conservative_result() -> None:
    request = _sample_request()

    original_should_use_native_timeout = marabou_verify._should_use_native_timeout
    original_get_or_create_worker = marabou_verify._get_or_create_worker
    original_invalidate_active_worker = marabou_verify._invalidate_active_worker
    try:
        marabou_verify._should_use_native_timeout = lambda timeout_seconds: False
        marabou_verify._get_or_create_worker = lambda req: _FakeWorker(
            {
                "kind": "error",
                "type": "MalformedBasisException",
                "message": "bad basis",
                "traceback": "traceback",
            }
        )
        marabou_verify._invalidate_active_worker = lambda: None

        result = marabou_verify._decide_block_skip(request)
    finally:
        marabou_verify._should_use_native_timeout = original_should_use_native_timeout
        marabou_verify._get_or_create_worker = original_get_or_create_worker
        marabou_verify._invalidate_active_worker = original_invalidate_active_worker

    assert result.status == "solver_error"
    assert not result.should_skip
    assert "MalformedBasisException" in result.summary


def test_native_solver_exception_returns_conservative_result() -> None:
    request = _sample_request()

    original_read_onnx = marabou_verify.Marabou.read_onnx
    try:
        def _raise(_path):
            raise RuntimeError("MalformedBasisException")

        marabou_verify.Marabou.read_onnx = _raise
        result = marabou_verify._solve_block_with_marabou(request)
    finally:
        marabou_verify.Marabou.read_onnx = original_read_onnx

    assert result.status == "solver_error"
    assert not result.should_skip
    assert "MalformedBasisException" in result.summary
