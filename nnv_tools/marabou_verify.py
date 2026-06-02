from __future__ import annotations

import atexit
import contextlib
import io
import math
import multiprocessing
import os
import queue
import time
import traceback
import warnings
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Tensorflow parser is unavailable because tensorflow package is not installed",
)

from maraboupy import Marabou

from nnv_tools.block_verifier import BlockSkipResult
from nnv_tools.block_verifier import BlockVerificationRequest
from nnv_tools.block_verifier import format_verifier_status
from nnv_tools.block_metadata import PairGeometry
from nnv_tools.block_metadata import hull_halfspaces_ccw
from nnv_tools.function_catalog import FeatureSpec
from nnv_tools.function_catalog import FunctionSpec


@dataclass(frozen=True)
class _MarabouSessionKey:
    model_path: str
    spec_name: str
    task_type: str
    feature_names: tuple[str, ...]
    num_classes: int | None


@dataclass
class _PersistentMarabouWorker:
    key: _MarabouSessionKey
    ctx: multiprocessing.context.BaseContext
    request_queue: object
    response_queue: object
    process: multiprocessing.Process


_ACTIVE_WORKER: _PersistentMarabouWorker | None = None


@contextlib.contextmanager
def _suppress_native_output():
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    stdout_fd = sys.__stdout__.fileno()
    stderr_fd = sys.__stderr__.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.__stdout__.flush()
        sys.__stderr__.flush()
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            yield
    finally:
        sys.__stdout__.flush()
        sys.__stderr__.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def decide_block_skip(request: BlockVerificationRequest) -> BlockSkipResult:
    target_desc = (
        f"class={request.target_class}"
        if request.spec.task_type == "classifier"
        else f"range=[{request.predicate_lower}, {request.predicate_upper}]"
    )
    if request.verbose:
        print(f"[marabou] Checking block_id={request.block_id} for predicate {target_desc}")
    result = _decide_block_skip(request)
    if request.verbose:
        print(
            f"[marabou] Result for block {request.block_id}: "
            f"should_skip={result.should_skip} ({format_verifier_status(result)})"
        )
    return result


def _decide_block_skip(request: BlockVerificationRequest) -> BlockSkipResult:
    if _should_use_native_timeout(request.timeout_seconds):
        return _solve_block_with_marabou(request)

    worker = _get_or_create_worker(request)
    started_at = time.perf_counter()
    worker.request_queue.put({"kind": "solve", "request": asdict(request)})
    try:
        payload = worker.response_queue.get(timeout=request.timeout_seconds)
    except queue.Empty:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        _invalidate_active_worker()
        return _timeout_result(request, elapsed_ms)

    if payload["kind"] == "result":
        return _block_skip_result_from_payload(payload["value"])

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    _invalidate_active_worker()
    return _solver_error_result(
        request,
        elapsed_ms,
        error_type=payload.get("type", "MarabouError"),
        error_message=payload.get("message", "unknown worker failure"),
    )


def _solve_block_with_marabou(request: BlockVerificationRequest) -> BlockSkipResult:
    started_at = time.perf_counter()
    try:
        with _suppress_native_output():
            network = Marabou.read_onnx(str(request.model_path))
        return _solve_with_network(
            network=network,
            request=request,
            base_lower_bounds=dict(network.lowerBounds),
            base_upper_bounds=dict(network.upperBounds),
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return _solver_error_result(
            request,
            elapsed_ms,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _solve_with_network(
    *,
    network,
    request: BlockVerificationRequest,
    base_lower_bounds: dict[int, float],
    base_upper_bounds: dict[int, float],
) -> BlockSkipResult:
    setup_started_at = time.perf_counter()
    _reset_network_property(
        network,
        base_lower_bounds=base_lower_bounds,
        base_upper_bounds=base_upper_bounds,
    )
    _apply_request_constraints(network, request)
    setup_ms = (time.perf_counter() - setup_started_at) * 1000.0

    options = Marabou.createOptions(
        verbosity=0,
        timeoutInSeconds=_marabou_timeout_option(request.timeout_seconds),
    )

    solve_started_at = time.perf_counter()
    with _suppress_native_output():
        sat_status, _, _ = network.solve(options=options)
    solve_ms = (time.perf_counter() - solve_started_at) * 1000.0
    should_skip = sat_status == "unsat"

    return BlockSkipResult(
        backend="marabou",
        block_id=request.block_id,
        predicate_lower=(
            float(request.predicate_lower) if request.predicate_lower is not None else None
        ),
        predicate_upper=(
            float(request.predicate_upper) if request.predicate_upper is not None else None
        ),
        target_class=request.target_class,
        status=sat_status,
        should_skip=should_skip,
        elapsed_ms=float(setup_ms + solve_ms),
        summary=(
            "Marabou found no feasible output for the requested predicate, so the block can be skipped."
            if should_skip
            else "Marabou found a feasible output for the requested predicate, so the block should be kept."
        ),
        setup_ms=float(setup_ms),
        solve_ms=float(solve_ms),
    )


def _reset_network_property(
    network,
    *,
    base_lower_bounds: dict[int, float],
    base_upper_bounds: dict[int, float],
) -> None:
    network.clearProperty()
    network.lowerBounds.update(base_lower_bounds)
    network.upperBounds.update(base_upper_bounds)


def _apply_request_constraints(network, request: BlockVerificationRequest) -> None:
    input_vars = [network.inputVars[0][0][index] for index in range(len(request.spec.features))]
    feature_to_var = {}
    for feature, variable in zip(request.spec.features, input_vars, strict=True):
        lower_bound, upper_bound = request.input_bounds[feature.name]
        network.setLowerBound(variable, float(lower_bound))
        network.setUpperBound(variable, float(upper_bound))
        feature_to_var[feature.name] = variable

    _apply_pair_geometry_constraints(network, request, feature_to_var)

    if request.spec.task_type == "regressor":
        if request.predicate_lower is None or request.predicate_upper is None:
            raise ValueError("Regressor verification requires predicate bounds.")
        output_var = network.outputVars[0][0][0]
        network.setLowerBound(output_var, float(request.predicate_lower))
        network.setUpperBound(output_var, float(request.predicate_upper))
        return

    if request.spec.task_type == "classifier":
        if request.target_class is None:
            raise ValueError("Classifier verification requires a target_class.")
        output_vars = network.outputVars[0][0]
        target_var = output_vars[request.target_class]
        for class_index, class_var in enumerate(output_vars):
            if class_index == request.target_class:
                continue
            network.addInequality(
                [target_var, class_var],
                [1.0, -1.0],
                0.0,
                isProperty=True,
            )
        return

    raise ValueError(f"Unsupported task_type: {request.spec.task_type}")


def _apply_pair_geometry_constraints(
    network,
    request: BlockVerificationRequest,
    feature_to_var: dict[str, int],
) -> None:
    for geometry in request.pair_geometries or []:
        hull = geometry.bounded_convex_hull or geometry.hull
        if not hull:
            continue
        x_var = feature_to_var[geometry.feature_x]
        y_var = feature_to_var[geometry.feature_y]
        for a, b, c in hull_halfspaces_ccw(hull):
            network.addInequality(
                [x_var, y_var],
                [float(a), float(b)],
                float(c),
                isProperty=True,
            )


def _marabou_timeout_option(timeout_seconds: float) -> int:
    if timeout_seconds <= 0:
        return 0
    return max(1, math.ceil(timeout_seconds))


def _should_use_native_timeout(timeout_seconds: float) -> bool:
    return timeout_seconds <= 0 or float(timeout_seconds).is_integer()


def _get_or_create_worker(request: BlockVerificationRequest) -> _PersistentMarabouWorker:
    global _ACTIVE_WORKER

    key = _worker_key(request)
    if _ACTIVE_WORKER is not None and _ACTIVE_WORKER.key == key and _ACTIVE_WORKER.process.is_alive():
        return _ACTIVE_WORKER

    _invalidate_active_worker()

    ctx = multiprocessing.get_context(_timeout_start_method())
    request_queue = ctx.Queue(maxsize=1)
    response_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_persistent_worker_entrypoint,
        args=(request_queue, response_queue, str(request.model_path)),
        daemon=True,
    )
    process.start()
    _ACTIVE_WORKER = _PersistentMarabouWorker(
        key=key,
        ctx=ctx,
        request_queue=request_queue,
        response_queue=response_queue,
        process=process,
    )
    return _ACTIVE_WORKER


def _worker_key(request: BlockVerificationRequest) -> _MarabouSessionKey:
    return _MarabouSessionKey(
        model_path=str(request.model_path),
        spec_name=request.spec.name,
        task_type=request.spec.task_type,
        feature_names=tuple(feature.name for feature in request.spec.features),
        num_classes=request.spec.num_classes,
    )


def _persistent_worker_entrypoint(request_queue, response_queue, model_path: str) -> None:
    try:
        with _suppress_native_output():
            network = Marabou.read_onnx(model_path)
        base_lower_bounds = dict(network.lowerBounds)
        base_upper_bounds = dict(network.upperBounds)
        while True:
            message = request_queue.get()
            if message["kind"] == "close":
                break
            request = _request_from_payload(message["request"])
            result = _solve_with_network(
                network=network,
                request=request,
                base_lower_bounds=base_lower_bounds,
                base_upper_bounds=base_upper_bounds,
            )
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
        pair_geometries=[
            PairGeometry(
                feature_x=item["feature_x"],
                feature_y=item["feature_y"],
                hull=(
                    [tuple(map(float, point)) for point in item["hull"]]
                    if item.get("hull") is not None
                    else None
                ),
                grid_depth=item.get("grid_depth"),
                grid_cells=item.get("grid_cells"),
                bounded_convex_hull=(
                    [
                        tuple(map(float, point))
                        for point in item["bounded_convex_hull"]
                    ]
                    if item.get("bounded_convex_hull") is not None
                    else None
                ),
            )
            for item in payload.get("pair_geometries") or []
        ],
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
        status_detail=payload.get("status_detail"),
    )


def _invalidate_active_worker() -> None:
    global _ACTIVE_WORKER
    if _ACTIVE_WORKER is None:
        return
    _stop_worker(_ACTIVE_WORKER)
    _ACTIVE_WORKER = None


def _stop_worker(worker: _PersistentMarabouWorker) -> None:
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


def _timeout_start_method() -> str:
    available_methods = multiprocessing.get_all_start_methods()
    if "fork" in available_methods:
        return "fork"
    return "spawn"


def _close_queue(work_queue) -> None:
    work_queue.close()
    work_queue.join_thread()





def _solver_error_result(
    request: BlockVerificationRequest,
    elapsed_ms: float,
    *,
    error_type: str,
    error_message: str,
) -> BlockSkipResult:
    return BlockSkipResult(
        backend="marabou",
        block_id=request.block_id,
        predicate_lower=(
            float(request.predicate_lower) if request.predicate_lower is not None else None
        ),
        predicate_upper=(
            float(request.predicate_upper) if request.predicate_upper is not None else None
        ),
        target_class=request.target_class,
        status="solver_error",
        should_skip=False,
        elapsed_ms=float(elapsed_ms),
        status_detail=error_type,
        summary=(
            f"Marabou failed with {error_type}: {error_message}. "
            "The block is kept conservatively because skipping could not be proved."
        ),
    )

def _timeout_result(request: BlockVerificationRequest, elapsed_ms: float) -> BlockSkipResult:
    return BlockSkipResult(
        backend="marabou",
        block_id=request.block_id,
        predicate_lower=(
            float(request.predicate_lower) if request.predicate_lower is not None else None
        ),
        predicate_upper=(
            float(request.predicate_upper) if request.predicate_upper is not None else None
        ),
        target_class=request.target_class,
        status="timeout",
        should_skip=False,
        elapsed_ms=float(elapsed_ms),
        summary=(
            "Marabou hit the wall-clock timeout before proving the requested predicate impossible, "
            "so the block is kept conservatively."
        ),
    )


atexit.register(_invalidate_active_worker)
