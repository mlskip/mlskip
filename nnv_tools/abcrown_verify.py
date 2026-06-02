from __future__ import annotations

import contextlib
import importlib
import io
import sys
import time
from pathlib import Path

import torch

from nnv_tools.block_verifier import BlockSkipResult
from nnv_tools.block_verifier import BlockVerificationRequest
from nnv_tools.block_verifier import format_verifier_status
from nnv_tools.metadata_paths import REPO_ROOT

_SAFE_STATUSES = {"verified", "safe", "safe-incomplete"}


def decide_block_skip(request: BlockVerificationRequest) -> BlockSkipResult:
    with _abcrown_output_context(request.verbose):
        abcrown_module = _import_abcrown_api()
        ABCrownSolver = abcrown_module.ABCrownSolver
        ConfigBuilder = abcrown_module.ConfigBuilder
        VerificationSpec = abcrown_module.VerificationSpec
        input_vars = abcrown_module.input_vars
        output_vars = abcrown_module.output_vars

        if request.spec.task_type == "classifier" and request.spec.num_classes is None:
            raise ValueError("Classifier verification requires FunctionSpec.num_classes.")

        x = input_vars(len(request.spec.features))
        y = output_vars(_output_dim(request))
        lower, upper = _input_bound_tensors(request)

        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=(x > lower) & (x < upper),
            output_constraint=_output_constraint(request, y),
        )

        builder = (
            ConfigBuilder.from_defaults()
            .set(general__device="cpu")
            .set(general__seed=123)
            .set(attack__pgd_order="before")
        )
        if request.timeout_seconds > 0:
            builder.set(bab__override_timeout=float(request.timeout_seconds))
        config = builder()

        if request.verbose:
            print(
                f"[abcrown] Checking block_id={request.block_id} for "
                f"{_target_description(request)}"
            )

        started_at = time.perf_counter()
        solver = ABCrownSolver(
            spec,
            str(request.model_path),
            config=config,
            name=f"block-{request.block_id}",
        )
        result = solver.solve()
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

    should_skip = result.success and result.status in _SAFE_STATUSES
    if should_skip:
        summary = (
            "alpha-beta-CROWN proved the target predicate is impossible over the block bounds, "
            "so the block can be skipped."
        )
    else:
        summary = (
            "alpha-beta-CROWN did not prove the target predicate impossible over the block bounds, "
            "so the block should be kept."
        )

    if request.verbose:
        print(
            f"[abcrown] Result for block {request.block_id}: "
            f"should_skip={should_skip} ({format_verifier_status(result)})"
        )

    return BlockSkipResult(
        backend="abcrown",
        block_id=request.block_id,
        predicate_lower=(
            float(request.predicate_lower) if request.predicate_lower is not None else None
        ),
        predicate_upper=(
            float(request.predicate_upper) if request.predicate_upper is not None else None
        ),
        target_class=request.target_class,
        status=str(result.status),
        should_skip=bool(should_skip),
        elapsed_ms=float(elapsed_ms),
        summary=summary,
    )


@contextlib.contextmanager
def _abcrown_output_context(verbose: bool):
    if verbose:
        yield
        return

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def _import_abcrown_api():
    abcrown_dir = REPO_ROOT / "third-party" / "alpha-beta-CROWN"
    if not abcrown_dir.exists():
        raise FileNotFoundError(
            f"Could not find third-party alpha-beta-CROWN checkout at {abcrown_dir}."
        )

    for extra_path in (abcrown_dir, abcrown_dir / "complete_verifier"):
        extra_path_str = str(extra_path)
        if extra_path_str not in sys.path:
            sys.path.insert(0, extra_path_str)
    importlib.invalidate_caches()

    last_error: Exception | None = None
    for module_name in ("complete_verifier", "abcrown", "complete_verifier.abcrown"):
        try:
            module = importlib.import_module(module_name)
            required_attrs = (
                "ABCrownSolver",
                "ConfigBuilder",
                "VerificationSpec",
                "input_vars",
                "output_vars",
            )
            if all(hasattr(module, attr) for attr in required_attrs):
                return module
        except Exception as exc:  # pragma: no cover - import errors depend on local env
            last_error = exc
    raise RuntimeError("Could not import alpha-beta-CROWN API.") from last_error


def _output_dim(request: BlockVerificationRequest) -> int:
    if request.spec.task_type == "classifier":
        assert request.spec.num_classes is not None
        return request.spec.num_classes
    return 1


def _input_bound_tensors(
    request: BlockVerificationRequest,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = []
    upper = []
    for feature in request.spec.features:
        feature_lower, feature_upper = request.input_bounds[feature.name]
        lower.append(float(feature_lower))
        upper.append(float(feature_upper))
    return (
        torch.tensor(lower, dtype=torch.float32),
        torch.tensor(upper, dtype=torch.float32),
    )


def _output_constraint(request: BlockVerificationRequest, y):
    if request.spec.task_type == "regressor":
        if request.predicate_lower is None or request.predicate_upper is None:
            raise ValueError("Regressor verification requires predicate bounds.")
        return (y[0] < request.predicate_lower) | (y[0] > request.predicate_upper)

    if request.spec.task_type == "classifier":
        if request.target_class is None:
            raise ValueError("Classifier verification requires a target_class.")
        other_classes = [
            class_index
            for class_index in range(_output_dim(request))
            if class_index != request.target_class
        ]
        if not other_classes:
            raise ValueError("Classifier verification requires at least two classes.")
        constraint = y[other_classes[0]] > y[request.target_class]
        for class_index in other_classes[1:]:
            constraint = constraint | (y[class_index] > y[request.target_class])
        return constraint

    raise ValueError(f"Unsupported task_type: {request.spec.task_type}")


def _target_description(request: BlockVerificationRequest) -> str:
    if request.spec.task_type == "classifier":
        return f"class={request.target_class}"
    return f"range=[{request.predicate_lower}, {request.predicate_upper}]"
