from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nnv_tools.function_catalog import FunctionSpec
from nnv_tools.block_metadata import PairGeometry


VerifierBackend = Literal["marabou", "abcrown", "geomcad", "pytorch"]


@dataclass(frozen=True)
class BlockVerificationRequest:
    model_path: str | Path
    spec: FunctionSpec
    input_bounds: dict[str, tuple[float, float]]
    block_id: int
    pair_geometries: list[PairGeometry] | None = None
    predicate_lower: float | None = None
    predicate_upper: float | None = None
    target_class: int | None = None
    timeout_seconds: float = 0.0
    verbose: bool = True


@dataclass(frozen=True)
class BlockSkipResult:
    backend: str
    block_id: int
    predicate_lower: float | None
    predicate_upper: float | None
    target_class: int | None
    status: str
    should_skip: bool
    elapsed_ms: float
    summary: str
    setup_ms: float | None = None
    solve_ms: float | None = None
    status_detail: str | None = None


def format_verifier_status(result: BlockSkipResult | None) -> str | None:
    if result is None:
        return None
    if result.status == "timeout":
        return "TIMEOUT"
    if result.status == "solver_error":
        if result.status_detail:
            return f"ERROR ({result.status_detail})"
        return "ERROR"
    return result.status


def supported_verifier_backends() -> tuple[VerifierBackend, ...]:
    return ("marabou", "abcrown", "geomcad", "pytorch")


def decide_block_skip(
    *,
    backend: VerifierBackend,
    request: BlockVerificationRequest,
    output_path: str | Path | None = None,
) -> BlockSkipResult:
    if backend == "marabou":
        from nnv_tools.marabou_verify import decide_block_skip as decide_with_marabou

        result = decide_with_marabou(request)
    elif backend == "abcrown":
        from nnv_tools.abcrown_verify import decide_block_skip as decide_with_abcrown

        result = decide_with_abcrown(request)
    elif backend == "geomcad":
        from nnv_tools.geomcad_verify import decide_block_skip as decide_with_geomcad

        result = decide_with_geomcad(request)
    else:
        raise ValueError(f"Unsupported verifier backend: {backend}")

    if output_path is not None:
        Path(output_path).write_text(json.dumps(asdict(result), indent=2))
    return result
