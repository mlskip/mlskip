from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


PredicateKind = Literal["probability_at_least", "probability_below"]


@dataclass(frozen=True)
class LogisticBlockCheck:
    block_id: int
    predicate: PredicateKind
    threshold: float
    decision_logit: float
    min_logit: float
    max_logit: float
    possible_by_interval: bool
    skip: bool


@dataclass(frozen=True)
class HolderBlockMetadata:
    center: np.ndarray
    radius_q: float
    q: float


@dataclass(frozen=True)
class HolderMultiNormBlockMetadata:
    center: np.ndarray
    radii_by_weight_norm_p: dict[float, float]


@dataclass(frozen=True)
class HolderSlicedMultiNormBlockMetadata:
    center: np.ndarray
    radii_by_weight_norm_p: dict[float, np.ndarray]
    slice_size: int
    slices: tuple[slice, ...]


@dataclass(frozen=True)
class HolderBlockCheck:
    block_id: int
    predicate: PredicateKind
    threshold: float
    decision_logit: float
    lower_logit: float
    upper_logit: float
    weight_norm_p: float
    radius_norm_q: float
    possible: bool
    skip: bool


@dataclass(frozen=True)
class HolderMultiNormBlockCheck:
    block_id: int
    predicate: PredicateKind
    threshold: float
    decision_logit: float
    lower_logit: float
    upper_logit: float
    selected_weight_norm_p: float
    selected_radius_norm_q: float
    upper_by_weight_norm_p: dict[float, float]
    lower_by_weight_norm_p: dict[float, float]
    possible: bool
    skip: bool


@dataclass(frozen=True)
class HolderSlicedMultiNormBlockCheck:
    block_id: int
    predicate: PredicateKind
    threshold: float
    decision_logit: float
    lower_logit: float
    upper_logit: float
    selected_weight_norm_p: float
    selected_radius_norm_q: float
    slice_size: int
    upper_by_weight_norm_p: dict[float, float]
    lower_by_weight_norm_p: dict[float, float]
    possible: bool
    skip: bool


def probability_to_logit(threshold: float) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold!r}")
    return math.log(threshold / (1.0 - threshold))


def dual_norm_exponent(exponent: float) -> float:
    if exponent == 1.0:
        return math.inf
    if math.isinf(exponent):
        return 1.0
    if exponent <= 1.0:
        raise ValueError(f"norm exponent must be >= 1, got {exponent!r}")
    return exponent / (exponent - 1.0)


def finite_weight_norm_exponents(max_p: int = 10) -> tuple[float, ...]:
    if max_p < 1:
        raise ValueError(f"max_p must be >= 1, got {max_p}")
    return tuple(float(p) for p in range(1, max_p + 1))


def vector_norm(values: np.ndarray, exponent: float) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float), ord=exponent))


def block_bounds(embeddings: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    lower = []
    upper = []
    for start in range(0, len(embeddings), block_size):
        block = embeddings[start : start + block_size]
        lower.append(np.min(block, axis=0))
        upper.append(np.max(block, axis=0))
    return np.vstack(lower), np.vstack(upper)


def build_holder_metadata(embeddings: np.ndarray, radius_norm_q: float = 2.0) -> HolderBlockMetadata:
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(embeddings) == 0:
        raise ValueError("embeddings block must not be empty")

    center = embeddings.mean(axis=0)
    residuals = embeddings - center
    radius_q = float(np.max(np.linalg.norm(residuals, ord=radius_norm_q, axis=1)))
    return HolderBlockMetadata(center=center, radius_q=radius_q, q=radius_norm_q)


def build_holder_multi_norm_metadata(
    embeddings: np.ndarray,
    max_weight_norm_p: int = 10,
) -> HolderMultiNormBlockMetadata:
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(embeddings) == 0:
        raise ValueError("embeddings block must not be empty")

    center = embeddings.mean(axis=0)
    residuals = embeddings - center
    radii_by_weight_norm_p = {}
    for weight_norm_p in finite_weight_norm_exponents(max_weight_norm_p):
        radius_norm_q = dual_norm_exponent(weight_norm_p)
        radii_by_weight_norm_p[weight_norm_p] = float(
            np.max(np.linalg.norm(residuals, ord=radius_norm_q, axis=1))
        )
    return HolderMultiNormBlockMetadata(
        center=center,
        radii_by_weight_norm_p=radii_by_weight_norm_p,
    )


def dimension_slices(dimensions: int, slice_size: int) -> tuple[slice, ...]:
    if dimensions <= 0:
        raise ValueError(f"dimensions must be positive, got {dimensions}")
    if slice_size <= 0:
        raise ValueError(f"slice_size must be positive, got {slice_size}")
    return tuple(slice(start, min(start + slice_size, dimensions)) for start in range(0, dimensions, slice_size))


def _slice_starts(slices: tuple[slice, ...]) -> np.ndarray:
    return np.array([dim_slice.start for dim_slice in slices], dtype=int)


def sliced_vector_norms(values: np.ndarray, slices: tuple[slice, ...], exponent: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    abs_values = np.abs(values)
    starts = _slice_starts(slices)
    if math.isinf(exponent):
        return np.maximum.reduceat(abs_values, starts)
    return np.add.reduceat(abs_values**exponent, starts) ** (1.0 / exponent)


def sliced_row_norms(values: np.ndarray, slices: tuple[slice, ...], exponent: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    abs_values = np.abs(values)
    starts = _slice_starts(slices)
    if math.isinf(exponent):
        return np.maximum.reduceat(abs_values, starts, axis=1)
    return np.add.reduceat(abs_values**exponent, starts, axis=1) ** (1.0 / exponent)


def build_holder_sliced_multi_norm_metadata(
    embeddings: np.ndarray,
    slice_size: int = 8,
    max_weight_norm_p: int = 10,
) -> HolderSlicedMultiNormBlockMetadata:
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(embeddings) == 0:
        raise ValueError("embeddings block must not be empty")

    center = embeddings.mean(axis=0)
    residuals = embeddings - center
    slices = dimension_slices(embeddings.shape[1], slice_size)
    radii_by_weight_norm_p = {}
    for weight_norm_p in finite_weight_norm_exponents(max_weight_norm_p):
        radius_norm_q = dual_norm_exponent(weight_norm_p)
        radii_by_weight_norm_p[weight_norm_p] = np.max(
            sliced_row_norms(residuals, slices, radius_norm_q),
            axis=0,
        )

    return HolderSlicedMultiNormBlockMetadata(
        center=center,
        radii_by_weight_norm_p=radii_by_weight_norm_p,
        slice_size=slice_size,
        slices=slices,
    )


def holder_block_metadata(
    embeddings: np.ndarray,
    block_size: int,
    radius_norm_q: float = 2.0,
) -> list[HolderBlockMetadata]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return [
        build_holder_metadata(embeddings[start : start + block_size], radius_norm_q)
        for start in range(0, len(embeddings), block_size)
    ]


def holder_multi_norm_block_metadata(
    embeddings: np.ndarray,
    block_size: int,
    max_weight_norm_p: int = 10,
) -> list[HolderMultiNormBlockMetadata]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return [
        build_holder_multi_norm_metadata(
            embeddings[start : start + block_size],
            max_weight_norm_p=max_weight_norm_p,
        )
        for start in range(0, len(embeddings), block_size)
    ]


def holder_sliced_multi_norm_block_metadata(
    embeddings: np.ndarray,
    block_size: int,
    slice_size: int = 8,
    max_weight_norm_p: int = 10,
) -> list[HolderSlicedMultiNormBlockMetadata]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return [
        build_holder_sliced_multi_norm_metadata(
            embeddings[start : start + block_size],
            slice_size=slice_size,
            max_weight_norm_p=max_weight_norm_p,
        )
        for start in range(0, len(embeddings), block_size)
    ]


def holder_logit_range(
    metadata: HolderBlockMetadata,
    coefficients: np.ndarray,
    intercept: float,
) -> tuple[float, float, float]:
    weight_norm_p = dual_norm_exponent(metadata.q)
    midpoint = float(np.dot(coefficients, metadata.center) + intercept)
    margin = vector_norm(coefficients, weight_norm_p) * metadata.radius_q
    return midpoint - margin, midpoint + margin, weight_norm_p


def holder_multi_norm_logit_ranges(
    metadata: HolderMultiNormBlockMetadata,
    coefficients: np.ndarray,
    intercept: float,
) -> tuple[dict[float, float], dict[float, float]]:
    midpoint = float(np.dot(coefficients, metadata.center) + intercept)
    lower_by_p = {}
    upper_by_p = {}
    for weight_norm_p, radius_q in metadata.radii_by_weight_norm_p.items():
        margin = vector_norm(coefficients, weight_norm_p) * radius_q
        lower_by_p[weight_norm_p] = midpoint - margin
        upper_by_p[weight_norm_p] = midpoint + margin
    return lower_by_p, upper_by_p


def holder_sliced_multi_norm_logit_ranges(
    metadata: HolderSlicedMultiNormBlockMetadata,
    coefficients: np.ndarray,
    intercept: float,
) -> tuple[dict[float, float], dict[float, float]]:
    coefficients = np.asarray(coefficients, dtype=float)
    midpoint = float(np.dot(coefficients, metadata.center) + intercept)
    lower_by_p = {}
    upper_by_p = {}
    for weight_norm_p, radii_by_slice in metadata.radii_by_weight_norm_p.items():
        total_margin = float(
            np.dot(sliced_vector_norms(coefficients, metadata.slices, weight_norm_p), radii_by_slice)
        )
        lower_by_p[weight_norm_p] = midpoint - total_margin
        upper_by_p[weight_norm_p] = midpoint + total_margin
    return lower_by_p, upper_by_p


def decide_holder_block(
    block_id: int,
    coefficients: np.ndarray,
    intercept: float,
    metadata: HolderBlockMetadata,
    threshold: float = 0.5,
    predicate: PredicateKind = "probability_at_least",
) -> HolderBlockCheck:
    decision_logit = probability_to_logit(threshold)
    lower_logit, upper_logit, weight_norm_p = holder_logit_range(
        metadata, coefficients, intercept
    )
    possible = interval_predicate_possible(
        lower_logit,
        upper_logit,
        decision_logit,
        predicate,
    )
    return HolderBlockCheck(
        block_id=block_id,
        predicate=predicate,
        threshold=threshold,
        decision_logit=decision_logit,
        lower_logit=lower_logit,
        upper_logit=upper_logit,
        weight_norm_p=weight_norm_p,
        radius_norm_q=metadata.q,
        possible=possible,
        skip=not possible,
    )


def _select_holder_bound(
    lower_by_p: dict[float, float],
    upper_by_p: dict[float, float],
    predicate: PredicateKind,
) -> tuple[float, float, float]:
    if predicate == "probability_at_least":
        selected_weight_norm_p = min(upper_by_p, key=upper_by_p.get)
    elif predicate == "probability_below":
        selected_weight_norm_p = max(lower_by_p, key=lower_by_p.get)
    else:
        raise ValueError(f"unknown predicate: {predicate!r}")
    return (
        selected_weight_norm_p,
        lower_by_p[selected_weight_norm_p],
        upper_by_p[selected_weight_norm_p],
    )


def decide_holder_multi_norm_block(
    block_id: int,
    coefficients: np.ndarray,
    intercept: float,
    metadata: HolderMultiNormBlockMetadata,
    threshold: float = 0.5,
    predicate: PredicateKind = "probability_at_least",
) -> HolderMultiNormBlockCheck:
    decision_logit = probability_to_logit(threshold)
    lower_by_p, upper_by_p = holder_multi_norm_logit_ranges(
        metadata,
        coefficients,
        intercept,
    )

    selected_weight_norm_p, lower_logit, upper_logit = _select_holder_bound(
        lower_by_p,
        upper_by_p,
        predicate,
    )

    possible = interval_predicate_possible(
        lower_logit,
        upper_logit,
        decision_logit,
        predicate,
    )
    return HolderMultiNormBlockCheck(
        block_id=block_id,
        predicate=predicate,
        threshold=threshold,
        decision_logit=decision_logit,
        lower_logit=lower_logit,
        upper_logit=upper_logit,
        selected_weight_norm_p=selected_weight_norm_p,
        selected_radius_norm_q=dual_norm_exponent(selected_weight_norm_p),
        upper_by_weight_norm_p=upper_by_p,
        lower_by_weight_norm_p=lower_by_p,
        possible=possible,
        skip=not possible,
    )


def decide_holder_sliced_multi_norm_block(
    block_id: int,
    coefficients: np.ndarray,
    intercept: float,
    metadata: HolderSlicedMultiNormBlockMetadata,
    threshold: float = 0.5,
    predicate: PredicateKind = "probability_at_least",
) -> HolderSlicedMultiNormBlockCheck:
    decision_logit = probability_to_logit(threshold)
    lower_by_p, upper_by_p = holder_sliced_multi_norm_logit_ranges(
        metadata,
        coefficients,
        intercept,
    )
    selected_weight_norm_p, lower_logit, upper_logit = _select_holder_bound(
        lower_by_p,
        upper_by_p,
        predicate,
    )
    possible = interval_predicate_possible(
        lower_logit,
        upper_logit,
        decision_logit,
        predicate,
    )
    return HolderSlicedMultiNormBlockCheck(
        block_id=block_id,
        predicate=predicate,
        threshold=threshold,
        decision_logit=decision_logit,
        lower_logit=lower_logit,
        upper_logit=upper_logit,
        selected_weight_norm_p=selected_weight_norm_p,
        selected_radius_norm_q=dual_norm_exponent(selected_weight_norm_p),
        slice_size=metadata.slice_size,
        upper_by_weight_norm_p=upper_by_p,
        lower_by_weight_norm_p=lower_by_p,
        possible=possible,
        skip=not possible,
    )


def decide_holder_multi_norm_blocks(
    embeddings: np.ndarray,
    block_size: int,
    coefficients: np.ndarray,
    intercept: float,
    threshold: float = 0.5,
    predicate: PredicateKind = "probability_at_least",
    max_weight_norm_p: int = 10,
) -> list[HolderMultiNormBlockCheck]:
    metadata_by_block = holder_multi_norm_block_metadata(
        embeddings,
        block_size,
        max_weight_norm_p=max_weight_norm_p,
    )
    return [
        decide_holder_multi_norm_block(
            block_id=block_id,
            coefficients=coefficients,
            intercept=intercept,
            metadata=metadata,
            threshold=threshold,
            predicate=predicate,
        )
        for block_id, metadata in enumerate(metadata_by_block)
    ]


def decide_holder_sliced_multi_norm_blocks(
    embeddings: np.ndarray,
    block_size: int,
    coefficients: np.ndarray,
    intercept: float,
    threshold: float = 0.5,
    predicate: PredicateKind = "probability_at_least",
    slice_size: int = 8,
    max_weight_norm_p: int = 10,
) -> list[HolderSlicedMultiNormBlockCheck]:
    metadata_by_block = holder_sliced_multi_norm_block_metadata(
        embeddings,
        block_size,
        slice_size=slice_size,
        max_weight_norm_p=max_weight_norm_p,
    )
    return [
        decide_holder_sliced_multi_norm_block(
            block_id=block_id,
            coefficients=coefficients,
            intercept=intercept,
            metadata=metadata,
            threshold=threshold,
            predicate=predicate,
        )
        for block_id, metadata in enumerate(metadata_by_block)
    ]


def decide_holder_blocks_for_weight_norms(
    embeddings: np.ndarray,
    block_size: int,
    coefficients: np.ndarray,
    intercept: float,
    threshold: float = 0.5,
    predicate: PredicateKind = "probability_at_least",
    max_weight_norm_p: int = 10,
) -> dict[float, list[HolderBlockCheck]]:
    checks_by_p = {}
    for weight_norm_p in finite_weight_norm_exponents(max_weight_norm_p):
        radius_norm_q = dual_norm_exponent(weight_norm_p)
        metadata_by_block = holder_block_metadata(embeddings, block_size, radius_norm_q)
        checks_by_p[weight_norm_p] = [
            decide_holder_block(
                block_id=block_id,
                coefficients=coefficients,
                intercept=intercept,
                metadata=metadata,
                threshold=threshold,
                predicate=predicate,
            )
            for block_id, metadata in enumerate(metadata_by_block)
        ]
    return checks_by_p


def linear_logit_range(
    coefficients: np.ndarray,
    intercept: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, float]:
    coefficients = np.asarray(coefficients, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if coefficients.shape != lower.shape or coefficients.shape != upper.shape:
        raise ValueError(
            "coefficients, lower, and upper must have matching shapes; "
            f"got {coefficients.shape}, {lower.shape}, {upper.shape}"
        )

    min_terms = np.where(coefficients >= 0.0, coefficients * lower, coefficients * upper)
    max_terms = np.where(coefficients >= 0.0, coefficients * upper, coefficients * lower)
    return float(intercept + np.sum(min_terms)), float(intercept + np.sum(max_terms))


def interval_predicate_possible(
    min_logit: float,
    max_logit: float,
    decision_logit: float,
    predicate: PredicateKind = "probability_at_least",
) -> bool:
    if predicate == "probability_at_least":
        return max_logit >= decision_logit
    if predicate == "probability_below":
        return min_logit < decision_logit
    raise ValueError(f"unknown predicate: {predicate!r}")


def decide_logistic_block(
    block_id: int,
    coefficients: np.ndarray,
    intercept: float,
    lower: np.ndarray,
    upper: np.ndarray,
    threshold: float = 0.5,
    predicate: PredicateKind = "probability_at_least",
) -> LogisticBlockCheck:
    decision_logit = probability_to_logit(threshold)
    min_logit, max_logit = linear_logit_range(coefficients, intercept, lower, upper)
    possible_by_interval = interval_predicate_possible(
        min_logit, max_logit, decision_logit, predicate
    )

    return LogisticBlockCheck(
        block_id=block_id,
        predicate=predicate,
        threshold=threshold,
        decision_logit=decision_logit,
        min_logit=min_logit,
        max_logit=max_logit,
        possible_by_interval=possible_by_interval,
        skip=not possible_by_interval,
    )
