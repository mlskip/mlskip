import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnv_tools.embedding_logistic_pruning import (
    build_holder_metadata,
    build_holder_multi_norm_metadata,
    build_holder_sliced_multi_norm_metadata,
    block_bounds,
    decide_holder_block,
    decide_holder_blocks_for_weight_norms,
    decide_holder_multi_norm_block,
    decide_holder_multi_norm_blocks,
    decide_holder_sliced_multi_norm_blocks,
    dimension_slices,
    decide_logistic_block,
    dual_norm_exponent,
    finite_weight_norm_exponents,
    holder_logit_range,
    holder_multi_norm_logit_ranges,
    holder_sliced_multi_norm_logit_ranges,
    linear_logit_range,
    probability_to_logit,
)


def test_probability_to_logit():
    assert probability_to_logit(0.5) == 0.0
    assert math.isclose(probability_to_logit(0.7), math.log(0.7 / 0.3))


def test_dual_norm_exponent_special_cases():
    assert dual_norm_exponent(1.0) == math.inf
    assert dual_norm_exponent(math.inf) == 1.0
    assert dual_norm_exponent(2.0) == 2.0
    assert math.isclose(dual_norm_exponent(10.0), 10.0 / 9.0)


def test_finite_weight_norm_exponents_defaults_to_one_through_ten():
    assert finite_weight_norm_exponents() == tuple(float(p) for p in range(1, 11))


def test_linear_logit_range_uses_weight_signs():
    coefficients = np.array([2.0, -3.0])
    lower = np.array([1.0, 10.0])
    upper = np.array([4.0, 20.0])

    min_logit, max_logit = linear_logit_range(coefficients, 5.0, lower, upper)

    assert min_logit == 5.0 + 2.0 * 1.0 - 3.0 * 20.0
    assert max_logit == 5.0 + 2.0 * 4.0 - 3.0 * 10.0


def test_block_bounds_groups_rows():
    embeddings = np.array(
        [
            [1.0, 2.0],
            [3.0, 0.0],
            [-1.0, 4.0],
        ]
    )

    lower, upper = block_bounds(embeddings, block_size=2)

    np.testing.assert_allclose(lower, np.array([[1.0, 0.0], [-1.0, 4.0]]))
    np.testing.assert_allclose(upper, np.array([[3.0, 2.0], [-1.0, 4.0]]))


def test_decide_logistic_block_uses_interval_bounds():
    result = decide_logistic_block(
        block_id=0,
        coefficients=np.array([1.0]),
        intercept=0.0,
        lower=np.array([-2.0]),
        upper=np.array([-1.0]),
        threshold=0.5,
    )

    assert result.possible_by_interval is False
    assert result.skip is True


def test_holder_logit_range_bounds_rows():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [2.0, 1.0],
        ]
    )
    coefficients = np.array([2.0, -1.0])
    intercept = 0.5
    metadata = build_holder_metadata(embeddings, radius_norm_q=2.0)

    lower, upper, weight_norm_p = holder_logit_range(metadata, coefficients, intercept)
    row_logits = embeddings @ coefficients + intercept

    assert weight_norm_p == 2.0
    assert lower <= row_logits.min()
    assert upper >= row_logits.max()


def test_decide_holder_block_can_skip_when_upper_is_below_threshold():
    metadata = build_holder_metadata(np.array([[-2.0], [-1.0]]), radius_norm_q=2.0)

    result = decide_holder_block(
        block_id=0,
        coefficients=np.array([1.0]),
        intercept=0.0,
        metadata=metadata,
        threshold=0.5,
    )

    assert result.upper_logit < 0.0
    assert result.skip is True


def test_decide_holder_blocks_for_weight_norms_uses_p_one_through_max():
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])

    checks_by_p = decide_holder_blocks_for_weight_norms(
        embeddings=embeddings,
        block_size=1,
        coefficients=np.array([1.0, -1.0]),
        intercept=0.0,
        max_weight_norm_p=3,
    )

    assert tuple(checks_by_p) == (1.0, 2.0, 3.0)
    assert all(len(checks) == 2 for checks in checks_by_p.values())


def test_holder_multi_norm_metadata_stores_all_p_radii():
    metadata = build_holder_multi_norm_metadata(
        np.array([[0.0, 0.0], [2.0, 0.0]]),
        max_weight_norm_p=3,
    )

    assert tuple(metadata.radii_by_weight_norm_p) == (1.0, 2.0, 3.0)
    assert metadata.radii_by_weight_norm_p[1.0] == 1.0
    assert metadata.radii_by_weight_norm_p[2.0] == 1.0
    assert metadata.radii_by_weight_norm_p[3.0] == 1.0


def test_decide_holder_multi_norm_block_selects_tightest_upper_bound():
    embeddings = np.array(
        [
            [2.0, 0.0],
            [-2.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    metadata = build_holder_multi_norm_metadata(embeddings, max_weight_norm_p=3)
    coefficients = np.array([1.0, 1.0])

    lower_by_p, upper_by_p = holder_multi_norm_logit_ranges(metadata, coefficients, 0.0)
    result = decide_holder_multi_norm_block(
        block_id=0,
        coefficients=coefficients,
        intercept=0.0,
        metadata=metadata,
        threshold=0.99,
    )

    assert result.upper_logit == min(upper_by_p.values())
    assert result.selected_weight_norm_p == min(upper_by_p, key=upper_by_p.get)
    assert result.lower_logit == lower_by_p[result.selected_weight_norm_p]


def test_decide_holder_multi_norm_blocks_returns_one_check_per_block():
    checks = decide_holder_multi_norm_blocks(
        embeddings=np.array([[0.0], [1.0], [2.0]]),
        block_size=2,
        coefficients=np.array([1.0]),
        intercept=0.0,
        max_weight_norm_p=4,
    )

    assert len(checks) == 2
    assert all(check.upper_by_weight_norm_p.keys() == {1.0, 2.0, 3.0, 4.0} for check in checks)


def test_dimension_slices_partitions_dimensions():
    assert dimension_slices(10, 4) == (slice(0, 4), slice(4, 8), slice(8, 10))


def test_sliced_holder_bound_is_not_looser_than_global_for_same_p():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, -3.0, 0.0],
        ]
    )
    coefficients = np.array([1.0, 1.0, 1.0, 1.0])
    global_metadata = build_holder_multi_norm_metadata(embeddings, max_weight_norm_p=2)
    sliced_metadata = build_holder_sliced_multi_norm_metadata(
        embeddings,
        slice_size=2,
        max_weight_norm_p=2,
    )

    _, global_upper_by_p = holder_multi_norm_logit_ranges(global_metadata, coefficients, 0.0)
    _, sliced_upper_by_p = holder_sliced_multi_norm_logit_ranges(
        sliced_metadata,
        coefficients,
        0.0,
    )

    assert sliced_upper_by_p[2.0] <= global_upper_by_p[2.0]


def test_decide_holder_sliced_multi_norm_blocks_returns_one_check_per_block():
    checks = decide_holder_sliced_multi_norm_blocks(
        embeddings=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        block_size=2,
        coefficients=np.array([1.0, -1.0]),
        intercept=0.0,
        slice_size=1,
        max_weight_norm_p=3,
    )

    assert len(checks) == 2
    assert all(check.slice_size == 1 for check in checks)
    assert all(check.upper_by_weight_norm_p.keys() == {1.0, 2.0, 3.0} for check in checks)
