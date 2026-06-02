from __future__ import annotations

from nnv_tools.block_metadata import hull_halfspaces_ccw
from nnv_tools.block_metadata import monotone_chain_hull
from nnv_tools.block_metadata import occupied_grid_cells


def test_monotone_chain_hull_drops_interior_points() -> None:
    hull = monotone_chain_hull(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.5),
        ]
    )

    assert hull == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_hull_halfspaces_contain_the_ccw_square() -> None:
    hull = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    halfspaces = hull_halfspaces_ccw(hull)

    assert len(halfspaces) == 4
    for a, b, c in halfspaces:
        assert (a * 0.5) + (b * 0.5) <= c


def test_occupied_grid_cells_uses_2d_depth() -> None:
    cells = occupied_grid_cells(
        [(0.1, 0.1), (0.9, 0.9)],
        input_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        feature_x="x",
        feature_y="y",
        depth=1,
    )

    assert cells == [0, 3]
