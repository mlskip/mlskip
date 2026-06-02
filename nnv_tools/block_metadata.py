from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Literal

import duckdb
try:
    from scipy.spatial import ConvexHull
    from scipy.spatial import QhullError
except ImportError:  # pragma: no cover - fallback for environments without scipy
    ConvexHull = None
    QhullError = None

from nnv_tools.dataset_duckdb import ROW_ID_COLUMN
from nnv_tools.dataset_duckdb import build_block_id_predicate
from nnv_tools.dataset_duckdb import ensure_dataset_loaded
from nnv_tools.dataset_duckdb import ensure_row_id_column
from nnv_tools.dataset_duckdb import fetch_blocks_bounds
from nnv_tools.function_catalog import FunctionSpec


BlockMetadataKind = Literal["minmax", "convex_hull", "grid", "bounded_convex_hull"]


@dataclass(frozen=True)
class PairGeometry:
    feature_x: str
    feature_y: str
    hull: list[tuple[float, float]] | None = None
    grid_depth: int | None = None
    grid_cells: list[int] | None = None
    bounded_convex_hull: list[tuple[float, float]] | None = None


@dataclass(frozen=True)
class BlockMetadata:
    kind: BlockMetadataKind
    input_bounds: dict[str, tuple[float, float]]
    pair_geometries: list[PairGeometry]


@dataclass(frozen=True)
class BlockMetadataBundle:
    metadata_by_block: dict[int, BlockMetadata]
    collection_ms: float
    collection_ms_by_block: dict[int, float]


def collect_block_metadata(
    *,
    spec: FunctionSpec,
    block_ids: list[int],
    db_path: str | Path,
    block_size: int,
    kind: BlockMetadataKind,
    grid_depth: int,
) -> BlockMetadataBundle:
    metadata_by_block: dict[int, BlockMetadata] = {}
    collection_ms_by_block: dict[int, float] = {}
    total_collection_ms = 0.0
    with duckdb.connect(str(db_path), read_only=True) as con:
        ensure_dataset_loaded(con, spec)
        ensure_row_id_column(con, spec.table)
        for block_id in block_ids:
            started_at = perf_counter()
            metadata = _collect_block_metadata_for_block(
                spec=spec,
                block_id=block_id,
                db_path=db_path,
                block_size=block_size,
                kind=kind,
                grid_depth=grid_depth,
                con=con,
            )
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            if metadata is None:
                continue
            metadata_by_block[block_id] = metadata
            collection_ms_by_block[block_id] = elapsed_ms
            total_collection_ms += elapsed_ms
    return BlockMetadataBundle(
        metadata_by_block=metadata_by_block,
        collection_ms=total_collection_ms,
        collection_ms_by_block=collection_ms_by_block,
    )


def _collect_block_metadata_for_block(
    *,
    spec: FunctionSpec,
    block_id: int,
    db_path: str | Path,
    block_size: int,
    kind: BlockMetadataKind,
    grid_depth: int,
    con: duckdb.DuckDBPyConnection | None = None,
) -> BlockMetadata | None:
    bounds_by_block = fetch_blocks_bounds(
        spec,
        [block_id],
        db_path,
        block_size,
        con=con,
    )
    bounds = bounds_by_block.get(block_id)
    if bounds is None:
        return None
    if kind == "minmax" or len(spec.features) < 2:
        return BlockMetadata(
            kind=kind,
            input_bounds=bounds,
            pair_geometries=[],
        )

    points_by_block = _fetch_feature_points_by_block(
        spec=spec,
        block_ids=[block_id],
        db_path=db_path,
        block_size=block_size,
        con=con,
    )
    rows = points_by_block.get(block_id, [])
    feature_names = [feature.name for feature in spec.features]
    pair_indices = list(combinations(range(len(feature_names)), 2))
    pair_geometries: list[PairGeometry] = []
    for x_index, y_index in pair_indices:
        feature_x = feature_names[x_index]
        feature_y = feature_names[y_index]
        pair_points = [
            (float(row[x_index]), float(row[y_index]))
            for row in rows
            if row[x_index] is not None and row[y_index] is not None
        ]
        hull = (
            monotone_chain_hull(pair_points)
            if kind == "convex_hull"
            else None
        )
        cells = (
            occupied_grid_cells(
                pair_points,
                input_bounds=bounds,
                feature_x=feature_x,
                feature_y=feature_y,
                depth=grid_depth,
            )
            if kind == "grid"
            else None
        )
        bounded_hull = None
        if kind == "bounded_convex_hull":
            bounded_cells = occupied_grid_cells(
                pair_points,
                input_bounds=bounds,
                feature_x=feature_x,
                feature_y=feature_y,
                depth=grid_depth,
            )
            bounded_hull = discretized_rectangles_hull(
                bounded_cells,
                input_bounds=bounds,
                feature_x=feature_x,
                feature_y=feature_y,
                depth=grid_depth,
            )
        pair_geometries.append(
            PairGeometry(
                feature_x=feature_x,
                feature_y=feature_y,
                hull=hull,
                grid_depth=(
                    grid_depth
                    if kind in {"grid", "bounded_convex_hull"}
                    else None
                ),
                grid_cells=cells,
                bounded_convex_hull=bounded_hull,
            )
        )
    return BlockMetadata(
        kind=kind,
        input_bounds=bounds,
        pair_geometries=pair_geometries,
    )


def monotone_chain_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if ConvexHull is not None:
        scipy_hull = _scipy_convex_hull(points)
        if scipy_hull is not None:
            return scipy_hull
    return _monotone_chain_hull(points)


def _scipy_convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return [(float(x), float(y)) for x, y in unique]
    try:
        hull = ConvexHull(unique)
    except QhullError:
        return None
    return [(float(unique[index][0]), float(unique[index][1])) for index in hull.vertices]


def _monotone_chain_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return [(float(x), float(y)) for x, y in unique]

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return [(float(x), float(y)) for x, y in lower[:-1] + upper[:-1]]


def hull_halfspaces_ccw(
    hull: list[tuple[float, float]],
) -> list[tuple[float, float, float]]:
    if len(hull) < 3:
        return []
    halfspaces = []
    for index, current in enumerate(hull):
        nxt = hull[(index + 1) % len(hull)]
        dx = float(nxt[0] - current[0])
        dy = float(nxt[1] - current[1])
        a = dy
        b = -dx
        c = (dy * float(current[0])) - (dx * float(current[1]))
        halfspaces.append((a, b, c))
    return halfspaces


def occupied_grid_cells(
    points: list[tuple[float, float]],
    *,
    input_bounds: dict[str, tuple[float, float]],
    feature_x: str,
    feature_y: str,
    depth: int,
) -> list[int]:
    if not points:
        return []
    grid_n = 2 ** int(depth)
    x_lo, x_hi = input_bounds[feature_x]
    y_lo, y_hi = input_bounds[feature_y]
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo
    if x_span <= 0.0 or y_span <= 0.0:
        return [0]

    cell_ids = set()
    for x_value, y_value in points:
        x_idx = int(((x_value - x_lo) / x_span) * grid_n)
        y_idx = int(((y_value - y_lo) / y_span) * grid_n)
        x_idx = min(max(x_idx, 0), grid_n - 1)
        y_idx = min(max(y_idx, 0), grid_n - 1)
        cell_ids.add((y_idx * grid_n) + x_idx)
    return sorted(cell_ids)


def grid_cell_rect(
    cell_id: int,
    *,
    input_bounds: dict[str, tuple[float, float]],
    feature_x: str,
    feature_y: str,
    depth: int,
) -> dict[str, tuple[float, float]]:
    grid_n = 2 ** int(depth)
    ix = int(cell_id) % grid_n
    iy = int(cell_id) // grid_n
    x_lo, x_hi = input_bounds[feature_x]
    y_lo, y_hi = input_bounds[feature_y]
    x_step = (x_hi - x_lo) / grid_n if grid_n else 0.0
    y_step = (y_hi - y_lo) / grid_n if grid_n else 0.0
    return {
        feature_x: (float(x_lo + ix * x_step), float(x_lo + (ix + 1) * x_step)),
        feature_y: (float(y_lo + iy * y_step), float(y_lo + (iy + 1) * y_step)),
    }


def discretized_rectangles_hull(
    cell_ids: list[int],
    *,
    input_bounds: dict[str, tuple[float, float]],
    feature_x: str,
    feature_y: str,
    depth: int,
) -> list[tuple[float, float]]:
    corners: list[tuple[float, float]] = []
    for cell_id in cell_ids:
        rect = grid_cell_rect(
            cell_id,
            input_bounds=input_bounds,
            feature_x=feature_x,
            feature_y=feature_y,
            depth=depth,
        )
        x_lo, x_hi = rect[feature_x]
        y_lo, y_hi = rect[feature_y]
        corners.extend([(x_lo, y_lo), (x_hi, y_lo), (x_hi, y_hi), (x_lo, y_hi)])
    return monotone_chain_hull(corners)


def _fetch_feature_points_by_block(
    *,
    spec: FunctionSpec,
    block_ids: list[int],
    db_path: str | Path,
    block_size: int,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[int, list[tuple[float, ...]]]:
    if not block_ids:
        return {}
    feature_select = ", ".join(
        f"CAST({feature.expression} AS DOUBLE) AS {feature.name}"
        for feature in spec.features
    )
    valid_predicate = " AND ".join(
        f"{feature.name} IS NOT NULL AND isfinite({feature.name})"
        for feature in spec.features
    )
    owns_connection = con is None
    if con is None:
        con = duckdb.connect(str(db_path), read_only=True)
    try:
        ensure_dataset_loaded(con, spec)
        ensure_row_id_column(con, spec.table)
        block_predicate = build_block_id_predicate(block_ids, block_size)
        rows = con.execute(
            f"""
            WITH block_points AS (
                SELECT
                    CAST(FLOOR({ROW_ID_COLUMN} / {int(block_size)}) AS BIGINT) AS block_id,
                    {ROW_ID_COLUMN} AS row_id,
                    {feature_select}
                FROM {spec.table}
                WHERE {block_predicate}
            )
            SELECT block_id, {", ".join(feature.name for feature in spec.features)}
            FROM block_points
            WHERE {valid_predicate}
            ORDER BY block_id, row_id
            """
        ).fetchall()
    finally:
        if owns_connection:
            con.close()

    points_by_block: dict[int, list[tuple[float, ...]]] = {}
    for row in rows:
        block_id = int(row[0])
        points_by_block.setdefault(block_id, []).append(
            tuple(float(value) for value in row[1:])
        )
    return points_by_block
