-- duckdb db/hybrid.db < <(cat queries/hybrid.sql | sed 's/$x_min/20/g; s/$x_max/80/g; s/$y_min/20/g; s/$y_max/80/g; s/$f_min/0/g; s/$f_max/100/g')

WITH
-- See bounded.sql
finite_vertices AS (
    SELECT
        GREATEST(LEAST(x1, $x_max), $x_min) AS x1,
        GREATEST(LEAST(y1, $y_max), $y_min) AS y1,
        GREATEST(LEAST(x2, $x_max), $x_min) AS x2,
        GREATEST(LEAST(y2, $y_max), $y_min) AS y2,
        GREATEST(LEAST(x3, $x_max), $x_min) AS x3,
        GREATEST(LEAST(y3, $y_max), $y_min) AS y3,
        GREATEST(LEAST(x4, $x_max), $x_min) AS x4,
        GREATEST(LEAST(y4, $y_max), $y_min) AS y4,
        component_function_id
    FROM FinitePolytope
    WHERE
        (x1 >= $x_min AND x1 <= $x_max AND y1 >= $y_min AND y1 <= $y_max)
        OR (x2 >= $x_min AND x2 <= $x_max AND y2 >= $y_min AND y2 <= $y_max)
        OR (x3 >= $x_min AND x3 <= $x_max AND y3 >= $y_min AND y3 <= $y_max)
        OR (x4 >= $x_min AND x4 <= $x_max AND y4 >= $y_min AND y4 <= $y_max)
        OR (
            $x_min >= LEAST(x1, x2, x3, x4)
            AND $x_min <= GREATEST(x1, x2, x3, x4)
            AND
            (
                (
                    $y_min >= y1 + (y3 - y1) * ($x_min - x1) / (x3 - x1)
                    AND $y_min <= y2 + (y4 - y2) * ($x_min - x2) / (x4 - x2)
                )
                OR
                (
                    $y_max >= y1 + (y3 - y1) * ($x_min - x1) / (x3 - x1)
                    AND $y_max <= y2 + (y4 - y2) * ($x_min - x2) / (x4 - x2)
                )
            )
        )
        OR (
            $x_max >= LEAST(x1, x2, x3, x4)
            AND $x_max <= GREATEST(x1, x2, x3, x4)
            AND
            (
                (
                    $y_min >= y1 + (y3 - y1) * ($x_max - x1) / (x3 - x1)
                    AND $y_min <= y2 + (y4 - y2) * ($x_max - x2) / (x4 - x2)
                )
                OR
                (
                    $y_max >= y1 + (y3 - y1) * ($x_max - x1) / (x3 - x1)
                    AND $y_max <= y2 + (y4 - y2) * ($x_max - x2) / (x4 - x2)
                )
            )
        )
),
-- See geomcad.sql
infinite_vertices AS (
    WITH matching_dim1_cells AS (
        SELECT
            id,
            GREATEST(a0_lower, $x_min) AS x_left,
            LEAST(a0_upper, $x_max) AS x_right,
        FROM Cell_Dim1
        WHERE
            (a0_lower >= $x_min AND a0_lower <= $x_max)
            OR (a0_upper >= $x_min AND a0_upper <= $x_max)
            OR (
                $x_min >= a0_lower AND $x_min <= a0_upper
                AND $x_max >= a0_lower AND $x_max <= a0_upper
            )
    ),
    matching_dim2_cells AS (
        SELECT
            d1.x_left,
            d1.x_right,
            d2.a0_lower + d2.a1_lower * d1.x_left AS y_bottom_left,
            d2.a0_lower + d2.a1_lower * d1.x_right AS y_bottom_right,
            d2.a0_upper + d2.a1_upper * d1.x_left AS y_top_left,
            d2.a0_upper + d2.a1_upper * d1.x_right AS y_top_right,
            d2.component_function_id
        FROM Cell_Dim2 d2
        JOIN matching_dim1_cells d1 ON d2.parent_cell_id = d1.id
        WHERE
            (y_bottom_left >= $y_min AND y_bottom_left <= $y_max)
            OR (y_bottom_right >= $y_min AND y_bottom_right <= $y_max)
            OR (y_top_left >= $y_min AND y_top_left <= $y_max)
            OR (y_top_right >= $y_min AND y_top_right <= $y_max)
            OR ($y_min >= y_bottom_left AND $y_min <= y_top_left)
            OR ($y_max >= y_bottom_left AND $y_max <= y_top_left)
            OR ($y_min >= y_bottom_right AND $y_min <= y_top_right)
            OR ($y_max >= y_bottom_right AND $y_max <= y_top_right)
    )
    SELECT
        x_left AS x1,
        GREATEST(y_bottom_left, $y_min) AS y1,
        x_left AS x2,
        LEAST(y_top_left, $y_max) AS y2,
        x_right AS x3,
        GREATEST(y_bottom_right, $y_min) AS y3,
        x_right AS x4,
        LEAST(y_top_right, $y_max) AS y4,
        component_function_id
    FROM matching_dim2_cells
),
all_vertices AS (
    SELECT x1, y1, x2, y2, x3, y3, x4, y4, component_function_id
    FROM finite_vertices

    UNION

    SELECT x1, y1, x2, y2, x3, y3, x4, y4, component_function_id
    FROM infinite_vertices
),
evaluated_vertices AS (
    SELECT
        a0 + a1*x1 + a2*y1 AS v1,
        a0 + a1*x2 + a2*y2 AS v2,
        a0 + a1*x3 + a2*y3 AS v3,
        a0 + a1*x4 + a2*y4 AS v4
    FROM all_vertices
    JOIN ComponentFunction
        ON all_vertices.component_function_id = ComponentFunction.id
)
SELECT NOT EXISTS
(
    SELECT 1
    FROM evaluated_vertices
    WHERE
        v1 < $f_min OR v1 > $f_max
        OR v2 < $f_min OR v2 > $f_max
        OR v3 < $f_min OR v3 > $f_max
        OR v4 < $f_min OR v4 > $f_max
) AS f_in_bounds
