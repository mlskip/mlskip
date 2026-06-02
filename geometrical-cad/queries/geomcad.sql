-- duckdb db/geomcad.db < <(cat queries/geomcad.sql | sed 's/$x_min/20/g; s/$x_max/80/g; s/$y_min/20/g; s/$y_max/80/g; s/$f_min/0/g; s/$f_max/100/g')

WITH matching_dim1_cells AS (
    SELECT
        id,
        GREATEST(a0_lower, $x_min) AS x_left,
        LEAST(a0_upper, $x_max) AS x_right,
    FROM Cell_Dim1
    WHERE a0_upper >= $x_min AND a0_lower <= $x_max
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
    WHERE GREATEST(y_top_left, y_top_right) >= $y_min
        AND LEAST(y_bottom_left, y_bottom_right) <= $y_max
),
vertices_in_bounds AS (
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
evaluated_polytopes AS (
    SELECT
        a0 + a1*x1 + a2*y1 AS v1,
        a0 + a1*x2 + a2*y2 AS v2,
        a0 + a1*x3 + a2*y3 AS v3,
        a0 + a1*x4 + a2*y4 AS v4
    FROM vertices_in_bounds
    JOIN ComponentFunction
        ON vertices_in_bounds.component_function_id = ComponentFunction.id
)
SELECT EXISTS (
    SELECT 1 FROM evaluated_polytopes
    WHERE GREATEST(v1, v2, v3, v4) >= $f_min AND LEAST(v1, v2, v3, v4) <= $f_max
) AS f_reachable
