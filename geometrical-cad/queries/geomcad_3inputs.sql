-- duckdb db/3inputs.db < <(cat queries/geomcad_3inputs.sql | sed 's/$x1_min/20/g; s/$x1_max/80/g; s/$x2_min/20/g; s/$x2_max/80/g; s/$x3_min/20/g; s/$x3_max/80/g; s/$f_min/0/g; s/$f_max/100/g')

WITH matching_dim1_cells AS (
    SELECT
        id,
        GREATEST(a0_lower, $x1_min) AS x1_left,
        LEAST(a0_upper, $x1_max) AS x1_right
    FROM Cell_Dim1
    WHERE a0_upper >= $x1_min AND a0_lower <= $x1_max
),
matching_dim2_cells AS (
    SELECT
        d2.id AS dim2_id,
        d1.x1_left,
        d1.x1_right,
        d2.a0_lower + d2.a1_lower * d1.x1_left AS x2_bottom_left,
        d2.a0_lower + d2.a1_lower * d1.x1_right AS x2_bottom_right,
        d2.a0_upper + d2.a1_upper * d1.x1_left AS x2_top_left,
        d2.a0_upper + d2.a1_upper * d1.x1_right AS x2_top_right
    FROM Cell_Dim2 d2
    JOIN matching_dim1_cells d1 ON d2.parent_cell_id = d1.id
    WHERE GREATEST(x2_top_left, x2_top_right) >= $x2_min
        AND LEAST(x2_bottom_left, x2_bottom_right) <= $x2_max
),
matching_dim2_cells_bounded AS (
    SELECT
        dim2_id,
        x1_left,
        x1_right,
        GREATEST(x2_bottom_left, $x2_min) AS x2_bottom_left,
        LEAST(x2_top_left, $x2_max) AS x2_top_left,
        GREATEST(x2_bottom_right, $x2_min) AS x2_bottom_right,
        LEAST(x2_top_right, $x2_max) AS x2_top_right,
    FROM matching_dim2_cells
),
matching_dim3_cells AS (
    SELECT
        d2.x1_left,
        d2.x1_right,
        d2.x2_bottom_left,
        d2.x2_bottom_right,
        d2.x2_top_left,
        d2.x2_top_right,
        d3.a0_lower + d3.a1_lower * d2.x1_left + d3.a2_lower * d2.x2_bottom_left AS x3_lower_bottom_left,
        d3.a0_lower + d3.a1_lower * d2.x1_left + d3.a2_lower * d2.x2_top_left AS x3_lower_top_left,
        d3.a0_lower + d3.a1_lower * d2.x1_right + d3.a2_lower * d2.x2_bottom_right AS x3_lower_bottom_right,
        d3.a0_lower + d3.a1_lower * d2.x1_right + d3.a2_lower * d2.x2_top_right AS x3_lower_top_right,
        d3.a0_upper + d3.a1_upper * d2.x1_left + d3.a2_upper * d2.x2_bottom_left AS x3_upper_bottom_left,
        d3.a0_upper + d3.a1_upper * d2.x1_left + d3.a2_upper * d2.x2_top_left AS x3_upper_top_left,
        d3.a0_upper + d3.a1_upper * d2.x1_right + d3.a2_upper * d2.x2_bottom_right AS x3_upper_bottom_right,
        d3.a0_upper + d3.a1_upper * d2.x1_right + d3.a2_upper * d2.x2_top_right AS x3_upper_top_right,
        d3.component_function_id
    FROM Cell_Dim3 d3
    JOIN matching_dim2_cells_bounded d2 ON d3.parent_cell_id = d2.dim2_id
    WHERE GREATEST(x3_upper_bottom_left, x3_upper_top_left, x3_upper_bottom_right, x3_upper_top_right) >= $x3_min
        AND LEAST(x3_lower_bottom_left, x3_lower_top_left, x3_lower_bottom_right, x3_lower_top_right) <= $x3_max
),
vertices_in_bounds AS (
    SELECT
        x1_left AS x1,
        GREATEST(x2_bottom_left, $x2_min) AS y1,
        GREATEST(LEAST(x3_lower_bottom_left, $x3_max), $x3_min) AS z1,
        x1_left AS x2,
        LEAST(x2_top_left, $x2_max) AS y2,
        GREATEST(LEAST(x3_lower_top_left, $x3_max), $x3_min) AS z2,
        x1_right AS x3,
        GREATEST(x2_bottom_right, $x2_min) AS y3,
        GREATEST(LEAST(x3_lower_bottom_right, $x3_max), $x3_min) AS z3,
        x1_right AS x4,
        LEAST(x2_top_right, $x2_max) AS y4,
        GREATEST(LEAST(x3_lower_top_right, $x3_max), $x3_min) AS z4,
        x1_left AS x5,
        GREATEST(x2_bottom_left, $x2_min) AS y5,
        GREATEST(LEAST(x3_upper_bottom_left, $x3_max), $x3_min) AS z5,
        x1_left AS x6,
        LEAST(x2_top_left, $x2_max) AS y6,
        GREATEST(LEAST(x3_upper_top_left, $x3_max), $x3_min) AS z6,
        x1_right AS x7,
        GREATEST(x2_bottom_right, $x2_min) AS y7,
        GREATEST(LEAST(x3_upper_bottom_right, $x3_max), $x3_min) AS z7,
        x1_right AS x8,
        LEAST(x2_top_right, $x2_max) AS y8,
        GREATEST(LEAST(x3_upper_top_right, $x3_max), $x3_min) AS z8,
        component_function_id
    FROM matching_dim3_cells
),
evaluated_polytopes AS (
    SELECT
        a0 + a1*x1 + a2*y1 + a3*z1 AS v1,
        a0 + a1*x2 + a2*y2 + a3*z2 AS v2,
        a0 + a1*x3 + a2*y3 + a3*z3 AS v3,
        a0 + a1*x4 + a2*y4 + a3*z4 AS v4,
        a0 + a1*x5 + a2*y5 + a3*z5 AS v5,
        a0 + a1*x6 + a2*y6 + a3*z6 AS v6,
        a0 + a1*x7 + a2*y7 + a3*z7 AS v7,
        a0 + a1*x8 + a2*y8 + a3*z8 AS v8
    FROM vertices_in_bounds
    JOIN ComponentFunction
        ON vertices_in_bounds.component_function_id = ComponentFunction.id
)
SELECT EXISTS (
    SELECT 1 FROM evaluated_polytopes
    WHERE GREATEST(v1, v2, v3, v4, v5, v6, v7, v8) >= $f_min
      AND LEAST(v1, v2, v3, v4, v5, v6, v7, v8) <= $f_max
) AS f_reachable
