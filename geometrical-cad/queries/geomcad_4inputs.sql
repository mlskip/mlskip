-- duckdb db/4inputs.db < <(cat queries/geomcad_4inputs.sql | sed 's/$x1_min/20/g; s/$x1_max/80/g; s/$x2_min/20/g; s/$x2_max/80/g; s/$x3_min/20/g; s/$x3_max/80/g; s/$x4_min/20/g; s/$x4_max/80/g; s/$f_min/0/g; s/$f_max/100/g')

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
        d3.id AS dim3_id,
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
        d3.a0_upper + d3.a1_upper * d2.x1_right + d3.a2_upper * d2.x2_top_right AS x3_upper_top_right
    FROM Cell_Dim3 d3
    JOIN matching_dim2_cells_bounded d2 ON d3.parent_cell_id = d2.dim2_id
    WHERE GREATEST(x3_upper_bottom_left, x3_upper_top_left, x3_upper_bottom_right, x3_upper_top_right) >= $x3_min
        AND LEAST(x3_lower_bottom_left, x3_lower_top_left, x3_lower_bottom_right, x3_lower_top_right) <= $x3_max
),
matching_dim3_cells_bounded AS (
    SELECT
        dim3_id,
        d2.x1_left,
        d2.x1_right,
        d2.x2_bottom_left,
        d2.x2_bottom_right,
        d2.x2_top_left,
        d2.x2_top_right,
        GREATEST(x2_bottom_left, $x2_min) AS x2_bottom_left,
        GREATEST(LEAST(x3_lower_bottom_left, $x3_max), $x3_min) AS x3_lower_bottom_left,
        LEAST(x2_top_left, $x2_max) AS x2_top_left,
        GREATEST(LEAST(x3_lower_top_left, $x3_max), $x3_min) AS x3_lower_top_left,
        GREATEST(x2_bottom_right, $x2_min) AS x2_bottom_right,
        GREATEST(LEAST(x3_lower_bottom_right, $x3_max), $x3_min) AS x3_lower_bottom_right,
        LEAST(x2_top_right, $x2_max) AS x2_top_right,
        GREATEST(LEAST(x3_lower_top_right, $x3_max), $x3_min) AS x3_lower_top_right,
        GREATEST(x2_bottom_left, $x2_min) AS x2_bottom_left,
        GREATEST(LEAST(x3_upper_bottom_left, $x3_max), $x3_min) AS x3_upper_bottom_left,
        LEAST(x2_top_left, $x2_max) AS x2_top_left,
        GREATEST(LEAST(x3_upper_top_left, $x3_max), $x3_min) AS x3_upper_top_left,
        GREATEST(x2_bottom_right, $x2_min) AS x2_bottom_right,
        GREATEST(LEAST(x3_upper_bottom_right, $x3_max), $x3_min) AS x3_upper_bottom_right,
        LEAST(x2_top_right, $x2_max) AS x2_top_right,
        GREATEST(LEAST(x3_upper_top_right, $x3_max), $x3_min) AS x3_upper_top_right
    FROM matching_dim3_cells
),
matching_dim4_cells AS (
    SELECT
        d3.x1_left,
        d3.x1_right,
        d3.x2_bottom_left,
        d3.x2_bottom_right,
        d3.x2_top_left,
        d3.x2_top_right,
        d3.x3_lower_bottom_left,
        d3.x3_lower_top_left,
        d3.x3_lower_bottom_right,
        d3.x3_lower_top_right,
        d3.x3_upper_bottom_left,
        d3.x3_upper_top_left,
        d3.x3_upper_bottom_right,
        d3.x3_upper_top_right,
        d4.a0_lower + d4.a1_lower * d3.x1_left + d4.a2_lower * d3.x2_bottom_left + d4.a3_lower * d3.x3_lower_bottom_left AS x4_lower_lb_lower,
        d4.a0_lower + d4.a1_lower * d3.x1_left + d4.a2_lower * d3.x2_bottom_left + d4.a3_lower * d3.x3_upper_bottom_left AS x4_lower_lb_upper,
        d4.a0_lower + d4.a1_lower * d3.x1_left + d4.a2_lower * d3.x2_top_left + d4.a3_lower * d3.x3_lower_top_left AS x4_lower_lt_lower,
        d4.a0_lower + d4.a1_lower * d3.x1_left + d4.a2_lower * d3.x2_top_left + d4.a3_lower * d3.x3_upper_top_left AS x4_lower_lt_upper,
        d4.a0_lower + d4.a1_lower * d3.x1_right + d4.a2_lower * d3.x2_bottom_right + d4.a3_lower * d3.x3_lower_bottom_right AS x4_lower_rb_lower,
        d4.a0_lower + d4.a1_lower * d3.x1_right + d4.a2_lower * d3.x2_bottom_right + d4.a3_lower * d3.x3_upper_bottom_right AS x4_lower_rb_upper,
        d4.a0_lower + d4.a1_lower * d3.x1_right + d4.a2_lower * d3.x2_top_right + d4.a3_lower * d3.x3_lower_top_right AS x4_lower_rt_lower,
        d4.a0_lower + d4.a1_lower * d3.x1_right + d4.a2_lower * d3.x2_top_right + d4.a3_lower * d3.x3_upper_top_right AS x4_lower_rt_upper,
        d4.a0_upper + d4.a1_upper * d3.x1_left + d4.a2_upper * d3.x2_bottom_left + d4.a3_upper * d3.x3_lower_bottom_left AS x4_upper_lb_lower,
        d4.a0_upper + d4.a1_upper * d3.x1_left + d4.a2_upper * d3.x2_bottom_left + d4.a3_upper * d3.x3_upper_bottom_left AS x4_upper_lb_upper,
        d4.a0_upper + d4.a1_upper * d3.x1_left + d4.a2_upper * d3.x2_top_left + d4.a3_upper * d3.x3_lower_top_left AS x4_upper_lt_lower,
        d4.a0_upper + d4.a1_upper * d3.x1_left + d4.a2_upper * d3.x2_top_left + d4.a3_upper * d3.x3_upper_top_left AS x4_upper_lt_upper,
        d4.a0_upper + d4.a1_upper * d3.x1_right + d4.a2_upper * d3.x2_bottom_right + d4.a3_upper * d3.x3_lower_bottom_right AS x4_upper_rb_lower,
        d4.a0_upper + d4.a1_upper * d3.x1_right + d4.a2_upper * d3.x2_bottom_right + d4.a3_upper * d3.x3_upper_bottom_right AS x4_upper_rb_upper,
        d4.a0_upper + d4.a1_upper * d3.x1_right + d4.a2_upper * d3.x2_top_right + d4.a3_upper * d3.x3_lower_top_right AS x4_upper_rt_lower,
        d4.a0_upper + d4.a1_upper * d3.x1_right + d4.a2_upper * d3.x2_top_right + d4.a3_upper * d3.x3_upper_top_right AS x4_upper_rt_upper,
        d4.component_function_id
    FROM Cell_Dim4 d4
    JOIN matching_dim3_cells d3 ON d4.parent_cell_id = d3.dim3_id
    WHERE
        GREATEST(
            x4_upper_lb_lower, x4_upper_lb_upper,
            x4_upper_lt_lower, x4_upper_lt_upper,
            x4_upper_rb_lower, x4_upper_rb_upper,
            x4_upper_rt_lower, x4_upper_rt_upper
        ) >= $x4_min
    AND LEAST(
            x4_lower_lb_lower, x4_lower_lb_upper,
            x4_lower_lt_lower, x4_lower_lt_upper,
            x4_lower_rb_lower, x4_lower_rb_upper,
            x4_lower_rt_lower, x4_lower_rt_upper
        ) <= $x4_max
),
vertices_in_bounds AS (
    SELECT
        x1_left AS x1,
        GREATEST(x2_bottom_left, $x2_min) AS y1,
        GREATEST(LEAST(x3_lower_bottom_left, $x3_max), $x3_min) AS z1,
        GREATEST(LEAST(x4_lower_lb_lower, $x4_max), $x4_min) AS w1,
        x1_left AS x2,
        GREATEST(x2_bottom_left, $x2_min) AS y2,
        GREATEST(LEAST(x3_upper_bottom_left, $x3_max), $x3_min) AS z2,
        GREATEST(LEAST(x4_lower_lb_upper, $x4_max), $x4_min) AS w2,
        x1_left AS x3,
        LEAST(x2_top_left, $x2_max) AS y3,
        GREATEST(LEAST(x3_lower_top_left, $x3_max), $x3_min) AS z3,
        GREATEST(LEAST(x4_lower_lt_lower, $x4_max), $x4_min) AS w3,
        x1_left AS x4,
        LEAST(x2_top_left, $x2_max) AS y4,
        GREATEST(LEAST(x3_upper_top_left, $x3_max), $x3_min) AS z4,
        GREATEST(LEAST(x4_lower_lt_upper, $x4_max), $x4_min) AS w4,
        x1_right AS x5,
        GREATEST(x2_bottom_right, $x2_min) AS y5,
        GREATEST(LEAST(x3_lower_bottom_right, $x3_max), $x3_min) AS z5,
        GREATEST(LEAST(x4_lower_rb_lower, $x4_max), $x4_min) AS w5,
        x1_right AS x6,
        GREATEST(x2_bottom_right, $x2_min) AS y6,
        GREATEST(LEAST(x3_upper_bottom_right, $x3_max), $x3_min) AS z6,
        GREATEST(LEAST(x4_lower_rb_upper, $x4_max), $x4_min) AS w6,
        x1_right AS x7,
        LEAST(x2_top_right, $x2_max) AS y7,
        GREATEST(LEAST(x3_lower_top_right, $x3_max), $x3_min) AS z7,
        GREATEST(LEAST(x4_lower_rt_lower, $x4_max), $x4_min) AS w7,
        x1_right AS x8,
        LEAST(x2_top_right, $x2_max) AS y8,
        GREATEST(LEAST(x3_upper_top_right, $x3_max), $x3_min) AS z8,
        GREATEST(LEAST(x4_lower_rt_upper, $x4_max), $x4_min) AS w8,
        x1_left AS x9,
        GREATEST(x2_bottom_left, $x2_min) AS y9,
        GREATEST(LEAST(x3_lower_bottom_left, $x3_max), $x3_min) AS z9,
        GREATEST(LEAST(x4_upper_lb_lower, $x4_max), $x4_min) AS w9,
        x1_left AS x10,
        GREATEST(x2_bottom_left, $x2_min) AS y10,
        GREATEST(LEAST(x3_upper_bottom_left, $x3_max), $x3_min) AS z10,
        GREATEST(LEAST(x4_upper_lb_upper, $x4_max), $x4_min) AS w10,
        x1_left AS x11,
        LEAST(x2_top_left, $x2_max) AS y11,
        GREATEST(LEAST(x3_lower_top_left, $x3_max), $x3_min) AS z11,
        GREATEST(LEAST(x4_upper_lt_lower, $x4_max), $x4_min) AS w11,
        x1_left AS x12,
        LEAST(x2_top_left, $x2_max) AS y12,
        GREATEST(LEAST(x3_upper_top_left, $x3_max), $x3_min) AS z12,
        GREATEST(LEAST(x4_upper_lt_upper, $x4_max), $x4_min) AS w12,
        x1_right AS x13,
        GREATEST(x2_bottom_right, $x2_min) AS y13,
        GREATEST(LEAST(x3_lower_bottom_right, $x3_max), $x3_min) AS z13,
        GREATEST(LEAST(x4_upper_rb_lower, $x4_max), $x4_min) AS w13,
        x1_right AS x14,
        GREATEST(x2_bottom_right, $x2_min) AS y14,
        GREATEST(LEAST(x3_upper_bottom_right, $x3_max), $x3_min) AS z14,
        GREATEST(LEAST(x4_upper_rb_upper, $x4_max), $x4_min) AS w14,
        x1_right AS x15,
        LEAST(x2_top_right, $x2_max) AS y15,
        GREATEST(LEAST(x3_lower_top_right, $x3_max), $x3_min) AS z15,
        GREATEST(LEAST(x4_upper_rt_lower, $x4_max), $x4_min) AS w15,
        x1_right AS x16,
        LEAST(x2_top_right, $x2_max) AS y16,
        GREATEST(LEAST(x3_upper_top_right, $x3_max), $x3_min) AS z16,
        GREATEST(LEAST(x4_upper_rt_upper, $x4_max), $x4_min) AS w16,
        component_function_id
    FROM matching_dim4_cells
),
evaluated_polytopes AS (
    SELECT
        a0 + a1*x1 + a2*y1 + a3*z1 + a4*w1 AS v1,
        a0 + a1*x2 + a2*y2 + a3*z2 + a4*w2 AS v2,
        a0 + a1*x3 + a2*y3 + a3*z3 + a4*w3 AS v3,
        a0 + a1*x4 + a2*y4 + a3*z4 + a4*w4 AS v4,
        a0 + a1*x5 + a2*y5 + a3*z5 + a4*w5 AS v5,
        a0 + a1*x6 + a2*y6 + a3*z6 + a4*w6 AS v6,
        a0 + a1*x7 + a2*y7 + a3*z7 + a4*w7 AS v7,
        a0 + a1*x8 + a2*y8 + a3*z8 + a4*w8 AS v8,
        a0 + a1*x9 + a2*y9 + a3*z9 + a4*w9 AS v9,
        a0 + a1*x10 + a2*y10 + a3*z10 + a4*w10 AS v10,
        a0 + a1*x11 + a2*y11 + a3*z11 + a4*w11 AS v11,
        a0 + a1*x12 + a2*y12 + a3*z12 + a4*w12 AS v12,
        a0 + a1*x13 + a2*y13 + a3*z13 + a4*w13 AS v13,
        a0 + a1*x14 + a2*y14 + a3*z14 + a4*w14 AS v14,
        a0 + a1*x15 + a2*y15 + a3*z15 + a4*w15 AS v15,
        a0 + a1*x16 + a2*y16 + a3*z16 + a4*w16 AS v16
    FROM vertices_in_bounds
    JOIN ComponentFunction
        ON vertices_in_bounds.component_function_id = ComponentFunction.id
)
SELECT EXISTS
(
    SELECT 1
    FROM evaluated_polytopes
    WHERE GREATEST(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16) >= $f_min
      AND LEAST(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16) <= $f_max
) AS f_reachable
