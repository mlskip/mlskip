-- duckdb db/vertices.db < <(cat queries/bounded.sql | sed 's/$x_min/20/g; s/$x_max/80/g; s/$y_min/20/g; s/$y_max/80/g; s/$f_min/0/g; s/$f_max/100/g')

WITH polytopes_in_bounds AS (
    SELECT
        -- Correct for polytopes that are not fully within the bounds.
        GREATEST(LEAST(x1, $x_max), $x_min) AS x1,
        GREATEST(LEAST(y1, $y_max), $y_min) AS y1,
        GREATEST(LEAST(x2, $x_max), $x_min) AS x2,
        GREATEST(LEAST(y2, $y_max), $y_min) AS y2,
        GREATEST(LEAST(x3, $x_max), $x_min) AS x3,
        GREATEST(LEAST(y3, $y_max), $y_min) AS y3,
        GREATEST(LEAST(x4, $x_max), $x_min) AS x4,
        GREATEST(LEAST(y4, $y_max), $y_min) AS y4,
        a0, a1, a2
    FROM PWL
    WHERE
        -- We check if there is an intersection between the rectangle formed by
        -- the x- and y-bounds and our polytope.
        --
        -- Case 1: the rectangle contains one of our polytope's vertices.
        (x1 >= $x_min AND x1 <= $x_max AND y1 >= $y_min AND y1 <= $y_max)
        OR (x2 >= $x_min AND x2 <= $x_max AND y2 >= $y_min AND y2 <= $y_max)
        OR (x3 >= $x_min AND x3 <= $x_max AND y3 >= $y_min AND y3 <= $y_max)
        OR (x4 >= $x_min AND x4 <= $x_max AND y4 >= $y_min AND y4 <= $y_max)

        -- Case 2: the polytope contains one of the rectangle's vertices.
        --
        -- This is harder. For x_min and x_max things are easy, we simply check
        -- if they fall within the smallest and largest x-coordinates of the
        -- vertices. We can do this because we know the left and right edges of
        -- the polytopes are vertical.
        --
        -- For y_min and y_max, we evaluate the linear equations (y=ax+b) of the
        -- lower and upper bounds for x_min (resp. x_max) and check if either
        -- y_min or y_max is contained within these bounds.
        OR (
            -- Check x_min
            $x_min >= LEAST(x1, x2, x3, x4)
            AND $x_min <= GREATEST(x1, x2, x3, x4)
            -- Now check if either y_min or y_max is within the bounds formed by x_min.
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
            -- Check x_max
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
evaluated_polytopes AS (
    SELECT
        a0 + a1*x1 + a2*y1 AS v1,
        a0 + a1*x2 + a2*y2 AS v2,
        a0 + a1*x3 + a2*y3 AS v3,
        a0 + a1*x4 + a2*y4 AS v4
    FROM polytopes_in_bounds
)
SELECT NOT EXISTS
(
    SELECT 1
    FROM evaluated_polytopes
    WHERE
        v1 < $f_min OR v1 > $f_max
        OR v2 < $f_min OR v2 > $f_max
        OR v3 < $f_min OR v3 > $f_max
        OR v4 < $f_min OR v4 > $f_max
) AS f_in_bounds
