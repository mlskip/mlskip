# ⏩️ MLSkip

MLSkip uses neural network verification tools to prune Parquet row groups when querying with ML/AI table predicates, using just lightweight metadata such as min-max statistics.

## Setup

The project is set up using `uv`. To fetch all dependencies:

```sh
uv sync
```

Requires python 3.11 (mainly because of Marabou)

## Structure

- `nnv-tools`: code related to using NNV tools within DuckDB
- `nnv_tools`: modular multi-database training and Marabou verification pipeline
- `metadata/setup/<database>.json`: how a database is set up and where its DuckDB cache lives
- `metadata/functions/<database>/`: model configs grouped by database and optionally by table
- `metadata/filters/<database>/`: benchmark predicates that reference trained models
- `metadata/models/<database>/`: exported ONNX models, summaries, and metadata for reuse
- `data/<database>/`: raw table files such as CSVs when a dataset is file-backed

## Dataset Layout

The repo is now organized so databases can be added independently, for example:

- `metadata/setup/tpch.json`
- `metadata/functions/tpch/lineitem.json`
- `metadata/filters/tpch/lineitem.json`
- `metadata/models/tpch/lineitem/...`
- `data/tpch/`
- `data/tpcds/`

DuckDB cache files are stored separately under `.cache/duckdb/` and ignored by git.

## 1. Data Setup

The data setup step should be done once per database. It prepares CSV files under
`data/<database>/` and loads them into the ignored DuckDB cache.

For TPCH:

```sh
bash scripts/setup_database.sh tpch
```

For TPC-DS:

```sh
bash scripts/setup_database.sh tpcds
```

This step:

- generates CSV files under `data/<database>/`
- loads those CSV files into `.cache/duckdb/<database>_sf1.duckdb`
- adds a `row_id` column to each loaded table

You can rebuild the preprocessing outputs with:

```sh
bash scripts/setup_database.sh <database> --force-csv --force-duckdb
```

## 2. Models Setup

After preprocessing, train or reuse models with:

```sh
uv run python train.py --database tpch
```

This step:

- loads function definitions from `metadata/functions/<database>/`
- supports both `regressor` and `classifier` model definitions
- trains missing models and writes them to `metadata/models/<database>/<table>/<function>/`
- trains on the first `training_row_count` rows configured in `metadata/setup/<database>.json`
- reuses an existing model unless you pass `--force-retrain`

Some model definitions are deterministic functions of their inputs, while others are
proxy-label predictors that infer a target column from correlated features. The latter
are benchmarked as empirical predictors on held-out data, not as exact semantic
replacements for the original SQL expression.

For a single function:

```sh
uv run python train.py --database tpch --function discounted_price
```

## 3. Benchmarks

The benchmark step uses separate filter metadata from `metadata/filters/<database>/`.
Each filter points at a trained model and specifies either:

- a regressor predicate such as `lower <= M(x) <= upper`
- a classifier predicate such as `M(x) = class_id`

Run benchmarks with:

```sh
uv run python bench.py --database tpch --block-size 1000 --max-rows-total 50000
```

For a single filter:

```sh
uv run python bench.py --database tpch --block-size 1000 --max-rows-total 50000 --filter discounted_price_band
```

To run only one model family:

```sh
uv run python bench.py --database tpcds --block-size 1000 --max-rows-total 50000 --task-type regressor
uv run python bench.py --database tpcds --block-size 1000 --max-rows-total 50000 --task-type classifier
```

This step:

- runs the baseline SQL predicate directly
- runs the model inside DuckDB as a Python UDF
- excludes the training prefix defined in `metadata/setup/<database>.json` from evaluation
- derives block membership dynamically from per-row `row_id` metadata and the required `--block-size`
- collects the selected per-block metadata before the verifier pruning pass and logs that prepass
- optionally uses Marabou to prune blocks before running the UDF path
- compares model-assisted results against SQL ground truth

### GeomCAD Compilation

Train `shallow` or `deep` models explicitly with:

```sh
uv run python train.py --database tpch --model-kind shallow --force-retrain
uv run python train.py --database tpch --model-kind deep --force-retrain
```

Artifacts are stored under `metadata/models/<database>/<model-kind>/...`.

The GeomCAD verifier uses precompiled model databases stored under
`metadata/compiled-models/<database>/...`, with separate `shallow` and `deep`
subdirectories to mirror the trained model layout. GeomCAD compilation only
targets the `shallow` models. Compile all shallow regressor models for a
database with:

```sh
scripts/compile_geomcad_models.sh tpch
```

To compile exactly one model:

```sh
scripts/compile_geomcad_models.sh --model metadata/models/tpcds/shallow/regressor/catalog_sales/catalog_sales_net_paid/catalog_sales_net_paid.onnx
```

Use `--force` to rebuild existing compiled artifacts.

Once the compiled artifacts exist, run benchmarks with GeomCAD using:

```sh
uv run python bench.py --database tpcds --model-kind shallow --block-size 1000 --verifier-backend geomcad --task-type regressor
```

`bench.py` will stop immediately if a required compiled GeomCAD artifact is missing.

### Block Metadata API

`bench.py` can build different per-block metadata before verifier checks:

```sh
uv run python bench.py --database tpcds --block-size 1000 --max-rows-total 50000 --block-metadata minmax
uv run python bench.py --database tpcds --block-size 1000 --max-rows-total 50000 --block-metadata convex_hull
uv run python bench.py --database tpcds --block-size 1000 --max-rows-total 50000 --block-metadata grid --grid-depth 4
uv run python bench.py --database tpcds --block-size 1000 --max-rows-total 50000 --block-metadata bounded_convex_hull --grid-depth 4
```

Supported metadata kinds:

- `minmax`: the existing axis-aligned feature bounds.
- `convex_hull`: exact convex hull constraints for each 2D feature pair in a block.
- `grid`: occupied cells in a `2^grid_depth x 2^grid_depth` grid for each 2D feature pair.
- `bounded_convex_hull`: convex hull around the occupied grid-cell rectangles, using `--grid-depth`.

If `--block-metadata` is omitted, `bench.py` first looks for a filter-level default in the filter JSON:

```json
{
  "name": "discounted_price_band",
  "block_metadata": {
    "kind": "bounded_convex_hull",
    "grid_depth": 4
  }
}
```

If no filter-level default is present, it falls back to `minmax`. For filters with more than two model inputs, the framework builds all 2D feature pairs; for example, a 3D model gets `(x0, x1)`, `(x0, x2)`, and `(x1, x2)` metadata.

Benchmark result JSON records the selected `block_metadata`, `grid_depth`, `metadata_collection_ms`, and `metadata_pair_count` alongside the usual pruning counters.

For proxy-label models, benchmark results should be interpreted as predictive accuracy
against the held-out SQL labels rather than proof that the model encodes the underlying
rule exactly.

## Notes

By default `train.py` runs the `tpch` setup. You can choose another database later with:

```sh
uv run python train.py --database tpcds
```
