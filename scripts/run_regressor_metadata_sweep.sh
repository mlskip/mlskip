#!/usr/bin/env bash
set -euo pipefail

DATABASES="tpch,tpcds"
BLOCK_SIZES="1000"
MAX_ROWS_TOTALS="100000"
METADATA_TYPES="minmax"
MODEL_KINDS="shallow"
JOBS="20"
GRID_DEPTH="4"
VERIFIER_BACKEND="marabou"
RANGE_ALPHA="2"
RANGE_START_SAMPLES="10"
RANGE_SEED="0"
DRY_RUN="0"
BATCHED_GEOMCAD="0"

usage() {
  cat <<'EOF'
Usage: scripts/run_regressor_metadata_sweep.sh [options]

Run `bench.py` for the regressor task across the cross product of:
- databases
- block sizes
- max-rows totals
- metadata types
- model kinds

Options:
  --databases LIST           Comma-separated databases. Default: tpch,tpcds
  --block-sizes LIST         Comma-separated block sizes. Default: 1000
  --max-rows-totals LIST     Comma-separated row budgets. Default: 50000
  --metadata-types LIST      Comma-separated metadata kinds.
                             Default: minmax,convex_hull,grid,bounded_convex_hull
  --model-kinds LIST         Comma-separated model kinds. Default: shallow,deep for
                             marabou and shallow for geomcad
  --jobs N                   Parallel jobs for bench.py. Default: 20
  --grid-depth N             Grid depth for grid/bounded_convex_hull. Default: 4
  --verifier-backend NAME    Verifier backend. Default: geomcad
  --range-alpha VALUE        Range alpha for generated regressor filters. Default: 2
  --range-start-samples N    Start samples for generated regressor filters. Default: 10
  --range-seed N             Range seed for generated regressor filters. Default: 0
  --batched-geomcad          Add --batched-geomcad to bench.py invocations
  --dry-run                  Print commands without running them
  --help                     Show this help
EOF
}

split_csv() {
  local raw="$1"
  local -n out_ref=$2
  IFS=',' read -r -a out_ref <<<"$raw"
  for i in "${!out_ref[@]}"; do
    out_ref[$i]="$(echo "${out_ref[$i]}" | xargs)"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --databases)
      DATABASES="$2"
      shift 2
      ;;
    --block-sizes)
      BLOCK_SIZES="$2"
      shift 2
      ;;
    --max-rows-totals)
      MAX_ROWS_TOTALS="$2"
      shift 2
      ;;
    --metadata-types)
      METADATA_TYPES="$2"
      shift 2
      ;;
    --model-kinds)
      MODEL_KINDS="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --grid-depth)
      GRID_DEPTH="$2"
      shift 2
      ;;
    --verifier-backend)
      VERIFIER_BACKEND="$2"
      if [[ "$VERIFIER_BACKEND" == "batched-geomcad" ]]; then
        VERIFIER_BACKEND="geomcad"
        BATCHED_GEOMCAD="1"
      fi
      shift 2
      ;;
    --range-alpha)
      RANGE_ALPHA="$2"
      shift 2
      ;;
    --range-start-samples)
      RANGE_START_SAMPLES="$2"
      shift 2
      ;;
    --range-seed)
      RANGE_SEED="$2"
      shift 2
      ;;
    --batched-geomcad)
      BATCHED_GEOMCAD="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

split_csv "$DATABASES" DATABASE_LIST
split_csv "$BLOCK_SIZES" BLOCK_SIZE_LIST
split_csv "$MAX_ROWS_TOTALS" MAX_ROWS_LIST
split_csv "$METADATA_TYPES" METADATA_LIST

if [[ -n "$MODEL_KINDS" ]]; then
  split_csv "$MODEL_KINDS" MODEL_KIND_LIST
else
  case "$VERIFIER_BACKEND" in
    marabou)
      MODEL_KIND_LIST=(shallow deep)
      ;;
    geomcad)
      MODEL_KIND_LIST=(shallow)
      ;;
    *)
      echo "Unsupported verifier backend for this sweep: $VERIFIER_BACKEND" >&2
      exit 1
      ;;
  esac
fi

for model_kind in "${MODEL_KIND_LIST[@]}"; do
  if [[ "$VERIFIER_BACKEND" == "geomcad" && "$model_kind" != "shallow" ]]; then
    echo "[sweep] Skipping unsupported combination verifier_backend=$VERIFIER_BACKEND model_kind=$model_kind"
    continue
  fi

  for database in "${DATABASE_LIST[@]}"; do
    for block_size in "${BLOCK_SIZE_LIST[@]}"; do
      for max_rows_total in "${MAX_ROWS_LIST[@]}"; do
        for metadata_type in "${METADATA_LIST[@]}"; do
          cmd=(
            uv run python bench.py
            --database "$database"
            --model-kind "$model_kind"
            --block-size "$block_size"
            --max-rows-total "$max_rows_total"
            --task-type regressor
            --jobs "$JOBS"
            --verifier-backend "$VERIFIER_BACKEND"
            --block-metadata "$metadata_type"
            --range-alpha "$RANGE_ALPHA"
            --range-start-samples "$RANGE_START_SAMPLES"
            --range-seed "$RANGE_SEED"
          )
          if [[ "$metadata_type" == "grid" || "$metadata_type" == "bounded_convex_hull" ]]; then
            cmd+=(--grid-depth "$GRID_DEPTH")
          fi
          if [[ "$BATCHED_GEOMCAD" == "1" ]]; then
            cmd+=(--batched-geomcad)
          fi

          printf '[sweep] '
          printf '%q ' "${cmd[@]}"
          printf '\n'

          if [[ "$DRY_RUN" != "1" ]]; then
            "${cmd[@]}"
          fi
        done
      done
    done
  done
done
