#!/usr/bin/env bash
set -euo pipefail

DATABASES="tpcds"
MODEL_KINDS="deep"
BLOCK_SIZES="1000"
MAX_ROWS_TOTALS="100000"
METADATA_TYPES="baseline,minmax,bounded_convex_hull"
TASK_TYPE="regressor"
JOBS="20"
RANGE_ALPHA="2"
RANGE_START_SAMPLES="10"
RANGE_SEED="0"
VERIFIER_TIMEOUT_SECONDS="1.0"
GRID_DEPTH="4"
FILTERS_PATH=""
FILTER_NAME=""
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage: scripts/run_e2e_block_skipping_sweep.sh [options]

Run `bench.py` for end-to-end block-skipping with per-block PyTorch execution
across the cross product of:
- databases
- block sizes
- max-rows totals
- metadata types
- model kinds

Options:
  --databases LIST             Comma-separated databases. Default: tpch
  --database NAME              Alias for --databases with one value
  --model-kinds LIST           Comma-separated model kinds. Default: deep
  --model-kind NAME            Alias for --model-kinds with one value
  --block-sizes LIST           Comma-separated block sizes. Default: 1000
  --block-size N               Alias for --block-sizes with one value
  --max-rows-totals LIST       Comma-separated row budgets. Default: 100000
  --max-rows-total N           Alias for --max-rows-totals with one value
  --metadata-types LIST        Comma-separated metadata kinds. Use `baseline`
                               for no skipping. Default:
                               baseline,minmax,bounded_convex_hull
  --task-type NAME             Task type. Default: regressor
  --jobs N                     Parallel jobs for bench.py. Default: 20
  --range-alpha VALUE          Range alpha for generated regressor filters. Default: 2
  --range-start-samples N      Start samples for generated regressor filters. Default: 10
  --range-seed N               Range seed for generated regressor filters. Default: 0
  --verifier-timeout-seconds N Per-block Marabou timeout. Default: 1.0
  --grid-depth N               Grid depth for grid/bounded_convex_hull. Default: 4
  --filters-path PATH          Optional explicit filters JSON
  --filter NAME                Optional single filter name to pass through
  --dry-run                    Print commands without running them
  --help                       Show this help
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
    --database)
      DATABASES="$2"
      shift 2
      ;;
    --model-kinds)
      MODEL_KINDS="$2"
      shift 2
      ;;
    --model-kind)
      MODEL_KINDS="$2"
      shift 2
      ;;
    --block-sizes)
      BLOCK_SIZES="$2"
      shift 2
      ;;
    --block-size)
      BLOCK_SIZES="$2"
      shift 2
      ;;
    --max-rows-totals)
      MAX_ROWS_TOTALS="$2"
      shift 2
      ;;
    --max-rows-total)
      MAX_ROWS_TOTALS="$2"
      shift 2
      ;;
    --metadata-types)
      METADATA_TYPES="$2"
      shift 2
      ;;
    --task-type)
      TASK_TYPE="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
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
    --verifier-timeout-seconds)
      VERIFIER_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --grid-depth)
      GRID_DEPTH="$2"
      shift 2
      ;;
    --filters-path)
      FILTERS_PATH="$2"
      shift 2
      ;;
    --filter)
      FILTER_NAME="$2"
      shift 2
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
split_csv "$MODEL_KINDS" MODEL_KIND_LIST
split_csv "$BLOCK_SIZES" BLOCK_SIZE_LIST
split_csv "$MAX_ROWS_TOTALS" MAX_ROWS_LIST
split_csv "$METADATA_TYPES" METADATA_LIST

for database in "${DATABASE_LIST[@]}"; do
  for model_kind in "${MODEL_KIND_LIST[@]}"; do
    for block_size in "${BLOCK_SIZE_LIST[@]}"; do
      for max_rows_total in "${MAX_ROWS_LIST[@]}"; do
        for metadata_type in "${METADATA_LIST[@]}"; do
          cmd=(
            uv run python bench.py
            --database "$database"
            --model-kind "$model_kind"
            --block-size "$block_size"
            --max-rows-total "$max_rows_total"
            --task-type "$TASK_TYPE"
            --jobs "$JOBS"
            --measure-e2e
            --verifier-timeout-seconds "$VERIFIER_TIMEOUT_SECONDS"
          )

          if [[ -n "$FILTERS_PATH" ]]; then
            cmd+=(--filters-path "$FILTERS_PATH")
          fi
          if [[ -n "$FILTER_NAME" ]]; then
            cmd+=(--filter "$FILTER_NAME")
          fi
          if [[ "$TASK_TYPE" == "regressor" ]]; then
            cmd+=(
              --range-alpha "$RANGE_ALPHA"
              --range-start-samples "$RANGE_START_SAMPLES"
              --range-seed "$RANGE_SEED"
            )
          fi

          if [[ "$metadata_type" == "baseline" ]]; then
            cmd+=(--verifier-backend marabou --block-metadata minmax --disable-skipping)
          else
            cmd+=(--verifier-backend marabou --block-metadata "$metadata_type")
            if [[ "$metadata_type" == "bounded_convex_hull" || "$metadata_type" == "grid" ]]; then
              cmd+=(--grid-depth "$GRID_DEPTH")
            fi
          fi

          printf '[e2e-sweep] '
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
