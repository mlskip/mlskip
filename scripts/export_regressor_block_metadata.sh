#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/lib/resolve_uv.sh
source "$SCRIPT_DIR/lib/resolve_uv.sh"

DATABASES="tpch,tpcds"
MODEL_KINDS="shallow,deep"
METADATA_TYPES="minmax,convex_hull,grid,bounded_convex_hull"
BLOCK_SIZE="1000"
MAX_ROWS_TOTAL="50000"
GRID_DEPTH="4"
EXPORT_ROOT="export"

usage() {
  cat <<'EOF'
Usage: scripts/export_regressor_block_metadata.sh [options]

Export per-block metadata for regressor models across:
- databases
- shallow/deep model kinds
- minmax, convex_hull, grid, and bounded_convex_hull metadata

Options:
  --databases LIST        Comma-separated databases. Default: tpch,tpcds
  --model-kinds LIST      Comma-separated model kinds. Default: shallow,deep
  --metadata-types LIST   Comma-separated metadata kinds.
                          Default: minmax,convex_hull,grid,bounded_convex_hull
  --block-size N          Block size passed to bench.py. Default: 1000
  --max-rows-total N      Row budget passed to bench.py. Default: 50000
  --grid-depth N          Grid depth for grid/bounded_convex_hull. Default: 4
  --export-root PATH      Export root directory. Default: export
  --help                  Show this help
EOF
}

split_csv() {
  local raw="$1"
  SPLIT_CSV_RESULT=()
  IFS=',' read -r -a SPLIT_CSV_RESULT <<<"$raw"
  for i in "${!SPLIT_CSV_RESULT[@]}"; do
    SPLIT_CSV_RESULT[$i]="$(echo "${SPLIT_CSV_RESULT[$i]}" | xargs)"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --databases)
      DATABASES="$2"
      shift 2
      ;;
    --model-kinds)
      MODEL_KINDS="$2"
      shift 2
      ;;
    --metadata-types)
      METADATA_TYPES="$2"
      shift 2
      ;;
    --block-size)
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --max-rows-total)
      MAX_ROWS_TOTAL="$2"
      shift 2
      ;;
    --grid-depth)
      GRID_DEPTH="$2"
      shift 2
      ;;
    --export-root)
      EXPORT_ROOT="$2"
      shift 2
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

split_csv "$DATABASES"
DATABASE_LIST=("${SPLIT_CSV_RESULT[@]}")
split_csv "$MODEL_KINDS"
MODEL_KIND_LIST=("${SPLIT_CSV_RESULT[@]}")
split_csv "$METADATA_TYPES"
METADATA_LIST=("${SPLIT_CSV_RESULT[@]}")

UV_BIN="$(resolve_uv)"

cd "$REPO_ROOT"

for database in "${DATABASE_LIST[@]}"; do
  for model_kind in "${MODEL_KIND_LIST[@]}"; do
    for metadata_type in "${METADATA_LIST[@]}"; do
      cmd=(
        "$UV_BIN" run python bench.py
        --database "$database"
        --model-kind "$model_kind"
        --block-size "$BLOCK_SIZE"
        --max-rows-total "$MAX_ROWS_TOTAL"
        --task-type regressor
        --block-metadata "$metadata_type"
        --export "$EXPORT_ROOT"
      )

      if [[ "$metadata_type" == "grid" || "$metadata_type" == "bounded_convex_hull" ]]; then
        cmd+=(--grid-depth "$GRID_DEPTH")
      fi

      printf '[export] '
      printf '%q ' "${cmd[@]}"
      printf '\n'

      "${cmd[@]}"
    done
  done
done
