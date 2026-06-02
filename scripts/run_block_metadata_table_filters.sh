#!/usr/bin/env bash
set -euo pipefail

METADATA_TYPES="minmax,convex_hull,bounded_convex_hull"
JOBS="20"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage: scripts/run_block_metadata_table_filters.sh [options]

Rerun the exact benchmark configurations used by
`notebooks/block-metadata-size-build-table.ipynb` for the two selected deep
2-feature regressor filters:
- `tpch` / `discounted_price`
- `tpcds` / `store_sales_net_profit`

This script runs the following fixed configuration:
- verifier backend: `marabou`
- model kind: `deep`
- block size: `1000`
- max rows total: `100000`
- task type: `regressor`
- range alpha: `2`
- range start samples: `10`
- range seed: `0`

Options:
  --metadata-types LIST   Comma-separated metadata kinds.
                          Default: minmax,convex_hull,bounded_convex_hull
  --jobs N                Parallel jobs for bench.py. Default: 20
  --dry-run               Print commands without running them
  --help                  Show this help
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
    --metadata-types)
      METADATA_TYPES="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
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

split_csv "$METADATA_TYPES" METADATA_LIST

DATABASES=(tpch tpcds)
FILTERS=(discounted_price store_sales_net_profit)

for index in "${!DATABASES[@]}"; do
  database="${DATABASES[$index]}"
  filter_name="${FILTERS[$index]}"

  for metadata_type in "${METADATA_LIST[@]}"; do
    cmd=(
      uv run python bench.py
      --database "$database"
      --filter "$filter_name"
      --model-kind deep
      --block-size 1000
      --max-rows-total 100000
      --task-type regressor
      --jobs "$JOBS"
      --verifier-backend marabou
      --block-metadata "$metadata_type"
      --range-alpha 2
      --range-start-samples 10
      --range-seed 0
    )

    if [[ "$metadata_type" == "grid" || "$metadata_type" == "bounded_convex_hull" ]]; then
      cmd+=(--grid-depth 4)
    fi

    printf '[table-rerun] '
    printf '%q ' "${cmd[@]}"
    printf '
'

    if [[ "$DRY_RUN" != "1" ]]; then
      "${cmd[@]}"
    fi
  done
done
