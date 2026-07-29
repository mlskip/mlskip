#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib/resolve-uv.sh
source "$SCRIPT_DIR/lib/resolve-uv.sh"

DATABASES="tpch,tpcds"
MODEL_KINDS="shallow,deep"
SAMPLE_SIZE=""
EPOCHS=""
FORCE_RETRAIN=0
SETUP_FORCE_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/setup-benchmarks.sh [options]

Set up benchmark DuckDB files and train both shallow/deep models.

Options:
  --databases LIST       Comma-separated databases. Default: tpch,tpcds
  --model-kinds LIST     Comma-separated model kinds. Default: shallow,deep
  --sample-size N        Pass --sample-size to train.py
  --epochs N             Pass --epochs to train.py
  --force-retrain        Pass --force-retrain to train.py
  --force-csv            Pass --force-csv to database preprocessing
  --force-duckdb         Pass --force-duckdb to database preprocessing
  -h, --help             Show this help

Examples:
  scripts/setup-benchmarks.sh
  scripts/setup-benchmarks.sh --databases tpch --model-kinds deep
  scripts/setup-benchmarks.sh --force-retrain --epochs 120
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --databases)
      DATABASES="${2:?Missing value for --databases}"
      shift 2
      ;;
    --model-kinds)
      MODEL_KINDS="${2:?Missing value for --model-kinds}"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="${2:?Missing value for --sample-size}"
      shift 2
      ;;
    --epochs)
      EPOCHS="${2:?Missing value for --epochs}"
      shift 2
      ;;
    --force-retrain)
      FORCE_RETRAIN=1
      shift
      ;;
    --force-csv)
      SETUP_FORCE_ARGS+=("--force-csv")
      shift
      ;;
    --force-duckdb)
      SETUP_FORCE_ARGS+=("--force-duckdb")
      shift
      ;;
    -h|--help)
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

UV="$(resolve_uv)"
cd "$REPO_ROOT"

IFS=',' read -r -a DATABASE_LIST <<< "$DATABASES"
IFS=',' read -r -a MODEL_KIND_LIST <<< "$MODEL_KINDS"

for database in "${DATABASE_LIST[@]}"; do
  if [[ -z "$database" ]]; then
    continue
  fi
  printf '[setup] Preparing database=%s\n' "$database"
  setup_args=(
    run python -m nnv_tools.database_preprocess
    --database "$database"
  )
  if [[ "${#SETUP_FORCE_ARGS[@]}" -gt 0 ]]; then
    setup_args+=("${SETUP_FORCE_ARGS[@]}")
  fi
  "$UV" "${setup_args[@]}"

  for model_kind in "${MODEL_KIND_LIST[@]}"; do
    if [[ -z "$model_kind" ]]; then
      continue
    fi
    printf '[setup] Training database=%s model_kind=%s\n' "$database" "$model_kind"
    train_args=(
      run python train.py
      --database "$database"
      --model-kind "$model_kind"
    )
    if [[ -n "$SAMPLE_SIZE" ]]; then
      train_args+=(--sample-size "$SAMPLE_SIZE")
    fi
    if [[ -n "$EPOCHS" ]]; then
      train_args+=(--epochs "$EPOCHS")
    fi
    if [[ "$FORCE_RETRAIN" -eq 1 ]]; then
      train_args+=(--force-retrain)
    fi
    "$UV" "${train_args[@]}"
  done
done

printf '[setup] Benchmark setup complete\n'
