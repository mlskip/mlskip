#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_ROOT="${ROOT_DIR}/metadata/models"
COMPILED_ROOT="${ROOT_DIR}/metadata/compiled-models"
FORCE=0
DATABASE=""
MODEL_INPUT_PATH=""
MAX_COLUMNS="2"
MAX_COLUMNS_EXPLICIT="0"

usage() {
  cat <<'USAGE'
Usage: scripts/compile_geomcad_models.sh [database] [--model metadata/models/.../shallow/.../*.onnx | --max-columns N] [--force]

Compiles shallow regressor ONNX models under metadata/models into
metadata/compiled-models using geometrical-cad/compile_onnx_model.py.
By default, only models with at most 2 input features are compiled.
`--model` must be a repo-root path beginning with metadata/models/ and must point
to a shallow regressor model.

Examples:
  scripts/compile_geomcad_models.sh
  scripts/compile_geomcad_models.sh tpch
  scripts/compile_geomcad_models.sh tpcds --force
  scripts/compile_geomcad_models.sh --max-columns 2
  scripts/compile_geomcad_models.sh --model metadata/models/tpcds/shallow/regressor/store_sales/store_sales_net_profit/store_sales_net_profit.onnx
USAGE
}

feature_count_for_onnx() {
  local onnx_file="$1"
  python3 - "$onnx_file" <<'PY'
import json
import sys
from pathlib import Path
onnx_path = Path(sys.argv[1])
metadata_path = onnx_path.with_suffix('.metadata.json')
if not metadata_path.exists():
    print('')
    raise SystemExit(0)
metadata = json.loads(metadata_path.read_text())
print(len(metadata.get('features', [])))
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --model)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --model" >&2
        usage >&2
        exit 1
      fi
      MODEL_INPUT_PATH="$2"
      shift 2
      ;;
    --max-columns)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --max-columns" >&2
        usage >&2
        exit 1
      fi
      MAX_COLUMNS="$2"
      MAX_COLUMNS_EXPLICIT="1"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$DATABASE" ]]; then
        echo "Expected at most one database argument, got: $1" >&2
        usage >&2
        exit 1
      fi
      DATABASE="$1"
      shift
      ;;
  esac
done

if ! [[ "$MAX_COLUMNS" =~ ^[0-9]+$ ]] || [[ "$MAX_COLUMNS" -lt 1 ]]; then
  echo "--max-columns must be a positive integer, got: $MAX_COLUMNS" >&2
  exit 1
fi

if [[ -n "$MODEL_INPUT_PATH" && "$MAX_COLUMNS_EXPLICIT" == "1" ]]; then
  echo "--max-columns cannot be combined with --model; the selected model is already explicit." >&2
  exit 1
fi

SEARCH_ROOT="$MODELS_ROOT"
if [[ -n "$DATABASE" ]]; then
  SEARCH_ROOT="$MODELS_ROOT/$DATABASE"
fi

if [[ ! -d "$SEARCH_ROOT" ]]; then
  echo "Model directory not found: $SEARCH_ROOT" >&2
  exit 1
fi

if [[ -n "$MODEL_INPUT_PATH" ]]; then
  if [[ "$MODEL_INPUT_PATH" != metadata/models/* ]]; then
    echo "--model must point to metadata/models/... , got: $MODEL_INPUT_PATH" >&2
    exit 1
  fi

  ONNX_FILE="$ROOT_DIR/$MODEL_INPUT_PATH"
  if [[ ! -f "$ONNX_FILE" ]]; then
    echo "Model file not found: $ONNX_FILE" >&2
    exit 1
  fi

  MODEL_REPO_PATH="${ONNX_FILE#${MODELS_ROOT}/}"
  case "$MODEL_REPO_PATH" in
    */shallow/regressor/*/*.onnx)
      ;;
    *)
      echo "--model must point to a shallow regressor ONNX path under metadata/models, got: $MODEL_INPUT_PATH" >&2
      exit 1
      ;;
  esac

  if [[ -n "$DATABASE" && "$MODEL_REPO_PATH" != "$DATABASE/"* ]]; then
    echo "--model path must be inside database $DATABASE, got: $MODEL_INPUT_PATH" >&2
    exit 1
  fi

  ONNX_FILES=("$ONNX_FILE")
else
  mapfile -t ALL_ONNX_FILES < <(find "$SEARCH_ROOT" -type f -path '*/shallow/regressor/*/*.onnx' | sort)
  if [[ ${#ALL_ONNX_FILES[@]} -eq 0 ]]; then
    echo "No shallow regressor ONNX models found under $SEARCH_ROOT"
    exit 0
  fi

  ONNX_FILES=()
  skipped_by_columns=0
  for onnx_file in "${ALL_ONNX_FILES[@]}"; do
    feature_count="$(feature_count_for_onnx "$onnx_file")"
    if [[ -n "$feature_count" && "$feature_count" -gt "$MAX_COLUMNS" ]]; then
      skipped_by_columns=$((skipped_by_columns + 1))
      echo "[geomcad-compile] Skipping ${onnx_file#${ROOT_DIR}/}: ${feature_count} feature(s) exceeds --max-columns $MAX_COLUMNS"
      continue
    fi
    ONNX_FILES+=("$onnx_file")
  done

  if [[ ${#ONNX_FILES[@]} -eq 0 ]]; then
    echo "No shallow regressor ONNX models matched --max-columns $MAX_COLUMNS under $SEARCH_ROOT"
    exit 0
  fi

  if [[ "$skipped_by_columns" -gt 0 ]]; then
    echo "[geomcad-compile] Retained ${#ONNX_FILES[@]} model(s) after applying --max-columns $MAX_COLUMNS; skipped $skipped_by_columns model(s)"
  fi
fi

total_models=${#ONNX_FILES[@]}

for index in "${!ONNX_FILES[@]}"; do
  onnx_file="${ONNX_FILES[$index]}"
  rel_path="${onnx_file#${MODELS_ROOT}/}"
  compiled_file="${COMPILED_ROOT}/${rel_path%.onnx}.geomcad.db"
  metrics_file="${compiled_file%.db}.json"

  if [[ -f "$compiled_file" && -f "$metrics_file" && "$FORCE" -ne 1 ]]; then
    echo "[geomcad-compile] Skipping existing ${compiled_file#${ROOT_DIR}/} and ${metrics_file#${ROOT_DIR}/}"
    continue
  fi

  mkdir -p "$(dirname "$compiled_file")"
  human_index=$((index + 1))
  echo "[geomcad-compile] [${human_index}/${total_models}] ${rel_path} -> ${compiled_file#${ROOT_DIR}/}"
  (
    cd "${ROOT_DIR}/geometrical-cad"
    if [[ -n "$MODEL_INPUT_PATH" ]]; then
      uv run python compile_onnx_model.py \
        --onnx-file "$onnx_file" \
        --duckdb-file "$compiled_file" \
        --metrics-file "$metrics_file"
    else
      uv run python compile_onnx_model.py \
        --onnx-file "$onnx_file" \
        --duckdb-file "$compiled_file" \
        --metrics-file "$metrics_file" \
        --max-columns "$MAX_COLUMNS"
    fi
  )
done
