#!/usr/bin/env bash
set -euo pipefail

DATABASE="${1:-tpch}"
shift || true

uv run python -m nnv_tools.database_preprocess --database "${DATABASE}" "$@"
