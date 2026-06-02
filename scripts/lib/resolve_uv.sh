#!/usr/bin/env bash

resolve_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi

  if [[ -x "$HOME/.local/bin/uv" ]]; then
    printf '%s\n' "$HOME/.local/bin/uv"
    return 0
  fi

  echo "uv not found in PATH or at $HOME/.local/bin/uv" >&2
  return 1
}
