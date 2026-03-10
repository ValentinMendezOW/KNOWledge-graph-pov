#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR/src" "$ROOT_DIR/.venv/bin/python" -m knowledge_graph_tool "$@"
