#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$repo_dir/scripts/reproduce_paper.py" "$@"
