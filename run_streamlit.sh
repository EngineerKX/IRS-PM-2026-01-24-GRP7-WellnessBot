#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

export PYTHONPATH="$repo_root"
export MOCK_NLU="${MOCK_NLU:-1}"

python -m streamlit run app/streamlit_app.py