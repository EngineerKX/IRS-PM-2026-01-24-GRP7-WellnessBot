$ErrorActionPreference = "Stop"

# Ensure we run from repo root
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

# Activate venv if exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
  . .\.venv\Scripts\Activate.ps1
}

# Make package importable
$env:PYTHONPATH = "$repoRoot"

# Optional: set MOCK_NLU if not already set
if (-not $env:MOCK_NLU) { $env:MOCK_NLU = "1" }

# Optional placeholders for corporate proxy overrides (only if you need)
# $env:HTTP_PROXY="http://proxy:port"
# $env:HTTPS_PROXY="http://proxy:port"

streamlit run .\app\streamlit_app.py