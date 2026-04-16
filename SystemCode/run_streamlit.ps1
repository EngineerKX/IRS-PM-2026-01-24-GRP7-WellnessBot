$ErrorActionPreference = "Stop"

# Always run from the SystemCode directory (where this script lives)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $scriptDir

# Load .env into environment variables
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $parts = $_ -split '=', 2
    if ($parts.Length -eq 2) {
        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

if (Test-Path ".env") {
  Get-Content .env | ForEach-Object {
    if ($_ -match "^\s*([^#][^=]*)=(.*)$") {
      $name = $matches[1]
      $value = $matches[2]
      [System.Environment]::SetEnvironmentVariable($name, $value)
    }
  }
}

# Activate venv if exists
if (Test-Path "$repoRoot\.venv\Scripts\Activate.ps1") {
  . "$repoRoot\.venv\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
  . .\.venv\Scripts\Activate.ps1
}

# Make package importable — PYTHONPATH must be the repo root so
# "SystemCode.wellnessbot.*" imports resolve correctly
$env:PYTHONPATH = "$repoRoot"

# Optional: set MOCK_NLU if not already set
if (-not $env:MOCK_NLU) { $env:MOCK_NLU = "1" }

# Optional placeholders for corporate proxy overrides (only if you need)
# $env:HTTP_PROXY="http://proxy:port"
# $env:HTTPS_PROXY="http://proxy:port"

streamlit run .\wellnessbot\Wellnessbot.py