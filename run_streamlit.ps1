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

$ErrorActionPreference = "Stop"

# --- Step 3: Set working directory to project root ----------------------------
# Resolves the directory where this script lives, ensuring all relative paths
# (like .env, .venv, app/) work correctly regardless of where you call it from.
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

# --- Step 4: Activate Python virtual environment (if present) -----------------
# Looks for a .venv folder in the project root. If found, activates it so that
# the correct Python interpreter and installed packages are used.
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
  . .\.venv\Scripts\Activate.ps1
}

# --- Step 5: Set PYTHONPATH ---------------------------------------------------
# Adds the project root to PYTHONPATH so Python can find the 'wellnessbot' package.
# Without this, imports like 'from wellnessbot.pipeline.run import ...' would fail.
$env:PYTHONPATH = "$repoRoot"

# --- Step 6: Default to mock NLU mode ----------------------------------------
# If MOCK_NLU is not already set (e.g., via .env), default to "1" (enabled).
# MOCK_NLU=1 uses regex-based extraction (no OpenAI API key required).
# MOCK_NLU=0 uses OpenAI for NLU (requires OPENAI_API_KEY in .env).
if (-not $env:MOCK_NLU) { $env:MOCK_NLU = "1" }

# --- Optional: Corporate proxy overrides (uncomment if needed) ----------------
# $env:HTTP_PROXY="http://proxy:port"
# $env:HTTPS_PROXY="http://proxy:port"

# --- Step 7: Launch the Streamlit app -----------------------------------------
# Starts the Streamlit web server, which opens the chat UI in your browser.
# The app is defined in app/streamlit_app.py.
streamlit run .\app\streamlit_app.py
