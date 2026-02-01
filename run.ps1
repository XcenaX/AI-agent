$ErrorActionPreference = "Stop"

# venv
.\.venv\Scripts\Activate.ps1

# чтобы Python видел src/webpilot как пакет webpilot
$env:PYTHONPATH = "$PSScriptRoot\src"

python -m webpilot.cli