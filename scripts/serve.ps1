# Build the frontend and serve the whole app from the backend, as in Docker: fast, no dev tooling.
# Use it to work with the app; use dev.ps1 to work on the code.
# App: http://127.0.0.1:<Port>  (8000 by default; pass -Port 8001 if 8000 is taken)

param([int]$Port = 8000)

$root = Split-Path $PSScriptRoot -Parent

Push-Location "$root\frontend"
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

$env:QUANTILO_STATIC_DIR = "$root\frontend\dist"
Push-Location "$root\backend"
try {
    Write-Host "Quantilo: http://127.0.0.1:$Port"
    uv run uvicorn app.main:app --host 127.0.0.1 --port $Port
}
finally {
    Pop-Location
}
