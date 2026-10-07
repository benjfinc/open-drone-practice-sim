$ErrorActionPreference = "Stop"

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py "$PSScriptRoot\run.py" @args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python "$PSScriptRoot\run.py" @args
} else {
    Write-Error "Python was not found. Run setup.ps1 after installing Python 3.12."
    exit 1
}
exit $LASTEXITCODE
