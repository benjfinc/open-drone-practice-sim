$ErrorActionPreference = "Stop"

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.12 "$PSScriptRoot\bootstrap.py" @args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python "$PSScriptRoot\bootstrap.py" @args
} else {
    Write-Error "Python was not found. Install 64-bit Python 3.12 first."
    exit 1
}
exit $LASTEXITCODE
