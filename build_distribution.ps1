$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectDir ".venv\Scripts\python.exe"
$spec = Join-Path $projectDir "GridResistanceAnalyzer.spec"
$distDir = Join-Path $projectDir "dist"
$packageDir = Join-Path $distDir "GridResistanceAnalyzer"
$zipPath = Join-Path $distDir "GridResistanceAnalyzer-windows-x64.zip"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment was not found: $python"
}

& $python -m PyInstaller --noconfirm --clean $spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath (Join-Path $packageDir "GridResistanceAnalyzer.exe"))) {
    throw "The expected executable was not produced."
}

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -LiteralPath $packageDir -DestinationPath $zipPath -CompressionLevel Optimal

Write-Host "Portable directory: $packageDir"
Write-Host "Distribution ZIP: $zipPath"
