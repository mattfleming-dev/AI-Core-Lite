param(
    [string]$PythonRuntime = "$PSScriptRoot\..\..\AI-Core\Runtime\python",
    [string]$Compiler = "$env:LOCALAPPDATA\AI-CoreBuildTools\Inno7\ISCC.exe"
)
$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot\.."
$python = Resolve-Path $PythonRuntime
if (-not (Test-Path $Compiler)) { throw "Inno Setup compiler not found: $Compiler (pass -Compiler)" }

$stage = Join-Path $root "build\payload"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force "$stage\runtime", "$stage\app", "$stage\config" | Out-Null

# Trimmed stdlib-only Python: no tkinter, tests, pip or site-packages.
robocopy $python "$stage\runtime" /E /XD tcl test tests idlelib turtledemo site-packages ensurepip __pycache__ /XF "*.pyc" /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
New-Item -ItemType Directory -Force "$stage\runtime\Lib\site-packages" | Out-Null

robocopy "$root\lite_rag" "$stage\app\lite_rag" /E /XD __pycache__ /XF "*.pyc" /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
Copy-Item "$root\installer\config.default.json" "$stage\config\config.json"
Copy-Item "$root\README.md" "$stage\README.md"

# The staged runtime must be able to run the app before we package it.
Push-Location "$stage\app"
try {
    & "$stage\runtime\python.exe" -c "import lite_rag.cli, sqlite3, http.server; print('payload ok')"
    if ($LASTEXITCODE -ne 0) { throw "staged runtime cannot import lite_rag" }
} finally { Pop-Location }

$dist = Join-Path $root "dist"
New-Item -ItemType Directory -Force $dist | Out-Null
& $Compiler "/DPayloadDir=$stage" "/DInstallerOutputDir=$dist" "$root\installer\Lite-RAG.iss"
if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }
Get-ChildItem $dist -Filter *.exe | Select-Object Name, @{n="MB";e={[math]::Round($_.Length/1MB,1)}}
