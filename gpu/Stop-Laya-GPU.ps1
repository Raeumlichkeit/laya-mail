$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $repoRoot '.venv-gpu\Scripts\python.exe'
$scriptPath = Join-Path $PSScriptRoot 'serve_gpu.py'
$configPath = Join-Path $PSScriptRoot 'config.local.json'
$configJson = & $pythonPath $scriptPath --config $configPath --print-config
if ($LASTEXITCODE -ne 0) { throw 'GPU-Konfiguration ungültig; keine Prozesse beendet.' }
$config = $configJson | ConvertFrom-Json
$processes = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains('"' + $scriptPath + '"') -and $_.CommandLine.Contains('"' + $configPath + '"') }
$processes | ForEach-Object { Stop-Process -Id $_.ProcessId }
Write-Output ('Laya GPU für {0}:{1} ist beendet.' -f $config.host, $config.port)
