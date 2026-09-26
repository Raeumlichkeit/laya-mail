$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $repoRoot '.venv-gpu\Scripts\python.exe'
$scriptPath = Join-Path $PSScriptRoot 'serve_gpu.py'
$configPath = Join-Path $PSScriptRoot 'config.local.json'
$configJson = & $pythonPath $scriptPath --config $configPath --print-config
if ($LASTEXITCODE -ne 0) { throw 'GPU-Konfiguration ungültig; Dienst nicht gestartet.' }
$config = $configJson | ConvertFrom-Json
if ([IO.File]::ReadAllText($config.key_file).Trim().Length -lt 24) {
    throw 'Der zufällige API-Schlüssel muss mindestens 24 Zeichen lang sein.'
}
if (-not (Test-Path -LiteralPath $config.model_path -PathType Container)) {
    throw 'Der konfigurierte Modellordner fehlt.'
}
$urlHost = if ($config.host.Contains(':')) { '[' + $config.host + ']' } else { $config.host }
try {
    $status = Invoke-RestMethod -Uri ('http://{0}:{1}/health' -f $urlHost, $config.port) -TimeoutSec 2
    if ($status.device -eq 'cuda') { Write-Output 'Laya GPU läuft bereits.'; exit 0 }
} catch {}
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains('"' + $scriptPath + '"') -and $_.CommandLine.Contains('"' + $configPath + '"') }
if ($existing) { Write-Output 'Laya GPU startet bereits.'; exit 0 }
$logPath = Join-Path $repoRoot 'runtime\gpu'
New-Item -ItemType Directory -Path $logPath -Force | Out-Null
Start-Process -FilePath $pythonPath -ArgumentList @('-u', ('"' + $scriptPath + '"'), '--config', ('"' + $configPath + '"')) `
    -WorkingDirectory $repoRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logPath 'server.log') `
    -RedirectStandardError (Join-Path $logPath 'server-error.log') | Out-Null
Write-Output 'Laya GPU startet. Status und Fehler stehen unter runtime/gpu.'
