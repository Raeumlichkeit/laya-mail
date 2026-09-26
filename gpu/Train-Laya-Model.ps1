$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Laya Mail - Modelltraining auf NVIDIA RTX 4070 Ti (CUDA)  " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Workspace-Pfade ermitteln
$searchDir = $PSScriptRoot
$taskRoot = $null
while ($searchDir) {
    if ((Test-Path (Join-Path $searchDir 'work')) -and (Test-Path (Join-Path $searchDir 'laya-mail'))) {
        $taskRoot = $searchDir
        break
    }
    $searchDir = Split-Path $searchDir -Parent
}
if (-not $taskRoot) {
    $taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}

$pythonPath = Join-Path $taskRoot 'work\laya-gpu-venv\Scripts\python.exe'
$repoRoot = Join-Path $taskRoot 'laya-mail'
$trainScript = Join-Path $repoRoot 'gpu\train_laya.py'
$evalScript = Join-Path $repoRoot 'gpu\evaluate_model.py'
$baseModel = Join-Path $taskRoot 'work\laya-gpu\multilingual'
$outputDir = Join-Path $repoRoot 'runtime\models\laya-mail-custom'
$dashboardKeyPath = Join-Path $taskRoot 'work\gmail-private\dashboard-key'
$dashboardUrl = if ($env:LAYA_DASHBOARD_URL) { $env:LAYA_DASHBOARD_URL } else {
    $cfgPath = Join-Path $taskRoot 'work\gmail-private\dashboard-url.txt'
    if (Test-Path $cfgPath) { (Get-Content $cfgPath -Raw).Trim() } else { 'http://127.0.0.1:8001' }
}
$dataDir = Join-Path $repoRoot 'runtime\training'
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
$trainData = Join-Path $dataDir 'laya-mail-train.jsonl'
$valData = Join-Path $dataDir 'laya-mail-val.jsonl'

# 1. Neueste Daten vom Dashboard abrufen
Write-Host "[1/4] Lade aktuelle Bewertungen vom Mac mini Dashboard..." -ForegroundColor Yellow
$exportPy = @"
import urllib.request, urllib.parse, http.cookiejar, pathlib, sys
jar = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
key = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').strip()
origin = sys.argv[4].rstrip('/')
try:
    login_data = urllib.parse.urlencode({'code': key}).encode()
    op.open(urllib.request.Request(origin + '/login', data=login_data, headers={'Origin': origin}), timeout=5)
    train_bytes = op.open(origin + '/api/training-export?split=train', timeout=10).read()
    val_bytes = op.open(origin + '/api/training-export?split=val', timeout=10).read()
    if len(train_bytes) > 0 and len(val_bytes) > 0:
        pathlib.Path(sys.argv[2]).write_bytes(train_bytes)
        pathlib.Path(sys.argv[3]).write_bytes(val_bytes)
        print(f'Erfolg: {len(train_bytes.splitlines())} Trainingsbeispiele, {len(val_bytes.splitlines())} Validierungsbeispiele geladen.')
    else:
        print('Warnung: Dashboard lieferte leere Daten.', file=sys.stderr)
except Exception as e:
    print(f'Hinweis: Dashboard nicht erreichbar ({e}). Verwende vorhandene lokale Trainingsdaten.', file=sys.stderr)
"@

$exportScriptPath = Join-Path $env:TEMP 'laya_export_runner.py'
Set-Content -Path $exportScriptPath -Value $exportPy -Encoding UTF8
& $pythonPath $exportScriptPath $dashboardKeyPath $trainData $valData $dashboardUrl
Remove-Item -Path $exportScriptPath -Force -ErrorAction SilentlyContinue

# Fallback auf Projekt-Root falls runtime/training leer
if (-not (Test-Path $trainData) -or (Get-Item $trainData).Length -eq 0) {
    $altTrain = Join-Path $repoRoot 'laya-mail-train.jsonl'
    $altVal = Join-Path $repoRoot 'laya-mail-val.jsonl'
    if ((Test-Path $altTrain) -and (Test-Path $altVal)) {
        Copy-Item $altTrain $trainData -Force
        Copy-Item $altVal $valData -Force
    }
}

if (-not (Test-Path $trainData) -or (Get-Item $trainData).Length -eq 0) {
    Write-Host "Fehler: Keine Trainingsdaten verfuegbar. Bitte zuerst Mails im Dashboard bewerten!" -ForegroundColor Red
    Write-Host ""
    Read-Host "Druecke Enter zum Schliessen..."
    exit 1
}

# 2. Feinabstimmung auf RTX 4070 Ti
Write-Host ""
Write-Host "[2/4] Starte Feinabstimmung (Head-Tuning) auf RTX 4070 Ti..." -ForegroundColor Yellow
& $pythonPath -u $trainScript `
  --data $trainData `
  --val-data $valData `
  --base-model $baseModel `
  --output-dir $outputDir `
  --epochs 8 `
  --lr 3e-4 `
  --batch-size 8

if ($LASTEXITCODE -ne 0) {
    Write-Host "Fehler beim Modelltraining aufgetreten!" -ForegroundColor Red
    Write-Host ""
    Read-Host "Druecke Enter zum Schliessen..."
    exit 1
}

# 3. Evaluation
Write-Host ""
Write-Host "[3/4] Evaluierung des trainierten Modells gegen das Basismodell..." -ForegroundColor Yellow
& $pythonPath -u $evalScript `
  --data $valData `
  --model $outputDir `
  --baseline $baseModel

# 4. GPU-Dienst bei Bedarf neu laden
Write-Host ""
Write-Host "[4/4] Pruefe GPU-Dienst Status..." -ForegroundColor Yellow
$outputsDir = Join-Path $taskRoot 'outputs\laya-gpu'
$stopScript = Join-Path $outputsDir 'Stop-Laya-GPU.ps1'
$startScript = Join-Path $outputsDir 'Start-Laya-GPU.ps1'

$wasRunning = $false
try {
    $status = Invoke-RestMethod -Uri 'http://127.0.0.1:8002/health' -TimeoutSec 2
    if ($status.status -eq 'ok') { $wasRunning = $true }
} catch {}

if ($wasRunning) {
    Write-Host "Laufender GPU-Dienst wird neu geladen, um das neue Modell zu nutzen..." -ForegroundColor Cyan
    if (Test-Path $stopScript) { & $stopScript }
    Start-Sleep -Seconds 1
    if (Test-Path $startScript) { & $startScript }
    Write-Host "Warte kurz auf CUDA-Initialisierung..." -ForegroundColor Gray
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        try {
            $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8002/health' -TimeoutSec 1
            if ($h.status -eq 'ok' -and $h.device -eq 'cuda') { break }
        } catch {}
    }
    Write-Host "GPU-Dienst laeuft jetzt mit dem neu trainierten Modell." -ForegroundColor Green
} else {
    Write-Host "GPU-Dienst ist derzeit gestoppt (wird beim Klick auf 'Start GPU' automatisch mit dem neuen Modell gestartet)." -ForegroundColor Gray
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Training erfolgreich abgeschlossen!" -ForegroundColor Green
Write-Host " Das neue Modell ist ab sofort einsatzbereit." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Druecke Enter zum Beenden..."
