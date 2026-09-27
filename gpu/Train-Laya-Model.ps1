param([switch]$CheckOnly, [switch]$NoPause)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$exitCode = 0
$runLock = $null
$runDir = $null
$transcriptStarted = $false
$resultMessage = 'Lauf wurde nicht abgeschlossen.'
$report = $null

function Invoke-TrainingPython {
    # Windows PowerShell transcripts omit direct native output. Stream it through the host;
    # stderr warnings are not failures unless Python actually returns a nonzero exit code.
    $ErrorActionPreference = 'Continue'
    & $pythonPath @args 2>&1 | ForEach-Object { Write-Host "$_" }
    $script:LASTEXITCODE = $LASTEXITCODE
}

try {
    Write-Host "`nLaya Mail - GPU-Modelltraining" -ForegroundColor Cyan

    # Works from the repository and the installed desktop launcher directory.
    $searchDir = $PSScriptRoot
    $taskRoot = $null
    while ($searchDir) {
        if ((Test-Path (Join-Path $searchDir 'work')) -and (Test-Path (Join-Path $searchDir 'laya-mail'))) {
            $taskRoot = $searchDir
            break
        }
        $searchDir = Split-Path $searchDir -Parent
    }
    if (-not $taskRoot) { throw 'Laya-Arbeitsordner mit work und laya-mail wurde nicht gefunden.' }

    $pythonPath = Join-Path $taskRoot 'work\laya-gpu-venv\Scripts\python.exe'
    $repoRoot = Join-Path $taskRoot 'laya-mail'
    $trainScript = Join-Path $repoRoot 'gpu\train_laya.py'
    $evalScript = Join-Path $repoRoot 'gpu\evaluate_model.py'
    $exportScript = Join-Path $repoRoot 'gpu\export_training_data.py'
    $baseModel = Join-Path $taskRoot 'work\laya-gpu\multilingual'
    $activeModel = Join-Path $repoRoot 'runtime\models\laya-mail-custom'
    $dashboardKeyPath = Join-Path $taskRoot 'work\gmail-private\dashboard-key'

    $runsRoot = Join-Path $repoRoot 'runtime\training'
    New-Item -ItemType Directory -Path $runsRoot -Force | Out-Null
    try {
        $runLock = [System.IO.File]::Open((Join-Path $runsRoot '.training.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch { throw 'Ein anderer Trainingslauf ist bereits geoeffnet. Bitte zuerst dessen Fenster schliessen.' }
    $runId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
    $runDir = Join-Path $runsRoot $runId
    # The exporter must create a fresh directory itself. Keep the log beside it until export ends.
    $pendingLog = Join-Path $runsRoot ($runId + '.log')
    Start-Transcript -LiteralPath $pendingLog | Out-Null
    $transcriptStarted = $true

    $dashboardUrl = $env:LAYA_DASHBOARD_URL
    if (-not $dashboardUrl) {
        $urlFile = Join-Path $taskRoot 'work\gmail-private\dashboard-url.txt'
        if (-not (Test-Path -LiteralPath $urlFile)) { throw 'Dashboard-Adresse fehlt in dashboard-url.txt oder LAYA_DASHBOARD_URL.' }
        $dashboardUrl = (Get-Content -LiteralPath $urlFile -Raw).Trim()
    }
    foreach ($path in @($pythonPath, $trainScript, $evalScript, $exportScript, $dashboardKeyPath,
                         (Join-Path $baseModel 'model.safetensors'), (Join-Path $baseModel 'rl_agent_config.json'))) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Erforderliche Datei fehlt: $path" }
    }

    $trainData = Join-Path $runDir 'laya-mail-train.jsonl'
    $valData = Join-Path $runDir 'laya-mail-val.jsonl'
    $candidate = Join-Path $runDir 'model'
    $reportFile = Join-Path $runDir 'evaluation.json'

    Write-Host "`n[1/4] Lade und pruefe die aktuellen Bewertungen..." -ForegroundColor Yellow
    Invoke-TrainingPython $exportScript --url $dashboardUrl --key-file $dashboardKeyPath --output-dir $runDir
    if ($LASTEXITCODE -ne 0) { throw 'Datenexport fehlgeschlagen. Es wird kein Training gestartet.' }
    Invoke-TrainingPython $trainScript --data $trainData --val-data $valData --dry-run
    if ($LASTEXITCODE -ne 0) { throw 'Die Trainingsdaten sind ungueltig.' }

    if ($CheckOnly) {
        $resultMessage = 'Pruefung erfolgreich. Kein Training und keine Modellaenderung ausgefuehrt.'
        Write-Host "`n$resultMessage" -ForegroundColor Green
    } else {
        $env:HF_HUB_OFFLINE = '1'
        $env:TRANSFORMERS_OFFLINE = '1'
        $env:TORCH_DISABLE_NATIVE_JIT = '1'
        Write-Host "`n[2/4] Trainiere einen separaten Kandidaten auf CUDA..." -ForegroundColor Yellow
        Invoke-TrainingPython -u $trainScript --data $trainData --val-data $valData --base-model $baseModel `
            --output-dir $candidate --epochs 8 --lr 3e-4 --batch-size 8 --device cuda
        if ($LASTEXITCODE -ne 0) { throw 'Training fehlgeschlagen. Das bisherige Modell bleibt erhalten.' }

        $baseline = $baseModel
        if (Test-Path -LiteralPath (Join-Path $activeModel 'model.safetensors')) { $baseline = $activeModel }
        Write-Host "`n[3/4] Vergleiche den Kandidaten mit dem bisherigen Modell..." -ForegroundColor Yellow
        Invoke-TrainingPython -u $evalScript --data $valData --model $candidate --baseline $baseline --device cuda --report $reportFile
        if ($LASTEXITCODE -ne 0) { throw 'Modellvergleich fehlgeschlagen. Der Kandidat wird nicht uebernommen.' }
        $report = Get-Content -LiteralPath $reportFile -Raw | ConvertFrom-Json

        Write-Host "`n[4/4] Training und Modellvergleich abgeschlossen." -ForegroundColor Green
        Write-Host "Kandidat: $candidate"
        Write-Host "Vergleich: $reportFile"
        $resultMessage = 'Kandidat gespeichert; das bisherige Modell bleibt ausgewaehlt.'
        if ($report.improved -eq $true -and $report.nonregressed -eq $true) {
            Write-Host 'Im Validierungssatz insgesamt besser und bei keiner Frage schlechter.' -ForegroundColor Green
            $answer = if ($NoPause) { 'n' } else { Read-Host 'Neues Modell fuer den naechsten GPU-Start vormerken? [j/N]' }
            if ($answer -match '^(j|ja)$') {
                $backup = Join-Path $runDir 'previous-model'
                $runtimePrefix = [System.IO.Path]::GetFullPath((Join-Path $repoRoot 'runtime')) + [System.IO.Path]::DirectorySeparatorChar
                foreach ($modelPath in @($activeModel, $candidate, $backup)) {
                    if (-not [System.IO.Path]::GetFullPath($modelPath).StartsWith($runtimePrefix, [StringComparison]::OrdinalIgnoreCase)) {
                        throw 'Modellpfad liegt ausserhalb des lokalen runtime-Ordners.'
                    }
                }
                $hadModel = Test-Path -LiteralPath $activeModel
                if ($hadModel) { Move-Item -LiteralPath $activeModel -Destination $backup }
                try {
                    New-Item -ItemType Directory -Path (Split-Path $activeModel -Parent) -Force | Out-Null
                    Move-Item -LiteralPath $candidate -Destination $activeModel
                } catch {
                    if ($hadModel -and -not (Test-Path -LiteralPath $activeModel)) {
                        Move-Item -LiteralPath $backup -Destination $activeModel
                    }
                    throw
                }
                Write-Host 'Neues Modell vorgemerkt. Ein laufender GPU-Dienst nutzt weiterhin das alte Modell.' -ForegroundColor Cyan
                $resultMessage = 'Neues Modell fuer den naechsten GPU-Start vorgemerkt; ein laufender Dienst nutzt weiterhin das alte Modell.'
                Write-Host 'Wenn keine Mails ausgewertet werden und der Live-Modus aus ist:'
                Write-Host 'Auf dem Desktop zuerst Laya GPU beenden, dann Laya GPU starten.'
                if ($hadModel) { Write-Host "Bisheriges Modell gesichert: $backup" }
            } else {
                Write-Host 'Kandidat gespeichert; das bisherige Modell bleibt ausgewaehlt.'
            }
        } else {
            $resultMessage = 'Kein eindeutiger Fortschritt im Validierungssatz. Das bisherige Modell bleibt ausgewaehlt.'
            Write-Host $resultMessage -ForegroundColor Yellow
        }
        Write-Host 'Der GPU-Dienst wurde nicht neu gestartet.'
    }
} catch {
    $exitCode = 1
    $resultMessage = "Abgebrochen: $($_.Exception.Message)"
    Write-Host "`n$resultMessage" -ForegroundColor Red
} finally {
    try {
        if ($runDir) {
            New-Item -ItemType Directory -Path $runDir -Force | Out-Null
            $summary = @('Laya Mail - Trainingsergebnis', ('Zeit: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
                         ('Exitcode: ' + $exitCode), $resultMessage)
            if ($report) {
                $summary += 'Validierung (bisheriges Modell -> Kandidat):'
                $summary += ('Gesamt: {0:P2} -> {1:P2}' -f $report.baseline.overall_accuracy, $report.candidate.overall_accuracy)
                foreach ($question in $report.candidate.accuracies.PSObject.Properties.Name) {
                    $summary += ('{0}: {1:P2} -> {2:P2}' -f $question, $report.baseline.accuracies.$question, $report.candidate.accuracies.$question)
                }
                $spamConfusion = $report.candidate.confusion_by_question.spamverdacht
                if ($spamConfusion -and -not ($spamConfusion.ja.PSObject.Properties.Value | Where-Object { $_ -gt 0 })) {
                    $summary += 'Hinweis: Keine echten Spam-Beispiele im Validierungssatz; Spam-Erkennung ist damit nicht nachgewiesen.'
                }
                $summary += 'Details: evaluation.json (Validierung, kein unabhaengiger Test)'
            }
            $summary += 'Protokoll: training.log'
            $summary | Set-Content -LiteralPath (Join-Path $runDir 'ergebnis.txt') -Encoding UTF8
            Write-Host "`nErgebnis gespeichert: $(Join-Path $runDir 'ergebnis.txt')" -ForegroundColor Cyan
        }
    } catch {
        $exitCode = 1
        Write-Warning "Ergebnisdatei konnte nicht gespeichert werden: $($_.Exception.Message)"
    } finally {
        try {
            if ($transcriptStarted) {
                Stop-Transcript | Out-Null
                Move-Item -LiteralPath $pendingLog -Destination (Join-Path $runDir 'training.log')
            }
        } catch {
            $exitCode = 1
            Write-Warning "Protokoll bleibt unter ${pendingLog}: $($_.Exception.Message)"
        } finally {
            if ($runLock) { $runLock.Dispose() }
        }
    }
}

if ($NoPause) { exit $exitCode }
# The Desktop shortcut uses powershell.exe -NoExit. Do not override it with an explicit exit.
Write-Host "`nFertig. Du kannst dieses Fenster jetzt schliessen."
$global:LASTEXITCODE = $exitCode
