# Optionale GPU-Auswertung unter Windows

Dieser Dienst lädt Laya auf eine NVIDIA-GPU. Dashboard, Gmail-Zugriff und Bewertungen bleiben auf dem Mailserver. Die Bewertung sammelt Trainingsbeispiele; sie verändert die Modellgewichte nicht.

Getesteter Stand: Python 3.12, Torch **2.14.0+cu126**, Transformers **5.17.0**, Laya **0.3.20** auf einer RTX 4070 Ti mit 12 GB VRAM. Die Skripte starten keinen Dienst automatisch nach einem Windows-Neustart.

## Einrichtung

PowerShell im Repository-Hauptordner öffnen. Python 3.12, Git und ein passender NVIDIA-Treiber müssen installiert sein:

```powershell
py -3.12 -m venv .venv-gpu
& .\.venv-gpu\Scripts\python.exe -m pip install 'torch==2.14.0+cu126' --index-url https://download.pytorch.org/whl/cu126
& .\.venv-gpu\Scripts\python.exe -m pip install 'transformers==5.17.0' 'laya[serve] @ git+https://github.com/NandhaKishorM/laya.git@23a17522aa4942da6cce53a995a275760320b691'
Copy-Item -LiteralPath .\gpu\config.example.json -Destination .\gpu\config.local.json
& .\.venv-gpu\Scripts\hf.exe download convaiinnovations/laya --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 --include 'multilingual/*' --local-dir .\runtime\models
New-Item -ItemType Directory -Path .\runtime\private -Force | Out-Null
& .\.venv-gpu\Scripts\python.exe -c "import secrets; from pathlib import Path; p=Path('runtime/private/gpu-api-key'); p.open('x', encoding='utf-8').write(secrets.token_urlsafe(32))"
```

Die Schlüssel-Erstellung ist einmalig und überschreibt keine vorhandene Datei. Gewichte und fremder Laya-Quellcode gehören nicht in dieses Repository. Beim laufenden GPU-Dienst sind Modelldownloads abgeschaltet.

`gpu/config.local.json` enthält deine privaten Einstellungen. Relative Modell- und Schlüsselpfade beziehen sich auf den Repository-Hauptordner; absolute Pfade sind ebenfalls möglich. Für Windows-Pfade in JSON entweder `/` oder doppelte Backslashes verwenden.

- Standard: `127.0.0.1:8002`, nur lokale Zugriffe, keine Firewallöffnung.
- Für den Mailserver im LAN: `host` auf die konkrete LAN-IP dieses Windows-PCs setzen; in `allowed_clients` die konkrete IP des Mailservers und die eigene Host-IP für den Start-Healthcheck eintragen. Keine Platzhalter, Subnetze oder `0.0.0.0` verwenden.
- Den zufälligen API-Schlüssel geschützt auf den Mailserver übertragen. Dort den GPU-Endpunkt `http://<GPU-HOST>:8002/v1/systemone` und eine lokale Kopie derselben Schlüsseldatei konfigurieren; siehe Haupt-README. Es werden keine Gmail-Zugangsdaten an den GPU-Dienst gesendet.
- Bei LAN-Nutzung `gpu/Enable-Laya-Firewall.ps1` einmal in einer PowerShell mit Administratorrechten ausführen. Die Regel gilt nur im privaten Windows-Netzwerkprofil und nur für konfigurierte Client-IPs. Nach IP- oder Portänderungen erneut ausführen. HTTP bleibt auf das vertrauenswürdige lokale Netz beschränkt; keine öffentliche Portfreigabe einrichten.

## Start, Stopp und Prüfung

```powershell
& .\gpu\Start-Laya-GPU.ps1
& .\gpu\Stop-Laya-GPU.ps1
```

Vor dem Beenden einen laufenden Mailstapel abschließen lassen. Start und Stopp betreffen den Dienst aus diesem Checkout. Startprotokolle liegen unter `runtime/gpu/`; der Start wartet nicht auf das vollständige Laden des Modells. Der Health-Endpunkt ist `http://<host>:<port>/health`. Der Server prüft zusätzlich zur Firewall jede Client-IP; Auswertungen benötigen den Bearer-Schlüssel mit mindestens 24 Zeichen.

Offline-Konfigurationsprüfung, ohne GPU, Modell- oder Mailzugriff:

```powershell
py -3.12 .\gpu\serve_gpu.py --config .\gpu\config.example.json --print-config
py -3.12 -m unittest discover -s .\gpu -p 'test_*.py'
```
