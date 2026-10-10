$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $repoRoot '.venv-gpu\Scripts\python.exe'
$scriptPath = Join-Path $PSScriptRoot 'serve_gpu.py'
$configPath = Join-Path $PSScriptRoot 'config.local.json'
$configJson = & $pythonPath $scriptPath --config $configPath --print-config
if ($LASTEXITCODE -ne 0) { throw 'GPU-Konfiguration ungültig; Firewall unverändert.' }
$config = $configJson | ConvertFrom-Json
if ([Net.IPAddress]::IsLoopback([Net.IPAddress]::Parse($config.host))) {
    Write-Output 'Loopback-Konfiguration: keine Firewallfreigabe nötig.'
    exit 0
}
$remoteAddresses = @($config.allowed_clients | Where-Object {
    -not [Net.IPAddress]::IsLoopback([Net.IPAddress]::Parse($_))
})
if ($remoteAddresses.Count -eq 0) { throw 'Keine konkrete vertrauenswürdige Client-IP für die LAN-Freigabe konfiguriert.' }
$interfaceAliases = @(Get-NetIPAddress -IPAddress $config.host -ErrorAction Stop |
    Select-Object -ExpandProperty InterfaceAlias -Unique)
if ($interfaceAliases.Count -ne 1) { throw 'Die konfigurierte Host-IP gehört zu keinem eindeutigen Netzwerkadapter.' }
$ruleName = 'Laya-Mail-GPU'
Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -Name $ruleName -DisplayName 'Laya Mail GPU (konfigurierte Clients)' `
    -Direction Inbound -Action Allow -Protocol TCP -LocalPort $config.port `
    -LocalAddress $config.host -RemoteAddress $remoteAddresses `
    -InterfaceAlias $interfaceAliases[0] `
    -Profile Private | Out-Null
Write-Output 'Laya GPU: Nur konfigurierte Client-IP-Adressen im privaten Netzwerk sind freigegeben.'
