# Laya Mail

Lokale Weboberfläche zum Sortieren von Gmail-Nachrichten mit [Laya](https://github.com/NandhaKishorM/laya). Sie zeigt Vorschläge, unsichere Fälle und manuelle Korrekturen. Ein kleiner Ubuntu-Server kann Dashboard und Daten verwalten, während ein Windows-PC mit NVIDIA-GPU die Auswertung übernimmt.

## Funktionen

- Pro Durchlauf insgesamt 1–100 Mails aus den ausgewählten Posteingängen laden; neue Stapel überspringen bereits angezeigte Nachrichten getrennt je Konto.
- Inhalt: **Rechnung**, **Allgemeine Information**, **Werbung**, **Sonstiges**.
- **Wichtig** und **Spamverdacht** werden unabhängig davon bewertet.
- **⚡ Live GPU-Statusanzeige**: Header-Badge zeigt in Echtzeit den Verbindungsstatus und die Latenz der NVIDIA-Grafikkarte (z. B. `RTX 4070 Ti verbunden · 3 ms`), mit automatischem Fallback auf CPU.
- **✉️ Multi-Account-Modell**: Beliebig viele E-Mail-Adressen über `accounts.json` verwalten; Abfrage einzelner Postfächer oder aller Konten kombiniert direkt im Dashboard.
- **🏷️ Postfach-Sortier-Toggle**: Optionales automatisches Zuweisen von Labels (`Rechnung`, `Werbung`, `Info`, `Laya/Wichtig`, `Spamverdacht`) im Gmail-Postfach – absolut nicht-destruktiv (Mails bleiben im Posteingang, kein Löschen/Archivieren). `Laya/Wichtig` vermeidet die Kollision mit Gmails reserviertem Systemlabel „Wichtig“.
- **🔄 Live-Modus (Autopilot)**: Prüft die ausgewählten aktiven Postfächer alle 3 Minuten in Stapeln von bis zu 10 Mails. Er nutzt denselben Ablauf, die GPU-Auswahl mit CPU-Fallback und die Ergebnisübersicht wie manuelle Durchläufe. Gmail-Labels werden nur bei eingeschalteter Postfach-Sortierung geändert; während eines laufenden Stapels wartet der nächste Live-Durchlauf.
- **Speed-Annotation & Tastaturkürzel**: Schnelles Bewerten mit `1` (Stimmt), `2` (Korrigieren), `3` (Überspringen) und `J`/`K` zur Navigation.
- **Active Learning**: Eigener Filter für „Wackelkandidaten“ (knappe Modellentscheidungen) zum gezielten Anlernen an Grenzfällen.
- **Signalwort-Hervorhebung**: Automatische Markierung von Signalbegriffen (Rechnungsbegriffe, Dringlichkeit, Phishing/Spam) im Mailtext.
- **Datensatz-Verteilung & Split-Export**: Visuelle Klassen-Balance im Dashboard und deterministischer Export mit ungefähr 80% Training und 20% Validierung. Identische Eingabetexte werden immer derselben Gruppe zugeordnet, auch über Konten hinweg. Bestehende Exporte bleiben unverändert; für einen neuen Split beide Dateien gemeinsam neu exportieren.
- **Lokale Fine-Tuning-Pipeline**: 1-Klick-Skripte für GPU-Training (`gpu/train_laya.py`) und Modell-Evaluation (`gpu/evaluate_model.py`) auf dem Windows-PC.
- GPU vor jedem Stapel automatisch auswählen; bei nicht verfügbarem GPU-Dienst CPU verwenden.

Im Standardmodus liest Laya Mail mit `gmail.readonly`. Bei aktiviertem Sortier-Toggle werden ausschließlich Klassifizierungs-Labels vergeben (`gmail.modify`). Das Programm versendet und löscht unter keinen Umständen E-Mails; alle Nachrichten verbleiben sicher im Posteingang. Anhänge und verlinkte Seiten werden nicht gelesen. Modell-Scores sind keine gemessenen Zuverlässigkeitsquoten.

Beim Aktualisieren werden die fünf verwalteten Gmail-Labels abgeglichen: Überholte Zuordnungen werden entfernt, andere Labels bleiben erhalten. Vor Schreibzugriffen wird geprüft, ob die Google-Anmeldung zum jeweiligen Postfach gehört. Bewertungen und Ladehistorie bleiben je Konto getrennt, auch wenn Nachrichten dieselbe ID haben.

## Aufbau

```text
mail/       Dashboard, Gmail-Abruf, Label-Manager, Multi-Account, Watcher, Weboberfläche und Tests
gpu/        Optionaler Windows-GPU-Dienst, 1-Click-Trainingsskripte und Start-/Stoppskripte
deploy/     Vorlagen für systemd und lokale Konfiguration
runtime/    Lokale Laufzeitdaten (nicht im Repository)
```

Das Repository enthält den Anwendungscode, keine Postfachdaten, Zugangsschlüssel, Python-Umgebungen oder Modellgewichte. Laya wird als separate Abhängigkeit installiert; dessen Quellcode und Gewichte werden hier nicht mitgeliefert.

## Dashboard einrichten

Voraussetzungen: Python 3.12 oder neuer, ein eigener Google-Desktop-OAuth-Client mit aktivierter Gmail API und ein laufender Laya-Server mit geladenem Modell `multilingual`. Getestet wurde Laya 0.3.20. Laya muss über eine konkrete lokale IP-Adresse unter `/v1/systemone` erreichbar sein und einen API-Schlüssel verlangen. Die [Laya-Dokumentation](https://nandhakishorm.github.io/laya/) beschreibt dessen Installation.

Im geklonten Repository unter Ubuntu:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r mail/requirements.txt
install -d -m 700 runtime/private
umask 077
.venv/bin/python -c "from pathlib import Path; import secrets; Path('runtime/private/dashboard-key').write_text(secrets.token_urlsafe(32))"
```

Den letzten Befehl nur beim ersten Einrichten ausführen: Er erzeugt den Anmeldecode des Dashboards. Den API-Schlüssel des bestehenden Laya-Servers in `runtime/private/laya-api-key` speichern und die heruntergeladene Google-Clientdatei als `runtime/private/client.json` ablegen. Unter Windows vor dem Speichern der Dateien die Verzeichnisrechte auf den eigenen Benutzer und SYSTEM beschränken.

Die Google-Freigabe auf dem Rechner durchführen, auf dem auch der Browser geöffnet wird:

```sh
.venv/bin/python mail/authorize_gmail.py \
  --client runtime/private/client.json \
  --token runtime/private/token.json \
  --project-id DEINE_GOOGLE_PROJEKT_ID
```

Die im Terminal angezeigte Google-Adresse persönlich im Browser öffnen und die Leseberechtigung freigeben. Der Helfer überschreibt keine bestehende Tokendatei. Bei getrennten Rechnern `token.json` anschließend über eine geschützte Verbindung auf den Dashboard-Rechner übertragen. Für eine bereits eingerichtete Installation können die vorhandenen privaten Dateien weiterverwendet werden; keine neue Google-Freigabe allein wegen dieses Repositorys erforderlich.

Für Gmail-Labels benötigt jedes betroffene Konto eine Freigabe mit `--modify`. Eine vorhandene reine Lesefreigabe reicht dafür nicht. Weitere Konten zuerst im Dashboard hinzufügen; den dort angegebenen Token-Dateinamen für die jeweilige Autorisierung verwenden.

Starten, dabei Adresse und Modell-URL anpassen:

```sh
.venv/bin/python mail/dashboard.py \
  --data runtime/private \
  --account you@gmail.com \
  --host 127.0.0.1 \
  --port 8001 \
  --laya-url http://127.0.0.1:8000/v1/systemone
```

Dann `http://127.0.0.1:8001` öffnen und den Code aus `runtime/private/dashboard-key` eingeben. Für andere Geräte im Heimnetz `--host` auf die konkrete LAN-IP des Servers setzen und exakt diese Adresse im Browser verwenden. Das Dashboard ist für ein vertrauenswürdiges Heimnetz gedacht, nicht für eine öffentliche Portfreigabe; HTTP verschlüsselt die Verbindung nicht.

## GPU auf einem zweiten Rechner

Die [GPU-Anleitung](gpu/README.md) beschreibt die Windows-Einrichtung. Beispieladressen in diesem Repository sind Platzhalter und müssen zur eigenen Umgebung passen.

Auf dem Dashboard-Rechner `deploy/gpu-backend.example.json` als `runtime/private/gpu-backend.json` ablegen und die GPU-IP anpassen. Den gleichen GPU-API-Schlüssel wie auf dem Windows-PC als `runtime/private/gpu-api-key` speichern. Vor dem nächsten Stapel prüft das Dashboard `/health`; bei erreichbarem CUDA-Modell nutzt es die GPU. Während eines laufenden Stapels wird nicht automatisch auf ein anderes Gerät gewechselt.

In der ursprünglichen Installation brauchten zehn identische Mail-Eingaben auf einer RTX 4070 Ti 1,1 Sekunden statt 45,3 Sekunden auf einem Mac mini von 2014. Ein vollständiger Stapel mit 100 Nachrichten einschließlich Gmail-Abruf dauerte 26 Sekunden statt zuvor 8 Minuten 43 Sekunden. Das sind Messwerte dieser Installation, keine Garantie für andere Geräte oder Postfächer.

## Als Ubuntu-Dienst betreiben

Die Vorlagen in `deploy/` gehen von folgenden Pfaden aus:

- Repository und virtuelle Umgebung: `/opt/laya-mail` und `/opt/laya-mail/.venv`.
- Eigener Systembenutzer und Gruppe: `laya-mail`.
- Privates Datenverzeichnis: `/var/lib/laya-mail`, Eigentümer `laya-mail`, Modus `0700`; enthaltene Geheimnisse Modus `0600`.
- Konfiguration: `/etc/laya-mail/dashboard.env`, aus `deploy/dashboard.env.example` kopieren und anpassen.

Die Servicevorlage als `/etc/systemd/system/laya-mail-dashboard.service` installieren, anschließend `sudo systemctl daemon-reload` und `sudo systemctl enable --now laya-mail-dashboard` ausführen. Die Einheit erlaubt Schreibzugriffe nur im privaten Datenverzeichnis. Vor Umstellungen laufende Stapel abschließen lassen und das private Datenverzeichnis sichern.

`deploy/run-pilot.sh.example` ist ein optionaler manueller CPU-Durchlauf. Unter dem Dienstbenutzer ausführen, `MAIL_ACCOUNT` setzen und bei Bedarf Pfade und `LAYA_URL` anpassen.

## Entwicklung und Tests

Die Tests verwenden synthetische Nachrichten, keine Google-Anmeldung und kein geladenes Modell:

```sh
.venv/bin/python -m unittest discover -s mail -v
.venv/bin/python -m unittest discover -s gpu -v
node mail/web/check-ui.cjs
```

Node.js wird nur für den UI-Test benötigt; die Oberfläche braucht keinen Build-Schritt. Geprüft werden unter anderem Lesebeschränkungen, OAuth-Konfiguration, Anmeldung, Herkunftsprüfung, Stapelgrößen, Pagination, GPU-Auswahl, gespeicherte Korrekturen, Trainingsdatenexport und sicheres Anzeigen von Mailtext.

## Daten und Grenzen

`pilot-*.json` enthält Mailauszüge und die genaue begrenzte Modelleingabe. `feedback.json` enthält menschliche Bewertungen samt Eingabe. Beides gehört ausschließlich ins private Datenverzeichnis. `.gitignore` schließt Laufzeitdaten, Schlüssel, lokale Konfiguration, Modellgewichte und Trainingsdateien aus; vor jedem Commit trotzdem die vorgemerkten Dateien prüfen.

Sehr lange Texte werden gekürzt; das Modell kann zusätzlich sein Tokenlimit anwenden. Unsichere oder unvollständige Eingaben sollten geprüft oder übersprungen werden. Ein späteres Fine-Tuning benötigt einen eigenen Trainingsablauf und getrennte Prüfdaten. Die Lebensdauer der Google-Freigabe hängt unter anderem vom OAuth-Veröffentlichungsstatus ab; die App kann bei abgelaufener Freigabe eine erneute Anmeldung benötigen.
