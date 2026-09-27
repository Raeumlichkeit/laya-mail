'use strict';

const $ = (id) => document.getElementById(id);
const draftKey = 'laya-account-setup';
const permissionText = { readonly: 'Freigabe: nur lesen', modify: 'Freigabe: lesen und Gmail-Labels setzen' };
let step = 1;
let prepared = null;
let busy = false;
let setupAvailable = false;

function showError(message = '') {
  $('setup-error').textContent = message;
  $('setup-error').hidden = !message;
  if (message) $('setup-error').scrollIntoView?.({ block: 'nearest' });
}

async function api(path, method = 'GET', body) {
  const options = { method, credentials: 'same-origin', cache: 'no-store' };
  if (body !== undefined) {
    options.headers = { 'Content-Type': 'application/json' };
    options.body = JSON.stringify(body);
  }
  let response;
  try { response = await fetch(path, options); }
  catch { throw new Error('Der Laya-Server ist gerade nicht erreichbar. Bitte prüfe die Verbindung und versuche es erneut.'); }
  if (response.status === 401) throw new Error('Deine Sitzung ist abgelaufen. Melde dich in der Übersicht erneut an und öffne danach die Einrichtung.');
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || 'Die Einrichtung konnte nicht abgeschlossen werden. Bitte erneut versuchen.');
  return data;
}

function setBusy(value) {
  busy = value;
  for (const id of ['prepare-account', 'grant-ready', 'back-to-account', 'back-to-grant', 'connect-account', 'connect-another', 'grant-file']) $(id).disabled = value;
  $('account-form').setAttribute('aria-busy', String(value));
  $('connect-form').setAttribute('aria-busy', String(value));
  $('prepare-account').textContent = value && step === 1 ? 'Wird vorbereitet …' : 'Weiter zur Google-Freigabe';
  $('connect-account').textContent = value && step === 3 ? 'Verbindung wird geprüft …' : 'Datei prüfen & verbinden';
}

function saveDraft() {
  try {
    // Store only the selected address/permission and the non-secret instructions, never a token.
    sessionStorage.setItem(draftKey, JSON.stringify({ email: $('setup-email').value, permission: selectedPermission(), step, prepared }));
  } catch { /* The flow also works with browser storage disabled. */ }
}

function selectedPermission() {
  return document.querySelector('input[name="permission"]:checked')?.value === 'modify' ? 'modify' : 'readonly';
}

function showStep(value, focus = true) {
  step = value;
  for (let number = 1; number <= 3; number++) {
    $('setup-step-' + number).hidden = value !== number;
    const marker = $('step-marker-' + number);
    marker.classList.toggle('complete', value === 4 || number < value);
    if (number === value) marker.setAttribute('aria-current', 'step');
    else marker.removeAttribute('aria-current');
  }
  $('setup-success').hidden = value !== 4;
  showError();
  if (value < 4) saveDraft();
  if (focus) $({ 1: 'step-one-heading', 2: 'step-two-heading', 3: 'step-three-heading', 4: 'success-heading' }[value]).focus();
}

function renderPrepared() {
  for (const id of ['grant-account', 'google-account', 'connection-email']) $(id).textContent = prepared.email;
  for (const id of ['grant-permission', 'connection-permission']) $(id).textContent = permissionText[prepared.permission];
  $('setup-command').textContent = prepared.command;
  $('grant-filename').textContent = prepared.token_filename;
  // Only link to Google's console, never an arbitrary destination from stored state.
  const consoleUrl = new URL('https://console.cloud.google.com/auth/audience');
  if (typeof prepared.project_id === 'string' && /^[a-z0-9-]+$/.test(prepared.project_id)) consoleUrl.searchParams.set('project', prepared.project_id);
  $('google-console-link').href = consoleUrl.href;
}

function validPrepared(value) {
  return value && typeof value === 'object' && typeof value.email === 'string'
    && value.email.length <= 254 && Object.hasOwn(permissionText, value.permission)
    && typeof value.command === 'string' && value.command.length < 8192
    && typeof value.token_filename === 'string' && value.token_filename.length < 200;
}

async function initialize() {
  $('setup-loading').hidden = false;
  $('setup-unavailable').hidden = true;
  showError();
  try {
    const config = await api('/api/accounts/setup');
    setupAvailable = config.available === true;
    $('setup-loading').hidden = true;
    if (!setupAvailable) {
      $('setup-unavailable').hidden = false;
      $('unavailable-reason').textContent = config.reason || 'Die Verbindungshilfe ist auf diesem Laya-Server noch nicht eingerichtet.';
      return;
    }
    let draft;
    try { draft = JSON.parse(sessionStorage.getItem(draftKey) || 'null'); } catch { draft = null; }
    if (draft && typeof draft === 'object') {
      if (typeof draft.email === 'string') $('setup-email').value = draft.email.slice(0, 254);
      for (const radio of document.querySelectorAll('input[name="permission"]')) radio.checked = radio.value === (draft.permission === 'modify' ? 'modify' : 'readonly');
      if (validPrepared(draft.prepared) && draft.prepared.email === $('setup-email').value && draft.prepared.permission === selectedPermission()) {
        prepared = draft.prepared;
        renderPrepared();
        showStep(draft.step === 3 ? 3 : draft.step === 2 ? 2 : 1, false);
        return;
      }
    }
    showStep(1, false);
  } catch (error) {
    $('setup-loading').hidden = true;
    $('setup-unavailable').hidden = false;
    showError(error.message);
  }
}

$('account-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (busy || !setupAvailable) return;
  const email = $('setup-email').value.trim().toLowerCase();
  const permission = selectedPermission();
  if (!$('setup-email').reportValidity()) return;
  setBusy(true);
  showError();
  try {
    const response = await api('/api/accounts/setup', 'POST', { email, permission });
    if (!validPrepared(response) || response.email !== email || response.permission !== permission) throw new Error('Der Server hat keine gültige Einrichtungsanleitung geliefert. Bitte erneut versuchen.');
    prepared = response;
    $('setup-email').value = email;
    renderPrepared();
    showStep(2);
  } catch (error) { showError(error.message); }
  finally { setBusy(false); }
});

$('copy-command').addEventListener('click', async () => {
  if (!prepared) return;
  try {
    await navigator.clipboard.writeText(prepared.command);
    $('copy-status').textContent = 'Befehl kopiert. Jetzt in PowerShell einfügen.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents($('setup-command'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    $('setup-command').focus();
    $('copy-status').textContent = 'Befehl markiert. Drücke Strg+C zum Kopieren.';
  }
});
$('grant-ready').addEventListener('click', () => { if (!busy && prepared) showStep(3); });
$('back-to-account').addEventListener('click', () => { if (!busy) showStep(1); });
$('back-to-grant').addEventListener('click', () => { if (!busy && prepared) showStep(2); });
$('setup-reload').addEventListener('click', initialize);

$('connect-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (busy || !prepared) return;
  const file = $('grant-file').files?.[0];
  if (!file) { showError('Bitte wähle zuerst die von der Google-Anmeldung erzeugte JSON-Datei aus.'); return; }
  if (file.size > 12000) { showError('Diese Datei ist zu groß für eine Google-Freigabe. Bitte wähle die erzeugte Token-Datei aus (höchstens 12 KB).'); return; }
  setBusy(true);
  showError();
  $('connection-status').textContent = 'Laya prüft die Adresse und Freigabe bei Google. Das kann einen Moment dauern.';
  $('connection-status').hidden = false;
  let token = null;
  try {
    try { token = JSON.parse(await file.text()); }
    catch { throw new Error('Die Datei enthält kein gültiges JSON. Bitte wähle die unveränderte Freigabedatei aus.'); }
    if (!token || typeof token !== 'object' || Array.isArray(token) || !token.refresh_token) throw new Error('Das ist keine Google-Freigabedatei. Benötigt wird die erzeugte Token-Datei, nicht client.json.');
    const result = await api('/api/accounts/connect', 'POST', { email: prepared.email, permission: prepared.permission, token });
    if (result.status !== 'connected' || result.email !== prepared.email) throw new Error('Die Verbindung konnte nicht eindeutig bestätigt werden. Bitte prüfe die Kontoauswahl in der Übersicht.');
    $('connected-email').textContent = result.email;
    $('connected-permission').textContent = result.permission === 'modify' ? 'Lesen und Labels setzen ist freigegeben. Die Sortierung steuerst du in der Übersicht.' : 'Das Postfach ist zum Lesen freigegeben. Zum Setzen von Gmail-Labels ist später eine zusätzliche Google-Freigabe nötig.';
    $('grant-file').value = '';
    try { sessionStorage.removeItem(draftKey); } catch { /* Optional browser storage. */ }
    showStep(4);
  } catch (error) { showError(error.message); }
  finally {
    token = null;
    $('connection-status').hidden = true;
    setBusy(false);
  }
});

$('connect-another').addEventListener('click', () => {
  if (busy) return;
  prepared = null;
  $('account-form').reset();
  $('grant-file').value = '';
  $('copy-status').textContent = '';
  showStep(1);
  $('setup-email').focus();
});

initialize();
