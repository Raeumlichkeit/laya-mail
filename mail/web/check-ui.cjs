// Offline state checks. Run: node mail/web/check-ui.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Node {
  constructor(tag = 'div') {
    Object.assign(this, { tag, tagName: tag.toUpperCase(), children: [], dataset: {}, attributes: {}, handlers: {}, value: '', textContent: '', parentElement: null, className: '' });
  }
  set innerHTML(_) { throw new Error('Unsafe HTML insertion'); }
  get value() { return this._value; }
  set value(value) { this._value = value; if (this.id === 'progress') this.attributes.value = String(value); }
  append(...nodes) { for (const node of nodes) { node.parentElement = this; this.children.push(node); } }
  replaceChildren(...nodes) { this.children.forEach((node) => { node.parentElement = null; }); this.children = []; this.append(...nodes); }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(key, handler) { this.handlers[key] = handler; }
  focus() { context.document.activeElement = this; }
  scrollIntoView() {}
  reportValidity() { this.validityReported = true; return false; }
  matches(selector) { return selector.startsWith('.') ? this.className.split(/\s+/).includes(selector.slice(1)) : this.tag === selector; }
  closest(selectors) {
    for (let node = this; node; node = node.parentElement) {
      if (selectors.split(',').some((selector) => node.matches(selector.trim()))) return node;
    }
    return null;
  }
  querySelector(selector) { return query([this], selector)[0] || null; }
  get classList() {
    return { toggle: (name, enabled) => { const names = this.className.split(/\s+/).filter((value) => value && value !== name); if (enabled) names.push(name); this.className = names.join(' '); } };
  }
}
function walk(node) { return [node, ...node.children.flatMap(walk)]; }
function query(roots, selector) {
  const parts = selector.split(' ');
  return roots.flatMap(walk).filter((node) => node.matches(parts.at(-1)) && (parts.length === 1 || node.parentElement?.closest(parts[0])));
}
const nodes = new Map();
const get = (id) => {
  const existing = [...nodes.values()].flatMap(walk).find((node) => node.id === id);
  if (existing) return existing;
  if (id.startsWith('card-')) return null;
  if (!nodes.has(id)) { const node = new Node(); node.id = id; nodes.set(id, node); }
  return nodes.get(id);
};
get('mail-limit').value = '20';
const filters = ['all', 'unreviewed', 'uncertain', 'review', 'rechnung', 'werbung', 'wichtig', 'spam'].map((value) => {
  const node = new Node('button'); node.dataset.filter = value; return node;
});
const hostile = '<img src=x onerror=alert(1)>';
const base = { report_id: 'synthetic-report', limit: 20, created_utc: '2026-09-26T10:00:00Z', messages: [
  { message_id: 'a', snapshot_id: 'snapshot-a', subject: hostile, sender: 'test@example.invalid', status: 'suggestion', training_ready: true, model_input: { state: hostile }, suggestions: { inhalt: 'rechnung', wichtig: 'ja', spamverdacht: 'nein' }, needs_review: true },
  { message_id: 'b', snapshot_id: null, subject: 'Älterer Bericht', status: 'suggestion', training_ready: false, suggestions: { inhalt: 'info', wichtig: 'nein', spamverdacht: 'nein' } },
  { message_id: 'c', snapshot_id: 'snapshot-c', subject: 'Fehler', status: 'error', training_ready: true },
] };
let serverReport = structuredClone(base);
let revision = 0;
let reportReads = 0;
let failSave = false;
let lastPayload;
let runPayloads = [];
let stopCalls = 0;
let serverStatus = {
  phase: 'complete', finished_at: base.created_utc, batch_mode: 'latest',
  accounts: { selected: 'all', accounts: [{ id: 'first', email: 'first@example.invalid', token_exists: true }, { id: 'second', email: 'second@example.invalid', token_exists: false }] },
  settings: { live_mode: false, sort_inbox: false },
};
let controlPayload;
const context = {
  document: {
    getElementById: get, createElement: (tag) => new Node(tag),
    createTextNode: (text) => { const node = new Node('#text'); node.textContent = text; return node; },
    querySelectorAll: (selector) => selector === '[data-filter]' ? filters : query([...nodes.values()], selector),
    querySelector: (selector) => query([...nodes.values()], selector)[0] || null,
    activeElement: new Node('body'), handlers: {},
    addEventListener(type, handler) { this.handlers[type] = handler; },
  },
  Intl, Date, JSON, Math, Number, String, Boolean, Object, Error,
  setTimeout: () => 1, clearTimeout: () => {}, location: { reload: () => { throw new Error('Unexpected reload'); } },
  fetch: async (url, options) => {
    let data;
    if (url === '/api/status') data = { ...serverStatus, feedback_revision: revision };
    else if (url === '/api/report') { reportReads++; data = structuredClone(serverReport); }
    else if (url === '/api/feedback') {
      assert.equal(get('run').disabled, true, 'Latest action must be locked while saving');
      assert.equal(get('run-next').disabled, true, 'Next action must be locked while saving');
      assert.equal(get('mail-limit').disabled, true, 'Count must be locked while saving');
      lastPayload = JSON.parse(options.body);
      assert.equal(options.credentials, 'same-origin');
      if (failSave) return { status: 500, ok: false, json: async () => ({ error: 'Synthetischer Speicherfehler' }) };
      const row = serverReport.messages.find((mail) => mail.message_id === lastPayload.message_id && (mail.account_email || '') === (lastPayload.account_email || ''));
      row.review = { action: lastPayload.action, labels: lastPayload.action === 'corrected' ? lastPayload.labels : lastPayload.action === 'accepted' ? row.suggestions : null, training_eligible: lastPayload.action !== 'skipped' };
      revision++;
      data = structuredClone(serverReport);
    } else if (url === '/api/run') {
      assert.equal(get('run').disabled, true, 'Latest action must be locked while starting');
      assert.equal(get('run-next').disabled, true, 'Next action must be locked while starting');
      assert.equal(get('mail-limit').disabled, true, 'Count must be locked while starting');
      assert.equal(get('run-all').disabled, true, 'All action must be locked while starting');
      assert.equal(get('account-select').disabled, true, 'Account selection must be locked while starting');
      const payload = JSON.parse(options.body);
      runPayloads.push(payload);
      serverStatus = { phase: 'running', finished_at: null, processed: 0, total: payload.limit ?? null, requested_limit: payload.limit ?? null, batch_mode: payload.mode, settings: { live_mode: false, sort_inbox: false } };
      data = { status: 'started' };
    } else if (url === '/api/run/stop') {
      assert.deepEqual(JSON.parse(options.body), {});
      assert.equal(get('stop-run').disabled, true, 'Stop is locked while its request is pending');
      assert.equal(vm.runInContext('status.phase', context), 'running', 'A stop request must not locally finish a run');
      stopCalls++;
      serverStatus.stop_requested = true;
      data = { status: 'stopping' };
    } else if (url === '/api/labels/apply-current') {
      assert.equal(get('run-all').disabled, true, 'All-run is locked while label application is pending');
      assert.equal(get('account-select').disabled, true);
      data = { status: 'ok', applied_count: serverReport.messages.length, total_messages: serverReport.messages.length, errors: [] };
    } else if (url === '/api/accounts/select') {
      controlPayload = JSON.parse(options.body);
      serverStatus.accounts.selected = controlPayload.selected;
      data = serverStatus.accounts;
    } else if (url === '/api/settings') {
      controlPayload = JSON.parse(options.body);
      Object.assign(serverStatus.settings, controlPayload);
      data = serverStatus.settings;
    } else throw new Error('Unexpected route ' + url);
    return { status: 200, ok: true, json: async () => structuredClone(data) };
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8'), context);

async function press(key, target = new Node('body'), options = {}) {
  context.document.activeElement = target;
  const event = { key, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...options };
  context.document.handlers.keydown(event);
  await new Promise(setImmediate);
  return event;
}

(async () => {
  await new Promise(setImmediate);
  for (const [code, expected] of [
    ['gmail_rate_limited', 'Gmail begrenzt derzeit die Anfragen. Bitte später erneut versuchen.'],
    ['gmail_quota_exceeded', 'Das Gmail-Tageskontingent ist ausgeschöpft. Bitte nach dessen Zurücksetzung erneut versuchen.'],
    ['gmail_temporary_error', 'Gmail meldet einen vorübergehenden Fehler. Bitte später erneut versuchen.'],
    ['gmail_message_not_found', 'Diese Nachricht ist in Gmail nicht mehr verfügbar.'],
    ['gmail_authentication_failed', 'Die Gmail-Anmeldung ist nicht mehr gültig. Bitte erneut anmelden.'],
    ['gmail_permission_denied', 'Gmail hat den Zugriff auf diese Nachricht verweigert. Bitte die Anmeldung prüfen.'],
    ['gmail_request_failed', 'Diese Nachricht konnte nicht von Gmail geladen werden.'],
    [hostile, 'Diese Nachricht konnte nicht ausgewertet werden.'],
    ['toString', 'Diese Nachricht konnte nicht ausgewertet werden.'],
    [undefined, 'Diese Nachricht konnte nicht ausgewertet werden.'],
  ]) {
    const card = context.renderMail({ message_id: 'synthetic-error', status: 'error', error: code });
    assert.equal(card.querySelector('.mail-error').textContent, expected, 'Mail errors use fixed German messages, never raw server text');
  }
  assert.equal(get('message-count').textContent, '3');
  assert.equal(get('run-label').textContent, 'Neueste 20 erneut prüfen');
  assert.equal(get('run-next').disabled, false);
  assert.equal(get('mail-limit').disabled, false);
  assert.equal(get('message-limit-note').textContent, 'bis zu 20 je Durchlauf');
  assert.equal(get('processing-location').textContent, 'Auswertung: Mac mini · CPU', 'Legacy status defaults to the Mac CPU');
  assert.equal(get('processing-note').hidden, true);
  serverStatus.model = { name: 'Laya multilingual', device: 'cuda', host: 'Hauptrechner' };
  await context.refreshStatus();
  assert.equal(get('processing-location').textContent, 'Auswertung: Hauptrechner · RTX 4070 Ti');
  serverStatus.model = { name: 'Laya multilingual', device: 'cpu', host: 'Mac mini' };
  serverStatus.processing_note = 'GPU nicht bereit. ' + hostile;
  await context.refreshStatus();
  assert.equal(get('processing-location').textContent, 'Auswertung: Mac mini · CPU');
  assert.equal(get('processing-note').hidden, false);
  assert.equal(get('processing-note').textContent, 'GPU nicht bereit. ' + hostile, 'Fallback note is literal text');
  serverStatus.processing_note = null;
  await context.refreshStatus();
  assert.equal(get('processing-note').hidden, true, 'A cleared fallback note disappears');
  assert.equal(get('account-select').children.length, 3);
  assert.equal(get('account-select').children[2].textContent, 'second@example.invalid ⚠️');
  get('account-select').value = 'second';
  await get('account-select').handlers.change({ target: get('account-select') });
  assert.deepEqual(controlPayload, { selected: 'second' });
  assert.equal(get('account-select').value, 'second');
  get('toggle-live-mode').checked = true;
  await get('toggle-live-mode').handlers.change({ target: get('toggle-live-mode') });
  assert.deepEqual(controlPayload, { live_mode: true });
  context.renderStatus();
  assert.equal(get('toggle-live-mode').checked, true, 'A local render must preserve the successful settings update');
  assert.equal(get('live-activity').hidden, false);
  get('toggle-sort-inbox').checked = true;
  await get('toggle-sort-inbox').handlers.change({ target: get('toggle-sort-inbox') });
  assert.deepEqual(controlPayload, { sort_inbox: true });
  context.renderStatus();
  assert.equal(get('toggle-sort-inbox').checked, true);
  assert.equal(get('mode-badge-text').textContent, 'Gmail sortieren (Labels aktiv)');
  assert.ok(walk(get('messages')).some((node) => node.textContent === hostile));
  const legacy = get('messages').children[1];
  assert.equal(walk(legacy).find((node) => node.textContent === 'Stimmt').disabled, true);
  assert.equal(walk(legacy).find((node) => node.textContent === 'Korrigieren').disabled, true);
  assert.equal(walk(legacy).find((node) => node.textContent === 'Überspringen').disabled, false);

  context.startCorrection(base.messages[0]);
  assert.equal(get('run').disabled, true, 'Latest action must be locked during correction');
  assert.equal(get('run-next').disabled, true, 'Next action must be locked during correction');
  assert.equal(get('mail-limit').disabled, true, 'Count must be locked during correction');
  const select = walk(get('messages')).find((node) => node.tag === 'select' && node.name === 'inhalt');
  select.value = 'werbung'; select.handlers.change();
  const readsBefore = reportReads;
  revision++;
  await context.refreshStatus();
  assert.equal(reportReads, readsBefore, 'Polling must not replace an open draft');
  assert.equal(vm.runInContext('editing.labels.inhalt', context), 'werbung');
  context.selectFilter('werbung');
  get('search').value = 'unpassender Suchbegriff';
  get('return-correction').handlers.click();
  assert.equal(get('search').value, '');
  assert.equal(walk(get('messages')).find((node) => node.tag === 'select' && node.name === 'inhalt').value, 'werbung');

  failSave = true;
  await context.saveFeedback(base.messages[0], 'corrected');
  assert.equal(vm.runInContext('editing.labels.inhalt', context), 'werbung', 'Failed save must retain draft');
  failSave = false;
  await context.saveFeedback(base.messages[0], 'corrected');
  assert.equal(lastPayload.report_id, 'synthetic-report');
  assert.equal(lastPayload.snapshot_id, 'snapshot-a');
  assert.equal(lastPayload.labels.inhalt, 'werbung');
  assert.equal(vm.runInContext('editing', context), null);
  assert.equal(serverReport.messages[0].suggestions.inhalt, 'rechnung', 'Original suggestion remains intact');
  context.selectFilter('werbung'); assert.equal(get('messages').children.length, 1);
  context.selectFilter('review'); assert.equal(get('messages').children.length, 1);
  context.selectFilter('unreviewed'); assert.equal(get('messages').children.length, 2);

  context.selectFilter('all');
  context.startCorrection(base.messages[2]);
  const payloadBefore = lastPayload;
  await context.saveFeedback(base.messages[2], 'corrected');
  assert.equal(lastPayload, payloadBefore, 'Incomplete labels must not be saved');
  assert.equal(vm.runInContext('editing.labels.inhalt', context), '');
  walk(get('messages')).find((node) => node.textContent === 'Abbrechen').handlers.click();
  await new Promise(setImmediate);
  await context.saveFeedback(base.messages[1], 'skipped');
  assert.equal(lastPayload.snapshot_id, null);
  assert.equal(lastPayload.action, 'skipped');
  assert.equal(Object.hasOwn(lastPayload, 'labels'), false);
  context.selectFilter('unreviewed'); assert.equal(get('messages').children.length, 1);
  context.selectFilter('all'); get('search').value = 'nicht vorhanden'; context.renderMessages();
  assert.equal(get('empty').hidden, false);

  for (const invalid of ['', '0', '101', '2.5', 'abc']) {
    get('mail-limit').value = invalid;
    await get('run').handlers.click();
    await get('run-next').handlers.click();
    assert.equal(runPayloads.length, 0, 'Invalid counts must never start a request');
  }
  assert.equal(get('mail-limit').validityReported, true);
  get('mail-limit').value = '37';
  get('mail-limit').handlers.input();
  assert.equal(get('run-label').textContent, 'Neueste 37 erneut prüfen');
  assert.equal(get('run-next-label').textContent, 'Nächste 37 Mails laden');
  await context.refreshStatus();
  assert.equal(get('mail-limit').value, '37', 'Status polling must preserve user count');
  await get('run').handlers.click();
  assert.deepEqual(runPayloads[0], { mode: 'latest', limit: 37 });
  assert.equal(get('run').disabled, true);
  assert.equal(get('run-next').disabled, true);
  assert.equal(get('mail-limit').disabled, true);
  assert.equal(get('run-label').textContent, 'Prüft bis zu 37 Mails …');
  await get('run-next').handlers.click();
  assert.equal(runPayloads.length, 1, 'A running request blocks both run actions');
  serverStatus = { phase: 'complete', finished_at: '2026-09-26T10:01:00Z', batch_mode: 'latest' };
  await context.refreshStatus();
  const originalReviews = JSON.stringify(serverReport.messages.map((mail) => mail.review));
  get('mail-limit').value = '12';
  get('mail-limit').handlers.input();
  await get('run-next').handlers.click();
  assert.deepEqual(runPayloads[1], { mode: 'next', limit: 12 });
  assert.equal(get('run-next-label').textContent, 'Lädt bis zu 12 Mails …');
  assert.equal(get('search').value, '');
  assert.equal(vm.runInContext('filter', context), 'unreviewed');
  assert.equal(JSON.stringify(serverReport.messages.map((mail) => mail.review)), originalReviews);
  serverReport = { report_id: 'empty-next', limit: 12, created_utc: '2026-09-26T10:02:00Z', batch_mode: 'next', messages: [], summary: { training_examples: 1 } };
  serverStatus = { phase: 'complete', finished_at: serverReport.created_utc, batch_mode: 'next' };
  await context.refreshStatus();
  assert.equal(get('empty-heading').textContent, 'Keine weiteren Mails im Posteingang');
  assert.equal(get('training-count').textContent, '1 Trainingsbeispiel insgesamt');
  assert.equal(get('run-next').disabled, false);
  assert.equal(get('mail-limit').disabled, false);
  assert.equal(get('message-limit-note').textContent, 'bis zu 12 je Durchlauf');

  serverReport = structuredClone(base);
  context.renderReport(serverReport);
  get('search').value = '';
  context.selectFilter('all');
  vm.runInContext('selectedMailKey = null; renderMessages();', context);
  const originalCards = [...get('messages').children];
  const otherDetails = originalCards[1].querySelector('.model-details');
  otherDetails.open = true;
  originalCards[1].handlers.click();
  assert.equal(get('messages').children[1], originalCards[1], 'Card selection must not replace native controls');
  assert.equal(otherDetails.open, true, 'Card selection must preserve open details');
  vm.runInContext('selectedMailKey = null; renderMessages();', context);

  for (const tag of ['button', 'a', 'summary']) {
    const nativeControl = new Node(tag);
    const nested = new Node('span'); nativeControl.append(nested);
    for (const target of [nativeControl, nested]) {
      for (const key of ['Enter', ' ']) {
        const before = lastPayload;
        const event = await press(key, target);
        assert.equal(event.defaultPrevented, false, `${key} on ${tag} must retain its native action`);
        assert.equal(lastPayload, before, `${key} on ${tag} must not review mail`);
      }
    }
  }
  for (const tag of ['input', 'select', 'textarea']) {
    const before = lastPayload;
    assert.equal((await press('1', new Node(tag))).defaultPrevented, false);
    assert.equal(lastPayload, before, 'Typing in a form field must not review mail');
  }
  const editable = new Node('div'); editable.isContentEditable = true;
  for (const [target, options] of [[editable, {}], [new Node('body'), { ctrlKey: true }], [new Node('body'), { metaKey: true }], [new Node('body'), { defaultPrevented: true }]]) {
    const before = lastPayload;
    await press('s', target, options);
    assert.equal(lastPayload, before, 'Native editing/browser shortcuts must not skip mail');
  }
  assert.equal((await press('j')).defaultPrevented, true);
  assert.equal(vm.runInContext('selectedMailKey', context), context.mailKey(base.messages[1]));
  await press('K');
  assert.equal(vm.runInContext('selectedMailKey', context), context.mailKey(base.messages[0]));
  await press('ArrowDown'); await press('ArrowUp');
  assert.equal(vm.runInContext('selectedMailKey', context), context.mailKey(base.messages[0]));
  const inputDetails = get('messages').children[0].querySelector('.input-details');
  await press(' ');
  assert.equal(inputDetails.open, true);
  await press('1'); assert.equal(lastPayload.action, 'accepted');
  await press('2'); assert.equal(vm.runInContext('editing.message_id', context), 'a');
  await press('Escape'); assert.equal(vm.runInContext('editing', context), null);
  await press('c'); assert.equal(vm.runInContext('editing.message_id', context), 'a');
  await press('Escape');
  await press('3'); assert.equal(lastPayload.action, 'skipped');
  await press('Enter'); assert.equal(lastPayload.action, 'accepted');

  const first = { ...structuredClone(base.messages[0]), account_email: 'first@example.invalid' };
  const second = { ...structuredClone(base.messages[0]), account_email: 'second@example.invalid' };
  serverReport = { ...structuredClone(base), messages: [first, second] };
  context.renderReport(serverReport);
  const cards = get('messages').children;
  assert.notEqual(cards[0].id, cards[1].id, 'Same message IDs in different accounts need unique DOM IDs');
  cards[1].handlers.click();
  await press('2');
  assert.equal(vm.runInContext('editing.account_email', context), 'second@example.invalid');
  assert.equal(query([get('messages')], '.correction-form').length, 1, 'Only the selected account opens a correction form');
  await context.saveFeedback(second, 'corrected');
  assert.equal(lastPayload.account_email, 'second@example.invalid');
  assert.equal(first.review, undefined);
  assert.equal(second.review.action, 'corrected');
  await press('1'); assert.equal(lastPayload.account_email, 'second@example.invalid');
  await press('3'); assert.equal(lastPayload.account_email, 'second@example.invalid');
  await get('run-rescan').handlers.click();
  assert.deepEqual(runPayloads.at(-1), { mode: 'rescan', limit: 12 });
  assert.equal(get('stop-run').hidden, true, 'Stop is only available for an all-mail run');
  await get('stop-run').handlers.click();
  assert.equal(stopCalls, 0);

  serverStatus = { phase: 'complete', finished_at: '2026-09-27T10:00:00Z', batch_mode: 'rescan' };
  await context.refreshStatus();
  await get('btn-apply-current').handlers.click();
  serverStatus.applying_labels = true;
  await context.refreshStatus();
  assert.equal(get('btn-apply-current').disabled, true);
  assert.equal(get('run-all').disabled, true);
  assert.equal(get('account-select').disabled, true);
  const beforeAll = runPayloads.length;
  await get('run-all').handlers.click();
  assert.equal(runPayloads.length, beforeAll, 'An all-mail run cannot start during label application');
  serverStatus.applying_labels = false;
  await context.refreshStatus();
  get('mail-limit').value = '';
  get('mail-limit').validityReported = false;
  await get('run-all').handlers.click();
  assert.deepEqual(runPayloads.at(-1), { mode: 'all' });
  assert.equal(get('mail-limit').validityReported, false, 'All-mail mode bypasses the batch limit field');
  assert.equal(get('status-heading').textContent, 'Alle Posteingänge werden gelabelt');
  assert.equal(get('status-description').textContent, '0 Mails verarbeitet');
  assert.equal(Object.hasOwn(get('progress').attributes, 'value'), false, 'Listing has indeterminate progress');
  assert.equal(get('stop-run').hidden, false);
  assert.equal(get('mode-badge-text').textContent, 'Gmail sortieren (Labels aktiv)', 'All mode labels even with sort_inbox off');
  serverStatus.processed = 12;
  serverStatus.total = 65;
  serverStatus.account_index = 1;
  serverStatus.accounts_total = 2;
  await context.refreshStatus();
  assert.equal(get('status-description').textContent, '12 Mails verarbeitet · Postfach 1 von 2');
  assert.equal(get('progress').value, 12);
  assert.equal(get('progress').max, 65);
  await get('stop-run').handlers.click();
  assert.equal(stopCalls, 1);
  assert.equal(vm.runInContext('status.phase', context), 'running');
  assert.equal(get('stop-run').disabled, true);
  assert.equal(get('stop-run').textContent, 'Stoppen …');
  await get('stop-run').handlers.click();
  assert.equal(stopCalls, 1, 'Do not repeat an acknowledged stop request');
  serverStatus.phase = 'stopped';
  serverStatus.finished_at = '2026-09-27T10:02:00Z';
  await context.refreshStatus();
  assert.equal(get('status-heading').textContent, 'Labeln gestoppt');
  assert.match(get('status-description').textContent, /^12 Mails verarbeitet\./);
  assert.equal(get('stop-run').hidden, true);
  assert.equal(get('run-all').disabled, false);
  assert.equal(get('account-select').disabled, false);
  serverStatus.phase = 'complete';
  serverStatus.finished_at = '2026-09-27T10:03:00Z';
  await context.refreshStatus();
  assert.equal(get('status-heading').textContent, 'Mails gelabelt');
  assert.match(get('status-description').textContent, /^12 Mails /);
  console.log('PASS: dashboard/keyboard/account regressions; all-mail start without limit; truthful progress; label/application locks; acknowledged stop and completion');
})().catch((error) => { console.error(error); process.exitCode = 1; });
