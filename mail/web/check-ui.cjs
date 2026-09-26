// Offline state checks. Run: node mail/web/check-ui.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Node {
  constructor(tag = 'div') {
    Object.assign(this, { tag, children: [], dataset: {}, attributes: {}, handlers: {}, value: '', textContent: '' });
  }
  set innerHTML(_) { throw new Error('Unsafe HTML insertion'); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(key, handler) { this.handlers[key] = handler; }
  focus() {}
  scrollIntoView() {}
  reportValidity() { this.validityReported = true; return false; }
}
const nodes = new Map();
const get = (id) => {
  if (!nodes.has(id)) nodes.set(id, new Node());
  return nodes.get(id);
};
get('mail-limit').value = '20';
const filters = ['all', 'unreviewed', 'uncertain', 'review', 'rechnung', 'werbung', 'wichtig', 'spam'].map((value) => {
  const node = new Node('button'); node.dataset.filter = value; return node;
});
const walk = (node) => [node, ...node.children.flatMap(walk)];
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
let serverStatus = { phase: 'complete', finished_at: base.created_utc, batch_mode: 'latest' };
const context = {
  document: { getElementById: get, createElement: (tag) => new Node(tag), querySelectorAll: () => filters, querySelector: () => new Node('select') },
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
      const row = serverReport.messages.find((mail) => mail.message_id === lastPayload.message_id);
      row.review = { action: lastPayload.action, labels: lastPayload.action === 'corrected' ? lastPayload.labels : lastPayload.action === 'accepted' ? row.suggestions : null, training_eligible: lastPayload.action !== 'skipped' };
      revision++;
      data = structuredClone(serverReport);
    } else if (url === '/api/run') {
      assert.equal(get('run').disabled, true, 'Latest action must be locked while starting');
      assert.equal(get('run-next').disabled, true, 'Next action must be locked while starting');
      assert.equal(get('mail-limit').disabled, true, 'Count must be locked while starting');
      const payload = JSON.parse(options.body);
      runPayloads.push(payload);
      serverStatus = { phase: 'running', finished_at: null, processed: 0, total: payload.limit, requested_limit: payload.limit, batch_mode: payload.mode };
      data = { status: 'started' };
    } else throw new Error('Unexpected route ' + url);
    return { status: 200, ok: true, json: async () => data };
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8'), context);

(async () => {
  await new Promise(setImmediate);
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
  console.log('PASS: safe text; GPU/CPU status and fallback; reviews; draft recovery; next/latest limits; invalid-count guards; input locks; count preservation; next-batch empty state');
})().catch((error) => { console.error(error); process.exitCode = 1; });
