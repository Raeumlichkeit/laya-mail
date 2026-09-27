// Offline onboarding checks. Run: node mail/web/check-onboarding.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, 'onboarding.html'), 'utf8');
const script = fs.readFileSync(path.join(__dirname, 'onboarding.js'), 'utf8');
const draftKey = 'laya-account-setup';
const hostile = '<img src=x onerror=alert(1)>';
const secret = 'SYNTHETIC-REFRESH-TOKEN-NEVER-PERSIST';
const prepared = {
  email: 'test@example.invalid', permission: 'modify', command: 'synthetic-command ' + hostile,
  token_filename: 'synthetic-grant.json', project_id: 'synthetic-project',
};
const response = (data, status = 200) => ({ status, ok: status >= 200 && status < 300, json: async () => structuredClone(data) });
const tick = () => new Promise(setImmediate);
const grant = (text, size = Buffer.byteLength(text)) => ({ size, text: async () => text });

async function page({ storage = new Map(), config = { available: true } } = {}) {
  const requests = [], storageWrites = [], nodes = new Map();
  let activeElement = null;
  for (const match of html.matchAll(/<\w+\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const node = {
      id: match[2], hidden: /\bhidden\b/.test(match[1]), value: '', textContent: '', files: [],
      disabled: false, attributes: {}, handlers: {}, classes: new Set(),
      set innerHTML(_) { throw new Error('Unsafe HTML insertion'); },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; },
      addEventListener(name, callback) { this.handlers[name] = callback; },
      scrollIntoView() {}, focus() { activeElement = this.id; }, reportValidity() { return true; },
      get classList() { return { toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name) }; },
    };
    nodes.set(node.id, node);
  }
  const get = (id) => { assert.ok(nodes.has(id), 'Missing HTML element: ' + id); return nodes.get(id); };
  const radios = [{ value: 'readonly', checked: true }, { value: 'modify', checked: false }];
  const permission = (value) => radios.forEach((radio) => { radio.checked = radio.value === value; });
  get('account-form').reset = () => { get('setup-email').value = ''; permission('readonly'); };
  const state = { requests, storage, storageWrites, get, permission, config };
  state.reply = async ({ url, method, body }) => {
    if (method === 'GET') return response(state.config);
    if (url === '/api/accounts/setup') return response({ ...prepared, ...body });
    return response({ status: 'connected', email: body.email, permission: body.permission });
  };
  const context = vm.createContext({
    URL, Object, JSON, Error,
    document: {
      getElementById: get,
      querySelector: (selector) => {
        assert.equal(selector, 'input[name="permission"]:checked');
        return radios.find((radio) => radio.checked);
      },
      querySelectorAll: (selector) => { assert.equal(selector, 'input[name="permission"]'); return radios; },
    },
    sessionStorage: {
      getItem: (key) => storage.get(key) ?? null,
      setItem: (key, value) => { storageWrites.push(value); storage.set(key, value); },
      removeItem: (key) => storage.delete(key),
    },
    fetch: async (url, options) => {
      assert.equal(options.credentials, 'same-origin');
      assert.ok(['/api/accounts/setup', '/api/accounts/connect'].includes(url), 'Onboarding must never start a mail run');
      const request = { url, method: options.method, body: options.body && JSON.parse(options.body) };
      requests.push(request);
      return state.reply(request);
    },
  });
  state.fire = (id, event = 'click') => get(id).handlers[event]({ preventDefault() {} });
  state.submit = (id) => state.fire(id, 'submit');
  state.assertStep = (expected) => {
    for (let number = 1; number <= 3; number++) assert.equal(get('setup-step-' + number).hidden, expected !== number);
    assert.equal(get('setup-success').hidden, expected !== 4);
    if (expected < 4) assert.equal(get('step-marker-' + expected).attributes['aria-current'], 'step');
  };
  state.activeElement = () => activeElement;
  vm.runInContext(script, context, { filename: 'onboarding.js' });
  await tick();
  return state;
}

const resumedPage = () => page({ storage: new Map([[draftKey, JSON.stringify({ email: prepared.email, permission: prepared.permission, prepared, step: 3 })]]) });

test('steps, safe instructions, resume, double-submit lock and connection without a mail run', async () => {
  const first = await page();
  first.assertStep(1);
  first.get('setup-email').value = '  TEST@EXAMPLE.INVALID  ';
  first.permission('modify');
  let finishPrepare;
  first.reply = () => new Promise((resolve) => { finishPrepare = resolve; });
  const pending = first.submit('account-form');
  await first.submit('account-form');
  assert.equal(first.requests.length, 2, 'Double submit sends one prepare request');
  assert.equal(first.get('prepare-account').disabled, true);
  assert.deepEqual(first.requests[1].body, { email: prepared.email, permission: 'modify' });
  finishPrepare(response({ ...prepared, project_id: 'https://evil.invalid/?secret=x' }));
  await pending;
  first.assertStep(2);
  assert.equal(first.activeElement(), 'step-two-heading');
  assert.equal(first.get('setup-command').textContent, prepared.command, 'HTML-like instructions are literal text');
  assert.equal(first.get('google-console-link').href, 'https://console.cloud.google.com/auth/audience');
  await first.fire('back-to-account'); first.assertStep(1);
  assert.equal(first.get('setup-email').value, prepared.email);
  first.reply = async () => response(prepared);
  await first.submit('account-form'); first.assertStep(2);
  await first.fire('grant-ready'); first.assertStep(3);
  await first.fire('back-to-grant'); first.assertStep(2);
  await first.fire('grant-ready');

  const resumed = await page({ storage: first.storage });
  resumed.assertStep(3);
  assert.equal(resumed.requests.length, 1, 'Resume only loads configuration');
  assert.equal(resumed.get('connection-email').textContent, prepared.email);
  const saved = JSON.parse(resumed.storage.get(draftKey));
  assert.deepEqual(Object.keys(saved).sort(), ['email', 'permission', 'prepared', 'step']);
  assert.deepEqual(Object.keys(saved.prepared).sort(), Object.keys(prepared).sort());
  const token = { refresh_token: secret, client_secret: 'SYNTHETIC-CLIENT-SECRET' };
  resumed.get('grant-file').files = [grant(JSON.stringify(token))];
  resumed.get('grant-file').value = 'synthetic-grant.json';
  await resumed.fire('back-to-grant');
  await resumed.fire('grant-ready');
  let finishConnect;
  resumed.reply = () => new Promise((resolve) => { finishConnect = resolve; });
  const connecting = resumed.submit('connect-form');
  await tick();
  await resumed.submit('connect-form');
  assert.equal(resumed.requests.length, 2, 'Double submit sends one connect request');
  assert.equal(resumed.get('connect-account').disabled, true);
  assert.equal(resumed.get('back-to-grant').disabled, true);
  assert.equal(resumed.get('grant-file').disabled, true);
  assert.equal(resumed.get('connection-status').hidden, false);
  assert.deepEqual(resumed.requests[1], { url: '/api/accounts/connect', method: 'POST', body: { email: prepared.email, permission: 'modify', token } });
  finishConnect(response({ status: 'connected', email: prepared.email, permission: 'modify' }));
  await connecting;
  resumed.assertStep(4);
  assert.equal(resumed.get('connected-email').textContent, prepared.email);
  assert.equal(resumed.get('grant-file').value, '');
  assert.equal(resumed.get('connection-status').hidden, true);
  assert.equal(resumed.storage.has(draftKey), false);
  assert.equal(resumed.get('connect-account').disabled, false);
  for (const stored of [...first.storageWrites, ...resumed.storageWrites]) {
    assert.ok(!stored.includes(secret) && !stored.includes(token.client_secret));
    assert.ok(!stored.includes('"refresh_token"') && !stored.includes('"access_token"'));
  }
  for (const id of ['connected-email', 'setup-error', 'setup-command']) assert.ok(!resumed.get(id).textContent.includes(secret));
  await resumed.fire('connect-another'); resumed.assertStep(1);
  assert.equal(resumed.get('setup-email').value, '');
});

test('missing, malformed, wrong and oversized files never reach the server', async () => {
  const current = await resumedPage();
  for (const [file, message] of [
    [null, /wähle zuerst/i],
    [grant('{broken'), /kein gültiges JSON/i],
    [grant(JSON.stringify({ installed: { client_id: 'synthetic-client' } })), /nicht client\.json/i],
    [grant('[]'), /keine Google-Freigabedatei/i],
    [{ size: 12001, text: async () => { throw Error('Oversized file must not be read'); } }, /höchstens 12 KB/i],
  ]) {
    current.get('grant-file').files = file ? [file] : [];
    await current.submit('connect-form');
    assert.match(current.get('setup-error').textContent, message);
    assert.equal(current.requests.length, 1);
    assert.equal(current.get('grant-file').disabled, false);
    current.assertStep(3);
  }
});

test('server and connection failures remain readable and cannot create success', async () => {
  const current = await resumedPage();
  current.get('grant-file').files = [grant(JSON.stringify({ refresh_token: secret }))];
  for (const [reply, message] of [
    [async () => response({ error: 'Adresse passt nicht zur Freigabe. ' + hostile }, 409), /Adresse passt nicht/],
    [async () => response({}, 401), /Sitzung ist abgelaufen/],
    [async () => { throw Error('private transport details'); }, /Server ist gerade nicht erreichbar/],
    [async () => response({}, 500), /Bitte erneut versuchen/],
    [async () => response({ status: 'connected', email: 'other@example.invalid' }), /nicht eindeutig bestätigt/],
  ]) {
    current.reply = reply;
    await current.submit('connect-form');
    assert.match(current.get('setup-error').textContent, message);
    assert.equal(current.get('setup-error').hidden, false);
    assert.equal(current.get('connect-account').disabled, false);
    assert.ok(!current.get('setup-error').textContent.includes('private transport details'));
    assert.ok(!current.get('setup-error').textContent.includes(secret));
    current.assertStep(3);
  }
});

test('unavailable configuration blocks preparation and retry can restore the flow', async () => {
  const current = await page({ config: { available: false, reason: hostile } });
  assert.equal(current.get('setup-unavailable').hidden, false);
  assert.equal(current.get('unavailable-reason').textContent, hostile);
  assert.equal(current.get('setup-loading').hidden, true);
  for (let number = 1; number <= 3; number++) assert.equal(current.get('setup-step-' + number).hidden, true);
  current.get('setup-email').value = prepared.email;
  await current.submit('account-form');
  assert.equal(current.requests.length, 1);
  current.config = { available: true };
  await current.fire('setup-reload');
  current.assertStep(1);
  assert.equal(current.get('setup-unavailable').hidden, true);
});
