'use strict';

const $ = (id) => document.getElementById(id);
let status = null;
let report = { report_id: null, created_utc: null, messages: [], limitations: [], summary: {} };
let filter = 'all';
let pollTimer;
let starting = false;
let startingLimit = null;
let leaving = false;
let reportLoaded = false;
let reportPending = true;
let editing = null;
let saving = false;
let reportVersion = 0;
let errorSource = null;
let selectedMailId = null;
const contentLabels = { rechnung: 'Rechnung', info: 'Allgemeine Information', werbung: 'Werbung', sonstiges: 'Sonstiges' };
const reviewLabels = { accepted: 'Bestätigt', corrected: 'Korrigiert', skipped: 'Übersprungen' };

function renderHighlightedText(container, text) {
  if (!text) return;
  const regex = /(rechnung(?:sbetrag|snummer|skopie)?|zahlungsziel|fällig(?:keit)?|überweisung|mahnung|quittung|gutschrift|mwst|\beur\b|\bchf\b|\busd\b|€|\$)|(dringend|frist|wichtig|sofort|handlungsbedarf|termin|erinnerung)|(passwort|konto gesperrt|sicherheitswarnung|zugangsdaten|unusual activity|verify your account)/gi;
  let lastIndex = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      container.append(document.createTextNode(text.slice(lastIndex, match.index)));
    }
    const matchedText = match[0];
    let markClass = 'hl-invoice';
    if (match[2]) markClass = 'hl-urgent';
    else if (match[3]) markClass = 'hl-security';
    container.append(element('mark', markClass, matchedText));
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) {
    container.append(document.createTextNode(text.slice(lastIndex)));
  }
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function showError(message, source = 'action') {
  $('api-error').textContent = message || '';
  $('api-error').hidden = !message;
  errorSource = message ? source : null;
}

async function api(path, method = 'GET', body = {}) {
  const options = { method, credentials: 'same-origin', cache: 'no-store' };
  if (method === 'POST') {
    options.headers = { 'Content-Type': 'application/json' };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  if (response.status === 401) {
    leaving = true;
    clearTimeout(pollTimer);
    location.reload();
    throw new Error('Die Sitzung ist abgelaufen. Bitte erneut anmelden.');
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || 'Die Anfrage konnte nicht abgeschlossen werden.');
  return data;
}

function needsReview(mail) {
  return !mail.review && (mail.status === 'error' || mail.needs_review === true);
}

function isUncertain(mail) {
  if (mail.review) return false;
  if (mail.raw_scores && typeof mail.raw_scores === 'object') {
    for (const q of Object.values(mail.raw_scores)) {
      if (q && q.probabilities) {
        const probs = Object.values(q.probabilities).sort((a, b) => b - a);
        if (probs.length > 1 && (probs[0] - probs[1]) < 0.20) return true;
      }
    }
  }
  return Boolean(mail.needs_review);
}

function dateParts(value) {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) return null;
  return {
    day: new Intl.DateTimeFormat('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }).format(date),
    time: new Intl.DateTimeFormat('de-DE', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' }).format(date),
  };
}

function renderStatus() {
  const running = status?.phase === 'running';
  const failed = status?.phase === 'error';
  const selected = Number($('mail-limit').value);
  const validLimit = Number.isInteger(selected) && selected >= 1 && selected <= 100;
  const selectedText = validLimit ? `${selected} ` : '';
  const processed = Math.max(0, Number(status?.processed) || 0);
  const locked = !status || running || Boolean(starting) || Boolean(editing) || saving;
  const activeMode = starting || status?.batch_mode || 'latest';
  const activeLimit = startingLimit || status?.requested_limit || 20;
  const total = Math.max(0, Number(status?.total ?? activeLimit) || 0);
  $('run').disabled = locked;
  $('run-next').disabled = locked;
  $('mail-limit').disabled = locked;
  $('run-label').textContent = (running || starting) && activeMode === 'latest' ? `Prüft bis zu ${activeLimit} Mails …` : report.report_id ? `Neueste ${selectedText}erneut prüfen` : `${selectedText}Mails prüfen`;
  $('run-next-label').textContent = (running || starting) && activeMode === 'next' ? `Lädt bis zu ${activeLimit} Mails …` : `Nächste ${selectedText}Mails laden`;
  $('status-symbol').className = `status-symbol${running ? ' running' : failed ? ' error' : ''}`;
  $('progress-wrap').hidden = !running;
  $('progress').max = Math.max(1, total);
  $('progress').value = Math.min(processed, total);
  const gpu = status?.model?.device === 'cuda';
  const processingHost = status?.model?.host || (gpu ? 'Hauptrechner' : 'Mac mini');
  $('processing-location').textContent = `Auswertung: ${processingHost} · ${gpu ? 'RTX 4070 Ti' : 'CPU'}`;
  const processingNote = typeof status?.processing_note === 'string' ? status.processing_note.trim() : '';
  $('processing-note').textContent = processingNote;
  $('processing-note').hidden = !processingNote;
  if (running) {
    $('status-heading').textContent = 'Laya prüft deine Mails';
    $('status-description').textContent = `${processed} von ${total} Nachrichten verarbeitet. Die Anzeige aktualisiert sich automatisch.`;
  } else if (failed) {
    $('status-heading').textContent = 'Die Auswertung wurde unterbrochen';
    $('status-description').textContent = status.error || 'Bitte starte die Prüfung erneut.';
  } else if (status) {
    $('status-heading').textContent = status.phase === 'complete' ? 'Die Auswertung ist abgeschlossen' : 'Bereit für deine Mails';
    $('status-description').textContent = 'Wähle die Anzahl der Nachrichten für deine nächste lokale Auswertung.';
  }
}

function label(text, kind) {
  return element('span', `label ${kind}`, text);
}

function appendLabels(container, values, showNegative = false) {
  if (values.inhalt) container.append(label(contentLabels[values.inhalt] || 'Nicht eingeordnet', Object.hasOwn(contentLabels, values.inhalt) ? values.inhalt : ''));
  if (values.wichtig === 'ja') container.append(label('Wichtig', 'wichtig'));
  else if (showNegative && values.wichtig === 'nein') container.append(label('Nicht wichtig', ''));
  if (values.spamverdacht === 'ja') container.append(label('Spamverdacht', 'spam'));
  else if (showNegative && values.spamverdacht === 'nein') container.append(label('Kein Spamverdacht', ''));
}

function actionButton(text, className, action, disabled = false) {
  const button = element('button', className, text);
  button.type = 'button';
  button.disabled = disabled;
  button.addEventListener('click', action);
  return button;
}

function startCorrection(mail) {
  reportVersion += 1;
  const previous = mail.review?.labels || mail.suggestions || {};
  editing = {
    report_id: report.report_id, message_id: mail.message_id, snapshot_id: mail.snapshot_id,
    labels: {
      inhalt: Object.hasOwn(contentLabels, previous.inhalt) ? previous.inhalt : '',
      wichtig: ['ja', 'nein'].includes(previous.wichtig) ? previous.wichtig : '',
      spamverdacht: ['ja', 'nein'].includes(previous.spamverdacht) ? previous.spamverdacht : '',
    },
  };
  renderMessages();
  renderStatus();
  document.querySelector('.correction-form select')?.focus();
}

async function saveFeedback(mail, action) {
  if (saving) return;
  const body = {
    report_id: editing?.report_id || report.report_id,
    message_id: mail.message_id,
    snapshot_id: mail.snapshot_id || null,
    action,
  };
  if (action === 'corrected') {
    if (!editing || !Object.hasOwn(contentLabels, editing.labels.inhalt) ||
      !['ja', 'nein'].includes(editing.labels.wichtig) || !['ja', 'nein'].includes(editing.labels.spamverdacht)) return;
    body.labels = { ...editing.labels };
    body.snapshot_id = editing.snapshot_id;
  }
  saving = true;
  reportVersion += 1;
  renderMessages();
  renderStatus();
  showError('');
  try {
    const updated = await api('/api/feedback', 'POST', body);
    editing = null;
    reportPending = false;
    renderReport(updated);
  } catch (error) {
    if (!leaving) showError(`Die Bewertung wurde nicht gespeichert. ${error.message}`);
  } finally {
    saving = false;
    renderMessages();
    renderStatus();
  }
}

function correctionForm(mail) {
  const form = element('form', 'correction-form');
  form.append(element('h4', '', 'Deine Korrektur'));
  const fields = element('div', 'correction-fields');
  const choices = {
    inhalt: ['Inhalt', contentLabels],
    wichtig: ['Wichtig', { ja: 'Ja', nein: 'Nein' }],
    spamverdacht: ['Spamverdacht', { ja: 'Ja', nein: 'Nein' }],
  };
  for (const [name, [title, options]] of Object.entries(choices)) {
    const field = element('label', 'correction-field');
    field.append(element('span', '', title));
    const select = element('select');
    select.name = name;
    select.required = true;
    select.disabled = saving;
    const placeholder = element('option', '', 'Bitte auswählen …');
    placeholder.value = '';
    placeholder.disabled = true;
    select.append(placeholder);
    for (const [value, text] of Object.entries(options)) {
      const option = element('option', '', text);
      option.value = value;
      select.append(option);
    }
    select.value = editing.labels[name];
    select.addEventListener('change', () => { editing.labels[name] = select.value; });
    field.append(select);
    fields.append(field);
  }
  form.append(fields);
  const actions = element('div', 'correction-actions');
  const save = element('button', 'primary-button', saving ? 'Wird gespeichert …' : 'Speichern');
  save.type = 'submit';
  save.disabled = saving;
  actions.append(save, actionButton('Abbrechen', 'secondary-button', () => {
    editing = null;
    renderMessages();
    renderStatus();
    if (reportPending) refreshStatus();
  }, saving));
  form.append(actions);
  form.addEventListener('submit', (event) => { event.preventDefault(); saveFeedback(mail, 'corrected'); });
  return form;
}

function renderMail(mail) {
  const isSelected = selectedMailId ? mail.message_id === selectedMailId : false;
  const card = element('article', `mail-card${needsReview(mail) ? ' review-card' : ''}${isSelected ? ' selected-card' : ''}`);
  if (mail.message_id) {
    card.id = `card-${mail.message_id}`;
    card.addEventListener('click', () => {
      if (selectedMailId !== mail.message_id) {
        selectedMailId = mail.message_id;
        renderMessages();
      }
    });
  }
  const top = element('div', 'mail-top');
  const identity = element('div');
  identity.append(element('h3', 'mail-subject', mail.subject || '(Ohne Betreff)'), element('p', 'mail-sender', mail.sender || 'Absender nicht verfügbar'));
  const labels = element('div', 'mail-labels');
  if (mail.account_email) {
    labels.append(element('span', 'label account-label', mail.account_email));
  }
  if (Array.isArray(mail.applied_labels) && mail.applied_labels.length) {
    for (const applied of mail.applied_labels) {
      labels.append(element('span', 'label label-applied', `🏷️ ${applied}`));
    }
  }
  const suggestions = mail.suggestions || {};
  if (mail.status === 'error') labels.append(label('Fehler', 'error'));
  else if (!suggestions.inhalt) labels.append(label('Nicht eingeordnet', ''));
  appendLabels(labels, suggestions);
  if (needsReview(mail)) labels.append(label('Bitte prüfen', 'review'));
  const original = element('div', 'original-suggestion');
  original.append(element('span', 'label-caption', 'Laya-Vorschlag'), labels);
  top.append(identity, original);
  card.append(top);
  if (mail.review) {
    const evaluation = element('div', 'saved-review');
    const evaluationLabels = element('div', 'evaluation-labels');
    evaluationLabels.append(label(reviewLabels[mail.review.action] || 'Bewertet', 'reviewed'));
    if (mail.review.labels) appendLabels(evaluationLabels, mail.review.labels, true);
    evaluation.append(element('span', 'label-caption', 'Deine Bewertung'), evaluationLabels);
    const date = dateParts(mail.review.updated_utc);
    const note = mail.review.action === 'skipped' ? 'Nicht als Trainingsbeispiel verwendet.' : mail.review.training_eligible ? 'Als lokales Trainingsbeispiel gespeichert.' : 'Bewertung gespeichert; kein exportierbares Trainingsbeispiel.';
    evaluation.append(element('p', 'evaluation-note', `${note}${date ? ` · ${date.day}, ${date.time}` : ''}`));
    card.append(evaluation);
  }
  if (mail.excerpt) {
    const excerpt = element('p', 'mail-excerpt');
    renderHighlightedText(excerpt, mail.excerpt);
    card.append(excerpt);
  }
  if (Array.isArray(mail.attachment_names) && mail.attachment_names.length) {
    card.append(element('p', 'attachments', `Anhänge: ${mail.attachment_names.join(' · ')}`));
  }
  if (mail.status === 'error') card.append(element('p', 'mail-error', mail.error || 'Diese Nachricht konnte nicht ausgewertet werden.'));
  const reasons = Array.isArray(mail.review_reasons) ? mail.review_reasons : [];
  if (reasons.length || (needsReview(mail) && mail.status !== 'error')) {
    const note = element('div', 'review-notes');
    note.append(element('strong', '', 'Ein zweiter Blick lohnt sich'));
    const list = element('ul');
    for (const reason of reasons.length ? reasons : ['Für diesen Vorschlag wird eine manuelle Prüfung empfohlen.']) list.append(element('li', '', reason));
    note.append(list);
    card.append(note);
  }
  if (mail.model_input && typeof mail.model_input.state === 'string') {
    const input = element('details', 'model-details input-details');
    input.append(element('summary', '', 'Ausgewerteter Text'), element('p', '', 'Genau dieser Text wurde an Laya übergeben. Deine Bewertung bezieht sich auf diese Eingabe.'), element('pre', '', mail.model_input.state));
    card.append(input);
  }
  const details = element('details', 'model-details');
  details.append(element('summary', '', 'Modelldetails'), element('p', '', 'Interne Modellwerte, keine gemessene Zuverlässigkeit. Die Sortierung ist ein Vorschlag.'));
  details.append(element('pre', '', JSON.stringify({
    modell: status?.model?.name || 'Laya multilingual',
    textquelle: mail.body_source ?? null,
    eingabetokens: mail.input_tokens ?? null,
    inhalt_ausgelassen: mail.content_omitted ?? false,
    text_gekuerzt: mail.truncated ?? false,
    modellwerte: mail.raw_scores ?? null,
  }, null, 2)));
  card.append(details);
  if (editing?.message_id === mail.message_id && editing?.report_id === report.report_id) {
    card.append(correctionForm(mail));
  } else {
    const actions = element('div', 'feedback-actions');
    const locked = saving || Boolean(editing) || !report.report_id || !mail.message_id;
    actions.append(
      actionButton('Stimmt', 'feedback-button accept-button', () => saveFeedback(mail, 'accepted'), locked || mail.status === 'error' || !mail.training_ready),
      actionButton('Korrigieren', 'feedback-button', () => startCorrection(mail), locked || !mail.training_ready),
      actionButton('Überspringen', 'text-button skip-button', () => saveFeedback(mail, 'skipped'), locked),
    );
    card.append(actions);
  }
  if (!mail.training_ready) card.append(element('p', 'legacy-note', 'Für diesen älteren oder unvollständigen Eintrag fehlt die genaue Modelleingabe. Starte einen neuen Lesetest, um ihn bestätigen oder korrigieren zu können.'));
  return card;
}

function getVisibleMessages() {
  const query = $('search').value.trim().toLocaleLowerCase('de');
  return report.messages.filter((mail) => {
    const suggestions = mail.review?.labels || mail.suggestions || {};
    const matchesFilter = filter === 'all' || (filter === 'review' && needsReview(mail)) ||
      (filter === 'unreviewed' && !mail.review) ||
      (filter === 'uncertain' && isUncertain(mail)) ||
      (filter === 'rechnung' && suggestions.inhalt === 'rechnung') ||
      (filter === 'werbung' && suggestions.inhalt === 'werbung') ||
      (filter === 'wichtig' && suggestions.wichtig === 'ja') ||
      (filter === 'spam' && suggestions.spamverdacht === 'ja');
    const haystack = [mail.subject, mail.sender, mail.excerpt, ...(mail.attachment_names || [])].join(' ').toLocaleLowerCase('de');
    return matchesFilter && haystack.includes(query);
  });
}

function renderMessages() {
  const messages = getVisibleMessages();
  if (messages.length > 0 && !messages.some((m) => m.message_id === selectedMailId)) {
    selectedMailId = messages[0].message_id;
  }
  $('messages').replaceChildren(...messages.map(renderMail));
  $('editing-note').hidden = !editing;
  $('messages').setAttribute('aria-busy', 'false');
  $('visible-count').textContent = String(messages.length);
  $('empty').hidden = messages.length > 0;
  const query = $('search').value.trim();
  const filtered = filter !== 'all' || Boolean(query);
  const nextBatchEmpty = reportLoaded && report.batch_mode === 'next' && report.messages.length === 0;
  $('empty-heading').textContent = nextBatchEmpty ? 'Keine weiteren Mails im Posteingang' : filtered ? 'Keine passenden Nachrichten' : reportLoaded && report.created_utc ? 'Keine Nachrichten im Bericht' : 'Noch keine Auswertung';
  $('empty-description').textContent = nextBatchEmpty ? 'Aktuell gibt es keine bisher nicht angezeigten Nachrichten im Posteingang. Deine Bewertungen bleiben erhalten.' : filtered ? 'Versuche einen anderen Filter oder Suchbegriff.' : reportLoaded && report.created_utc ? 'Die letzte Prüfung hat keine Nachrichten geliefert.' : 'Starte die erste Prüfung, um hier Vorschläge zu sehen.';
  $('reset-filters').hidden = !filtered || nextBatchEmpty;
}

function renderReport(data) {
  reportVersion += 1;
  report = {
    report_id: data.report_id || null,
    batch_mode: data.batch_mode || 'latest',
    limit: Number.isInteger(data.limit) && data.limit > 0 ? data.limit : null,
    created_utc: data.created_utc || null,
    messages: Array.isArray(data.messages) ? data.messages.filter((mail) => mail && typeof mail === 'object') : [],
    limitations: Array.isArray(data.limitations) ? data.limitations : [],
    summary: data.summary || {},
  };
  reportLoaded = true;
  $('message-count').textContent = String(report.messages.length);
  $('message-limit-note').textContent = report.limit ? `bis zu ${report.limit} je Durchlauf` : 'je Durchlauf';
  $('review-count').textContent = String(report.messages.filter(needsReview).length);
  const reviewed = report.summary.reviewed ?? report.messages.filter((mail) => ['accepted', 'corrected'].includes(mail.review?.action)).length;
  const skipped = report.summary.skipped ?? report.messages.filter((mail) => mail.review?.action === 'skipped').length;
  const pending = report.summary.pending ?? report.messages.filter((mail) => !mail.review).length;
  const training = report.summary.training_examples ?? 0;
  $('review-progress-text').textContent = `${reviewed} von ${report.messages.length} bewertet`;
  $('review-progress-detail').textContent = `${skipped} übersprungen · ${pending} unbewertet`;
  $('review-progress').max = Math.max(1, report.messages.length);
  $('review-progress').value = reviewed + skipped;
  $('training-count').textContent = `${training} ${training === 1 ? 'Trainingsbeispiel' : 'Trainingsbeispiele'} insgesamt`;
  $('training-export').hidden = !training;
  const trainSplit = $('training-export-train');
  const valSplit = $('training-export-val');
  if (trainSplit) trainSplit.hidden = !training;
  if (valSplit) valSplit.hidden = !training;

  const dist = report.summary.distribution;
  const distEl = $('category-distribution');
  if (dist && distEl) {
    const pills = [];
    const inhaltNames = { rechnung: 'Rechnungen', info: 'Info', werbung: 'Werbung', sonstiges: 'Sonstiges' };
    for (const [k, name] of Object.entries(inhaltNames)) {
      const count = dist.inhalt?.[k] ?? 0;
      pills.push(element('span', 'dist-pill', `${name}: ${count}`));
    }
    pills.push(element('span', 'dist-pill', `Wichtig: ${dist.wichtig?.ja ?? 0}`));
    pills.push(element('span', 'dist-pill', `Spam: ${dist.spamverdacht?.ja ?? 0}`));
    distEl.replaceChildren(...pills);
    distEl.hidden = !training;
  } else if (distEl) {
    distEl.hidden = true;
  }
  $('review-progress').value = reviewed + skipped;
  $('training-count').textContent = `${training} ${training === 1 ? 'Trainingsbeispiel' : 'Trainingsbeispiele'} insgesamt`;
  $('training-export').hidden = !training;
  const date = dateParts(report.created_utc);
  $('report-date').textContent = date?.day || 'Noch keine';
  $('report-time').textContent = date ? `${date.time} · Ortszeit dieses Geräts` : 'Starte eine erste Prüfung';
  $('limitations').hidden = !report.limitations.length;
  $('limitations-list').replaceChildren(...report.limitations.map((text) => element('li', '', text)));
  renderMessages();
  renderStatus();
}

function renderGpu(gpu) {
  const badge = $('gpu-badge');
  const text = $('gpu-text');
  const ping = $('gpu-ping');
  if (!badge || !text) return;
  if (!gpu) {
    badge.className = 'gpu-badge gpu-badge-probing';
    text.textContent = 'GPU prüfen …';
    if (ping) ping.hidden = true;
    return;
  }
  if (gpu.online && gpu.is_gpu) {
    badge.className = 'gpu-badge gpu-badge-online';
    text.textContent = 'RTX 4070 Ti verbunden';
    if (ping) {
      ping.hidden = false;
      ping.textContent = `${gpu.latency_ms ?? 12} ms`;
    }
    badge.title = `NVIDIA GeForce RTX 4070 Ti online · ${gpu.model || 'Laya multilingual'} · Latenz: ${gpu.latency_ms}ms`;
  } else if (gpu.online && !gpu.is_gpu) {
    badge.className = 'gpu-badge gpu-badge-cpu';
    text.textContent = 'CPU aktiv';
    if (ping) {
      ping.hidden = false;
      ping.textContent = `${gpu.latency_ms ?? 10} ms`;
    }
    badge.title = `Lokale CPU aktiv · Latenz: ${gpu.latency_ms}ms`;
  } else {
    badge.className = 'gpu-badge gpu-badge-offline';
    text.textContent = 'GPU offline (Mac CPU Fallback)';
    if (ping) ping.hidden = true;
    badge.title = 'Hauptrechner (RTX 4070 Ti) nicht erreichbar · Fallback Mac mini CPU aktiv';
  }
}

function renderAccounts(data) {
  const select = $('account-select');
  if (!select || !data || !Array.isArray(data.accounts)) return;
  const currentVal = data.selected || 'all';
  select.innerHTML = '';
  const allOpt = document.createElement('option');
  allOpt.value = 'all';
  allOpt.textContent = `Alle Postfächer (${data.accounts.length})`;
  select.append(allOpt);

  for (const acc of data.accounts) {
    const opt = document.createElement('option');
    opt.value = acc.id;
    opt.textContent = `${acc.email}${acc.token_exists ? '' : ' ⚠️'}`;
    select.append(opt);
  }
  select.value = currentVal;
}

function renderSettings(settings) {
  if (!settings) return;
  const liveToggle = $('toggle-live-mode');
  const sortToggle = $('toggle-sort-inbox');
  const modeBadge = $('mode-badge');
  const modeBadgeText = $('mode-badge-text');
  const liveActivity = $('live-activity');
  const liveActivityText = $('live-activity-text');

  if (liveToggle) liveToggle.checked = Boolean(settings.live_mode);
  if (sortToggle) sortToggle.checked = Boolean(settings.sort_inbox);

  if (modeBadge && modeBadgeText) {
    if (settings.sort_inbox) {
      modeBadge.className = 'readonly-badge sort-active';
      modeBadgeText.textContent = 'Gmail sortieren (Labels aktiv)';
    } else {
      modeBadge.className = 'readonly-badge';
      modeBadgeText.textContent = 'Gmail nur lesen';
    }
  }

  if (liveActivity && liveActivityText) {
    if (settings.live_mode) {
      liveActivity.hidden = false;
      const watcher = settings.watcher;
      const recent = watcher?.recent_activities?.[0];
      if (recent) {
        liveActivityText.textContent = `Live: ${recent.message}`;
      } else {
        liveActivityText.textContent = 'Live-Watcher aktiv · Prüft alle 3 Minuten';
      }
    } else {
      liveActivity.hidden = true;
    }
  }
}

async function refreshStatus() {
  clearTimeout(pollTimer);
  try {
    const next = await api('/api/status');
    const reportChanged = next.finished_at && next.finished_at !== status?.finished_at;
    const justFinished = status?.phase === 'running' && next.phase !== 'running';
    const feedbackChanged = next.feedback_revision !== status?.feedback_revision;
    reportPending = reportPending || reportChanged || justFinished || feedbackChanged;
    status = next;
    renderStatus();
    if (next.gpu) renderGpu(next.gpu);
    if (next.accounts) renderAccounts(next.accounts);
    if (next.settings) renderSettings(next.settings);
    if (reportPending && !editing && !saving) {
      const version = reportVersion;
      const updated = await api('/api/report');
      if (!editing && !saving && version === reportVersion) {
        renderReport(updated);
        reportPending = false;
      }
    }
    if (errorSource === 'status') showError('');
  } catch (error) {
    if (!leaving) showError(`Der aktuelle Stand konnte nicht geladen werden. ${error.message}`, 'status');
  } finally {
    if (!leaving) pollTimer = setTimeout(refreshStatus, status?.phase === 'running' ? 2000 : 15000);
  }
}

function selectFilter(value) {
  filter = value;
  document.querySelectorAll('[data-filter]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.filter === value)));
  renderMessages();
}

document.querySelectorAll('[data-filter]').forEach((button) => button.addEventListener('click', () => selectFilter(button.dataset.filter)));
$('search').addEventListener('input', renderMessages);
$('reset-filters').addEventListener('click', () => { $('search').value = ''; selectFilter('all'); });
$('return-correction').addEventListener('click', () => {
  $('search').value = '';
  selectFilter('all');
  document.querySelector('.correction-form')?.scrollIntoView({ block: 'center' });
  document.querySelector('.correction-form select')?.focus();
});
async function startRun(mode) {
  if (starting || status?.phase === 'running' || editing || saving) return;
  const input = $('mail-limit');
  const limit = Number(input.value);
  if (!Number.isInteger(limit) || limit < 1 || limit > 100) {
    input.reportValidity();
    return;
  }
  starting = mode;
  startingLimit = limit;
  renderStatus();
  showError('');
  try {
    await api('/api/run', 'POST', { mode, limit });
    if (mode === 'next') {
      $('search').value = '';
      selectFilter('unreviewed');
    }
    await refreshStatus();
  } catch (error) {
    if (!leaving) showError(error.message);
  } finally {
    starting = false;
    startingLimit = null;
    renderStatus();
  }
}
$('run').addEventListener('click', () => startRun('latest'));
$('run-next').addEventListener('click', () => startRun('next'));
$('mail-limit').addEventListener('input', renderStatus);

$('account-select')?.addEventListener('change', async (e) => {
  try {
    const res = await api('/api/accounts/select', 'POST', { selected: e.target.value });
    renderAccounts(res);
    await refreshStatus();
  } catch (err) {
    showError(`Konto-Auswahl fehlgeschlagen: ${err.message}`);
  }
});

$('toggle-live-mode')?.addEventListener('change', async (e) => {
  try {
    const res = await api('/api/settings', 'POST', { live_mode: e.target.checked });
    renderSettings(res);
  } catch (err) {
    e.target.checked = !e.target.checked;
    showError(`Live-Modus konnte nicht geändert werden: ${err.message}`);
  }
});

$('toggle-sort-inbox')?.addEventListener('change', async (e) => {
  try {
    const res = await api('/api/settings', 'POST', { sort_inbox: e.target.checked });
    renderSettings(res);
  } catch (err) {
    e.target.checked = !e.target.checked;
    showError(`Sortier-Einstellung konnte nicht geändert werden: ${err.message}`);
  }
});
$('logout').addEventListener('click', async () => {
  $('logout').disabled = true;
  try {
    await api('/api/logout', 'POST');
    leaving = true;
    clearTimeout(pollTimer);
    location.reload();
  } catch (error) {
    if (!leaving) showError(error.message);
    $('logout').disabled = false;
  }
});

document.addEventListener('keydown', (event) => {
  if (leaving || saving) return;
  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  const isInteractive = Boolean(
    document.activeElement?.closest
      ? document.activeElement.closest('button, a, summary')
      : ['BUTTON', 'A', 'SUMMARY'].includes(tag)
  );
  if (isInteractive && (event.key === 'Enter' || event.key === ' ')) return;
  if (editing) {
    if (event.key === 'Escape') {
      editing = null;
      renderMessages();
      renderStatus();
    }
    return;
  }
  const currentFiltered = getVisibleMessages();
  if (!currentFiltered.length) return;
  const currentIndex = currentFiltered.findIndex((m) => m.message_id === selectedMailId);
  const activeMail = currentIndex >= 0 ? currentFiltered[currentIndex] : currentFiltered[0];

  if (event.key === 'j' || event.key === 'J' || event.key === 'ArrowDown') {
    event.preventDefault();
    const nextIndex = Math.min(currentFiltered.length - 1, (currentIndex >= 0 ? currentIndex : 0) + 1);
    selectedMailId = currentFiltered[nextIndex].message_id;
    renderMessages();
    document.getElementById(`card-${selectedMailId}`)?.scrollIntoView?.({ block: 'nearest' });
  } else if (event.key === 'k' || event.key === 'K' || event.key === 'ArrowUp') {
    event.preventDefault();
    const prevIndex = Math.max(0, (currentIndex >= 0 ? currentIndex : 0) - 1);
    selectedMailId = currentFiltered[prevIndex].message_id;
    renderMessages();
    document.getElementById(`card-${selectedMailId}`)?.scrollIntoView?.({ block: 'nearest' });
  } else if (event.key === '1' || event.key === 'Enter') {
    if (activeMail && activeMail.training_ready && activeMail.status !== 'error') {
      event.preventDefault();
      saveFeedback(activeMail, 'accepted');
    }
  } else if (event.key === '2' || event.key === 'c' || event.key === 'C') {
    if (activeMail && activeMail.training_ready) {
      event.preventDefault();
      startCorrection(activeMail);
    }
  } else if (event.key === '3' || event.key === 's' || event.key === 'S') {
    if (activeMail && report.report_id) {
      event.preventDefault();
      saveFeedback(activeMail, 'skipped');
    }
  } else if (event.key === ' ') {
    event.preventDefault();
    const cardEl = document.getElementById(`card-${selectedMailId}`);
    const details = cardEl?.querySelector?.('.input-details');
    if (details) details.open = !details.open;
  }
});

refreshStatus();
