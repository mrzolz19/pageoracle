let appState = null;

// Centralized element registry to avoid repeated DOM queries.
const els = {
  statusText: document.getElementById('statusText'),
  statusDot: document.getElementById('statusDot'),
  booksList: document.getElementById('booksList'),
  chatContainer: document.getElementById('chatContainer'),
  logsBox: document.getElementById('logsBox'),
  questionInput: document.getElementById('questionInput'),
  btnSend: document.getElementById('btnSend'),
  btnLoad: document.getElementById('btnLoad'),
  bookInput: document.getElementById('bookInput'),
  btnClearChat: document.getElementById('btnClearChat'),
  btnClearHistory: document.getElementById('btnClearHistory'),
  btnClearLogs: document.getElementById('btnClearLogs'),
  btnSettings: document.getElementById('btnSettings'),
  settingsOverlay: document.getElementById('settingsOverlay'),
  settingsForm: document.getElementById('settingsForm'),
  providerSelect: document.getElementById('providerSelect'),
  modelSelect: document.getElementById('modelSelect'),
  embeddingSelect: document.getElementById('embeddingSelect'),
  llmApiInput: document.getElementById('llmApiInput'),
  embeddingApiInput: document.getElementById('embeddingApiInput'),
  ycFolderInput: document.getElementById('ycFolderInput'),
  showKeysCheckbox: document.getElementById('showKeysCheckbox'),
  temperatureInput: document.getElementById('temperatureInput'),
  maxTokensInput: document.getElementById('maxTokensInput'),
  topPInput: document.getElementById('topPInput'),
  scoreThresholdInput: document.getElementById('scoreThresholdInput'),
  btnCancelSettings: document.getElementById('btnCancelSettings'),
};

function createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) {
    el.className = className;
  }
  if (typeof text === 'string') {
    el.textContent = text;
  }
  return el;
}

async function api(path, options = {}) {
  // Unified fetch helper with consistent error extraction.
  const response = await fetch(path, {
    headers: {
      ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
    },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Ошибка запроса: ${response.status}`);
  }
  return data;
}

function setInteractiveState() {
  // When backend is busy, lock controls to prevent conflicting operations.
  if (!appState) {
    return;
  }
  const disabled = appState.is_busy;
  els.btnSend.disabled = disabled;
  els.btnLoad.disabled = disabled;
  els.btnClearChat.disabled = disabled;
  els.btnClearHistory.disabled = disabled;
  els.btnClearLogs.disabled = disabled;
  els.questionInput.disabled = disabled;

  document.querySelectorAll('.mode-pill').forEach((btn) => {
    btn.disabled = disabled;
  });
}

function renderStatus() {
  els.statusText.textContent = appState.status.text;
  els.statusText.style.color = appState.status.color;
  els.statusDot.style.background = appState.status.color;
}

function renderMode() {
  document.querySelectorAll('.mode-pill').forEach((btn) => {
    const active = btn.dataset.mode === appState.mode;
    btn.classList.toggle('active', active);
  });
}

function renderBooks() {
  // Sidebar book list always re-rendered from server state.
  els.booksList.innerHTML = '';
  if (!appState.books.length) {
    els.booksList.appendChild(createEl('div', 'empty-text', 'Книги не загружены'));
    return;
  }

  appState.books.forEach((name) => {
    const row = createEl('div', 'book-item');
    const icon = createEl('div', 'book-icon', '📖');
    const title = createEl('div', 'book-name', name);
    row.append(icon, title);
    els.booksList.appendChild(row);
  });
}

function renderChat() {
  // Preserve autoscroll only when user is near the bottom.
  const isAtBottom =
    Math.abs(
      els.chatContainer.scrollHeight -
        els.chatContainer.clientHeight -
        els.chatContainer.scrollTop
    ) < 24;

  els.chatContainer.innerHTML = '';

  appState.chat.forEach((message) => {
    const wrapper = createEl('div', `message ${message.is_ai ? 'ai' : 'user'}`);
    const avatar = createEl('div', 'avatar', message.is_ai ? '🤖' : '🧑');
    const content = createEl('div', 'msg-content');
    const sender = createEl('span', 'sender', message.sender);
    const text = createEl('div', '', message.text);
    content.append(sender, text);
    wrapper.append(avatar, content);
    els.chatContainer.appendChild(wrapper);
  });

  if (appState.is_busy && appState.status.text.toLowerCase().includes('дума')) {
    const thinking = createEl('div', 'thinking');
    const dots = createEl('div', 'thinking-dots');
    dots.append(createEl('span'), createEl('span'), createEl('span'));
    const text = createEl('span', '', 'PageOracle думает...');
    thinking.append(dots, text);
    els.chatContainer.appendChild(thinking);
  }

  if (isAtBottom || appState.is_busy) {
    els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
  }
}

function renderLogs() {
  // Logs are append-only on backend; frontend paints full current snapshot.
  const isAtBottom =
    Math.abs(els.logsBox.scrollHeight - els.logsBox.clientHeight - els.logsBox.scrollTop) < 20;
  els.logsBox.innerHTML = '';

  appState.logs.forEach((entry) => {
    const line = createEl('div', 'log-line');
    const ts = createEl('span', 'log-ts', `[${entry.ts}]`);
    const text = createEl('span', `log-text ${entry.level || 'info'}`, entry.text);
    line.append(ts, text);
    els.logsBox.appendChild(line);
  });

  if (isAtBottom) {
    els.logsBox.scrollTop = els.logsBox.scrollHeight;
  }
}

function fillProviderOptions() {
  const providers = Object.keys(appState.providers);
  els.providerSelect.innerHTML = '';
  providers.forEach((name) => {
    const option = createEl('option', '', name);
    option.value = name;
    els.providerSelect.appendChild(option);
  });
}

function fillEmbeddingOptions() {
  els.embeddingSelect.innerHTML = '';
  appState.embedding_options.forEach((name) => {
    const option = createEl('option', '', name);
    option.value = name;
    els.embeddingSelect.appendChild(option);
  });
}

function fillModelOptions(providerName, selectedModel) {
  // Model options depend on provider; keep selected value when possible.
  const models = appState.providers[providerName]?.models || [];
  els.modelSelect.innerHTML = '';
  models.forEach((modelName) => {
    const option = createEl('option', '', modelName);
    option.value = modelName;
    els.modelSelect.appendChild(option);
  });

  const modelToSelect = models.includes(selectedModel) ? selectedModel : models[0] || '';
  els.modelSelect.value = modelToSelect;
}

function openSettings() {
  // Populate form from current persisted server-side settings.
  fillProviderOptions();
  fillEmbeddingOptions();

  const settings = appState.settings;
  els.providerSelect.value = settings.provider;
  fillModelOptions(settings.provider, settings.model);

  els.embeddingSelect.value = settings.embedding_model;
  els.llmApiInput.value = settings.llm_api_key || '';
  els.embeddingApiInput.value = settings.embedding_api_key || '';
  els.ycFolderInput.value = settings.yc_folder_id || '';
  els.temperatureInput.value = String(settings.temperature ?? 0.2);
  els.maxTokensInput.value = String(settings.max_tokens ?? 4096);
  els.topPInput.value = String(settings.top_p ?? 0.9);
  els.scoreThresholdInput.value = String(settings.score_threshold ?? 0.6);

  els.showKeysCheckbox.checked = false;
  els.llmApiInput.type = 'password';
  els.embeddingApiInput.type = 'password';
  els.settingsOverlay.classList.remove('hidden');
}

function closeSettings() {
  els.settingsOverlay.classList.add('hidden');
}

async function refreshState() {
  // Poll-based sync: source of truth lives on backend.
  const result = await api('/api/state');
  appState = result.data;

  renderStatus();
  renderMode();
  renderBooks();
  renderChat();
  renderLogs();
  setInteractiveState();
}

async function handleSend() {
  // Submit question in current mode and refresh the whole UI state.
  const question = els.questionInput.value.trim();
  if (!question || !appState || appState.is_busy) {
    return;
  }
  els.questionInput.value = '';

  try {
    await api('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ question, mode: appState.mode }),
    });
    await refreshState();
  } catch (error) {
    alert(error.message);
    await refreshState();
  }
}

async function handleUpload(file) {
  // Upload only txt books because backend loader/indexer expects text files.
  if (!file) {
    return;
  }
  if (!file.name.toLowerCase().endsWith('.txt')) {
    alert('Поддерживаются только файлы .txt');
    return;
  }

  const formData = new FormData();
  formData.append('book', file);

  try {
    await api('/api/upload', {
      method: 'POST',
      body: formData,
    });
    await refreshState();
  } catch (error) {
    alert(error.message);
    await refreshState();
  } finally {
    els.bookInput.value = '';
  }
}

function bindEvents() {
  // Wire all UI controls once on boot.
  els.btnSend.addEventListener('click', handleSend);

  els.questionInput.addEventListener('keydown', async (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      await handleSend();
    }
  });

  els.btnLoad.addEventListener('click', () => {
    els.bookInput.click();
  });

  els.bookInput.addEventListener('change', async () => {
    const file = els.bookInput.files?.[0];
    await handleUpload(file);
  });

  document.querySelectorAll('.mode-pill').forEach((btn) => {
    btn.addEventListener('click', async () => {
      if (!appState || appState.is_busy) {
        return;
      }
      try {
        await api('/api/mode', {
          method: 'POST',
          body: JSON.stringify({ mode: btn.dataset.mode }),
        });
        await refreshState();
      } catch (error) {
        alert(error.message);
      }
    });
  });

  els.btnClearChat.addEventListener('click', async () => {
    try {
      await api('/api/chat/clear', { method: 'POST' });
      await refreshState();
    } catch (error) {
      alert(error.message);
    }
  });

  els.btnClearHistory.addEventListener('click', async () => {
    if (!confirm('Удалить историю диалога из файла?')) {
      return;
    }
    try {
      await api('/api/history/clear', { method: 'POST' });
      await refreshState();
    } catch (error) {
      alert(error.message);
      await refreshState();
    }
  });

  els.btnClearLogs.addEventListener('click', async () => {
    try {
      await api('/api/logs/clear', { method: 'POST' });
      await refreshState();
    } catch (error) {
      alert(error.message);
    }
  });

  els.btnSettings.addEventListener('click', () => {
    if (!appState) {
      return;
    }
    openSettings();
  });

  els.btnCancelSettings.addEventListener('click', closeSettings);

  els.settingsOverlay.addEventListener('click', (event) => {
    if (event.target === els.settingsOverlay) {
      closeSettings();
    }
  });

  els.providerSelect.addEventListener('change', () => {
    fillModelOptions(els.providerSelect.value, els.modelSelect.value);
  });

  els.showKeysCheckbox.addEventListener('change', () => {
    const type = els.showKeysCheckbox.checked ? 'text' : 'password';
    els.llmApiInput.type = type;
    els.embeddingApiInput.type = type;
  });

  els.settingsForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const payload = {
      provider: els.providerSelect.value,
      model: els.modelSelect.value,
      embedding_model: els.embeddingSelect.value,
      llm_api_key: els.llmApiInput.value.trim(),
      embedding_api_key: els.embeddingApiInput.value.trim(),
      yc_folder_id: els.ycFolderInput.value.trim(),
      temperature: els.temperatureInput.value.trim(),
      max_tokens: els.maxTokensInput.value.trim(),
      top_p: els.topPInput.value.trim(),
      score_threshold: els.scoreThresholdInput.value.trim(),
    };

    try {
      await api('/api/settings', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      closeSettings();
      await refreshState();
    } catch (error) {
      alert(error.message);
      await refreshState();
    }
  });
}

async function bootstrap() {
  // Initial render + periodic polling to reflect background operations.
  bindEvents();
  await refreshState();

  window.setInterval(async () => {
    try {
      await refreshState();
    } catch (error) {
      console.error(error);
    }
  }, 2500);
}

bootstrap();
