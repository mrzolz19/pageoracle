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
  btnManageCustomProviders: document.getElementById('btnManageCustomProviders'),
  providerSelect: document.getElementById('providerSelect'),
  llmEndpointGroup: document.getElementById('llmEndpointGroup'),
  llmEndpointInput: document.getElementById('llmEndpointInput'),
  modelInput: document.getElementById('modelInput'),
  modelDatalist: document.getElementById('modelDatalist'),
  llmApiGroup: document.getElementById('llmApiGroup'),
  llmApiInput: document.getElementById('llmApiInput'),
  ycFolderGroup: document.getElementById('ycFolderGroup'),
  ycFolderInput: document.getElementById('ycFolderInput'),
  embeddingInput: document.getElementById('embeddingInput'),
  embeddingDatalist: document.getElementById('embeddingDatalist'),
  embeddingSameAsLlmCheckbox: document.getElementById('embeddingSameAsLlmCheckbox'),
  embeddingEndpointGroup: document.getElementById('embeddingEndpointGroup'),
  embeddingEndpointInput: document.getElementById('embeddingEndpointInput'),
  embeddingApiGroup: document.getElementById('embeddingApiGroup'),
  embeddingApiInput: document.getElementById('embeddingApiInput'),
  showKeysCheckbox: document.getElementById('showKeysCheckbox'),
  temperatureInput: document.getElementById('temperatureInput'),
  maxTokensInput: document.getElementById('maxTokensInput'),
  topPInput: document.getElementById('topPInput'),
  scoreThresholdInput: document.getElementById('scoreThresholdInput'),
  btnCancelSettings: document.getElementById('btnCancelSettings'),

  // Custom providers modal elements
  customProvidersOverlay: document.getElementById('customProvidersOverlay'),
  btnCloseCustomProviders: document.getElementById('btnCloseCustomProviders'),
  customProvidersList: document.getElementById('customProvidersList'),
  customFormHeading: document.getElementById('customFormHeading'),
  customProviderForm: document.getElementById('customProviderForm'),
  customEditIndex: document.getElementById('customEditIndex'),
  customNameInput: document.getElementById('customNameInput'),
  customEndpointInput: document.getElementById('customEndpointInput'),
  customApiKeyInput: document.getElementById('customApiKeyInput'),
  customModelInput: document.getElementById('customModelInput'),
  customEmbeddingModelInput: document.getElementById('customEmbeddingModelInput'),
  btnCancelCustomForm: document.getElementById('btnCancelCustomForm'),
  btnSaveCustomProvider: document.getElementById('btnSaveCustomProvider'),
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

function getCustomProviders() {
  if (appState && appState.settings && Array.isArray(appState.settings.custom_providers)) {
    return appState.settings.custom_providers;
  }
  return [];
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
  const currentVal = els.providerSelect.value;
  els.providerSelect.innerHTML = '';

  // Built-in providers
  const builtins = Object.keys(appState.providers || {});
  builtins.forEach((name) => {
    const option = createEl('option', '', name);
    option.value = name;
    els.providerSelect.appendChild(option);
  });

  // Custom provider profiles
  const customProviders = getCustomProviders();
  if (customProviders.length > 0) {
    const optGroup = document.createElement('optgroup');
    optGroup.label = 'Кастомные провайдеры';
    customProviders.forEach((p) => {
      const option = createEl('option', '', p.name);
      option.value = p.name;
      optGroup.appendChild(option);
    });
    els.providerSelect.appendChild(optGroup);
  }

  // Generic "Custom" option if not present
  const hasCustomOption = Array.from(els.providerSelect.options).some((o) => o.value === 'Custom');
  if (!hasCustomOption) {
    const customOpt = createEl('option', '', 'Custom (OpenAI-совместимый)');
    customOpt.value = 'Custom';
    els.providerSelect.appendChild(customOpt);
  }

  if (currentVal && Array.from(els.providerSelect.options).some((o) => o.value === currentVal)) {
    els.providerSelect.value = currentVal;
  }
}

function fillModelSuggestions(providerName) {
  els.modelDatalist.innerHTML = '';
  const builtinModels = appState.providers?.[providerName]?.models || [];

  const modelsSet = new Set(builtinModels);
  const matchingCustom = getCustomProviders().find((p) => p.name === providerName);
  if (matchingCustom && matchingCustom.model) {
    modelsSet.add(matchingCustom.model);
  }

  if (providerName === 'Custom' || matchingCustom) {
    ['llama3.2', 'llama3.1:8b', 'mistral-large', 'qwen2.5-coder:7b', 'deepseek-r1'].forEach((m) =>
      modelsSet.add(m)
    );
  }

  modelsSet.forEach((modelName) => {
    const opt = document.createElement('option');
    opt.value = modelName;
    els.modelDatalist.appendChild(opt);
  });
}

function fillEmbeddingSuggestions() {
  els.embeddingDatalist.innerHTML = '';
  const suggestions = new Set([
    ...(appState.embedding_options || []),
    'text-embedding-3-small',
    'text-embedding-3-large',
    'nomic-embed-text',
    'bge-large-en-v1.5',
  ]);

  suggestions.forEach((modelName) => {
    const opt = document.createElement('option');
    opt.value = modelName;
    els.embeddingDatalist.appendChild(opt);
  });
}

function updateProviderVisibility() {
  const provider = els.providerSelect.value;
  const customProviders = getCustomProviders();
  const matchingCustom = customProviders.find((p) => p.name === provider);
  const isCustom = provider === 'Custom' || Boolean(matchingCustom);

  if (isCustom) {
    els.llmEndpointGroup.classList.remove('hidden');
    els.ycFolderGroup.classList.add('hidden');
    if (matchingCustom) {
      if (matchingCustom.endpoint && !els.llmEndpointInput.value) {
        els.llmEndpointInput.value = matchingCustom.endpoint;
      }
      if (matchingCustom.model && !els.modelInput.value) {
        els.modelInput.value = matchingCustom.model;
      }
      if (matchingCustom.api_key && !els.llmApiInput.value) {
        els.llmApiInput.value = matchingCustom.api_key;
      }
      if (matchingCustom.embedding_model && !els.embeddingInput.value) {
        els.embeddingInput.value = matchingCustom.embedding_model;
      }
    }
  } else {
    els.llmEndpointGroup.classList.add('hidden');
    const usesYandex = provider === 'YandexGPT' || els.embeddingInput.value === 'text-search-doc/latest';
    els.ycFolderGroup.classList.toggle('hidden', !usesYandex);
  }

  fillModelSuggestions(provider);
  if (els.embeddingSameAsLlmCheckbox.checked) {
    updateEmbeddingSyncVisibility();
  }
}

function updateEmbeddingSyncVisibility() {
  const isSame = els.embeddingSameAsLlmCheckbox.checked;
  els.embeddingEndpointGroup.classList.toggle('hidden', isSame);
  els.embeddingApiGroup.classList.toggle('hidden', isSame);

  if (isSame) {
    els.embeddingEndpointInput.value = els.llmEndpointInput.value;
    els.embeddingApiInput.value = els.llmApiInput.value;
  }
}

function openSettings() {
  // Populate form from current persisted server-side settings.
  fillProviderOptions();
  fillEmbeddingSuggestions();

  const settings = appState.settings || {};
  els.providerSelect.value = settings.provider || 'YandexGPT';
  els.llmEndpointInput.value = settings.llm_endpoint || '';
  els.modelInput.value = settings.model || '';
  els.llmApiInput.value = settings.llm_api_key || '';
  els.ycFolderInput.value = settings.yc_folder_id || '';

  els.embeddingInput.value = settings.embedding_model || 'nvidia/llama-nemotron-embed-vl-1b-v2:free';
  els.embeddingSameAsLlmCheckbox.checked = Boolean(settings.embedding_same_as_llm);
  els.embeddingEndpointInput.value = settings.embedding_endpoint || '';
  els.embeddingApiInput.value = settings.embedding_api_key || '';

  els.temperatureInput.value = String(settings.temperature ?? 0.2);
  els.maxTokensInput.value = String(settings.max_tokens ?? 4096);
  els.topPInput.value = String(settings.top_p ?? 0.9);
  els.scoreThresholdInput.value = String(settings.score_threshold ?? 0.6);

  els.showKeysCheckbox.checked = false;
  els.llmApiInput.type = 'password';
  els.embeddingApiInput.type = 'password';

  updateProviderVisibility();
  updateEmbeddingSyncVisibility();

  els.settingsOverlay.classList.remove('hidden');
}

function closeSettings() {
  els.settingsOverlay.classList.add('hidden');
}

function renderCustomProvidersList() {
  els.customProvidersList.innerHTML = '';
  const providers = getCustomProviders();

  if (providers.length === 0) {
    els.customProvidersList.appendChild(
      createEl('div', 'empty-text', 'Кастомные провайдеры еще не добавлены')
    );
    return;
  }

  providers.forEach((item, index) => {
    const card = createEl('div', 'custom-provider-item');

    const info = createEl('div', 'custom-provider-info');
    const name = createEl('div', 'custom-provider-name', item.name);
    const endpoint = createEl('div', 'custom-provider-endpoint', item.endpoint);
    const models = createEl(
      'div',
      'custom-provider-models',
      `LLM: ${item.model || 'не указана'} | Эмбеддинги: ${item.embedding_model || 'не указана'}`
    );
    info.append(name, endpoint, models);

    const actions = createEl('div', 'custom-provider-actions');
    const btnEdit = createEl('button', 'btn-icon-action', '✏ Ред.');
    btnEdit.type = 'button';
    btnEdit.addEventListener('click', () => editCustomProvider(index));

    const btnDelete = createEl('button', 'btn-icon-action delete', '🗑 Удалить');
    btnDelete.type = 'button';
    btnDelete.addEventListener('click', () => deleteCustomProvider(index));

    actions.append(btnEdit, btnDelete);
    card.append(info, actions);
    els.customProvidersList.appendChild(card);
  });
}

function resetCustomProviderForm() {
  els.customEditIndex.value = '-1';
  els.customFormHeading.textContent = 'Добавить провайдер';
  els.customNameInput.value = '';
  els.customEndpointInput.value = '';
  els.customApiKeyInput.value = '';
  els.customModelInput.value = '';
  els.customEmbeddingModelInput.value = '';
}

function editCustomProvider(index) {
  const providers = getCustomProviders();
  const item = providers[index];
  if (!item) {
    return;
  }

  els.customEditIndex.value = String(index);
  els.customFormHeading.textContent = 'Редактировать провайдер';
  els.customNameInput.value = item.name || '';
  els.customEndpointInput.value = item.endpoint || '';
  els.customApiKeyInput.value = item.api_key || '';
  els.customModelInput.value = item.model || '';
  els.customEmbeddingModelInput.value = item.embedding_model || '';
  els.customNameInput.focus();
}

async function deleteCustomProvider(index) {
  const providers = [...getCustomProviders()];
  const item = providers[index];
  if (!item) {
    return;
  }

  if (!confirm(`Удалить кастомный провайдер "${item.name}"?`)) {
    return;
  }

  providers.splice(index, 1);
  const currentSettings = { ...(appState?.settings || {}), custom_providers: providers };

  if (currentSettings.provider === item.name) {
    currentSettings.provider = 'YandexGPT';
    currentSettings.llm_endpoint = '';
  }

  try {
    await api('/api/settings', {
      method: 'POST',
      body: JSON.stringify(currentSettings),
    });
    if (appState?.settings) {
      appState.settings.custom_providers = providers;
    }
    renderCustomProvidersList();
    fillProviderOptions();
    resetCustomProviderForm();
  } catch (error) {
    alert(error.message);
  }
}

async function saveCustomProviderProfile(event) {
  event.preventDefault();
  const name = els.customNameInput.value.trim();
  const endpoint = els.customEndpointInput.value.trim();

  if (!name) {
    alert('Укажите название провайдера.');
    els.customNameInput.focus();
    return;
  }
  if (!endpoint) {
    alert('Укажите Endpoint URL.');
    els.customEndpointInput.focus();
    return;
  }

  const apiKey = els.customApiKeyInput.value.trim();
  const model = els.customModelInput.value.trim();
  const embeddingModel = els.customEmbeddingModelInput.value.trim();

  const providers = [...getCustomProviders()];
  const editIndex = parseInt(els.customEditIndex.value, 10);
  const profile = {
    name,
    endpoint,
    api_key: apiKey,
    model,
    embedding_model: embeddingModel,
  };

  if (editIndex >= 0 && editIndex < providers.length) {
    providers[editIndex] = profile;
  } else {
    // Avoid duplicate names
    const existingIdx = providers.findIndex((p) => p.name === name);
    if (existingIdx >= 0) {
      providers[existingIdx] = profile;
    } else {
      providers.push(profile);
    }
  }

  const currentSettings = { ...(appState?.settings || {}), custom_providers: providers };

  try {
    await api('/api/settings', {
      method: 'POST',
      body: JSON.stringify(currentSettings),
    });
    if (appState?.settings) {
      appState.settings.custom_providers = providers;
    }
    renderCustomProvidersList();
    resetCustomProviderForm();
    fillProviderOptions();
    els.providerSelect.value = name;
    updateProviderVisibility();
  } catch (error) {
    alert(error.message);
  }
}

function openCustomProviders() {
  renderCustomProvidersList();
  resetCustomProviderForm();
  els.customProvidersOverlay.classList.remove('hidden');
}

function closeCustomProviders() {
  els.customProvidersOverlay.classList.add('hidden');
  fillProviderOptions();
  updateProviderVisibility();
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

  els.btnManageCustomProviders.addEventListener('click', openCustomProviders);
  els.btnCloseCustomProviders.addEventListener('click', closeCustomProviders);
  els.customProvidersOverlay.addEventListener('click', (event) => {
    if (event.target === els.customProvidersOverlay) {
      closeCustomProviders();
    }
  });

  els.customProviderForm.addEventListener('submit', saveCustomProviderProfile);
  els.btnCancelCustomForm.addEventListener('click', resetCustomProviderForm);

  els.providerSelect.addEventListener('change', updateProviderVisibility);

  els.embeddingSameAsLlmCheckbox.addEventListener('change', updateEmbeddingSyncVisibility);

  els.llmEndpointInput.addEventListener('input', () => {
    if (els.embeddingSameAsLlmCheckbox.checked) {
      els.embeddingEndpointInput.value = els.llmEndpointInput.value;
    }
  });

  els.llmApiInput.addEventListener('input', () => {
    if (els.embeddingSameAsLlmCheckbox.checked) {
      els.embeddingApiInput.value = els.llmApiInput.value;
    }
  });

  els.embeddingInput.addEventListener('input', () => {
    const usesYandex =
      els.providerSelect.value === 'YandexGPT' ||
      els.embeddingInput.value.trim() === 'text-search-doc/latest';
    els.ycFolderGroup.classList.toggle('hidden', !usesYandex);
  });

  els.showKeysCheckbox.addEventListener('change', () => {
    const type = els.showKeysCheckbox.checked ? 'text' : 'password';
    els.llmApiInput.type = type;
    els.embeddingApiInput.type = type;
  });

  els.settingsForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const isSame = els.embeddingSameAsLlmCheckbox.checked;
    const llmEndpoint = els.llmEndpointInput.value.trim();
    const llmApiKey = els.llmApiInput.value.trim();

    const payload = {
      provider: els.providerSelect.value,
      model: els.modelInput.value.trim(),
      embedding_model: els.embeddingInput.value.trim(),
      llm_api_key: llmApiKey,
      embedding_api_key: isSame ? llmApiKey : els.embeddingApiInput.value.trim(),
      llm_endpoint: llmEndpoint,
      embedding_endpoint: isSame ? llmEndpoint : els.embeddingEndpointInput.value.trim(),
      embedding_same_as_llm: isSame,
      yc_folder_id: els.ycFolderInput.value.trim(),
      custom_providers: getCustomProviders(),
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
