// In-browser AI chat using WebLLM (no API key)
// Uses a web worker so the UI stays responsive.

import * as webllm from "https://esm.run/@mlc-ai/web-llm";

const els = {
  model: document.getElementById("model"),
  temperature: document.getElementById("temperature"),
  tempValue: document.getElementById("tempValue"),
  systemPrompt: document.getElementById("systemPrompt"),
  loadModel: document.getElementById("loadModel"),
  clearChat: document.getElementById("clearChat"),
  exportTxt: document.getElementById("exportTxt"),
  exportJson: document.getElementById("exportJson"),
  progress: document.getElementById("progress"),
  progressBar: document.getElementById("progressBar"),
  progressText: document.getElementById("progressText"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  userInput: document.getElementById("userInput"),
  send: document.getElementById("send"),
  stop: document.getElementById("stop"),
};

const LS_KEYS = {
  settings: "tc_settings",
  chat: "tc_chat",
};

let state = {
  messages: [],
  engine: null,
  loading: false,
  streaming: false,
  abortController: null,
};

function saveSettings() {
  const settings = {
    model: els.model.value,
    temperature: parseFloat(els.temperature.value),
    system: els.systemPrompt.value,
  };
  localStorage.setItem(LS_KEYS.settings, JSON.stringify(settings));
}

function loadSettings() {
  const s = JSON.parse(localStorage.getItem(LS_KEYS.settings) || "{}");
  if (s.model) els.model.value = s.model;
  if (typeof s.temperature === "number") {
    els.temperature.value = String(s.temperature);
    els.tempValue.textContent = s.temperature.toFixed(2);
  }
  if (s.system) els.systemPrompt.value = s.system;
}

function saveChat() {
  localStorage.setItem(LS_KEYS.chat, JSON.stringify(state.messages));
}

function loadChat() {
  try {
    const msgs = JSON.parse(localStorage.getItem(LS_KEYS.chat) || "[]");
    state.messages = Array.isArray(msgs) ? msgs : [];
  } catch {
    state.messages = [];
  }
  renderMessages();
}

function addMessage(role, content) {
  const msg = { role, content };
  state.messages.push(msg);
  saveChat();
  renderMessages();
  scrollToBottom();
}

function updateLastAssistant(content) {
  const idx = state.messages.length - 1;
  if (idx >= 0 && state.messages[idx].role === "assistant") {
    state.messages[idx].content = content;
    renderMessages();
    scrollToBottom();
    saveChat();
  }
}

function renderMessages() {
  els.messages.innerHTML = "";
  state.messages.forEach((m) => {
    const item = document.createElement("div");
    item.className = `message ${m.role}`;
    const role = document.createElement("div");
    role.className = "role";
    role.textContent = m.role === "user" ? "U" : m.role === "assistant" ? "A" : "S";
    const content = document.createElement("div");
    content.className = "content";
    content.textContent = m.content;
    item.append(role, content);
    els.messages.appendChild(item);
  });
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setUIBusy(busy) {
  els.send.disabled = busy;
  els.stop.disabled = busy;
  els.model.disabled = busy;
  els.loadModel.disabled = busy;
}

function showProgress(show) {
  els.progress.hidden = !show;
}

function setProgress(value, text) {
  if (typeof value === "number") {
    const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
    els.progressBar.style.width = pct + "%";
  }
  if (text) els.progressText.textContent = text;
}

async function initEngine() {
  if (state.loading) return;
  state.loading = true;
  setUIBusy(true);
  showProgress(true);
  setProgress(0, "Preparing model…");

  // Reset engine if switching model
  state.engine = null;

  // Create a worker and engine
  let worker;
  try {
    worker = new Worker("https://esm.run/@mlc-ai/web-llm/dist/worker.js", { type: "module" });
  } catch (e) {
    console.error("Worker creation failed", e);
    setProgress(0, "Worker failed to initialize.");
    state.loading = false;
    setUIBusy(false);
    return;
  }

  try {
    const selectedModel = els.model.value;
    const temp = parseFloat(els.temperature.value);
    // Init with progress callback
    const progressCallback = (report) => {
      // report fields: progress, text (varies by version)
      const v = typeof report.progress === "number" ? report.progress : null;
      const t = report.text || "Loading…";
      setProgress(v, t);
    };

    state.engine = await webllm.CreateWebWorkerMLCEngine(
      worker,
      { model: selectedModel },
      { initProgressCallback: progressCallback }
    );

    // Set default generation params
    await state.engine.reload({ temperature: temp }).catch(() => {});
    setProgress(1, "Model ready.");
  } catch (err) {
    console.error(err);
    setProgress(0, "Failed to load model. Try the smaller model.");
  } finally {
    state.loading = false;
    setUIBusy(false);
    setTimeout(() => showProgress(false), 800);
  }
}

async function sendMessage(text) {
  if (!text.trim()) return;
  if (!state.engine) await initEngine();
  if (!state.engine) return;

  const temp = parseFloat(els.temperature.value);
  const system = els.systemPrompt.value.trim();

  // Build messages per OpenAI Chat spec
  const messages = [];
  if (system) messages.push({ role: "system", content: system });
  state.messages.forEach((m) => messages.push({ role: m.role, content: m.content }));
  messages.push({ role: "user", content: text });

  // Render user and placeholder assistant
  addMessage("user", text);
  addMessage("assistant", "");

  // Stream response
  state.streaming = true;
  state.abortController = new AbortController();
  setUIBusy(true);
  els.stop.disabled = false;

  try {
    let full = "";
    // Preferred streaming path (if supported by current WebLLM version)
    const stream = await state.engine.chat.completions.create({
      model: els.model.value,
      messages,
      temperature: temp,
      stream: true,
      signal: state.abortController.signal,
    });

    // Consume async iterator
    for await (const chunk of stream) {
      const delta =
        chunk?.choices?.[0]?.delta?.content ??
        chunk?.choices?.[0]?.message?.content ??
        "";
      if (delta) {
        full += delta;
        updateLastAssistant(full);
      }
    }

    // If nothing streamed, try non-streaming fallback
    if (!full) {
      const res = await state.engine.chat.completions.create({
        model: els.model.value,
        messages,
        temperature: temp,
        stream: false,
      });
      full = res?.choices?.[0]?.message?.content || "";
      updateLastAssistant(full);
    }
  } catch (err) {
    if (err?.name === "AbortError") {
      updateLastAssistant("[stopped]");
    } else {
      console.error(err);
      updateLastAssistant("Sorry — the model failed to respond. Try again or switch model.");
    }
  } finally {
    state.streaming = false;
    setUIBusy(false);
    els.stop.disabled = true;
    state.abortController = null;
    saveChat();
  }
}

// Handlers
els.temperature.addEventListener("input", () => {
  els.tempValue.textContent = parseFloat(els.temperature.value).toFixed(2);
  saveSettings();
});
els.model.addEventListener("change", () => saveSettings());
els.systemPrompt.addEventListener("input", () => saveSettings());
els.loadModel.addEventListener("click", async () => {
  await initEngine();
});
els.clearChat.addEventListener("click", () => {
  state.messages = [];
  saveChat();
  renderMessages();
});
els.exportTxt.addEventListener("click", () => {
  const lines = state.messages.map((m) => `${m.role.toUpperCase()}:\n${m.content}\n`);
  const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "tejas-chatbot-chat.txt";
  a.click();
});
els.exportJson.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(state.messages, null, 2)], { type: "application/json;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "tejas-chatbot-chat.json";
  a.click();
});
els.stop.addEventListener("click", () => {
  if (state.abortController) state.abortController.abort();
});

els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = els.userInput.value;
  els.userInput.value = "";
  await sendMessage(text);
});

els.userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

// Init
loadSettings();
loadChat();
els.tempValue.textContent = parseFloat(els.temperature.value).toFixed(2);
