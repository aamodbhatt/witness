// Service worker: the only part of the extension that talks to the witness server.
// Content scripts run in the page's origin, so their requests to 127.0.0.1 would hit
// CORS and Private Network Access checks; the worker has host permissions instead.

const DEFAULT_SERVER = "http://127.0.0.1:8094";

async function serverUrl() {
  const { server } = await chrome.storage.sync.get({ server: DEFAULT_SERVER });
  return String(server || DEFAULT_SERVER).replace(/\/+$/, "");
}

async function postJson(path, body) {
  const resp = await fetch(`${await serverUrl()}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`witness server answered ${resp.status} on ${path}`);
  return resp;
}

// Parses a text/event-stream body and calls onEvent(name, data) per event.
async function readEvents(resp, onEvent) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const name = /^event: (.*)$/m.exec(block)?.[1];
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (name && data) onEvent(name, JSON.parse(data));
    }
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === "claims") {
    postJson("/claims", { title: msg.title, paragraphs: msg.paragraphs })
      .then((r) => r.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err.message || err) }));
    return true; // async response
  }
  if (msg?.type === "health") {
    serverUrl()
      .then((base) => fetch(`${base}/health`))
      .then((r) => r.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err.message || err) }));
    return true;
  }
  if (msg?.type === "badge" && sender.tab?.id !== undefined) {
    const refuted = msg.counts?.refuted ?? 0;
    chrome.action.setBadgeBackgroundColor({ tabId: sender.tab.id, color: "#c62828" });
    chrome.action.setBadgeText({ tabId: sender.tab.id, text: refuted ? String(refuted) : "" });
  }
  return false;
});

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "witness-check") return;
  port.onMessage.addListener(async (msg) => {
    try {
      const resp = await postJson("/check", { claims: msg.claims });
      await readEvents(resp, (event, data) => port.postMessage({ event, data }));
    } catch (err) {
      port.postMessage({ event: "error", data: { message: String(err.message || err) } });
    }
  });
});
