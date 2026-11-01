// Popup: page summary, server status and the on/off switch.

const STATES = {
  idle: "Waiting for the page…",
  extracting: "Finding claims…",
  checking: "Checking claims against Wikipedia…",
  done: "Done.",
  off: "Paused on all pages.",
};

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function refresh() {
  const status = document.getElementById("status");
  const tab = await activeTab();
  let summary = null;
  try {
    summary = await chrome.tabs.sendMessage(tab.id, { type: "summary" });
  } catch {
    status.textContent = "witness does not run on this page.";
    return;
  }
  for (const key of ["supported", "refuted", "unverifiable"]) {
    document.getElementById(key).textContent = String(summary.counts[key]);
  }
  if (summary.state === "error") status.textContent = `Error: ${summary.error}`;
  else if (summary.state === "checking") {
    const done = summary.counts.supported + summary.counts.refuted + summary.counts.unverifiable;
    status.textContent = `Checked ${done} of ${summary.total} claims…`;
  } else status.textContent = STATES[summary.state] ?? summary.state;
}

async function main() {
  const { enabled } = await chrome.storage.local.get({ enabled: true });
  const toggle = document.getElementById("enabled");
  toggle.checked = enabled;
  toggle.addEventListener("change", async () => {
    await chrome.storage.local.set({ enabled: toggle.checked });
    const tab = await activeTab();
    try {
      await chrome.tabs.sendMessage(tab.id, { type: "set-enabled", enabled: toggle.checked });
    } catch {
      // The content script is not injected on this page (e.g. chrome:// URLs).
    }
    refresh();
  });
  document.getElementById("options").addEventListener("click", (ev) => {
    ev.preventDefault();
    chrome.runtime.openOptionsPage();
  });
  const health = await chrome.runtime.sendMessage({ type: "health" });
  document.getElementById("server").textContent = health?.ok
    ? `server: ${health.data.status}`
    : "server: offline";
  refresh();
  setInterval(refresh, 1000);
}

main();
