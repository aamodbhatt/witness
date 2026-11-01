// Options page: where the local witness server lives.

const DEFAULT_SERVER = "http://127.0.0.1:8094";
const input = document.getElementById("server");
const result = document.getElementById("result");

function valid(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" && ["127.0.0.1", "localhost"].includes(parsed.hostname);
  } catch {
    return false;
  }
}

chrome.storage.sync.get({ server: DEFAULT_SERVER }).then(({ server }) => {
  input.value = server;
});

document.getElementById("save").addEventListener("click", async () => {
  const url = input.value.trim().replace(/\/+$/, "");
  if (!valid(url)) {
    result.textContent = "Use a local http URL, e.g. http://127.0.0.1:8094 (the extension only has permission for localhost).";
    return;
  }
  await chrome.storage.sync.set({ server: url });
  result.textContent = "Saved.";
});

document.getElementById("test").addEventListener("click", async () => {
  result.textContent = "Testing…";
  const reply = await chrome.runtime.sendMessage({ type: "health" });
  result.textContent = reply?.ok
    ? `Connected: ${reply.data.status}, verifier ${reply.data.verifier ?? "?"}`
    : `Could not reach the server: ${reply?.error ?? "no reply"}. Is \`witness serve\` running?`;
});
