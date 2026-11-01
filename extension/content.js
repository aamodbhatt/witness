// Content script: finds the article's paragraphs, asks the local server which sentences
// are check-worthy claims, underlines them in place and streams in verdicts.
(() => {
  if (window.__witnessLoaded) return;
  window.__witnessLoaded = true;

  const BLOCKS = "p, li, blockquote, dd";
  const SKIP_ANCESTORS = "nav, header, footer, aside, form, [role=navigation], [contenteditable]";
  const MIN_CHARS = 40;
  const MAX_BLOCKS = 120;
  const COLORS = { supported: "#1a7f4b", refuted: "#c62828", unverifiable: "#7d7d82" };
  const LABELS = { supported: "Supported", refuted: "Refuted", unverifiable: "Can't tell" };

  const verdicts = new Map(); // claim id -> verdict payload
  const claimsById = new Map();
  let state = "idle"; // idle | extracting | checking | done | error | off
  let errorMessage = "";
  let card = null;
  let hideTimer = null;

  // ---------------------------------------------------------------- page text

  function contentRoot() {
    return document.querySelector("article, main, [role=main], #mw-content-text, #content") || document.body;
  }

  function collectBlocks() {
    const root = contentRoot();
    const blocks = [];
    for (const el of root.querySelectorAll(BLOCKS)) {
      if (el.closest(SKIP_ANCESTORS)) continue;
      if (el.parentElement && el.parentElement.closest(BLOCKS)) continue; // nested block
      if (el.textContent.trim().length < MIN_CHARS) continue;
      if (!el.getClientRects().length) continue; // hidden
      blocks.push(el);
      if (blocks.length >= MAX_BLOCKS) break;
    }
    return blocks;
  }

  // Wraps characters [start, end) of block.textContent in spans, one per text node, so
  // links and inline formatting inside the claim survive.
  function wrapRange(block, start, end, id) {
    const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
    const targets = [];
    let offset = 0;
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const len = node.nodeValue.length;
      const s = Math.max(start, offset);
      const e = Math.min(end, offset + len);
      if (s < e) targets.push([node, s - offset, e - offset]);
      offset += len;
      if (offset >= end) break;
    }
    for (const [node, s, e] of targets) {
      let middle = node;
      if (e < middle.nodeValue.length) middle.splitText(e);
      if (s > 0) middle = middle.splitText(s);
      if (!middle.nodeValue.trim()) continue;
      const span = document.createElement("span");
      span.className = "witness-claim witness-pending";
      span.dataset.witnessId = id;
      middle.parentNode.insertBefore(span, middle);
      span.appendChild(middle);
    }
  }

  function unwrapAll() {
    for (const span of document.querySelectorAll("span.witness-claim")) {
      const parent = span.parentNode;
      while (span.firstChild) parent.insertBefore(span.firstChild, span);
      parent.removeChild(span);
      parent.normalize();
    }
    hideCard(true);
  }

  // ---------------------------------------------------------------- hover card

  function ensureCard() {
    if (card) return card;
    const host = document.createElement("div");
    host.id = "witness-card-host";
    host.style.cssText = "position:absolute;top:0;left:0;z-index:2147483647;display:none;";
    const shadow = host.attachShadow({ mode: "open" });
    shadow.innerHTML = `
      <style>
        .card { width: 380px; max-width: calc(100vw - 32px); background: #fff; color: #1d1d1f;
          border: 1px solid #e3e0d8; border-radius: 12px; padding: 14px 16px;
          box-shadow: 0 12px 40px rgba(0,0,0,.18); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
        .top { display: flex; align-items: center; gap: 8px; }
        .badge { color: #fff; font-weight: 700; font-size: 11px; letter-spacing: .06em;
          text-transform: uppercase; padding: 3px 9px; border-radius: 999px; }
        .conf { font-weight: 600; }
        .bar { height: 6px; background: #eeebe4; border-radius: 3px; margin: 10px 0; overflow: hidden; }
        .bar span { display: block; height: 100%; border-radius: 3px; }
        .query { color: #6b6b70; font-size: 12.5px; margin-bottom: 8px; }
        .ev { margin: 8px 0 0; padding-left: 10px; border-left: 3px solid #e3e0d8; }
        .ev a { color: #1d1d1f; font-weight: 650; text-decoration: none; }
        .ev a:hover { text-decoration: underline; }
        .foot { margin-top: 10px; color: #8a8a8f; font-size: 11.5px; }
        .empty { color: #6b6b70; font-style: italic; }
      </style>
      <div class="card" part="card"></div>`;
    host.addEventListener("mouseenter", () => clearTimeout(hideTimer));
    host.addEventListener("mouseleave", () => hideCard());
    document.documentElement.appendChild(host);
    card = host;
    return host;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  function showCard(span) {
    clearTimeout(hideTimer);
    const id = span.dataset.witnessId;
    const v = verdicts.get(id);
    const host = ensureCard();
    const body = host.shadowRoot.querySelector(".card");
    if (!v) {
      body.innerHTML = `<div class="empty">Checking this claim against Wikipedia…</div>`;
    } else {
      const color = COLORS[v.verdict];
      const pct = Math.round(v.confidence * 100);
      const evidence = v.evidence.length
        ? v.evidence.map((e) => `<div class="ev"><a href="${escapeHtml(e.url)}" target="_blank" rel="noopener">${escapeHtml(e.title)}</a>: ${escapeHtml(e.sentence)}</div>`).join("")
        : `<div class="empty">No relevant Wikipedia sentence found.</div>`;
      body.innerHTML = `
        <div class="top"><span class="badge" style="background:${color}">${LABELS[v.verdict]}</span>
          <span class="conf">${pct}% confidence</span></div>
        <div class="bar"><span style="width:${pct}%;background:${color}"></span></div>
        ${v.query !== v.claim ? `<div class="query">Checked as: “${escapeHtml(v.query)}”</div>` : ""}
        ${evidence}
        <div class="foot">witness · Wikipedia snapshot of June 2017</div>`;
    }
    const rect = span.getBoundingClientRect();
    host.style.display = "block";
    const width = Math.min(380, window.innerWidth - 32);
    const left = Math.min(Math.max(8, rect.left), window.innerWidth - width - 16);
    host.style.left = `${left + window.scrollX}px`;
    host.style.top = `${rect.bottom + window.scrollY + 8}px`;
    document.querySelectorAll(`span.witness-claim[data-witness-id="${id}"]`).forEach((n) => n.classList.add("witness-active"));
  }

  function hideCard(now = false) {
    clearTimeout(hideTimer);
    const hide = () => {
      if (card) card.style.display = "none";
      document.querySelectorAll("span.witness-active").forEach((n) => n.classList.remove("witness-active"));
    };
    if (now) hide();
    else hideTimer = setTimeout(hide, 250);
  }

  document.addEventListener("mouseover", (ev) => {
    const span = ev.target instanceof Element ? ev.target.closest("span.witness-claim") : null;
    if (span) showCard(span);
  });
  document.addEventListener("mouseout", (ev) => {
    const span = ev.target instanceof Element ? ev.target.closest("span.witness-claim") : null;
    if (span) hideCard();
  });

  // ---------------------------------------------------------------- checking

  function counts() {
    const out = { supported: 0, refuted: 0, unverifiable: 0 };
    for (const v of verdicts.values()) out[v.verdict] += 1;
    return out;
  }

  function applyVerdict(v) {
    verdicts.set(v.id, v);
    for (const span of document.querySelectorAll(`span.witness-claim[data-witness-id="${v.id}"]`)) {
      span.classList.remove("witness-pending");
      span.classList.add(`witness-${v.verdict}`);
    }
    chrome.runtime.sendMessage({ type: "badge", counts: counts() });
  }

  async function run() {
    verdicts.clear();
    claimsById.clear();
    state = "extracting";
    const blocks = collectBlocks();
    if (!blocks.length) { state = "done"; return; }
    const reply = await chrome.runtime.sendMessage({
      type: "claims", title: document.title, paragraphs: blocks.map((b) => b.textContent),
    });
    if (!reply?.ok) { state = "error"; errorMessage = reply?.error || "no reply"; return; }
    const claims = reply.data.claims;
    for (const c of claims) {
      claimsById.set(c.id, c);
      wrapRange(blocks[c.paragraph], c.start, c.end, c.id);
    }
    if (!claims.length) { state = "done"; return; }
    state = "checking";
    const port = chrome.runtime.connect({ name: "witness-check" });
    port.onMessage.addListener(({ event, data }) => {
      if (event === "verdict") applyVerdict(data);
      else if (event === "done") { state = "done"; port.disconnect(); }
      else if (event === "error") { state = "error"; errorMessage = data.message; }
    });
    port.postMessage({ claims: claims.map((c) => ({ id: c.id, text: c.text, query: c.query })) });
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.type === "summary") {
      sendResponse({ state, error: errorMessage, total: claimsById.size, counts: counts() });
    } else if (msg?.type === "set-enabled") {
      if (msg.enabled) { if (state === "off") run(); }
      else { unwrapAll(); verdicts.clear(); claimsById.clear(); state = "off"; chrome.runtime.sendMessage({ type: "badge", counts: counts() }); }
      sendResponse({ state });
    }
    return false;
  });

  chrome.storage.local.get({ enabled: true }).then(({ enabled }) => {
    if (enabled) run();
    else state = "off";
  });
})();
