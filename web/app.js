/* Engram UI: chat plus live memory inspector. */

const $ = (id) => document.getElementById(id);
const chatLog = $("chat-log");

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
  });
});

function switchTab(name) {
  document.querySelector(`.tab[data-tab="${name}"]`).click();
}

// ---------- helpers ----------
async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${path} → ${response.status}`);
  return response.json();
}

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

function retentionColor(retention) {
  if (retention > 0.6) return "var(--amber)";
  if (retention > 0.25) return "var(--amber-dim)";
  return "var(--rose)";
}

function fmtClock(hours) {
  if (hours < 1) return "+0h";
  if (hours < 48) return `+${Math.round(hours)}h`;
  return `+${(hours / 24).toFixed(1)}d`;
}

function addMsg(cls, text) {
  const node = el("div", "msg " + cls);
  node.textContent = text;
  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
  return node;
}

// ---------- stats ----------
async function refreshStats() {
  const stats = await api("/api/stats");
  $("stat-beliefs").textContent = stats.beliefs_current;
  $("stat-episodes").textContent = stats.episodes_active + stats.summaries;
  $("stat-archived").textContent = stats.episodes_archived;
  $("stat-clock").textContent = fmtClock(stats.clock_offset_hours);
}

// ---------- recall trace ----------
function renderRecall(trace, tokensUsed, budget) {
  $("budget-label").textContent = `memory budget · ${tokensUsed}/${budget} tok`;
  $("budget-fill").style.width = Math.min(100, (tokensUsed / budget) * 100) + "%";

  const list = $("recall-list");
  list.innerHTML = "";
  if (!trace.length) {
    list.appendChild(el("p", "empty", "Nothing surfaced from long-term memory this turn."));
    return;
  }
  trace.forEach((memory, i) => {
    const card = el("div", "card flash");
    card.style.animationDelay = `${i * 70}ms`;
    card.appendChild(el("div", "meta", `
      <span class="kind ${memory.kind}">${memory.kind}</span>
      <span>strength ${memory.score.toFixed(2)}</span>
      <span>sim ${memory.similarity.toFixed(2)}</span>
      <span>${memory.tokens} tok</span>`));
    card.appendChild(el("div", "text", esc(memory.text)));
    const bar = el("div", "retention-bar");
    const fill = el("i");
    fill.style.width = memory.retention * 100 + "%";
    fill.style.background = retentionColor(memory.retention);
    bar.appendChild(fill);
    card.appendChild(bar);
    card.appendChild(el("div", "reinforce-note",
      `recalled → reinforced: stability ${memory.stability_before_h}h → ${memory.stability_after_h}h`));
    list.appendChild(card);
  });
}

// ---------- beliefs ----------
async function refreshBeliefs() {
  const beliefs = await api("/api/beliefs?include_superseded=true");
  const list = $("beliefs-list");
  list.innerHTML = "";
  if (!beliefs.length) {
    list.appendChild(el("p", "empty", "No beliefs learned yet."));
    return;
  }
  beliefs.forEach((belief) => {
    const card = el("div", "card");
    card.appendChild(el("div", "meta", `
      <span class="kind belief">belief</span>
      <span>conf ${belief.confidence.toFixed(2)}</span>
      <span>recalled ×${belief.access_count}</span>
      <span>${belief.current ? "current" : "superseded"}</span>`));
    card.appendChild(el("div", `text ${belief.current ? "" : "gone"}`, esc(belief.statement)));
    if (!belief.current) {
      card.appendChild(el("div", "supersede-chain", `↳ no longer held (bi-temporal record kept)`));
    } else {
      const bar = el("div", "retention-bar");
      const fill = el("i");
      fill.style.width = belief.retention * 100 + "%";
      fill.style.background = retentionColor(belief.retention);
      bar.appendChild(fill);
      card.appendChild(bar);
    }
    list.appendChild(card);
  });
}

// ---------- episodes ----------
async function refreshEpisodes() {
  const episodes = await api("/api/memories?status=all");
  const list = $("episodes-list");
  list.innerHTML = "";
  if (!episodes.length) {
    list.appendChild(el("p", "empty", "No episodes stored yet."));
    return;
  }
  episodes.slice(0, 80).forEach((episode) => {
    const card = el("div", "card");
    const kind = episode.kind === "summary" ? "summary" : "episode";
    card.appendChild(el("div", "meta", `
      <span class="kind ${kind}">${kind}</span>
      <span>${esc(episode.session_id)}</span>
      <span>imp ${episode.importance.toFixed(2)}</span>
      <span>ret ${episode.retention.toFixed(2)}</span>
      <span>${episode.status}</span>`));
    const cls = episode.status === "archived" ? "text gone"
      : episode.retention < 0.25 ? "text faded" : "text";
    card.appendChild(el("div", cls, esc(`[${episode.role}] ${episode.content.slice(0, 220)}`)));
    if (episode.status !== "archived") {
      const bar = el("div", "retention-bar");
      const fill = el("i");
      fill.style.width = episode.retention * 100 + "%";
      fill.style.background = retentionColor(episode.retention);
      bar.appendChild(fill);
      card.appendChild(bar);
    }
    list.appendChild(card);
  });
}

// ---------- events ----------
async function refreshEvents() {
  const events = await api("/api/events");
  const list = $("events-list");
  list.innerHTML = "";
  if (!events.length) {
    list.appendChild(el("p", "empty", "Nothing logged yet."));
    return;
  }
  events.forEach((event) => {
    const row = el("div", "event-row");
    row.appendChild(el("span", "etype", esc(event.type)));
    row.appendChild(el("span", "epayload", esc(JSON.stringify(event.payload))));
    list.appendChild(row);
  });
}

async function refreshAll() {
  await Promise.all([refreshStats(), refreshBeliefs(), refreshEpisodes(), refreshEvents()]);
}

// ---------- chat ----------
const form = $("chat-form");
const input = $("chat-input");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  addMsg("user", message);
  const pending = addMsg("assistant thinking", "recalling…");
  $("btn-send").disabled = true;
  try {
    const result = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: $("session").value.trim() || "default", message }),
    });
    pending.classList.remove("thinking");
    pending.textContent = result.reply;
    renderRecall(result.recall_trace, result.memory_tokens_used, result.memory_token_budget);
    switchTab("recall");
    if (result.sleep_report) {
      addMsg("toast", `☾ auto sleep cycle: ${result.sleep_report.episodes_archived} episodes consolidated into ${result.sleep_report.summaries_created} summaries`);
    }
    // Perception runs in the background; refresh again shortly to catch new beliefs.
    setTimeout(refreshAll, 3500);
    setTimeout(refreshAll, 9000);
  } catch (error) {
    pending.textContent = "⚠ " + error.message;
  } finally {
    $("btn-send").disabled = false;
    refreshAll();
    input.focus();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

// ---------- controls ----------
$("btn-new-session").addEventListener("click", () => {
  const current = $("session").value.match(/(\d+)$/);
  const next = current ? Number(current[1]) + 1 : 2;
  $("session").value = `session-${next}`;
  addMsg("toast", `new session started: session-${next} (working memory cleared, long-term memory persists)`);
});

async function warp(hours) {
  const result = await api("/api/timewarp", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hours }),
  });
  addMsg("toast", `⏩ simulated clock advanced ${hours / 24} days (total ${fmtClock(result.total_offset_hours)}), watch retention decay`);
  refreshAll();
  switchTab("episodes");
}

$("btn-warp-week").addEventListener("click", () => warp(24 * 7));
$("btn-warp-month").addEventListener("click", () => warp(24 * 30));

$("btn-sleep").addEventListener("click", async () => {
  addMsg("toast", "☾ running sleep cycle, consolidating faded episodes…");
  const report = await api("/api/consolidate", { method: "POST" });
  addMsg("toast", `☾ sleep cycle done: ${report.episodes_archived} episodes → ${report.summaries_created} summaries (${report.faded_candidates} faded candidates)`);
  refreshAll();
  switchTab("episodes");
});

refreshAll();
setInterval(refreshStats, 15000);
