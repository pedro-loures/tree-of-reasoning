window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { utils } = TD;

const SURGEON_PROMPT =
  "A boy is in an accident. He is rushed to the hospital. The surgeon, who is the boy's father, enters the room and says, \"I cannot operate on this boy, he is my son.\" Who is the surgeon to the boy?";
const PROMPT_MASK_MIN_LEN = 4;
const SEQ_COLOR_META = [
  { id: "green", label: "Green · 100% good", cls: "col-green", fill: "var(--good-node)" },
  { id: "blue", label: "Blue · <25% bad", cls: "col-blue", fill: "var(--mostly-good)" },
  { id: "grey", label: "Grey · 25–75%", cls: "col-grey", fill: "var(--internal)" },
  { id: "yellow", label: "Yellow · >75% bad", cls: "col-yellow", fill: "var(--mostly-bad)" },
  { id: "red", label: "Red · 100% bad", cls: "col-red", fill: "var(--bad-node)" },
];

function displayTok(tok) {
  if (tok === "\n") return "\\n";
  if (tok === "\t") return "\\t";
  if (tok === " ") return "·";
  return tok;
}

function normalizePromptText(text) {
  return String(text)
    .replace(/·/g, " ")
    .replace(/\\n/g, "\n")
    .replace(/[’‘]/g, "'")
    .replace(/[“”]/g, '"');
}

function compactPromptText(text) {
  return normalizePromptText(text).toLowerCase().replace(/\s+/g, " ").trim();
}

function promptEdgeSpanOk(spanText, tokenN = 1, minLen = PROMPT_MASK_MIN_LEN) {
  const compact = compactPromptText(spanText);
  return compact.length >= minLen && (tokenN >= 2 || compact.length >= 6);
}

function collapsePromptMarks(text) {
  return String(text)
    .replace(/(\(prompt\))(?:\s*\(prompt\))+/g, "(prompt)")
    .replace(/\s+\(prompt\)/g, " (prompt)")
    .replace(/\(prompt\)(?=\S)/g, "(prompt) ")
    .replace(/\s+/g, " ")
    .trim();
}

function longestPromptSuffixPrefix(seq, promptCompact, minLen = PROMPT_MASK_MIN_LEN) {
  for (let len = seq.length; len >= minLen; len--) {
    const compact = compactPromptText(seq.slice(0, len));
    if (compact.length >= minLen && promptCompact.endsWith(compact)) return len;
  }
  return 0;
}

function longestPromptPrefixSuffix(seq, promptCompact, minLen = PROMPT_MASK_MIN_LEN) {
  for (let len = seq.length; len >= minLen; len--) {
    const compact = compactPromptText(seq.slice(seq.length - len));
    if (compact.length >= minLen && promptCompact.startsWith(compact)) return len;
  }
  return 0;
}

function maskPromptSubstrings(text, prompt = SURGEON_PROMPT, minLen = PROMPT_MASK_MIN_LEN) {
  const seq = normalizePromptText(text);
  const promptCompact = compactPromptText(prompt);
  if (promptCompact.includes(compactPromptText(seq))) return "(prompt)";
  const prefixLen = longestPromptSuffixPrefix(seq, promptCompact, minLen);
  const suffixLen = longestPromptPrefixSuffix(seq, promptCompact, minLen);
  if (prefixLen && suffixLen && prefixLen + suffixLen >= seq.length) return "(prompt)";
  const mid = seq.slice(prefixLen, suffixLen ? seq.length - suffixLen : seq.length);
  const parts = [];
  if (prefixLen) parts.push("(prompt)");
  if (mid) parts.push(mid);
  if (suffixLen) parts.push("(prompt)");
  return collapsePromptMarks(parts.join(""));
}

function maskPromptTokens(tokens, prompt = SURGEON_PROMPT) {
  if (!tokens || !tokens.length) return "";
  const promptCompact = compactPromptText(prompt);
  if (promptCompact.includes(compactPromptText(tokens.join("")))) return "(prompt)";
  let prefixN = 0;
  for (let n = tokens.length; n >= 1; n--) {
    const span = tokens.slice(0, n).join("");
    if (promptEdgeSpanOk(span, n) && promptCompact.endsWith(compactPromptText(span))) {
      prefixN = n;
      break;
    }
  }
  let suffixN = 0;
  for (let n = tokens.length; n >= 1; n--) {
    const span = tokens.slice(tokens.length - n).join("");
    if (promptEdgeSpanOk(span, n) && promptCompact.startsWith(compactPromptText(span))) {
      suffixN = n;
      break;
    }
  }
  if (prefixN && suffixN && prefixN + suffixN >= tokens.length) return "(prompt)";
  const parts = [];
  if (prefixN) parts.push("(prompt)");
  const end = tokens.length - (suffixN || 0);
  for (let i = prefixN; i < end; i++) parts.push(displayTok(tokens[i]));
  if (suffixN) parts.push("(prompt)");
  return collapsePromptMarks(parts.join(""));
}

function formatSeqDisplay(display) {
  return utils.escapeHtml(display).split("(prompt)").join('<span class="seq-prompt-token">(prompt)</span>');
}

function emptyColorCounts() {
  return Object.fromEntries(SEQ_COLOR_META.map(meta => [meta.id, 0]));
}

function finalizeSeqRow(row) {
  const colors = SEQ_COLOR_META.filter(meta => (row.counts[meta.id] || 0) > 0).map(meta => meta.id);
  row.total = SEQ_COLOR_META.reduce((sum, meta) => sum + (row.counts[meta.id] || 0), 0);
  row.colors = colors;
  row.color_n = colors.length;
  row.shared = colors.length > 1;
  return row;
}

function assignSeqRanks(rows) {
  for (const row of rows) row.ranks = {};
  for (const meta of SEQ_COLOR_META) {
    [...rows]
      .filter(row => (row.counts[meta.id] || 0) > 0)
      .sort((a, b) => (b.counts[meta.id] || 0) - (a.counts[meta.id] || 0) || a.display.localeCompare(b.display))
      .forEach((row, index) => { row.ranks[meta.id] = index + 1; });
  }
  return rows;
}

function aggregateMaskedSequences(rows) {
  const groups = new Map();
  for (const row of rows) {
    const display = maskSequenceDisplay(row);
    if (!groups.has(display)) {
      groups.set(display, {
        display,
        tokens: row.tokens,
        counts: emptyColorCounts(),
        leaf_counts: emptyColorCounts(),
        source_n: 0,
        masked: true,
      });
    }
    const group = groups.get(display);
    group.source_n += 1;
    for (const meta of SEQ_COLOR_META) {
      group.counts[meta.id] += row.counts[meta.id] || 0;
      group.leaf_counts[meta.id] = Math.max(
        group.leaf_counts[meta.id],
        (row.leaf_counts && row.leaf_counts[meta.id]) || 0
      );
    }
  }
  return assignSeqRanks([...groups.values()].map(finalizeSeqRow));
}

function maskSequenceDisplay(row) {
  if (row.entirely_prompt) return "(prompt)";
  const joined = row.tokens && row.tokens.length ? row.tokens.join("") : row.display;
  if (compactPromptText(SURGEON_PROMPT).includes(compactPromptText(joined))) return "(prompt)";
  if (row.tokens && row.tokens.length) return maskPromptTokens(row.tokens);
  return maskPromptSubstrings(row.display);
}

function formatPct(pct) {
  if (pct == null || Number.isNaN(pct)) return "—";
  if (pct === 0) return "0%";
  if (pct >= 10) return `${pct.toFixed(1)}%`;
  if (pct >= 1) return `${pct.toFixed(2)}%`;
  return `${pct.toFixed(3)}%`;
}

function formatLeafShare(numer, denom) {
  if (!denom) return "—";
  return `${formatPct(100 * numer / denom)} (${numer}/${denom})`;
}

TD.ColorSequencesPanel = class {
  constructor(root, cfg) {
    this.root = root;
    this.cfg = cfg;
    this.prefix = cfg.prefix;
    this.store = cfg.store;
    this.fileField = cfg.fileField;
    this.defaultFile = cfg.defaultFile;
    this.countLabel = cfg.countLabel;
    this.windowLabel = cfg.windowLabel;
    this.data = null;
    this.selectedKey = null;
    this.sortKey = "total";
    this.sortAsc = false;
    this.bound = false;
  }

  id(name) {
    return `${this.prefix}${name}`;
  }

  el(name) {
    return this.root.querySelector(`#${this.id(name)}`);
  }

  async ensureLoaded() {
    if (this.data) return this.data;
    const file = this.store.manifest?.[this.fileField] || this.defaultFile;
    const response = await fetch(`${this.store.baseUrl}/${file}?ts=${Date.now()}`);
    if (!response.ok) throw new Error(`${file}: ${response.statusText}`);
    this.data = await response.json();
    return this.data;
  }

  bindOnce() {
    if (this.bound) return;
    this.bound = true;
    this.populateModel();
    this.populateColorFilter();
    ["Model", "View", "ShareFilter", "ValueMode"].forEach(name => {
      this.el(name)?.addEventListener("change", () => {
        this.selectedKey = null;
        this.renderDetail(null);
        void this.render();
      });
    });
    this.el("Search")?.addEventListener("input", () => void this.render());
    this.el("MaskPrompt")?.addEventListener("change", () => {
      this.selectedKey = null;
      this.renderDetail(null);
      void this.render();
    });
  }

  populateModel() {
    const sel = this.el("Model");
    if (!sel) return;
    const seen = new Set();
    const options = [];
    for (const row of this.store.DATA?.tree_summaries || []) {
      const id = row.model_id || row.country_id;
      if (!id || seen.has(id)) continue;
      seen.add(id);
      options.push(`<option value="${utils.escapeHtml(id)}">${utils.escapeHtml(row.country_name || id)}</option>`);
    }
    sel.innerHTML = options.join("");
  }

  populateColorFilter() {
    const container = this.el("ColorFilter");
    if (!container) return;
    container.innerHTML = SEQ_COLOR_META.map(meta => `
      <label class="${meta.cls}">
        <input type="checkbox" id="${this.id(`Color_${meta.id}`)}" checked />
        ${meta.id}
      </label>
    `).join("") + `
      <button type="button" class="preset-btn" id="${this.id("ColorAll")}">All</button>
      <button type="button" class="preset-btn" id="${this.id("ColorNone")}">None</button>
    `;
    container.querySelectorAll("input[type=checkbox]").forEach(input => {
      input.addEventListener("change", () => void this.render());
    });
    this.el("ColorAll")?.addEventListener("click", () => {
      container.querySelectorAll("input[type=checkbox]").forEach(input => { input.checked = true; });
      void this.render();
    });
    this.el("ColorNone")?.addEventListener("click", () => {
      container.querySelectorAll("input[type=checkbox]").forEach(input => { input.checked = false; });
      void this.render();
    });
  }

  currentReport() {
    const modelId = this.el("Model")?.value;
    return this.data?.models?.[modelId] || null;
  }

  showPrevalence() {
    return this.el("ValueMode")?.value === "prevalence";
  }

  promptMaskEnabled() {
    return this.el("MaskPrompt")?.checked;
  }

  preparedSequences(report) {
    if (!report) return [];
    if (this.promptMaskEnabled()) return aggregateMaskedSequences(report.sequences || []);
    return report.sequences || [];
  }

  selectedColors() {
    return SEQ_COLOR_META
      .map(meta => meta.id)
      .filter(id => this.root.querySelector(`#${this.id(`Color_${id}`)}`)?.checked);
  }

  sequenceMatchesColors(row, selected) {
    if (!selected.length) return false;
    const present = row.colors || SEQ_COLOR_META.filter(meta => (row.counts[meta.id] || 0) > 0).map(meta => meta.id);
    return present.length > 0 && present.every(id => selected.includes(id));
  }

  filteredSequences(report) {
    if (!report) return [];
    const share = this.el("ShareFilter")?.value || "all";
    const query = (this.el("Search")?.value || "").trim().toLowerCase();
    const selected = this.selectedColors();
    return this.preparedSequences(report).filter(row => {
      if (share === "unique" && row.shared) return false;
      if (share === "shared" && !row.shared) return false;
      if (query && !row.display.toLowerCase().includes(query)) return false;
      return this.sequenceMatchesColors(row, selected);
    });
  }

  colorDenom(report, color) {
    return (report && report.leaf_totals && report.leaf_totals[color]) || 0;
  }

  leafCount(row, color) {
    return (row.leaf_counts && row.leaf_counts[color]) || 0;
  }

  colorPrevalence(row, color, report) {
    const denom = this.colorDenom(report, color);
    if (!denom) return null;
    return 100 * this.leafCount(row, color) / denom;
  }

  formatColorValue(row, color, report) {
    const count = row.counts[color] || 0;
    if (!this.showPrevalence()) return count || "—";
    return formatLeafShare(this.leafCount(row, color), this.colorDenom(report, color));
  }

  formatTotalValue(row, report) {
    if (!this.showPrevalence()) return row.total;
    const green = this.leafCount(row, "green");
    const red = this.leafCount(row, "red");
    const denom = this.colorDenom(report, "green") + this.colorDenom(report, "red");
    if (!denom) return "—";
    return formatLeafShare(green + red, denom);
  }

  colorPills(row, skipColor, report) {
    return SEQ_COLOR_META
      .filter(meta => row.counts[meta.id] > 0 && meta.id !== skipColor)
      .map(meta => `<span class="pill ${meta.id === "green" ? "good-node" : meta.id === "blue" ? "mostly-good" : meta.id === "yellow" ? "mostly-bad" : meta.id === "red" ? "bad-node" : ""}" style="${meta.id === "grey" ? "background:#2a2d36;color:#c4c9d4" : ""}">${meta.id} ${this.formatColorValue(row, meta.id, report)}</span>`)
      .join("");
  }

  renderSeqTokens(tokens) {
    return `<div class="seq-tokens">${tokens.map(tok => `<span class="seq-tok">${utils.escapeHtml(displayTok(tok))}</span>`).join("")}</div>`;
  }

  renderDetail(row) {
    const card = this.el("Detail");
    if (!card) return;
    if (!row) {
      card.hidden = true;
      card.innerHTML = "";
      return;
    }
    const report = this.currentReport();
    const values = SEQ_COLOR_META.map(meta => (
      this.showPrevalence() ? (this.colorPrevalence(row, meta.id, report) || 0) : (row.counts[meta.id] || 0)
    ));
    const maxValue = Math.max(...values, 1e-9);
    const bars = SEQ_COLOR_META.map((meta, index) => {
      const width = Math.round(100 * values[index] / maxValue);
      return `<div class="seq-bar-row">
        <span class="${meta.cls}">${meta.id}</span>
        <div class="seq-bar-track"><div class="seq-bar-fill" style="width:${width}%;background:${meta.fill}"></div></div>
        <strong>${this.formatColorValue(row, meta.id, report)}</strong>
      </div>`;
    }).join("");
    card.hidden = false;
    card.innerHTML = `
      <h3>${row.shared ? "Shared" : "Unique"} 15-token sequence · ${row.color_n} color${row.color_n === 1 ? "" : "s"} · ${this.formatTotalValue(row, report)}${this.showPrevalence() ? " of all leaves" : ` ${this.countLabel}`}${row.source_n > 1 ? ` · merged from ${row.source_n}` : ""}</h3>
      <div class="seq-text">${formatSeqDisplay(row.display)}</div>
      ${row.masked ? "" : this.renderSeqTokens(row.tokens)}
      <div class="seq-bars">${bars}</div>
    `;
  }

  selectSequence(row) {
    this.selectedKey = row ? row.display : null;
    void this.render();
    this.renderDetail(row);
  }

  renderStats(report, rows) {
    const statsEl = this.el("Stats");
    if (!statsEl) return;
    if (!report) {
      statsEl.innerHTML = "";
      return;
    }
    const shared = rows.filter(row => row.shared).length;
    const unique = rows.filter(row => !row.shared).length;
    statsEl.innerHTML = `
      <div class="stat"><strong>${rows.length}</strong><span class="muted">sequences shown</span></div>
      <div class="stat"><strong>${unique}</strong><span class="muted">unique to one color</span></div>
      <div class="stat"><strong>${shared}</strong><span class="muted">shared across colors</span></div>
      <div class="stat"><strong>${this.data?.top_n || 0}</strong><span class="muted">top N stored per color</span></div>
    `;
  }

  sortValue(row, key, report) {
    if (key === "kind") return row.shared ? 1 : 0;
    if (key === "color_n") return row.color_n;
    if (key === "display") return row.display;
    if (key === "total") {
      if (!this.showPrevalence()) return row.total;
      const green = this.leafCount(row, "green");
      const red = this.leafCount(row, "red");
      const denom = this.colorDenom(report, "green") + this.colorDenom(report, "red");
      return denom ? (green + red) / denom : 0;
    }
    if (this.showPrevalence() && SEQ_COLOR_META.some(meta => meta.id === key)) {
      return this.colorPrevalence(row, key, report) || 0;
    }
    return row.counts[key] || 0;
  }

  sortRows(rows, report) {
    return [...rows].sort((a, b) => {
      const av = this.sortValue(a, this.sortKey, report);
      const bv = this.sortValue(b, this.sortKey, report);
      if (av === bv) return b.total - a.total;
      if (typeof av === "string") return this.sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
      return this.sortAsc ? av - bv : bv - av;
    });
  }

  seqHeader(key, label, cls, align) {
    const arrow = this.sortKey === key ? (this.sortAsc ? " ▲" : " ▼") : "";
    const extra = cls ? ` ${cls}` : "";
    return `<th data-seq-sort="${key}" class="${extra.trim()}" style="text-align:${align || "right"}">${label}${arrow}</th>`;
  }

  renderByColor(report, rows) {
    const container = this.el("ByColor");
    if (!container) return;
    const visible = SEQ_COLOR_META.filter(meta => this.root.querySelector(`#${this.id(`Color_${meta.id}`)}`)?.checked);
    container.style.gridTemplateColumns = `repeat(${Math.max(visible.length, 1)}, minmax(210px, 1fr))`;
    if (!visible.length) {
      container.innerHTML = `<p class="muted">Check at least one color.</p>`;
      return;
    }
    container.innerHTML = visible.map(meta => {
      const ranked = rows
        .filter(row => row.ranks && row.ranks[meta.id] != null && row.ranks[meta.id] <= (this.data?.top_n || 50))
        .sort((a, b) => (a.ranks[meta.id] - b.ranks[meta.id]) || (b.counts[meta.id] - a.counts[meta.id]));
      const windows = report.window_counts[meta.id] || 0;
      const items = ranked.map((row, index) => {
        const also = this.colorPills(row, meta.id, report);
        return `<div class="seq-item ${row.display === this.selectedKey ? "selected" : ""}" data-color="${meta.id}" data-index="${index}">
          <div class="seq-item-head">
            <span>#${row.ranks[meta.id]} · ${this.formatColorValue(row, meta.id, report)}${this.showPrevalence() ? " of leaves" : ` ${this.countLabel}`}</span>
            <span>${row.shared ? "shared" : "unique"}</span>
          </div>
          <div class="seq-text">${formatSeqDisplay(row.display)}</div>
          ${also ? `<div class="seq-also">${also}</div>` : ""}
        </div>`;
      }).join("") || `<p class="muted">${windows ? "None match the current filters." : `No ${this.windowLabel} of this color with a 15-token span.`}</p>`;
      return `<section class="seq-col">
        <h3 class="${meta.cls}">${meta.label}</h3>
        <p class="muted">${windows.toLocaleString()} ${this.windowLabel} · ${report.unique_seq_counts[meta.id].toLocaleString()} distinct 15-grams</p>
        ${items}
      </section>`;
    }).join("");
    container.querySelectorAll(".seq-item").forEach(item => {
      const color = item.dataset.color;
      const ranked = rows
        .filter(row => row.ranks && row.ranks[color] != null && row.ranks[color] <= (this.data?.top_n || 50))
        .sort((a, b) => (a.ranks[color] - b.ranks[color]) || (b.counts[color] - a.counts[color]));
      item.addEventListener("click", () => this.selectSequence(ranked[Number(item.dataset.index)]));
    });
  }

  renderCompare(report, rows) {
    const container = this.el("Compare");
    if (!container) return;
    const sorted = this.sortRows(rows, report);
    const visible = SEQ_COLOR_META.filter(meta => this.root.querySelector(`#${this.id(`Color_${meta.id}`)}`)?.checked);
    const colorHeads = visible.map(meta => this.seqHeader(meta.id, this.showPrevalence() ? `${meta.id} %` : meta.id, meta.cls)).join("");
    const body = sorted.map((row, index) => {
      const cells = visible.map(meta => `<td class="${meta.cls}">${this.formatColorValue(row, meta.id, report)}</td>`).join("");
      return `<tr class="seq-row ${row.display === this.selectedKey ? "selected" : ""}" data-index="${index}">
        <td>${row.shared ? "shared" : "unique"}</td>
        <td>${row.color_n}</td>
        <td>${this.formatTotalValue(row, report)}</td>
        ${cells}
        <td class="seq-cell">${formatSeqDisplay(row.display)}</td>
      </tr>`;
    }).join("") || `<tr><td colspan="${4 + visible.length}" class="muted">No sequences match the current filters.</td></tr>`;
    container.innerHTML = `<table class="summary">
      <thead><tr>
        ${this.seqHeader("kind", "Kind", "", "left")}
        ${this.seqHeader("color_n", "# colors")}
        ${this.seqHeader("total", this.showPrevalence() ? "Total %" : "Total")}
        ${colorHeads}
        ${this.seqHeader("display", "15-token sequence", "", "left")}
      </tr></thead>
      <tbody>${body}</tbody>
    </table>`;
    container.querySelectorAll("th[data-seq-sort]").forEach(th => {
      th.addEventListener("click", () => {
        const key = th.dataset.seqSort;
        if (this.sortKey === key) this.sortAsc = !this.sortAsc;
        else {
          this.sortKey = key;
          this.sortAsc = key === "display" || key === "kind";
        }
        void this.render();
      });
    });
    container.querySelectorAll("tr.seq-row").forEach(rowEl => {
      rowEl.addEventListener("click", () => this.selectSequence(sorted[Number(rowEl.dataset.index)]));
    });
  }

  async render() {
    const statsEl = this.el("Stats");
    try {
      if (!this.data && statsEl) {
        statsEl.innerHTML = `<div class="stat"><strong>…</strong><span class="muted">loading sequences</span></div>`;
      }
      await this.ensureLoaded();
      const report = this.currentReport();
      const rows = this.filteredSequences(report);
      const view = this.el("View")?.value || "compare";
      this.renderStats(report, rows);
      const byColor = this.el("ByColor");
      const compare = this.el("Compare");
      if (byColor) byColor.hidden = view !== "by_color";
      if (compare) compare.hidden = view !== "compare";
      if (view === "by_color") this.renderByColor(report, rows);
      else this.renderCompare(report, rows);
      const selected = rows.find(row => row.display === this.selectedKey);
      if (selected) this.renderDetail(selected);
    } catch (err) {
      if (statsEl) {
        statsEl.innerHTML = `<div class="stat"><strong class="error">Failed</strong><span class="muted">${utils.escapeHtml(err.message)}</span></div>`;
      }
    }
  }

  async show() {
    this.bindOnce();
    await this.render();
  }
};
})(TreeDashboard);
