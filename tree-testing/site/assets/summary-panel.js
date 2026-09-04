window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { utils } = TD;

TD.SummaryPanel = class {
  constructor(root, columns) {
    this.root = root;
    this.columns = columns;
    this.sortKey = columns[0]?.key || "label";
    this.sortAsc = true;
    this.tableEl = root.querySelector("table.summary");
    this.statsEl = root.querySelector(".stat-grid");
    this.hintEl = root.querySelector(".filter-hint");
  }

  sortRows(rows) {
    const key = this.sortKey;
    const asc = this.sortAsc;
    return [...rows].sort((a, b) => {
      const av = a[key];
      const bv = b[key];
      if (av === bv) return 0;
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      return (av < bv ? -1 : 1) * (asc ? 1 : -1);
    });
  }

  renderTable(rows, onRowClick) {
    if (!this.tableEl) return;
    const thead = this.tableEl.querySelector("thead");
    const tbody = this.tableEl.querySelector("tbody");
    thead.innerHTML = `<tr>${this.columns.map(col => `<th data-key="${col.key}">${col.label}${this.sortKey === col.key ? (this.sortAsc ? " ▲" : " ▼") : ""}</th>`).join("")}</tr>`;
    thead.querySelectorAll("th").forEach(th => {
      th.addEventListener("click", () => {
        const key = th.dataset.key;
        if (this.sortKey === key) this.sortAsc = !this.sortAsc;
        else { this.sortKey = key; this.sortAsc = true; }
        this.renderTable(rows, onRowClick);
      });
    });
    const sorted = this.sortRows(rows);
    tbody.innerHTML = sorted.map(row => {
      const cells = this.columns.map(col => `<td>${utils.escapeHtml(String(utils.fmtVal(row[col.key])))}</td>`).join("");
      return `<tr class="clickable" data-tree-key="${utils.escapeHtml(row.tree_key || "")}">${cells}</tr>`;
    }).join("");
    tbody.querySelectorAll("tr").forEach(tr => {
      tr.addEventListener("click", () => {
        const treeKey = tr.dataset.treeKey;
        if (treeKey && onRowClick) onRowClick(treeKey);
      });
    });
    if (this.hintEl) this.hintEl.textContent = `${sorted.length} tree${sorted.length === 1 ? "" : "s"} · click a row to open Tree Explorer`;
  }

  renderStats(items) {
    if (!this.statsEl) return;
    this.statsEl.innerHTML = items.map(item => `
      <div class="stat"><strong>${utils.escapeHtml(String(item.value))}</strong><span class="muted">${utils.escapeHtml(item.label)}</span></div>
    `).join("");
  }
};

TD.renderDetail = function(container, { node, leaf, stats, status }) {
  if (!container) return;
  if (!node && !leaf) {
    container.innerHTML = '<p class="muted">Click a node in the graph or a table row.</p>';
    return;
  }
  let html = "";
  if (node) {
    html += `<div class="card"><h3>Node ${utils.escapeHtml(node.id)}</h3>`;
    html += `<div class="muted">token: ${utils.escapeHtml(utils.displayToken(node.t || node.token))}</div>`;
    if (node.p != null) html += `<div class="muted">p=${utils.fmtProb(node.p)}</div>`;
    if (stats) html += `<div><span class="pill ${stats.bad_pct >= 1 ? "bad-node" : "ditched"}">${utils.fmtVal(stats.bad_pct_display || stats.bad_pct)}% bad</span></div>`;
    if (status) html += `<div><span class="pill ditched">${utils.escapeHtml(status)}</span></div>`;
    html += "</div>";
  }
  if (leaf) {
    const correct = leaf.answer_correct ? "yes" : "no";
    html += `<div class="card"><h3>Leaf answer</h3>`;
    html += `<div><span class="pill ${correct}">${correct === "yes" ? "good" : "bad"}</span></div>`;
    html += `<div class="answer">${utils.escapeHtml(leaf.answer_text || "")}</div>`;
    if (leaf.completion_preview) {
      html += `<pre class="muted" style="white-space:pre-wrap;margin-top:8px">${utils.escapeHtml(leaf.completion_preview)}</pre>`;
    }
    html += "</div>";
  }
  container.innerHTML = html;
};
})(TreeDashboard);
