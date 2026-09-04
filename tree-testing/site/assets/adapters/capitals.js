window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const {
  DataStore, TreeGraph, SummaryPanel, bindSubTabs, switchSubTab, utils,
  aggregateSummaries, chartableLabel, renderBarChart,
  CapitalsBadFilters, renderDetailPanel,
} = TD;

const CAPITALS_TABLE_COLUMNS = [
  { key: "label", label: "Group", align: "left" },
  { key: "trees", label: "Trees" },
  { key: "avg_exclusively_bad", label: "Excl. bad", chartable: true, map: r => r.avg_exclusively_bad },
  { key: "prob_good_pct", label: "P(good) %", chartable: true, highlight: true },
  { key: "prob_bad_pct", label: "P(bad) %", chartable: true, highlight: true },
  { key: "greedy_accuracy_pct", label: "Greedy %", chartable: true },
  { key: "leaf_accuracy_pct", label: "Leaf acc %", chartable: true },
  { key: "avg_leaf_count", label: "τ-leaves" },
  { key: "avg_total_nodes", label: "Nodes" },
];

const CAPITALS_ROW_COLUMNS = [
  { key: "country_name", label: "Country" },
  { key: "prefix_length", label: "Prefix" },
  { key: "total_nodes", label: "Nodes" },
  { key: "leaf_count", label: "τ-leaves" },
  { key: "exclusively_bad_count", label: "Excl. bad" },
  { key: "prob_good_pct", label: "P(good) %", highlight: true },
  { key: "prob_bad_pct", label: "P(bad) %", highlight: true },
];

TD.CapitalsAdapter = class {
  constructor(root) {
    this.root = root;
    this.store = new DataStore("capitals", "data/capitals");
    this.graph = new TreeGraph("#capitalsGraph", "#capitalsTooltip");
    this.summary = new SummaryPanel(root, CAPITALS_ROW_COLUMNS);
    this.summary.tableEl = root.querySelector("#capitalsSummaryTable");
    this.summary.statsEl = root.querySelector("#capitalsGlobalStats");
    this.summary.hintEl = root.querySelector("#capitalsFilterHint");
    this.currentSummary = null;
    this.currentShard = null;
    this.sortKey = "label";
    this.sortAsc = true;
    this.graph.onSelect = nodeId => this.showNode(nodeId);
  }

  async init() {
    try {
      await this.store.loadManifest();
      this.populateFilters();
      this.renderSummary();
      this.bindEvents();
      const pct = this.store.manifest.mech_interp_trees
        ? Math.round(100 * this.store.manifest.bad_nodes_trees / this.store.manifest.mech_interp_trees) : 0;
      const fill = this.root.querySelector("#capitalsProgressFill");
      if (fill) fill.style.width = `${pct}%`;
      document.getElementById("capitalsSource").textContent =
        `Capitals experiment (plain) · ${this.store.manifest.viewer_runs || 0} trees · generated ${this.store.manifest.generated_at}`;
    } catch (err) {
      document.getElementById("capitalsSource").innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }

  filteredSummaries() {
    const prefix = this.root.querySelector("#capitalsPrefixFilter")?.value || "";
    const region = this.root.querySelector("#capitalsRegionFilter")?.value || "";
    const country = this.root.querySelector("#capitalsCountryFilter")?.value || "";
    return (this.store.DATA?.tree_summaries || []).filter(row => {
      if (prefix && String(row.prefix_length) !== prefix) return false;
      if (region && row.region_id !== region) return false;
      if (country && row.country_id !== country) return false;
      return true;
    });
  }

  aggRows() {
    const dimension = this.root.querySelector("#capitalsAggSelect")?.value || "whole";
    return aggregateSummaries(this.filteredSummaries(), dimension).map(row => ({
      ...row, label: row.label,
    }));
  }

  populateFilters() {
    const summaries = this.store.DATA.tree_summaries || [];
    const regions = [...new Set(summaries.map(r => r.region_id).filter(Boolean))];
    const countries = [...new Set(summaries.map(r => r.country_id).filter(Boolean))].sort();
    this.root.querySelector("#capitalsRegionFilter").innerHTML =
      '<option value="">All regions</option>' + regions.map(r => `<option value="${r}">${r}</option>`).join("");
    this.root.querySelector("#capitalsCountryFilter").innerHTML =
      '<option value="">All countries</option>' + countries.map(c => {
        const row = summaries.find(s => s.country_id === c);
        return `<option value="${c}">${row?.country_name || c}</option>`;
      }).join("");
    this.root.querySelector("#capitalsPrefixFilter").innerHTML =
      '<option value="">All prefixes</option>' + (this.store.manifest.prefix_lengths || []).map(pl =>
        `<option value="${pl}">${pl}</option>`).join("");
    const metricSel = this.root.querySelector("#capitalsMetricSelect");
    metricSel.innerHTML = CAPITALS_TABLE_COLUMNS.filter(c => c.chartable).map(c =>
      `<option value="${c.key}">${c.label}</option>`).join("");
    metricSel.value = "prob_good_pct";
    const runSel = this.root.querySelector("#capitalsRunSelect");
    runSel.innerHTML = summaries.map(row =>
      `<option value="${row.tree_key}">${row.country_name || row.country_id} · p${row.prefix_length}</option>`).join("");
  }

  renderAggTable(rows) {
    const table = this.root.querySelector("#capitalsSummaryTable");
    if (!table) return;
    const sorted = [...rows].sort((a, b) => {
      const av = a[this.sortKey], bv = b[this.sortKey];
      if (av === bv) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return (av < bv ? -1 : 1) * (this.sortAsc ? 1 : -1);
    });
    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");
    thead.innerHTML = `<tr>${CAPITALS_TABLE_COLUMNS.map(col =>
      `<th data-key="${col.key}" class="${col.highlight ? "highlight" : ""}">${col.label}</th>`).join("")}</tr>`;
    tbody.innerHTML = sorted.map(row => `<tr class="clickable" data-group-key="${utils.escapeHtml(String(row.key))}">${CAPITALS_TABLE_COLUMNS.map(col =>
      `<td class="${col.highlight ? "highlight" : ""}">${utils.escapeHtml(String(utils.fmtVal(row[col.key])))}</td>`).join("")}</tr>`).join("");
    tbody.querySelectorAll("tr").forEach(tr => {
      tr.addEventListener("click", () => {
        const key = tr.dataset.groupKey;
        const dimension = this.root.querySelector("#capitalsAggSelect")?.value || "whole";
        const summaries = this.filteredSummaries();
        let match = summaries[0];
        if (dimension === "by_country") match = summaries.find(s => s.country_id === key) || match;
        else if (dimension === "by_region") match = summaries.find(s => s.region_id === key) || match;
        else if (dimension === "by_prefix_length") match = summaries.find(s => String(s.prefix_length) === key) || match;
        if (match) this.openTree(match.tree_key);
      });
    });
    thead.querySelectorAll("th").forEach(th => {
      th.addEventListener("click", () => {
        const key = th.dataset.key;
        if (this.sortKey === key) this.sortAsc = !this.sortAsc;
        else { this.sortKey = key; this.sortAsc = key === "label"; }
        this.renderSummary();
      });
    });
  }

  renderGlobalStats() {
    const rows = this.filteredSummaries();
    const whole = aggregateSummaries(rows, "whole")[0] || {};
    const el = this.root.querySelector("#capitalsGlobalStats");
    if (!el) return;
    el.innerHTML = `
      <div class="stat"><strong>${this.store.manifest.mech_interp_trees || 0}</strong><span class="muted">τ-trees</span></div>
      <div class="stat"><strong>${this.store.manifest.bad_nodes_trees || 0}</strong><span class="muted">analyzed</span></div>
      <div class="stat"><strong>${utils.fmtVal(whole.prob_good_pct)}%</strong><span class="muted">P(good)</span></div>
      <div class="stat"><strong>${utils.fmtVal(whole.prob_bad_pct)}%</strong><span class="muted">P(bad)</span></div>
      <div class="stat"><strong>${utils.fmtVal(whole.greedy_accuracy_pct)}%</strong><span class="muted">greedy acc</span></div>
      <div class="stat"><strong>${rows.length}</strong><span class="muted">filtered</span></div>`;
  }

  renderSummary() {
    const rows = this.aggRows();
    const metricKey = this.root.querySelector("#capitalsMetricSelect")?.value || "prob_good_pct";
    const metricLabel = this.root.querySelector("#capitalsMetricSelect")?.selectedOptions[0]?.textContent || metricKey;
    const hint = this.root.querySelector("#capitalsFilterHint");
    if (hint) hint.textContent = `${this.filteredSummaries().length} trees · ${rows.length} groups · click aggregated row context via country filters`;
    this.renderAggTable(rows);
    renderBarChart("#capitalsBarChart", "#capitalsBarChartTitle", rows, metricKey, metricLabel);
    this.renderGlobalStats();
  }

  firstTreeKey() {
    const rows = this.filteredSummaries();
    return rows[0]?.tree_key || this.store.DATA?.tree_summaries?.[0]?.tree_key;
  }

  async ensureTreeLoaded() {
    if (this.currentSummary) return;
    const key = this.firstTreeKey();
    if (key) await this.loadRun(key);
  }

  async openTree(treeKey) {
    switchSubTab(this.root, "trees");
    const runSel = this.root.querySelector("#capitalsRunSelect");
    if (runSel) runSel.value = treeKey;
    await this.loadRun(treeKey);
  }

  refreshStatsForMode() {
    if (!this.currentShard) return;
    const nodes = this.currentShard.tree_nodes || [];
    const leafMap = new Map(Object.entries(this.currentShard.leaf_completions || {}));
    const statsMap = CapitalsBadFilters.computeNodeStats(nodes, leafMap, this.root);
    this.graph.statsMap = statsMap;
    const minProb = parseFloat(this.root.querySelector("#capitalsProbFilter")?.value || "0");
    void this.graph.draw(nodes, minProb).then(() => this.graph.fitToScreen());
    this.renderBadNodeList(nodes, statsMap, leafMap);
    this.renderRunStats(leafMap);
  }

  renderRunStats(leafMap) {
    const leaves = [...leafMap.values()];
    let good = 0, bad = 0, total = 0;
    for (const leaf of leaves) {
      const p = Number(leaf.path_prob) || 0;
      total += p;
      if (CapitalsBadFilters.isLeafCorrect(leaf)) good += p;
      else if (CapitalsBadFilters.isLeafBad(leaf, this.root)) bad += p;
    }
    const el = this.root.querySelector("#capitalsRunStats");
    if (!el) return;
    el.innerHTML = `
      <div class="stat"><strong>${total ? Math.round(1000 * good / total) / 10 : 0}%</strong><span class="muted">P(good)</span></div>
      <div class="stat"><strong>${total ? Math.round(1000 * bad / total) / 10 : 0}%</strong><span class="muted">P(bad)</span></div>`;
  }

  renderBadNodeList(nodes, statsMap, leafMap) {
    const el = this.root.querySelector("#capitalsBadNodeList");
    if (!el) return;
    const items = [...statsMap.entries()]
      .filter(([, s]) => s.n_bad > 0)
      .sort((a, b) => b[1].bad_pct - a[1].bad_pct)
      .slice(0, 15);
    el.innerHTML = items.length ? items.map(([id, s]) =>
      `<div class="node-card" data-node="${utils.escapeHtml(id)}" style="padding:6px;cursor:pointer">${utils.escapeHtml(id)} · ${s.bad_pct_display}% bad</div>`
    ).join("") : '<p class="muted">No bad nodes under current filters.</p>';
    el.querySelectorAll("[data-node]").forEach(div => {
      div.addEventListener("click", () => {
        const nodeId = div.dataset.node;
        this.graph.focusNode(nodeId, nodes, parseFloat(this.root.querySelector("#capitalsProbFilter")?.value || "0"));
        this.showNode(nodeId);
      });
    });
  }

  async loadRun(treeKey) {
    const summary = this.store.summaryFor(treeKey);
    this.currentSummary = summary;
    const meta = this.root.querySelector("#capitalsRunMeta");
    if (meta) meta.textContent = "Loading tree…";
    const shard = await this.store.loadTree(treeKey);
    this.currentShard = shard;
    const nodes = shard.tree_nodes || [];
    const leafMap = new Map(Object.entries(shard.leaf_completions || {}));
    const statusMap = new Map(Object.entries(shard.node_status || {}));
    const expansionMap = new Map(Object.entries(shard.node_expansions || {}));
    const statsMap = CapitalsBadFilters.computeNodeStats(nodes, leafMap, this.root);
    const highlightSet = new Set();
    if (this.root.querySelector("#capitalsHighlightBadOnly")?.checked) {
      for (const [id, st] of statusMap.entries()) if (st === "exclusively_bad") highlightSet.add(id);
    }
    this.graph.setTreeData({ nodes, leafMap, statusMap, statsMap, expansionMap });
    const minProb = parseFloat(this.root.querySelector("#capitalsProbFilter")?.value || "0");
    this.graph.setOptions({
      showBad: this.root.querySelector("#capitalsShowBad")?.checked !== false,
      showGood: this.root.querySelector("#capitalsShowGood")?.checked !== false,
      dimOther: this.root.querySelector("#capitalsHighlightBadOnly")?.checked,
      highlightSet,
    });
    await this.graph.draw(nodes, minProb);
    this.graph.fitToScreen();
    if (meta && summary) {
      meta.textContent = `${summary.country_name || summary.country_id} · prefix ${summary.prefix_length}`;
    }
    this.renderBadNodeList(nodes, statsMap, leafMap);
    this.renderRunStats(leafMap);
    renderDetailPanel(this.root.querySelector("#capitalsDetail"), {});
  }

  showNode(nodeId) {
    const node = this.graph.currentNodes.find(n => n.id === nodeId);
    const leaf = this.graph.leafMap.get(nodeId);
    const stats = this.graph.statsMap.get(nodeId);
    const status = this.graph.statusMap.get(nodeId);
    const leaves = this.graph.leafDescendants(this.graph.currentNodes, nodeId)
      .map(id => this.graph.leafMap.get(id)).filter(Boolean);
    renderDetailPanel(this.root.querySelector("#capitalsDetail"), { node, leaf, stats, status, leaves });
  }

  bindEvents() {
    bindSubTabs(this, this.root);
    ["capitalsPrefixFilter", "capitalsRegionFilter", "capitalsCountryFilter", "capitalsAggSelect", "capitalsMetricSelect"]
      .forEach(id => this.root.querySelector(`#${id}`)?.addEventListener("change", () => this.renderSummary()));
    this.root.querySelector("#capitalsRunSelect")?.addEventListener("change", e => this.loadRun(e.target.value));
    this.root.querySelector("#capitalsProbFilter")?.addEventListener("change", () => {
      if (this.currentSummary) this.loadRun(this.currentSummary.tree_key);
    });
    this.root.querySelector("#capitalsFitBtn")?.addEventListener("click", () => this.graph.fitToScreen());
    this.root.querySelector("#capitalsResetBtn")?.addEventListener("click", () => this.graph.resetZoom());
    ["capitalsShowBad", "capitalsShowGood", "capitalsHighlightBadOnly"].forEach(id => {
      this.root.querySelector(`#${id}`)?.addEventListener("change", () => {
        if (this.currentSummary) this.refreshStatsForMode();
      });
    });
    CapitalsBadFilters.SUBTYPES.forEach(item => {
      this.root.querySelector(`#badFilter_${item.id}`)?.addEventListener("change", () => this.refreshStatsForMode());
    });
  }
};

document.addEventListener("experiment-tab", event => {
  if (event.detail.experiment === "capitals" && !window._capitalsInit) {
    try {
      const adapter = new TD.CapitalsAdapter(document.getElementById("capitalsPanel"));
      window._capitalsInit = true;
      adapter.init();
    } catch (err) {
      const el = document.getElementById("capitalsSource");
      if (el) el.innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }
});
})(TreeDashboard);
