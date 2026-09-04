window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const {
  DataStore, TreeGraph, bindSubTabs, switchSubTab, utils,
  aggregateSummaries, renderBarChart, renderHorizontalBarChart,
  ElectionsCorrectness, renderDetailPanel,
} = TD;

const ELECTIONS_TABLE_COLUMNS = [
  { key: "label", label: "Group", chartable: false },
  { key: "trees", label: "Trees" },
  { key: "prob_good_pct", label: "P(good) %", chartable: true, highlight: true },
  { key: "prob_bad_pct", label: "P(bad) %", chartable: true, highlight: true },
  { key: "greedy_accuracy_pct", label: "Greedy %", chartable: true },
  { key: "leaf_accuracy_pct", label: "Leaf acc %", chartable: true },
];

TD.ElectionsAdapter = class {
  constructor(root) {
    this.root = root;
    this.store = new DataStore("elections", "data/elections");
    this.graph = new TreeGraph("#electionsGraph", "#electionsTooltip");
    this.correctness = null;
    this.currentSummary = null;
    this.currentShard = null;
    this.sortKey = "label";
    this.sortAsc = true;
    this.graph.onSelect = nodeId => this.showNode(nodeId);
  }

  async init() {
    try {
      await this.store.loadManifest();
      this.correctness = new ElectionsCorrectness(this.store.manifest.presidential_candidates || []);
      this.correctness.populateSelect(this.root.querySelector("#electionsCorrectnessMode"));
      this.correctness.populateSelect(this.root.querySelector("#electionsTreeCorrectnessMode"));
      this.populateFilters();
      this.renderSummary();
      this.bindEvents();
      document.getElementById("electionsSource").textContent =
        `Brazilian president experiment · ${this.store.manifest.viewer_runs || 0} trees · generated ${this.store.manifest.generated_at}`;
    } catch (err) {
      document.getElementById("electionsSource").innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }

  filteredSummaries() {
    const prefix = this.root.querySelector("#electionsPrefixFilter")?.value || "";
    return (this.store.DATA?.tree_summaries || []).filter(row => {
      if (prefix && String(row.prefix_length) !== prefix) return false;
      return true;
    });
  }

  aggRows() {
    const dimension = this.root.querySelector("#electionsAggSelect")?.value || "whole";
    return aggregateSummaries(this.filteredSummaries(), dimension);
  }

  populateFilters() {
    const summaries = this.store.DATA.tree_summaries || [];
    this.root.querySelector("#electionsPrefixFilter").innerHTML =
      '<option value="">All prefixes</option>' + (this.store.manifest.prefix_lengths || []).map(pl =>
        `<option value="${pl}">${pl}</option>`).join("");
    const metricSel = this.root.querySelector("#electionsMetricSelect");
    metricSel.innerHTML = ELECTIONS_TABLE_COLUMNS.filter(c => c.chartable).map(c =>
      `<option value="${c.key}">${c.label}</option>`).join("");
    metricSel.value = "prob_good_pct";
    this.root.querySelector("#electionsRunSelect").innerHTML = summaries.map(row =>
      `<option value="${row.tree_key}">${row.instruction_variant || row.model_id}</option>`).join("");
  }

  renderAggTable(rows) {
    const table = this.root.querySelector("#electionsSummaryTable");
    if (!table) return;
    const sorted = [...rows].sort((a, b) => {
      const av = a[this.sortKey], bv = b[this.sortKey];
      if (av === bv) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return (av < bv ? -1 : 1) * (this.sortAsc ? 1 : -1);
    });
    table.querySelector("thead").innerHTML = `<tr>${ELECTIONS_TABLE_COLUMNS.map(col =>
      `<th data-key="${col.key}" class="${col.highlight ? "highlight" : ""}">${col.label}</th>`).join("")}</tr>`;
    table.querySelector("tbody").innerHTML = sorted.map(row => `<tr class="clickable" data-group-key="${utils.escapeHtml(String(row.key))}">${ELECTIONS_TABLE_COLUMNS.map(col =>
      `<td class="${col.highlight ? "highlight" : ""}">${utils.escapeHtml(String(utils.fmtVal(row[col.key])))}</td>`).join("")}</tr>`).join("");
    table.querySelectorAll("tbody tr").forEach(tr => {
      tr.addEventListener("click", () => {
        const key = tr.dataset.groupKey;
        const dimension = this.root.querySelector("#electionsAggSelect")?.value || "whole";
        const summaries = this.filteredSummaries();
        let match = summaries[0];
        if (dimension === "by_instruction_variant") match = summaries.find(s => s.instruction_variant === key) || match;
        else if (dimension === "by_model") match = summaries.find(s => s.model_id === key) || match;
        if (match) this.openTree(match.tree_key);
      });
    });
    table.querySelectorAll("thead th").forEach(th => {
      th.addEventListener("click", () => {
        const key = th.dataset.key;
        if (this.sortKey === key) this.sortAsc = !this.sortAsc;
        else { this.sortKey = key; this.sortAsc = key === "label"; }
        this.renderSummary();
      });
    });
  }

  renderGlobalStats() {
    const whole = aggregateSummaries(this.filteredSummaries(), "whole")[0] || {};
    const el = this.root.querySelector("#electionsGlobalStats");
    if (!el) return;
    el.innerHTML = `
      <div class="stat"><strong>${utils.fmtVal(whole.prob_good_pct)}%</strong><span class="muted">P(good)</span></div>
      <div class="stat"><strong>${utils.fmtVal(whole.prob_bad_pct)}%</strong><span class="muted">P(bad)</span></div>
      <div class="stat"><strong>${this.filteredSummaries().length}</strong><span class="muted">trees</span></div>
      <div class="stat"><strong>${utils.escapeHtml(this.correctness.label())}</strong><span class="muted">criterion</span></div>`;
  }

  renderCandidateSummaryChart() {
    const summaries = this.filteredSummaries().filter(r => r.has_bad_nodes);
    const totals = new Map();
    for (const row of summaries) {
      const shard = this.store.cache.get(row.tree_key);
      const leaves = shard ? Object.values(shard.leaf_completions || {}) : [];
      const probs = this.correctness.computeCandidateMentionProbs(leaves);
      for (const item of probs) {
        const key = item.ballot_name || item.full_name || item.id;
        totals.set(key, (totals.get(key) || 0) + item.prob);
      }
    }
    const items = [...totals.entries()].map(([label, prob]) => ({ label, prob })).sort((a, b) => b.prob - a.prob);
    renderHorizontalBarChart("#electionsCandidateChart", "#electionsCandidateChartTitle", items, "prob", "label");
  }

  renderSummary() {
    const rows = this.aggRows();
    const metricKey = this.root.querySelector("#electionsMetricSelect")?.value || "prob_good_pct";
    const metricLabel = this.root.querySelector("#electionsMetricSelect")?.selectedOptions[0]?.textContent || metricKey;
    const hint = this.root.querySelector("#electionsFilterHint");
    if (hint) hint.textContent = `Correct if: ${this.correctness.label()} · ${this.filteredSummaries().length} trees · ${rows.length} groups`;
    this.renderAggTable(rows);
    renderBarChart("#electionsBarChart", "#electionsBarChartTitle", rows, metricKey, metricLabel);
    this.renderGlobalStats();
    this.renderCandidateSummaryChart();
  }

  firstTreeKey() {
    return this.filteredSummaries()[0]?.tree_key;
  }

  async ensureTreeLoaded() {
    if (this.currentSummary) return;
    const key = this.firstTreeKey();
    if (key) await this.loadRun(key);
  }

  async openTree(treeKey) {
    switchSubTab(this.root, "trees");
    this.root.querySelector("#electionsRunSelect").value = treeKey;
    await this.loadRun(treeKey);
  }

  computeNodeStats(nodes, leafMap) {
    const stats = new Map();
    const leafDesc = nodeId => this.graph.leafDescendants(nodes, nodeId);
    for (const node of nodes) {
      const leaves = leafDesc(node.id).map(id => leafMap.get(id)).filter(Boolean);
      if (!leaves.length) continue;
      const nGood = leaves.filter(l => this.correctness.isLeafGood(l)).length;
      const nBad = leaves.filter(l => this.correctness.isLeafBad(l)).length;
      const nLeaves = leaves.length;
      const badPct = nBad / nLeaves;
      let colorClass = "mixed";
      if (nBad === nLeaves) colorClass = "exclusively_bad";
      else if (nGood === nLeaves) colorClass = "exclusively_good";
      else if (nBad / nLeaves > 0.75) colorClass = "mostly_bad";
      stats.set(node.id, { n_leaves: nLeaves, n_bad: nBad, n_good: nGood, bad_pct: badPct, bad_pct_display: Math.round(badPct * 1000) / 10, color_class: colorClass });
    }
    return stats;
  }

  refreshStatsForMode() {
    if (!this.currentShard) return;
    const nodes = this.currentShard.tree_nodes || [];
    const leafMap = new Map(Object.entries(this.currentShard.leaf_completions || {}));
    const statsMap = this.computeNodeStats(nodes, leafMap);
    this.graph.statsMap = statsMap;
    const minProb = parseFloat(this.root.querySelector("#electionsProbFilter")?.value || "0");
    void this.graph.draw(nodes, minProb).then(() => this.graph.fitToScreen());
    this.renderRunCandidateChart(leafMap);
    this.renderRunStats(leafMap);
  }

  renderRunStats(leafMap) {
    const leaves = [...leafMap.values()];
    let good = 0, bad = 0, total = 0;
    for (const leaf of leaves) {
      const p = Number(leaf.path_prob) || 0;
      total += p;
      if (this.correctness.isLeafGood(leaf)) good += p;
      else if (this.correctness.isLeafBad(leaf)) bad += p;
    }
    const el = this.root.querySelector("#electionsRunStats");
    if (!el) return;
    el.innerHTML = `
      <div class="stat"><strong>${total ? Math.round(1000 * good / total) / 10 : 0}%</strong><span class="muted">P(good)</span></div>
      <div class="stat"><strong>${total ? Math.round(1000 * bad / total) / 10 : 0}%</strong><span class="muted">P(bad)</span></div>`;
  }

  renderRunCandidateChart(leafMap) {
    const items = this.correctness.computeCandidateMentionProbs([...leafMap.values()])
      .map(item => ({ label: item.ballot_name || item.full_name || item.id, prob: item.prob }));
    renderHorizontalBarChart("#electionsRunCandidateChart", "#electionsRunCandidateChartTitle", items, "prob", "label");
  }

  async loadRun(treeKey) {
    const summary = this.store.summaryFor(treeKey);
    this.currentSummary = summary;
    const meta = this.root.querySelector("#electionsRunMeta");
    if (meta) meta.textContent = "Loading tree…";
    const shard = await this.store.loadTree(treeKey);
    this.currentShard = shard;
    const nodes = shard.tree_nodes || [];
    const leafMap = new Map(Object.entries(shard.leaf_completions || {}));
    const statusMap = new Map(Object.entries(shard.node_status || {}));
    const statsMap = this.computeNodeStats(nodes, leafMap);
    this.graph.setTreeData({ nodes, leafMap, statusMap, statsMap });
    const minProb = parseFloat(this.root.querySelector("#electionsProbFilter")?.value || "0");
    await this.graph.draw(nodes, minProb);
    this.graph.fitToScreen();
    if (meta) meta.textContent = summary?.instruction || summary?.instruction_variant || treeKey;
    this.renderRunCandidateChart(leafMap);
    this.renderRunStats(leafMap);
    renderDetailPanel(this.root.querySelector("#electionsDetail"), { correctness: this.correctness.label() });
  }

  showNode(nodeId) {
    const node = this.graph.currentNodes.find(n => n.id === nodeId);
    const leaf = this.graph.leafMap.get(nodeId);
    const stats = this.graph.statsMap.get(nodeId);
    const status = this.graph.statusMap.get(nodeId);
    const leaves = this.graph.leafDescendants(this.graph.currentNodes, nodeId)
      .map(id => this.graph.leafMap.get(id)).filter(Boolean);
    renderDetailPanel(this.root.querySelector("#electionsDetail"), {
      node, leaf, stats, status, leaves, correctness: this.correctness.label(),
    });
  }

  onCorrectnessChange(value) {
    this.correctness.setMode(value);
    this.correctness.populateSelect(this.root.querySelector("#electionsCorrectnessMode"));
    this.correctness.populateSelect(this.root.querySelector("#electionsTreeCorrectnessMode"));
    this.renderSummary();
    if (this.currentShard) this.refreshStatsForMode();
  }

  bindEvents() {
    bindSubTabs(this, this.root);
    ["electionsPrefixFilter", "electionsAggSelect", "electionsMetricSelect"].forEach(id => {
      this.root.querySelector(`#${id}`)?.addEventListener("change", () => this.renderSummary());
    });
    this.root.querySelector("#electionsCorrectnessMode")?.addEventListener("change", e => this.onCorrectnessChange(e.target.value));
    this.root.querySelector("#electionsTreeCorrectnessMode")?.addEventListener("change", e => this.onCorrectnessChange(e.target.value));
    this.root.querySelector("#electionsRunSelect")?.addEventListener("change", e => this.loadRun(e.target.value));
    this.root.querySelector("#electionsProbFilter")?.addEventListener("change", () => {
      if (this.currentSummary) this.loadRun(this.currentSummary.tree_key);
    });
    this.root.querySelector("#electionsFitBtn")?.addEventListener("click", () => this.graph.fitToScreen());
    this.root.querySelector("#electionsResetBtn")?.addEventListener("click", () => this.graph.resetZoom());
  }
};

document.addEventListener("experiment-tab", event => {
  if (event.detail.experiment === "elections" && !window._electionsInit) {
    try {
      const adapter = new TD.ElectionsAdapter(document.getElementById("electionsPanel"));
      window._electionsInit = true;
      adapter.init();
    } catch (err) {
      const el = document.getElementById("electionsSource");
      if (el) el.innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }
});
})(TreeDashboard);
