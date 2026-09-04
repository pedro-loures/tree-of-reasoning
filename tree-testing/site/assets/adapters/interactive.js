window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const {
  DataStore, TreeGraph, SummaryPanel, bindSubTabs, switchSubTab, utils,
  renderBarChart, renderScatterChart, renderDetailPanel,
} = TD;

const INTERACTIVE_COLUMNS = [
  { key: "prompt_preview", label: "Prompt" },
  { key: "model_id", label: "Model" },
  { key: "tau", label: "τ" },
  { key: "total_nodes", label: "Nodes" },
  { key: "leaf_count", label: "τ-leaves" },
  { key: "exclusively_bad_count", label: "Excl. bad" },
  { key: "leaf_correct_pct", label: "Leaf good %" },
  { key: "prob_good_pct", label: "P(good) %", highlight: true },
  { key: "prob_bad_pct", label: "P(bad) %", highlight: true },
];

const CHARTABLE = INTERACTIVE_COLUMNS.filter(c => typeof c.key === "string" && c.key !== "prompt_preview");

TD.InteractiveAdapter = class {
  constructor(root) {
    this.root = root;
    this.store = new DataStore("interactive", "data/interactive");
    this.graph = new TreeGraph("#interactiveGraph", "#interactiveTooltip");
    this.summary = new SummaryPanel(root, INTERACTIVE_COLUMNS);
    this.summary.tableEl = root.querySelector("#interactiveSummaryTable");
    this.summary.statsEl = root.querySelector("#interactiveGlobalStats");
    this.summary.hintEl = root.querySelector("#interactiveFilterHint");
    this.currentSummary = null;
    this.tutorial = null;
    this.graph.onSelect = nodeId => this.showNode(nodeId);
  }

  enableViewOnlyMock() {
    const banner = this.root.querySelector(".view-only-banner");
    if (banner) banner.style.display = "block";
    ["interactivePrompt", "interactiveExpected", "interactiveModel", "interactiveTau",
      "interactiveGenerateBtn", "interactiveSaveBtn"].forEach(id => {
      const el = this.root.querySelector(`#${id}`);
      if (el) el.disabled = true;
    });
  }

  populateFormOptions() {
    const models = this.store.manifest?.models || [];
    const modelSel = this.root.querySelector("#interactiveModel");
    if (modelSel && models.length) {
      modelSel.innerHTML = models.map(m => `<option>${m}</option>`).join("");
    }
    const taus = this.store.manifest?.taus || [];
    const tauSel = this.root.querySelector("#interactiveTau");
    if (tauSel && taus.length) {
      tauSel.innerHTML = taus.map(t => `<option value="${t}">${t}</option>`).join("");
    }
    const modelFilter = this.root.querySelector("#interactiveModelFilter");
    if (modelFilter) {
      modelFilter.innerHTML = '<option value="">All models</option>' + models.map(m =>
        `<option value="${m}">${m}</option>`).join("");
    }
    const tauFilter = this.root.querySelector("#interactiveTauFilter");
    if (tauFilter) {
      tauFilter.innerHTML = '<option value="">All τ</option>' + taus.map(t =>
        `<option value="${t}">${t}</option>`).join("");
    }
    const metricOpts = CHARTABLE.map(c => `<option value="${c.key}">${c.label}</option>`).join("");
    this.root.querySelector("#interactiveMetricSelect").innerHTML = metricOpts;
    this.root.querySelector("#interactiveScatterX").innerHTML = metricOpts;
    this.root.querySelector("#interactiveScatterY").innerHTML = metricOpts;
    this.root.querySelector("#interactiveMetricSelect").value = "prob_good_pct";
    this.root.querySelector("#interactiveScatterX").value = "total_nodes";
    this.root.querySelector("#interactiveScatterY").value = "prob_good_pct";
  }

  populateMockForm(tutorial) {
    if (!tutorial) return;
    this.root.querySelector("#interactivePrompt").value = tutorial.prompt || "";
    this.root.querySelector("#interactiveExpected").value = tutorial.expected_answers || "";
    if (tutorial.model_id) this.root.querySelector("#interactiveModel").value = tutorial.model_id;
    if (tutorial.tau != null) this.root.querySelector("#interactiveTau").value = String(tutorial.tau);
  }

  async init() {
    this.enableViewOnlyMock();
    try {
      await this.store.loadManifest();
      document.getElementById("interactiveSource").textContent =
        `View-only demo · ${this.store.manifest.shards?.length || 0} saved trees · generated ${this.store.manifest.generated_at}`;
      const tutorialResp = await fetch("data/interactive/tutorial.json?ts=" + Date.now());
      if (tutorialResp.ok) this.tutorial = await tutorialResp.json();
      this.populateFormOptions();
      if (this.tutorial) this.populateMockForm(this.tutorial);
      this.populateRunSelect();
      this.renderSummary();
      this.bindEvents();
    } catch (err) {
      document.getElementById("interactiveSource").innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }

  filteredSummaries() {
    const model = this.root.querySelector("#interactiveModelFilter")?.value || "";
    const tau = this.root.querySelector("#interactiveTauFilter")?.value || "";
    return (this.store.DATA.tree_summaries || []).filter(row => {
      if (model && row.model_id !== model) return false;
      if (tau && String(row.tau) !== tau) return false;
      return true;
    });
  }

  chartRows() {
    return this.filteredSummaries().map(row => ({
      ...row,
      label: row.prompt_preview || row.tree_key,
    }));
  }

  populateRunSelect() {
    const runSel = this.root.querySelector("#interactiveRunSelect");
    const summaries = this.store.DATA.tree_summaries || [];
    runSel.innerHTML = summaries.map(row =>
      `<option value="${row.tree_key}">${row.prompt_preview || row.tree_key}</option>`).join("");
  }

  renderGlobalStats() {
    const rows = this.filteredSummaries();
    let good = 0, bad = 0, n = 0;
    for (const row of rows) {
      good += row.prob_good || 0;
      bad += row.prob_bad || 0;
      n += 1;
    }
    const el = this.root.querySelector("#interactiveGlobalStats");
    if (!el) return;
    el.innerHTML = `
      <div class="stat"><strong>${rows.length}</strong><span class="muted">trees</span></div>
      <div class="stat"><strong>${n ? Math.round(1000 * good / n) / 10 : 0}</strong><span class="muted">avg P(good)</span></div>
      <div class="stat"><strong>${n ? Math.round(1000 * bad / n) / 10 : 0}</strong><span class="muted">avg P(bad)</span></div>
      <div class="stat"><strong>view only</strong><span class="muted">mode</span></div>`;
  }

  renderSummary() {
    const rows = this.chartRows();
    const metricKey = this.root.querySelector("#interactiveMetricSelect")?.value || "prob_good_pct";
    const xKey = this.root.querySelector("#interactiveScatterX")?.value || "total_nodes";
    const yKey = this.root.querySelector("#interactiveScatterY")?.value || "prob_good_pct";
    const metricLabel = this.root.querySelector("#interactiveMetricSelect")?.selectedOptions[0]?.textContent;
    const xLabel = this.root.querySelector("#interactiveScatterX")?.selectedOptions[0]?.textContent;
    const yLabel = this.root.querySelector("#interactiveScatterY")?.selectedOptions[0]?.textContent;
    this.summary.renderTable(rows, treeKey => this.openTree(treeKey));
    renderBarChart("#interactiveBarChart", "#interactiveBarChartTitle", rows, metricKey, metricLabel);
    renderScatterChart("#interactiveScatterChart", "#interactiveScatterChartTitle", rows, xKey, yKey, xLabel, yLabel);
    this.renderGlobalStats();
  }

  firstTreeKey() {
    return this.filteredSummaries()[0]?.tree_key || this.store.DATA?.tree_summaries?.[0]?.tree_key;
  }

  async ensureTreeLoaded() {
    if (this.currentSummary) return;
    const key = this.firstTreeKey();
    if (key) await this.loadRun(key);
  }

  async openTree(treeKey) {
    switchSubTab(this.root, "trees");
    this.root.querySelector("#interactiveRunSelect").value = treeKey;
    await this.loadRun(treeKey);
  }

  async loadRun(treeKey) {
    const summary = this.store.summaryFor(treeKey);
    this.currentSummary = summary;
    const meta = this.root.querySelector("#interactiveRunMeta");
    if (meta) meta.textContent = "Loading tree…";
    const shard = await this.store.loadTree(treeKey);
    const nodes = shard.tree_nodes || [];
    const leafMap = new Map(Object.entries(shard.leaf_completions || {}));
    const statusMap = new Map(Object.entries(shard.node_status || {}));
    const statsMap = new Map(Object.entries(shard.node_stats || {}));
    this.graph.setTreeData({ nodes, leafMap, statusMap, statsMap });
    const minProb = parseFloat(this.root.querySelector("#interactiveProbFilter")?.value || "0");
    this.graph.setOptions({
      showBad: this.root.querySelector("#interactiveShowBad")?.checked !== false,
      showGood: this.root.querySelector("#interactiveShowGood")?.checked !== false,
      dimOther: this.root.querySelector("#interactiveDimOther")?.checked,
    });
    await this.graph.draw(nodes, minProb);
    this.graph.fitToScreen();
    if (meta && summary) {
      meta.textContent = `${summary.prompt || summary.prompt_preview || ""} · τ=${summary.tau} · ${summary.model_id}`;
    }
    renderDetailPanel(this.root.querySelector("#interactiveDetail"), {});
  }

  showNode(nodeId) {
    const node = this.graph.currentNodes.find(n => n.id === nodeId);
    const leaf = this.graph.leafMap.get(nodeId);
    const stats = this.graph.statsMap.get(nodeId);
    const status = this.graph.statusMap.get(nodeId);
    renderDetailPanel(this.root.querySelector("#interactiveDetail"), {
      node, leaf, stats, status, showReasoning: true,
    });
  }

  bindEvents() {
    bindSubTabs(this, this.root);
    ["interactiveModelFilter", "interactiveTauFilter", "interactiveMetricSelect", "interactiveScatterX", "interactiveScatterY"]
      .forEach(id => this.root.querySelector(`#${id}`)?.addEventListener("change", () => this.renderSummary()));
    this.root.querySelector("#interactiveRunSelect")?.addEventListener("change", e => this.loadRun(e.target.value));
    this.root.querySelector("#interactiveProbFilter")?.addEventListener("change", () => {
      if (this.currentSummary) this.loadRun(this.currentSummary.tree_key);
    });
    this.root.querySelector("#interactiveFitBtn")?.addEventListener("click", () => this.graph.fitToScreen());
    this.root.querySelector("#interactiveResetBtn")?.addEventListener("click", () => this.graph.resetZoom());
    ["interactiveShowBad", "interactiveShowGood", "interactiveDimOther"].forEach(id => {
      this.root.querySelector(`#${id}`)?.addEventListener("change", () => {
        if (this.currentSummary) this.loadRun(this.currentSummary.tree_key);
      });
    });
    this.root.querySelector("#interactiveGenerateBtn")?.addEventListener("click", () => {
      const status = this.root.querySelector("#interactiveStatus");
      if (status) status.textContent = "Generation is only available locally via serve_dashboard.py with a GPU.";
    });
  }
};

document.addEventListener("experiment-tab", event => {
  if (event.detail.experiment === "interactive" && !window._interactiveInit) {
    try {
      const adapter = new TD.InteractiveAdapter(document.getElementById("interactivePanel"));
      window._interactiveInit = true;
      adapter.init();
    } catch (err) {
      const el = document.getElementById("interactiveSource");
      if (el) el.innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }
});

document.addEventListener("DOMContentLoaded", () => {
  TD.switchTopTab("interactive");
});
})(TreeDashboard);
