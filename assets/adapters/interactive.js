window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const {
  DataStore, TreeGraph, SummaryPanel, bindSubTabs, switchSubTab, utils,
  renderBarChart, renderScatterChart, renderDetailPanel,
  INTERACTIVE_DEFAULT_TAU, interactiveDefaultTau, interactiveTauIsSlow, interactiveTauMatches,
} = TD;

const INTERACTIVE_COLUMNS = [
  { key: "prompt_preview", label: "Prompt", align: "left", freezeDefault: true },
  { key: "model_id", label: "Model", align: "left" },
  { key: "tau", label: "τ" },
  { key: "max_depth", label: "Depth" },
  { key: "total_nodes", label: "Nodes" },
  { key: "leaf_count", label: "τ-leaves" },
  { key: "max_breadth", label: "Max breadth" },
  { key: "mean_breadth_by_depth", label: "Breadth/depth" },
  { key: "breadth_warning_count", label: "Breadth warns" },
  { key: "mass_above_tau", label: "Mass above τ" },
  { key: "total_candidates", label: "Candidates" },
  { key: "exclusively_bad_count", label: "Excl. bad" },
  { key: "exclusively_bad_pct", label: "% bad cand." },
  { key: "leaf_correct_pct", label: "Leaf good %" },
  { key: "prob_good_pct", label: "P(good) %", highlight: true },
  { key: "prob_bad_pct", label: "P(bad) %", highlight: true },
];

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
    const taus = this.store.manifest?.taus || [];
    const defaultTau = interactiveDefaultTau(taus);
    const modelSel = this.root.querySelector("#interactiveModel");
    if (modelSel && models.length) {
      modelSel.innerHTML = models.map(m => `<option>${m}</option>`).join("");
    }
    const tauSel = this.root.querySelector("#interactiveTau");
    if (tauSel && taus.length) {
      tauSel.innerHTML = taus.map(t => `<option value="${t}">${t}</option>`).join("");
      tauSel.value = String(defaultTau);
    }
    const tauOptions = '<option value="">All τ</option>' + [...taus].sort((a, b) => Number(b) - Number(a)).map(t => {
      const slow = interactiveTauIsSlow(t);
      const label = slow ? `${t} (large tree — slow)` : `${t}`;
      return `<option value="${t}">${label}</option>`;
    }).join("");
    const tauFilter = this.root.querySelector("#interactiveTauFilter");
    if (tauFilter) {
      tauFilter.innerHTML = tauOptions;
      tauFilter.value = "";
    }
    const treeTauFilter = this.root.querySelector("#interactiveTreeTauFilter");
    if (treeTauFilter) treeTauFilter.remove();
    const modelFilter = this.root.querySelector("#interactiveModelFilter");
    if (modelFilter) {
      modelFilter.innerHTML = '<option value="">All models</option>' + models.map(m =>
        `<option value="${m}">${m}</option>`).join("");
    }
    TD.populateTauProbFilter(
      this.root.querySelector("#interactiveProbFilter"),
      [defaultTau],
      { selected: defaultTau },
    );
    const legend = this.root.querySelector("#interactiveLegend");
    if (legend) legend.innerHTML = TD.treeLegendHtml();
    const metricOpts = this.summaryColumns()
      .filter(c => typeof c.key === "string" && c.key !== "prompt_preview")
      .map(c => `<option value="${c.key}">${c.label}</option>`).join("");
    this.root.querySelector("#interactiveMetricSelect").innerHTML = metricOpts;
    this.root.querySelector("#interactiveScatterX").innerHTML = metricOpts;
    this.root.querySelector("#interactiveScatterY").innerHTML = metricOpts;
    this.root.querySelector("#interactiveMetricSelect").value = "prob_good_pct";
    this.root.querySelector("#interactiveScatterX").value = "total_nodes";
    this.root.querySelector("#interactiveScatterY").value = "prob_good_pct";
    this.updateTauWarning();
  }

  selectedTau() {
    return this.root.querySelector("#interactiveTauFilter")?.value ?? "";
  }

  allSummaries() {
    const model = this.root.querySelector("#interactiveModelFilter")?.value || "";
    return (this.store.DATA.tree_summaries || []).filter(row => {
      if (model && row.model_id !== model) return false;
      return true;
    });
  }

  defaultTreeKey() {
    const all = this.allSummaries();
    const defaultTau = interactiveDefaultTau(this.store.manifest?.taus);
    return all.find(row => interactiveTauMatches(row.tau, defaultTau))?.tree_key
      || all[0]?.tree_key
      || null;
  }

  updateTauWarning() {
    const warning = this.root.querySelector("#interactiveTauWarning");
    if (!warning) return;
    const tau = this.selectedTau();
    const show = interactiveTauIsSlow(tau);
    warning.hidden = !show;
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
      const src = document.getElementById("interactiveSource");
      if (src) {
        src.textContent =
          `View-only demo · ${this.allSummaries().length} saved trees (Tree dropdown lists all) · explorer defaults to τ=${interactiveDefaultTau(this.store.manifest?.taus)} · generated ${this.store.manifest.generated_at}`;
      }
    } catch (err) {
      document.getElementById("interactiveSource").innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }

  filteredSummaries() {
    const model = this.root.querySelector("#interactiveModelFilter")?.value || "";
    const tau = this.selectedTau();
    return (this.store.DATA.tree_summaries || []).filter(row => {
      if (model && row.model_id !== model) return false;
      if (tau && !interactiveTauMatches(row.tau, tau)) return false;
      return true;
    });
  }

  chartRows() {
    return this.filteredSummaries().map(row => TD.expandMotifHits({
      ...row,
      label: row.prompt_preview || row.tree_key,
    }));
  }

  summaryColumns() {
    return TD.tableColumnsWithMotifs(INTERACTIVE_COLUMNS, this.filteredSummaries());
  }

  populateRunSelect() {
    const runSel = this.root.querySelector("#interactiveRunSelect");
    const summaries = this.allSummaries();
    if (!runSel) return;
    const prev = runSel.value;
    runSel.innerHTML = summaries.length
      ? summaries.map(row => {
        const slow = interactiveTauIsSlow(row.tau);
        const suffix = slow ? " · large tree" : "";
        const label = `${row.prompt_preview || row.tree_key} · τ=${row.tau}${suffix}`;
        return `<option value="${row.tree_key}">${utils.escapeHtml(label)}</option>`;
      }).join("")
      : '<option value="">No saved trees</option>';
    if (prev && summaries.some(row => row.tree_key === prev)) {
      runSel.value = prev;
    } else {
      const defaultKey = this.defaultTreeKey();
      if (defaultKey) runSel.value = defaultKey;
    }
  }

  onFiltersChanged(fromId) {
    if (fromId === "interactiveTauFilter") this.updateTauWarning();
    this.renderSummary();
    this.populateRunSelect();
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
    const columns = this.summaryColumns();
    this.summary.columns = columns;
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
    const hint = this.root.querySelector("#interactiveFilterHint");
    if (hint) {
      const tau = this.selectedTau();
      const model = this.root.querySelector("#interactiveModelFilter")?.value;
      const parts = [`${rows.length} of ${this.allSummaries().length} trees in summary`];
      if (tau) parts.push(`τ=${tau}`);
      else parts.push("all τ");
      if (model) parts.push(model);
      hint.textContent = parts.join(" · ");
    }
  }

  firstTreeKey() {
    return this.defaultTreeKey();
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
    const slow = interactiveTauIsSlow(summary?.tau);
    if (meta) {
      meta.textContent = slow
        ? `Loading large tree (τ=${summary.tau}, ~${summary?.total_nodes || "?"} nodes) — may take 30+ seconds…`
        : "Loading tree…";
    }
    const shard = await this.store.loadTree(treeKey);
    const nodes = shard.tree_nodes || [];
    const leafMap = new Map(Object.entries(shard.leaf_completions || {}));
    const statusMap = new Map(Object.entries(shard.node_status || {}));
    const statsMap = new Map(Object.entries(shard.node_stats || {}));
    this.graph.setTreeData({ nodes, leafMap, statusMap, statsMap });
    TD.populateTauProbFilter(
      this.root.querySelector("#interactiveProbFilter"),
      summary?.tau != null ? [summary.tau] : [INTERACTIVE_DEFAULT_TAU],
      { selected: summary?.tau ?? INTERACTIVE_DEFAULT_TAU },
    );
    const minProb = parseFloat(this.root.querySelector("#interactiveProbFilter")?.value || "0");
    this.graph.setOptions({
      showBad: this.root.querySelector("#interactiveShowBad")?.checked !== false,
      showGood: this.root.querySelector("#interactiveShowGood")?.checked !== false,
      dimOther: this.root.querySelector("#interactiveDimOther")?.checked,
      isLeafGood: leaf => Boolean(leaf?.answer_correct),
      isLeafBad: leaf => leaf?.answer_correct === false,
    });
    await this.graph.draw(nodes, minProb);
    this.graph.fitToScreen();
    if (meta && summary) {
      meta.textContent = `${summary.prompt || summary.prompt_preview || ""} · τ=${summary.tau} · ${summary.model_id}`;
    }
    renderDetailPanel(this.root.querySelector("#interactiveDetail"), {
      correctness: this.correctnessLabel(),
    });
  }

  correctnessLabel() {
    const summary = this.currentSummary;
    if (summary?.expected_answers) {
      const mode = summary.answer_mode ? ` (${summary.answer_mode})` : "";
      return `${summary.expected_answers}${mode}`;
    }
    return null;
  }

  showNode(nodeId) {
    if (!nodeId) {
      renderDetailPanel(this.root.querySelector("#interactiveDetail"), {
        correctness: this.correctnessLabel(),
      });
      return;
    }
    const node = this.graph.currentNodes.find(n => n.id === nodeId);
    const leaf = this.graph.leafMap.get(nodeId);
    const stats = this.graph.statsMap.get(nodeId);
    const status = this.graph.statusMap.get(nodeId);
    renderDetailPanel(this.root.querySelector("#interactiveDetail"), {
      node, leaf, stats, status, nodes: this.graph.currentNodes,
      correctness: this.correctnessLabel(),
    });
  }

  bindEvents() {
    bindSubTabs(this, this.root);
    ["interactiveModelFilter", "interactiveTauFilter",
      "interactiveMetricSelect", "interactiveScatterX", "interactiveScatterY"]
      .forEach(id => this.root.querySelector(`#${id}`)?.addEventListener("change", () => this.onFiltersChanged(id)));
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

})(TreeDashboard);
