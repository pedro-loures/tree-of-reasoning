window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const {
  DataStore, TreeGraph, SummaryPanel, bindSubTabs, switchSubTab, utils,
  aggregateSummaries, renderBarChart, renderScatterChart, SCATTER_NUMERIC_COLUMNS,
  createCapitalsBadFilters, renderDetailPanel,
} = TD;

const CAPITALS_TABLE_COLUMNS = [
  { key: "label", label: "Group", align: "left", freezeDefault: true },
  { key: "trees", label: "Trees" },
  { key: "bad_nodes_analyzed", label: "Bad-nodes done", chartable: true },
  { key: "avg_max_depth", label: "Avg depth", chartable: true },
  { key: "avg_total_nodes", label: "Avg nodes", chartable: true },
  { key: "avg_leaf_count", label: "Avg τ-leaves", chartable: true },
  { key: "avg_mass_above_tau", label: "Avg mass above τ", chartable: true },
  { key: "avg_max_breadth", label: "Avg max breadth", chartable: true },
  { key: "avg_mean_breadth_by_depth", label: "Avg breadth/depth", chartable: true },
  { key: "avg_breadth_warnings", label: "Avg breadth warns", chartable: true },
  { key: "avg_exclusively_bad", label: "Avg excl. bad", chartable: true },
  { key: "avg_ditched", label: "Avg ditched", chartable: true },
  { key: "leaf_accuracy_pct", label: "Leaf acc %", chartable: true },
  { key: "prob_good_pct", label: "P(good) %", chartable: true, highlight: true },
  { key: "prob_bad_pct", label: "P(bad) %", chartable: true, highlight: true },
  { key: "greedy_accuracy_pct", label: "Greedy %", chartable: true },
];

const CAPITALS_ROW_COLUMNS = [
  { key: "country_name", label: "Country" },
  { key: "prefix_length", label: "Prefix" },
  { key: "max_depth", label: "Depth" },
  { key: "total_nodes", label: "Nodes" },
  { key: "leaf_count", label: "τ-leaves" },
  { key: "max_breadth", label: "Max breadth" },
  { key: "mean_breadth_by_depth", label: "Breadth/depth" },
  { key: "breadth_warning_count", label: "Breadth warns" },
  { key: "mass_above_tau", label: "Mass above τ" },
  { key: "exclusively_bad_count", label: "Excl. bad" },
  { key: "prob_good_pct", label: "P(good) %", highlight: true },
  { key: "prob_bad_pct", label: "P(bad) %", highlight: true },
];

const CAPITALS_PANEL_CONFIGS = {
  capitals: {
    tab: "capitals",
    initKey: "_capitalsInit",
    pfx: "capitals",
    dataPath: "data/capitals",
    experiment: "capitals",
    sourceLabel: "Capitals experiment (plain)",
    badFilterMode: "plain",
  },
  ignoreLorem: {
    tab: "ignoreLorem",
    initKey: "_ignoreLoremInit",
    pfx: "ignoreLorem",
    dataPath: "data/ignore-lorem",
    experiment: "ignore-lorem",
    sourceLabel: "Capitals experiment (legacy · ignore lorem)",
    badFilterMode: "ignore-lorem",
  },
};

TD.CapitalsAdapter = class {
  constructor(root, cfg) {
    this.root = root;
    this.cfg = cfg;
    this.badFilters = createCapitalsBadFilters(cfg.badFilterMode || "plain");
    this.id = name => `${cfg.pfx}${name}`;
    this.el = name => root.querySelector(`#${this.id(name)}`);
    this.store = new DataStore(cfg.experiment, cfg.dataPath);
    this.graph = new TreeGraph(`#${this.id("Graph")}`, `#${this.id("Tooltip")}`);
    this.summary = new SummaryPanel(root, CAPITALS_ROW_COLUMNS);
    this.summary.tableEl = this.el("SummaryTable");
    this.summary.statsEl = this.el("GlobalStats");
    this.summary.hintEl = this.el("FilterHint");
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
      const fill = this.el("ProgressFill");
      if (fill) fill.style.width = `${pct}%`;
      const source = this.el("Source");
      if (source) {
        source.textContent =
          `${this.cfg.sourceLabel} · ${this.store.manifest.viewer_runs || 0} trees · generated ${this.store.manifest.generated_at}`;
      }
    } catch (err) {
      const source = this.el("Source");
      if (source) source.innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }

  filteredSummaries() {
    const prefix = this.el("PrefixFilter")?.value || "";
    const region = this.el("RegionFilter")?.value || "";
    const country = this.el("CountryFilter")?.value || "";
    return (this.store.DATA?.tree_summaries || []).filter(row => {
      if (prefix && String(row.prefix_length) !== prefix) return false;
      if (region && row.region_id !== region) return false;
      if (country && row.country_id !== country) return false;
      return true;
    });
  }

  aggRows() {
    const dimension = this.el("AggSelect")?.value || "whole";
    return aggregateSummaries(this.filteredSummaries(), dimension).map(row => ({
      ...row, label: row.label,
    }));
  }

  treeRows() {
    return this.filteredSummaries().map(row => {
      const totalLeaves = row.total_leaves || 0;
      const leafCorrectPct = totalLeaves
        ? Math.round(1000 * (row.leaf_correct || 0) / totalLeaves) / 10
        : null;
      return TD.expandMotifHits({
        ...row,
        label: row.country_name || row.country_id || row.tree_key,
        leaf_correct_pct: leafCorrectPct,
      });
    });
  }

  summaryColumns() {
    return TD.tableColumnsWithMotifs(CAPITALS_TABLE_COLUMNS, this.filteredSummaries(), { chartable: true });
  }

  populateFilters() {
    const summaries = this.store.DATA.tree_summaries || [];
    const regions = [...new Set(summaries.map(r => r.region_id).filter(Boolean))];
    const countries = [...new Set(summaries.map(r => r.country_id).filter(Boolean))].sort();
    this.el("RegionFilter").innerHTML =
      '<option value="">All regions</option>' + regions.map(r => `<option value="${r}">${r}</option>`).join("");
    this.el("CountryFilter").innerHTML =
      '<option value="">All countries</option>' + countries.map(c => {
        const row = summaries.find(s => s.country_id === c);
        return `<option value="${c}">${row?.country_name || c}</option>`;
      }).join("");
    this.el("PrefixFilter").innerHTML =
      '<option value="">All prefixes</option>' + (this.store.manifest.prefix_lengths || []).map(pl =>
        `<option value="${pl}">${pl}</option>`).join("");
    const metricSel = this.el("MetricSelect");
    const tableColumns = TD.tableColumnsWithMotifs(CAPITALS_TABLE_COLUMNS, summaries, { chartable: true });
    metricSel.innerHTML = tableColumns.filter(c => c.chartable).map(c =>
      `<option value="${c.key}">${c.label}</option>`).join("");
    metricSel.value = "prob_good_pct";
    const scatterOpts = SCATTER_NUMERIC_COLUMNS.map(col =>
      `<option value="${col.key}">${col.label}</option>`).join("");
    this.el("ScatterX").innerHTML = scatterOpts;
    this.el("ScatterY").innerHTML = scatterOpts;
    this.el("ScatterX").value = "total_nodes";
    this.el("ScatterY").value = "prob_good_pct";
    this.el("RunSelect").innerHTML = summaries.map(row =>
      `<option value="${row.tree_key}">${row.country_name || row.country_id} · p${row.prefix_length}</option>`).join("");
  }

  renderAggTable(rows) {
    const table = this.el("SummaryTable");
    if (!table) return;
    const dimension = this.el("AggSelect")?.value || "whole";
    const columns = this.summaryColumns();
    const expandedRows = rows.map(row => TD.expandMotifHits(row));
    const sorted = [...expandedRows].sort((a, b) => {
      const av = a[this.sortKey], bv = b[this.sortKey];
      if (av === bv) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return (av < bv ? -1 : 1) * (this.sortAsc ? 1 : -1);
    });
    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");
    thead.innerHTML = TD.buildSummaryTableHead(columns, dimension);
    tbody.innerHTML = TD.renderSummaryTableBody(columns, sorted);
    TD.applyFrozenColumns(table);
    tbody.querySelectorAll("tr").forEach(tr => {
      tr.addEventListener("click", () => {
        const key = tr.dataset.groupKey;
        const dimension = this.el("AggSelect")?.value || "whole";
        const summaries = this.filteredSummaries();
        let match = summaries[0];
        if (dimension === "by_country") match = summaries.find(s => s.country_id === key) || match;
        else if (dimension === "by_region") match = summaries.find(s => s.region_id === key) || match;
        else if (dimension === "by_prefix_length") match = summaries.find(s => String(s.prefix_length) === key) || match;
        if (match) this.openTree(match.tree_key);
      });
    });
    thead.querySelectorAll("th[data-key]").forEach(th => {
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
    const el = this.el("GlobalStats");
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
    const aggRows = this.aggRows();
    const scatterRows = this.treeRows();
    const metricKey = this.el("MetricSelect")?.value || "prob_good_pct";
    const metricLabel = this.el("MetricSelect")?.selectedOptions[0]?.textContent || metricKey;
    const xKey = this.el("ScatterX")?.value || "total_nodes";
    const yKey = this.el("ScatterY")?.value || "prob_good_pct";
    const xLabel = this.el("ScatterX")?.selectedOptions[0]?.textContent;
    const yLabel = this.el("ScatterY")?.selectedOptions[0]?.textContent;
    const hint = this.el("FilterHint");
    if (hint) {
      hint.textContent = `${scatterRows.length} trees · ${aggRows.length} groups · scatter uses per-tree rows`;
    }
    this.renderAggTable(aggRows);
    renderBarChart(`#${this.id("BarChart")}`, `#${this.id("BarChartTitle")}`, aggRows, metricKey, metricLabel);
    renderScatterChart(`#${this.id("ScatterChart")}`, `#${this.id("ScatterChartTitle")}`, scatterRows, xKey, yKey, xLabel, yLabel);
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
    const runSel = this.el("RunSelect");
    if (runSel) runSel.value = treeKey;
    await this.loadRun(treeKey);
  }

  refreshStatsForMode() {
    if (!this.currentShard) return;
    const nodes = this.currentShard.tree_nodes || [];
    const leafMap = new Map(Object.entries(this.currentShard.leaf_completions || {}));
    const statsMap = this.badFilters.computeNodeStats(nodes, leafMap, this.root);
    this.graph.statsMap = statsMap;
    const highlightSet = new Set();
    if (this.el("HighlightBadOnly")?.checked) {
      for (const [id, st] of this.graph.statusMap.entries()) if (st === "exclusively_bad") highlightSet.add(id);
    }
    this.graph.setOptions({
      showBad: this.el("ShowBad")?.checked !== false,
      showGood: this.el("ShowGood")?.checked !== false,
      dimOther: this.el("HighlightBadOnly")?.checked,
      highlightSet,
      isLeafGood: leaf => this.badFilters.isLeafCorrect(leaf),
      isLeafBad: leaf => this.badFilters.isLeafBad(leaf, this.root),
    });
    const minProb = parseFloat(this.el("ProbFilter")?.value || "0");
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
      if (this.badFilters.isLeafCorrect(leaf)) good += p;
      else if (this.badFilters.isLeafBad(leaf, this.root)) bad += p;
    }
    const el = this.el("RunStats");
    if (!el) return;
    el.innerHTML = `
      <div class="stat"><strong>${total ? Math.round(1000 * good / total) / 10 : 0}%</strong><span class="muted">P(good)</span></div>
      <div class="stat"><strong>${total ? Math.round(1000 * bad / total) / 10 : 0}%</strong><span class="muted">P(bad)</span></div>`;
  }

  renderBadNodeList(nodes, statsMap, leafMap) {
    const el = this.el("BadNodeList");
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
        this.graph.focusNode(nodeId, nodes, parseFloat(this.el("ProbFilter")?.value || "0"));
        this.showNode(nodeId);
      });
    });
  }

  async loadRun(treeKey) {
    const summary = this.store.summaryFor(treeKey);
    this.currentSummary = summary;
    const meta = this.el("RunMeta");
    if (meta) meta.textContent = "Loading tree…";
    const shard = await this.store.loadTree(treeKey);
    this.currentShard = shard;
    const nodes = shard.tree_nodes || [];
    const leafMap = new Map(Object.entries(shard.leaf_completions || {}));
    const statusMap = new Map(Object.entries(shard.node_status || {}));
    const expansionMap = new Map(Object.entries(shard.node_expansions || {}));
    const statsMap = this.badFilters.computeNodeStats(nodes, leafMap, this.root);
    const highlightSet = new Set();
    if (this.el("HighlightBadOnly")?.checked) {
      for (const [id, st] of statusMap.entries()) if (st === "exclusively_bad") highlightSet.add(id);
    }
    this.graph.setTreeData({ nodes, leafMap, statusMap, statsMap, expansionMap });
    const minProb = parseFloat(this.el("ProbFilter")?.value || "0");
    this.graph.setOptions({
      showBad: this.el("ShowBad")?.checked !== false,
      showGood: this.el("ShowGood")?.checked !== false,
      dimOther: this.el("HighlightBadOnly")?.checked,
      highlightSet,
      isLeafGood: leaf => this.badFilters.isLeafCorrect(leaf),
      isLeafBad: leaf => this.badFilters.isLeafBad(leaf, this.root),
    });
    await this.graph.draw(nodes, minProb);
    this.graph.fitToScreen();
    if (meta && summary) {
      meta.textContent = `${summary.country_name || summary.country_id} · prefix ${summary.prefix_length}`;
    }
    this.renderBadNodeList(nodes, statsMap, leafMap);
    this.renderRunStats(leafMap);
    renderDetailPanel(this.el("Detail"), {});
  }

  showNode(nodeId) {
    if (!nodeId) {
      renderDetailPanel(this.el("Detail"), {});
      return;
    }
    const node = this.graph.currentNodes.find(n => n.id === nodeId);
    const leaf = this.graph.leafMap.get(nodeId);
    const stats = this.graph.statsMap.get(nodeId);
    const status = this.graph.statusMap.get(nodeId);
    const leaves = this.graph.leafDescendants(this.graph.currentNodes, nodeId)
      .map(id => this.graph.leafMap.get(id)).filter(Boolean);
    renderDetailPanel(this.el("Detail"), {
      node, leaf, stats, status, leaves, nodes: this.graph.currentNodes,
    });
  }

  bindEvents() {
    bindSubTabs(this, this.root);
    ["PrefixFilter", "RegionFilter", "CountryFilter", "AggSelect", "MetricSelect", "ScatterX", "ScatterY"]
      .forEach(name => this.el(name)?.addEventListener("change", () => this.renderSummary()));
    this.el("RunSelect")?.addEventListener("change", e => this.loadRun(e.target.value));
    this.el("ProbFilter")?.addEventListener("change", () => {
      if (this.currentSummary) this.loadRun(this.currentSummary.tree_key);
    });
    this.el("FitBtn")?.addEventListener("click", () => this.graph.fitToScreen());
    this.el("ResetBtn")?.addEventListener("click", () => this.graph.resetZoom());
    ["ShowBad", "ShowGood", "HighlightBadOnly"].forEach(name => {
      this.el(name)?.addEventListener("change", () => {
        if (this.currentSummary) this.refreshStatsForMode();
      });
    });
    this.badFilters.SUBTYPES.forEach(item => {
      this.root.querySelector(`#badFilter_${item.id}`)?.addEventListener("change", () => this.refreshStatsForMode());
    });
  }
};

function initCapitalsPanel(cfg) {
  document.addEventListener("experiment-tab", event => {
    if (event.detail.experiment !== cfg.tab || window[cfg.initKey]) return;
    try {
      const adapter = new TD.CapitalsAdapter(document.getElementById(`${cfg.pfx}Panel`), cfg);
      window[cfg.initKey] = true;
      adapter.init();
    } catch (err) {
      const el = document.getElementById(`${cfg.pfx}Source`);
      if (el) el.innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  });
}

initCapitalsPanel(CAPITALS_PANEL_CONFIGS.capitals);
initCapitalsPanel(CAPITALS_PANEL_CONFIGS.ignoreLorem);
})(TreeDashboard);
