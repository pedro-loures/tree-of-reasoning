window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { DataStore, TreeGraph, SummaryPanel, renderDetail, switchSubTab, utils } = TD;

const CAPITALS_COLUMNS = [
  { key: "country_name", label: "Country" },
  { key: "instruction_variant", label: "Variant" },
  { key: "prefix_length", label: "Prefix" },
  { key: "total_nodes", label: "Nodes" },
  { key: "leaf_count", label: "τ-leaves" },
  { key: "exclusively_bad_count", label: "Excl. bad" },
  { key: "prob_good_pct", label: "P(good) %" },
  { key: "prob_bad_pct", label: "P(bad) %" },
];

TD.CapitalsAdapter = class {
  constructor(root) {
    this.root = root;
    this.store = new DataStore("capitals", "data/capitals");
    this.graph = new TreeGraph("#capitalsGraph", "#capitalsTooltip");
    this.summary = new SummaryPanel(root, CAPITALS_COLUMNS);
    this.currentSummary = null;
    this.graph.onSelect = nodeId => this.showNode(nodeId);
  }

  async init() {
    try {
      await this.store.loadManifest();
      this.populateFilters();
      this.renderSummary();
      this.bindEvents();
      document.getElementById("capitalsSource").textContent =
        `Capitals experiment · ${this.store.manifest.viewer_runs || 0} trees · generated ${this.store.manifest.generated_at}`;
    } catch (err) {
      document.getElementById("capitalsSource").innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }

  filteredSummaries() {
    const region = this.root.querySelector("#capitalsRegionFilter")?.value || "";
    const country = this.root.querySelector("#capitalsCountryFilter")?.value || "";
    return (this.store.DATA?.tree_summaries || []).filter(row => {
      if (region && row.region_id !== region) return false;
      if (country && row.country_id !== country) return false;
      return true;
    });
  }

  populateFilters() {
    const summaries = this.store.DATA.tree_summaries || [];
    const regions = [...new Set(summaries.map(row => row.region_id).filter(Boolean))];
    const countries = [...new Set(summaries.map(row => row.country_id).filter(Boolean))].sort();
    const regionSel = this.root.querySelector("#capitalsRegionFilter");
    const countrySel = this.root.querySelector("#capitalsCountryFilter");
    regionSel.innerHTML = '<option value="">All regions</option>' + regions.map(r => `<option value="${r}">${r}</option>`).join("");
    countrySel.innerHTML = '<option value="">All countries</option>' + countries.map(c => {
      const row = summaries.find(s => s.country_id === c);
      return `<option value="${c}">${row?.country_name || c}</option>`;
    }).join("");
    const runs = summaries;
    const runSel = this.root.querySelector("#capitalsRunSelect");
    runSel.innerHTML = runs.map(row => {
      const label = `${row.country_name || row.country_id} · ${row.instruction_variant} · p${row.prefix_length}`;
      return `<option value="${row.tree_key}">${label}</option>`;
    }).join("");
  }

  renderSummary() {
    const rows = this.filteredSummaries();
    this.summary.renderTable(rows, treeKey => this.openTree(treeKey));
    this.summary.renderStats([
      { value: this.store.manifest.mech_interp_trees || 0, label: "τ-trees" },
      { value: this.store.manifest.bad_nodes_trees || 0, label: "analyzed" },
      { value: rows.length, label: "filtered" },
    ]);
  }

  async openTree(treeKey) {
    switchSubTab(this.root, "trees");
    const runSel = this.root.querySelector("#capitalsRunSelect");
    if (runSel) runSel.value = treeKey;
    await this.loadRun(treeKey);
  }

  async loadRun(treeKey) {
    const summary = this.store.summaryFor(treeKey);
    this.currentSummary = summary;
    const meta = this.root.querySelector("#capitalsRunMeta");
    if (meta) meta.textContent = "Loading tree…";
    const shard = await this.store.loadTree(treeKey);
    const nodes = shard.tree_nodes || [];
    const leafMap = new Map(Object.entries(shard.leaf_completions || {}));
    const statusMap = new Map(Object.entries(shard.node_status || {}));
    const statsMap = new Map(Object.entries(shard.node_stats || {}));
    this.graph.setTreeData({ nodes, leafMap, statusMap, statsMap });
    const minProb = parseFloat(this.root.querySelector("#capitalsProbFilter")?.value || "0");
    this.graph.setOptions({
      showBad: this.root.querySelector("#capitalsShowBad")?.checked !== false,
      showGood: this.root.querySelector("#capitalsShowGood")?.checked !== false,
    });
    await this.graph.draw(nodes, minProb);
    if (meta && summary) {
      meta.textContent = `${summary.country_name || summary.country_id} · ${summary.instruction_variant} · prefix ${summary.prefix_length}`;
    }
    renderDetail(this.root.querySelector("#capitalsDetail"), {});
  }

  showNode(nodeId) {
    const nodes = this.graph.currentNodes;
    const node = nodes.find(n => n.id === nodeId);
    const leaf = this.graph.leafMap.get(nodeId);
    const stats = this.graph.statsMap.get(nodeId);
    const status = this.graph.statusMap.get(nodeId);
    renderDetail(this.root.querySelector("#capitalsDetail"), { node, leaf, stats, status });
  }

  bindEvents() {
    this.root.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => switchSubTab(this.root, btn.dataset.tab));
    });
    ["capitalsRegionFilter", "capitalsCountryFilter"].forEach(id => {
      this.root.querySelector(`#${id}`)?.addEventListener("change", () => this.renderSummary());
    });
    this.root.querySelector("#capitalsRunSelect")?.addEventListener("change", e => this.loadRun(e.target.value));
    this.root.querySelector("#capitalsProbFilter")?.addEventListener("change", () => {
      if (this.currentSummary) this.loadRun(this.currentSummary.tree_key);
    });
    this.root.querySelector("#capitalsFitBtn")?.addEventListener("click", () => this.graph.fitToScreen());
    this.root.querySelector("#capitalsResetBtn")?.addEventListener("click", () => this.graph.resetZoom());
    ["capitalsShowBad", "capitalsShowGood"].forEach(id => {
      this.root.querySelector(`#${id}`)?.addEventListener("change", () => {
        if (this.currentSummary) this.loadRun(this.currentSummary.tree_key);
      });
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
