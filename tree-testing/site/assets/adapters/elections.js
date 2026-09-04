window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { DataStore, TreeGraph, SummaryPanel, renderDetail, switchSubTab, utils } = TD;

const ELECTIONS_COLUMNS = [
  { key: "model_id", label: "Model" },
  { key: "instruction_variant", label: "Prompt" },
  { key: "total_nodes", label: "Nodes" },
  { key: "leaf_count", label: "τ-leaves" },
  { key: "exclusively_bad_count", label: "Excl. bad" },
  { key: "prob_good_pct", label: "P(good) %" },
  { key: "prob_bad_pct", label: "P(bad) %" },
  { key: "greedy_mention_category", label: "Greedy mention" },
];

TD.ElectionsAdapter = class {
  constructor(root) {
    this.root = root;
    this.store = new DataStore("elections", "data/elections");
    this.graph = new TreeGraph("#electionsGraph", "#electionsTooltip");
    this.summary = new SummaryPanel(root, ELECTIONS_COLUMNS);
    this.currentSummary = null;
    this.graph.onSelect = nodeId => this.showNode(nodeId);
  }

  async init() {
    try {
      await this.store.loadManifest();
      this.populateRunSelect();
      this.renderSummary();
      this.bindEvents();
      document.getElementById("electionsSource").textContent =
        `Brazilian president experiment · ${this.store.manifest.viewer_runs || 0} trees · generated ${this.store.manifest.generated_at}`;
    } catch (err) {
      document.getElementById("electionsSource").innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }

  populateRunSelect() {
    const runSel = this.root.querySelector("#electionsRunSelect");
    const summaries = this.store.DATA.tree_summaries || [];
    runSel.innerHTML = summaries.map(row => {
      return `<option value="${row.tree_key}">${row.instruction_variant || row.model_id}</option>`;
    }).join("");
  }

  renderSummary() {
    const rows = this.store.DATA.tree_summaries || [];
    this.summary.renderTable(rows, treeKey => this.openTree(treeKey));
    this.summary.renderStats([
      { value: this.store.manifest.mech_interp_trees || 0, label: "τ-trees" },
      { value: this.store.manifest.viewer_runs || 0, label: "with bad-nodes" },
    ]);
  }

  async openTree(treeKey) {
    switchSubTab(this.root, "trees");
    this.root.querySelector("#electionsRunSelect").value = treeKey;
    await this.loadRun(treeKey);
  }

  async loadRun(treeKey) {
    const summary = this.store.summaryFor(treeKey);
    this.currentSummary = summary;
    const meta = this.root.querySelector("#electionsRunMeta");
    if (meta) meta.textContent = "Loading tree…";
    const shard = await this.store.loadTree(treeKey);
    const nodes = shard.tree_nodes || [];
    const leafMap = new Map(Object.entries(shard.leaf_completions || {}));
    const statusMap = new Map(Object.entries(shard.node_status || {}));
    const statsMap = new Map(Object.entries(shard.node_stats || {}));
    this.graph.setTreeData({ nodes, leafMap, statusMap, statsMap });
    const minProb = parseFloat(this.root.querySelector("#electionsProbFilter")?.value || "0");
    await this.graph.draw(nodes, minProb);
    if (meta) meta.textContent = summary?.instruction || summary?.instruction_variant || treeKey;
    const cand = this.root.querySelector("#electionsCandidates");
    if (cand && shard.candidate_mention_probs) {
      cand.innerHTML = shard.candidate_mention_probs.map(item => `
        <div class="muted">${utils.escapeHtml(item.ballot_name || item.full_name || item.id)}: ${utils.fmtVal(Math.round(item.prob * 10000) / 10000)}</div>
      `).join("");
    }
    renderDetail(this.root.querySelector("#electionsDetail"), {});
  }

  showNode(nodeId) {
    const node = this.graph.currentNodes.find(n => n.id === nodeId);
    const leaf = this.graph.leafMap.get(nodeId);
    const stats = this.graph.statsMap.get(nodeId);
    const status = this.graph.statusMap.get(nodeId);
    renderDetail(this.root.querySelector("#electionsDetail"), { node, leaf, stats, status });
  }

  bindEvents() {
    this.root.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => switchSubTab(this.root, btn.dataset.tab));
    });
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
