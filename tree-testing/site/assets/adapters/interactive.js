window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { DataStore, TreeGraph, SummaryPanel, renderDetail, switchSubTab, utils } = TD;

const INTERACTIVE_COLUMNS = [
  { key: "prompt_preview", label: "Prompt" },
  { key: "model_id", label: "Model" },
  { key: "tau", label: "τ" },
  { key: "total_nodes", label: "Nodes" },
  { key: "leaf_count", label: "τ-leaves" },
  { key: "exclusively_bad_count", label: "Excl. bad" },
  { key: "leaf_correct_pct", label: "Leaf good %" },
  { key: "prob_good_pct", label: "P(good) %" },
];

TD.InteractiveAdapter = class {
  constructor(root) {
    this.root = root;
    this.store = new DataStore("interactive", "data/interactive");
    this.graph = new TreeGraph("#interactiveGraph", "#interactiveTooltip");
    this.summary = new SummaryPanel(root, INTERACTIVE_COLUMNS);
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

  populateMockForm(tutorial) {
    if (!tutorial) return;
    this.root.querySelector("#interactivePrompt").value = tutorial.prompt || "";
    this.root.querySelector("#interactiveExpected").value = tutorial.expected_answers || "";
    const modelSel = this.root.querySelector("#interactiveModel");
    modelSel.innerHTML = `<option>${tutorial.model_id || "deepseek-r1-7b"}</option>`;
    const tauSel = this.root.querySelector("#interactiveTau");
    tauSel.innerHTML = `<option selected>${tutorial.tau ?? 0.01}</option>`;
  }

  async init() {
    this.enableViewOnlyMock();
    try {
      await this.store.loadManifest();
      document.getElementById("interactiveSource").textContent =
        `View-only demo · ${this.store.manifest.shards?.length || 0} saved trees · generated ${this.store.manifest.generated_at}`;
      const tutorialResp = await fetch("data/interactive/tutorial.json?ts=" + Date.now());
      if (tutorialResp.ok) this.tutorial = await tutorialResp.json();
      if (this.tutorial) this.populateMockForm(this.tutorial);
      this.populateRunSelect();
      this.renderSummary();
      this.bindEvents();
    } catch (err) {
      document.getElementById("interactiveSource").innerHTML = `<span class="error">${utils.escapeHtml(err.message)}</span>`;
    }
  }

  populateRunSelect() {
    const runSel = this.root.querySelector("#interactiveRunSelect");
    const summaries = this.store.DATA.tree_summaries || [];
    runSel.innerHTML = summaries.map(row => {
      return `<option value="${row.tree_key}">${row.prompt_preview || row.tree_key}</option>`;
    }).join("");
  }

  renderSummary() {
    const rows = this.store.DATA.tree_summaries || [];
    this.summary.renderTable(rows, treeKey => this.openTree(treeKey));
    this.summary.renderStats([
      { value: rows.length, label: "saved trees" },
      { value: "view only", label: "mode" },
    ]);
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
    });
    await this.graph.draw(nodes, minProb);
    if (meta && summary) {
      meta.textContent = `${summary.prompt || summary.prompt_preview || ""} · τ=${summary.tau} · ${summary.model_id}`;
    }
    renderDetail(this.root.querySelector("#interactiveDetail"), {});
  }

  showNode(nodeId) {
    const node = this.graph.currentNodes.find(n => n.id === nodeId);
    const leaf = this.graph.leafMap.get(nodeId);
    const stats = this.graph.statsMap.get(nodeId);
    const status = this.graph.statusMap.get(nodeId);
    renderDetail(this.root.querySelector("#interactiveDetail"), { node, leaf, stats, status });
  }

  bindEvents() {
    this.root.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => switchSubTab(this.root, btn.dataset.tab));
    });
    this.root.querySelector("#interactiveRunSelect")?.addEventListener("change", e => this.loadRun(e.target.value));
    this.root.querySelector("#interactiveProbFilter")?.addEventListener("change", () => {
      if (this.currentSummary) this.loadRun(this.currentSummary.tree_key);
    });
    this.root.querySelector("#interactiveFitBtn")?.addEventListener("click", () => this.graph.fitToScreen());
    this.root.querySelector("#interactiveResetBtn")?.addEventListener("click", () => this.graph.resetZoom());
    ["interactiveShowBad", "interactiveShowGood"].forEach(id => {
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

// Auto-load interactive tab on first visit (default demo)
document.addEventListener("DOMContentLoaded", () => {
  TD.switchTopTab("interactive");
});
})(TreeDashboard);
