window.TreeDashboard = window.TreeDashboard || {};

TreeDashboard.utils = {
  escapeHtml(text) {
    return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  },
  fmtVal(v) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") return Number.isInteger(v) ? v : Math.round(v * 1000) / 1000;
    return v;
  },
  fmtProb(p) {
    if (p >= 0.0001) return p.toFixed(4);
    return p.toExponential(2);
  },
  displayToken(token) {
    if (!token) return "⟨prompt⟩";
    if (token === "\n") return "\\n";
    if (token === "\t") return "\\t";
    if (token === " ") return "·";
    return token;
  },
};

TreeDashboard.DataStore = class {
  constructor(experiment, baseUrl) {
    this.experiment = experiment;
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.manifest = null;
    this.cache = new Map();
    this.DATA = null;
  }

  async loadManifest() {
    const response = await fetch(`${this.baseUrl}/manifest.json?ts=${Date.now()}`);
    if (!response.ok) throw new Error(`${this.experiment} manifest: ${response.statusText}`);
    this.manifest = await response.json();
    this.DATA = {
      ...this.manifest,
      trees: {},
      node_status: {},
      node_stats: {},
      leaf_completions: {},
      node_expansions: {},
      candidate_mention_probs: {},
    };
    return this.DATA;
  }

  shardFile(treeKey) {
    const entry = (this.manifest?.shards || []).find(item => item.tree_key === treeKey);
    return entry?.file || null;
  }

  async loadTree(treeKey) {
    if (this.cache.has(treeKey)) return this.cache.get(treeKey);
    const file = this.shardFile(treeKey);
    if (!file) throw new Error(`No shard for ${treeKey}`);
    const response = await fetch(`${this.baseUrl}/${file}?ts=${Date.now()}`);
    if (!response.ok) throw new Error(`Shard ${file}: ${response.statusText}`);
    const shard = await response.json();
    this.cache.set(treeKey, shard);
    this.DATA.trees[treeKey] = shard.tree_nodes;
    this.DATA.node_status[treeKey] = shard.node_status || {};
    this.DATA.node_stats[treeKey] = shard.node_stats || {};
    this.DATA.leaf_completions[treeKey] = shard.leaf_completions || {};
    if (shard.node_expansions) this.DATA.node_expansions[treeKey] = shard.node_expansions;
    if (shard.candidate_mention_probs) this.DATA.candidate_mention_probs[treeKey] = shard.candidate_mention_probs;
    return shard;
  }
};

TreeDashboard.switchSubTab = function(panelRoot, name) {
  panelRoot.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  panelRoot.querySelectorAll(".panel").forEach(panel => {
    panel.classList.toggle("active", panel.id === `${panelRoot.id}_${name}Panel`);
  });
};

TreeDashboard.switchTopTab = function(name) {
  document.querySelectorAll(".top-tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.experiment === name);
  });
  document.querySelectorAll(".experiment-panel").forEach(panel => {
    panel.classList.toggle("active", panel.id === `${name}Panel`);
  });
  window.dispatchEvent(new CustomEvent("experiment-tab", { detail: { experiment: name } }));
};

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".top-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => TreeDashboard.switchTopTab(btn.dataset.experiment));
  });
});
