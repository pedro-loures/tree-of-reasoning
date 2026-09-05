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

TreeDashboard.treeLegendHtml = function(extraItems = "") {
  return `
    <span><i style="background:var(--bad-node)"></i> 100% bad</span>
    <span><i style="background:var(--mostly-bad)"></i> &gt;75% bad</span>
    <span><i style="background:var(--good-node)"></i> 100% good</span>
    <span><i style="background:var(--internal)"></i> mixed</span>
    ${extraItems}`;
};

TreeDashboard.populateTauProbFilter = function(selectEl, taus, { selected } = {}) {
  if (!selectEl) return;
  const values = Array.isArray(taus) ? [...taus] : [];
  values.sort((a, b) => Number(a) - Number(b));
  const options = [{ value: "0", label: "0 (show all)" }];
  for (const tau of values) {
    options.push({ value: String(tau), label: `${tau} (τ)` });
  }
  selectEl.innerHTML = options.map(opt =>
    `<option value="${opt.value}">${opt.label}</option>`).join("");
  if (selected != null && options.some(opt => opt.value === String(selected))) {
    selectEl.value = String(selected);
  } else if (values.length) {
    selectEl.value = String(values[values.length - 1]);
  }
};

TreeDashboard.INTERACTIVE_DEFAULT_TAU = 0.01;

TreeDashboard.interactiveDefaultTau = function(taus) {
  const sorted = [...(taus || [])].map(Number).filter(n => !Number.isNaN(n)).sort((a, b) => b - a);
  if (!sorted.length) return TreeDashboard.INTERACTIVE_DEFAULT_TAU;
  return sorted.find(t => t <= TreeDashboard.INTERACTIVE_DEFAULT_TAU) ?? sorted[0];
};

TreeDashboard.interactiveTauIsSlow = function(tau) {
  const n = Number(tau);
  return !Number.isNaN(n) && n > 0 && n < TreeDashboard.INTERACTIVE_DEFAULT_TAU;
};

TreeDashboard.interactiveTauMatches = function(a, b) {
  if (a == null || b == null || b === "") return true;
  return Math.abs(Number(a) - Number(b)) < 1e-9;
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

  summaryFor(treeKey) {
    return (this.DATA?.tree_summaries || []).find(row => row.tree_key === treeKey) || null;
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
  document.dispatchEvent(new CustomEvent("experiment-tab", { detail: { experiment: name } }));
};

TreeDashboard.bindSubTabs = function(adapter, root) {
  root.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      TreeDashboard.switchSubTab(root, btn.dataset.tab);
      if (btn.dataset.tab === "trees" && adapter.ensureTreeLoaded) {
        await adapter.ensureTreeLoaded();
      }
    });
  });
};

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".top-tab-btn[data-experiment]").forEach(btn => {
    btn.addEventListener("click", () => TreeDashboard.switchTopTab(btn.dataset.experiment));
  });
  const active = document.querySelector(".top-tab-btn.active[data-experiment]");
  if (active) TreeDashboard.switchTopTab(active.dataset.experiment);
});
