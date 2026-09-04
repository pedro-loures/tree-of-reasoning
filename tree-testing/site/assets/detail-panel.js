window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { utils } = TD;

TD.CapitalsBadFilters = {
  SUBTYPES: [
    { id: "wrong_city", label: "wrong city" },
    { id: "empty", label: "no answer" },
    { id: "incomplete", label: "incomplete" },
    { id: "lorem_drift", label: "lorem drift" },
  ],

  activeSubtypes(root) {
    return this.SUBTYPES
      .filter(item => root.querySelector(`#badFilter_${item.id}`)?.checked)
      .map(item => item.id);
  },

  classifySubtypes(leaf) {
    if (!leaf) return ["empty"];
    if (leaf.answer_correct) return ["correct"];
    const subtypes = [];
    if (!leaf.reasoning_complete) subtypes.push("incomplete");
    if (!leaf.answer_text || !String(leaf.answer_text).trim()) subtypes.push("empty");
    if (leaf.mentions_lorem) subtypes.push("lorem_drift");
    if (leaf.reasoning_complete && leaf.answer_text && String(leaf.answer_text).trim()) {
      subtypes.push("wrong_city");
    }
    return subtypes;
  },

  isLeafBad(leaf, root) {
    const subtypes = this.classifySubtypes(leaf);
    if (subtypes.includes("correct")) return false;
    const active = this.activeSubtypes(root);
    if (!active.length) return false;
    return subtypes.some(s => active.includes(s));
  },

  isLeafCorrect(leaf) {
    return this.classifySubtypes(leaf).includes("correct");
  },

  computeNodeStats(nodes, leafMap, root) {
    const stats = new Map();
    const leafDesc = (nodeId) => {
      const byId = new Map(nodes.map(n => [n.id, n]));
      const walk = id => {
        const node = byId.get(id);
        const children = node?.c || node?.child_ids || [];
        if (!children.length) return [id];
        return children.flatMap(walk);
      };
      return walk(nodeId);
    };
    for (const node of nodes) {
      const leafIds = leafDesc(node.id);
      const leaves = leafIds.map(id => leafMap.get(id)).filter(Boolean);
      if (!leaves.length) continue;
      const nBad = leaves.filter(l => this.isLeafBad(l, root)).length;
      const nGood = leaves.filter(l => this.isLeafCorrect(l)).length;
      const nLeaves = leaves.length;
      const badPct = nBad / nLeaves;
      let colorClass = "mixed";
      if (nBad === nLeaves) colorClass = "exclusively_bad";
      else if (nGood === nLeaves) colorClass = "exclusively_good";
      else if (nBad / nLeaves > 0.75) colorClass = "mostly_bad";
      stats.set(node.id, {
        n_leaves: nLeaves, n_bad: nBad, n_good: nGood,
        bad_pct: badPct, bad_pct_display: Math.round(badPct * 1000) / 10,
        color_class: colorClass,
      });
    }
    return stats;
  },
};

TD.renderDetailPanel = function(container, ctx) {
  if (!container) return;
  const { node, leaf, stats, status, leaves, correctness, showReasoning } = ctx;
  if (!node && !leaf && !leaves?.length) {
    container.innerHTML = '<p class="muted">Click a node in the graph or a summary row.</p>';
    return;
  }
  let html = "";
  if (correctness) {
    html += `<div class="muted" style="margin-bottom:8px">Correct if: ${utils.escapeHtml(correctness)}</div>`;
  }
  if (node) {
    const cls = stats?.color_class === "exclusively_good" ? "exclusively-good"
      : stats?.color_class === "mostly_bad" ? "mostly-bad" : "";
    html += `<div class="card node-card ${cls}"><h3>Node ${utils.escapeHtml(node.id)}</h3>`;
    html += `<div class="muted">token: ${utils.escapeHtml(utils.displayToken(node.t || node.token))}</div>`;
    if (node.p != null) html += `<div class="muted">p=${utils.fmtProb(node.p)}</div>`;
    if (stats) html += `<div><span class="pill ${stats.bad_pct >= 1 ? "bad-node" : "ditched"}">${utils.fmtVal(stats.bad_pct_display ?? stats.bad_pct * 100)}% bad</span></div>`;
    if (status) html += `<div><span class="pill ditched">${utils.escapeHtml(status)}</span></div>`;
    html += "</div>";
  }
  const renderLeaf = (lf, title) => {
    if (!lf) return "";
    const good = lf.answer_correct;
    const cls = good ? "correct" : lf.reasoning_complete ? "incorrect" : "neutral";
    let block = `<div class="card leaf-card ${cls}"><h3>${title || "Leaf"}</h3>`;
    if (lf.mention_category) block += `<div><span class="pill ditched">${utils.escapeHtml(lf.mention_category)}</span></div>`;
    block += `<div><span class="pill ${good ? "yes" : "no"}">${good ? "good" : "bad"}</span></div>`;
    block += `<div class="answer">${utils.escapeHtml(lf.answer_text || "")}</div>`;
    if (lf.mentions?.length) {
      block += lf.mentions.map(m => `<span class="pill ditched">${utils.escapeHtml(m.ballot_name || m.full_name || m.id)}</span>`).join(" ");
    }
    if (showReasoning && lf.completion_preview) {
      block += `<pre class="reasoning-block">${utils.escapeHtml(lf.completion_preview)}</pre>`;
    }
    block += "</div>";
    return block;
  };
  if (leaf) html += renderLeaf(leaf, "Selected leaf");
  if (leaves?.length) {
    const sorted = [...leaves].sort((a, b) => (b.path_prob || 0) - (a.path_prob || 0));
    html += sorted.slice(0, 12).map((lf, i) => renderLeaf(lf, `Leaf ${lf.leaf_id || i + 1} · p=${utils.fmtProb(lf.path_prob || 0)}`)).join("");
    if (sorted.length > 12) html += `<p class="muted">+ ${sorted.length - 12} more leaves</p>`;
  }
  container.innerHTML = html;
};
})(TreeDashboard);
