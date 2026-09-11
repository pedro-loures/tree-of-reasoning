window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { utils } = TD;

const LOREM_MARKERS = ["lorem", "ipsum", "dolor", "amet", "consectetur", "adipiscing"];

TD.mentionsLorem = function(text) {
  if (!text) return false;
  const lowered = String(text).toLowerCase();
  return LOREM_MARKERS.some(marker => lowered.includes(marker));
};

function reasoningPreviewText(leaf) {
  const preview = leaf?.completion_preview || "";
  const answer = (leaf?.answer_text || "").trim();
  if (!preview) return "";
  if (answer && preview.endsWith(answer)) return preview.slice(0, preview.length - answer.length);
  if (answer) {
    const idx = preview.lastIndexOf(answer);
    if (idx >= 0) return preview.slice(0, idx);
  }
  return preview;
}

function classifyPlainSubtypes(leaf) {
  if (!leaf) return ["empty"];
  if (leaf.answer_correct) return ["correct"];
  const subtypes = [];
  if (!leaf.reasoning_complete) subtypes.push("incomplete");
  if (!leaf.answer_text || !String(leaf.answer_text).trim()) subtypes.push("empty");
  if (leaf.mentions_lorem || TD.mentionsLorem(leaf.completion_preview)) subtypes.push("lorem_drift");
  if (leaf.reasoning_complete && leaf.answer_text && String(leaf.answer_text).trim()) {
    subtypes.push("wrong_city");
  }
  return subtypes;
}

function classifyIgnoreLoremSubtypes(leaf) {
  if (!leaf) return ["empty"];
  if (leaf.answer_correct) return ["correct"];
  const subtypes = [];
  if (!leaf.reasoning_complete) subtypes.push("incomplete");
  if (!leaf.answer_text || !String(leaf.answer_text).trim()) subtypes.push("empty");
  if (TD.mentionsLorem(reasoningPreviewText(leaf))) subtypes.push("lorem_reasoning");
  if (TD.mentionsLorem(leaf.answer_text)) subtypes.push("lorem_answer");
  if (leaf.reasoning_complete && leaf.answer_text && String(leaf.answer_text).trim()) {
    subtypes.push("wrong_city");
  }
  return subtypes;
}

function buildBadFilters(mode, subtypes, classifySubtypes) {
  return {
    SUBTYPES: subtypes,
    mode,
    activeSubtypes(root) {
      return subtypes
        .filter(item => root.querySelector(`#badFilter_${item.id}`)?.checked)
        .map(item => item.id);
    },
    classifySubtypes,
    isLeafBad(leaf, root) {
      const tags = classifySubtypes(leaf);
      if (tags.includes("correct")) return false;
      const active = this.activeSubtypes(root);
      if (!active.length) return false;
      return tags.some(tag => active.includes(tag));
    },
    isLeafCorrect(leaf) {
      return classifySubtypes(leaf).includes("correct");
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
}

TD.createCapitalsBadFilters = function(mode = "plain") {
  if (mode === "ignore-lorem") {
    return buildBadFilters(mode, [
      { id: "wrong_city", label: "wrong city" },
      { id: "empty", label: "no answer" },
      { id: "incomplete", label: "incomplete" },
      { id: "lorem_reasoning", label: "mention lorem in reasoning" },
      { id: "lorem_answer", label: "mention lorem in answer" },
    ], classifyIgnoreLoremSubtypes);
  }
  return buildBadFilters(mode, [
    { id: "wrong_city", label: "wrong city" },
    { id: "empty", label: "no answer" },
    { id: "incomplete", label: "incomplete" },
    { id: "lorem_drift", label: "lorem drift" },
  ], classifyPlainSubtypes);
};

TD.CapitalsBadFilters = TD.createCapitalsBadFilters("plain");

function buildParentMap(nodes) {
  const parentOf = new Map();
  for (const n of nodes || []) {
    for (const childId of n.c || n.child_ids || []) parentOf.set(childId, n.id);
  }
  return parentOf;
}

function tokenPathToNode(nodes, nodeId) {
  const byId = new Map((nodes || []).map(n => [n.id, n]));
  const parentOf = buildParentMap(nodes);
  const parts = [];
  let cur = nodeId;
  while (cur && cur !== "root") {
    const n = byId.get(cur);
    if (n) parts.unshift(n.t || n.tok || n.token || "");
    cur = parentOf.get(cur);
  }
  return parts.join("");
}

function buildIncomingTokenMap(nodes) {
  const incoming = new Map();
  for (const n of nodes || []) {
    const children = n.c || n.child_ids || [];
    const tokens = n.ct || n.child_tokens || [];
    children.forEach((childId, index) => {
      if (tokens[index] != null) incoming.set(childId, tokens[index]);
    });
  }
  return incoming;
}

function nodeToken(node, nodes) {
  if (!node || node.id === "root") return null;
  const direct = node.t ?? node.token ?? node.tok;
  if (direct != null && direct !== "") return direct;
  return buildIncomingTokenMap(nodes).get(node.id) ?? null;
}

TD.nodeReasoningText = function(node, leaf, nodes) {
  const parts = [];
  if (node?.suffix) parts.push(node.suffix);
  else if (node && nodes?.length) parts.push(tokenPathToNode(nodes, node.id));
  if (leaf) {
    const extra = leaf.completion_text || leaf.completion_preview || "";
    if (extra) parts.push(extra);
  }
  return parts.join("");
};

TD.renderDetailPanel = function(container, ctx) {
  if (!container) return;
  const { node, leaf, stats, status, leaves, correctness, nodes } = ctx;
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
    html += `<div class="muted mono">${utils.escapeHtml(utils.displayToken(nodeToken(node, nodes)))}`;
    if (node.p != null) html += ` · p=${Number(node.p).toFixed(2)}`;
    if (node.d != null) html += ` · depth ${node.d}`;
    html += "</div>";
    if (stats) {
      html += `<div style="margin-top:6px"><span class="pill ${stats.bad_pct >= 1 ? "bad-node" : "ditched"}">${utils.fmtVal(stats.bad_pct_display ?? stats.bad_pct * 100)}% bad</span></div>`;
    }
    if (status) html += `<div><span class="pill ditched">${utils.escapeHtml(status)}</span></div>`;
    const reasoning = TD.nodeReasoningText(node, leaf, nodes);
    if (reasoning) {
      html += `<h4 class="reasoning-label">Chain of thought (up to this node)</h4>`;
      html += `<pre class="reasoning-block reasoning-block-full">${utils.escapeHtml(reasoning)}</pre>`;
    } else if (node.id !== "root") {
      html += '<p class="muted">No reasoning text stored for this node.</p>';
    } else {
      html += '<p class="muted">Root — click a child node to view reasoning along that path.</p>';
    }
    if (leaf?.answer_text != null && String(leaf.answer_text).trim()) {
      html += `<h4 class="reasoning-label">Answer</h4><div class="answer">${utils.escapeHtml(leaf.answer_text)}</div>`;
    } else if (leaf) {
      html += '<h4 class="reasoning-label">Answer</h4><div class="answer muted"><em>no answer</em></div>';
    }
    if (leaf?.answer_matches && Object.keys(leaf.answer_matches).length) {
      html += `<div style="margin-top:8px">${Object.entries(leaf.answer_matches).map(([term, hit]) =>
        `<span class="pill ${hit ? "yes" : "no"}">${utils.escapeHtml(term)}: ${hit ? "yes" : "no"}</span>`
      ).join(" ")}</div>`;
    }
    html += "</div>";
  }
  const renderLeaf = (lf, title, compact = false) => {
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
    if (!compact) {
      const leafNode = (nodes || []).find(n => n.id === lf.leaf_id);
      const reasoning = TD.nodeReasoningText(leafNode, lf, nodes);
      if (reasoning) block += `<pre class="reasoning-block reasoning-block-full">${utils.escapeHtml(reasoning)}</pre>`;
    }
    block += "</div>";
    return block;
  };
  if (leaves?.length) {
    const sorted = [...leaves].sort((a, b) => (b.path_prob || 0) - (a.path_prob || 0));
    const descendants = leaf ? sorted.filter(lf => lf.leaf_id !== node?.id) : sorted;
    if (descendants.length) {
      html += `<h4 class="reasoning-label" style="margin:12px 0 8px">Descendant leaves (${descendants.length})</h4>`;
      html += descendants.slice(0, 12).map((lf, i) =>
        renderLeaf(lf, `Leaf ${lf.leaf_id || i + 1} · p=${Number(lf.path_prob || 0).toFixed(2)}`, true)
      ).join("");
      if (descendants.length > 12) html += `<p class="muted">+ ${descendants.length - 12} more leaves</p>`;
    }
  }
  container.innerHTML = html;
};
})(TreeDashboard);
