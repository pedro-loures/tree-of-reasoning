const VIEW_ONLY = window.DASHBOARD_MODE === "pages";
const MODEL_LABELS = {"deepseek-r1-7b":"DeepSeek-R1-7B","qwq-32b-awq":"QwQ-32B-AWQ"};

const TABLE_COLUMNS = [
  { key: "prompt_preview", label: "Prompt", align: "left" },
  { key: "model_id", label: "Model", align: "left" },
  { key: "tau", label: "τ", chartable: true },
  { key: "total_nodes", label: "Nodes", chartable: true },
  { key: "leaf_count", label: "τ-leaves", chartable: true },
  { key: "mass_above_tau", label: "Mass above τ", chartable: true },
  { key: "total_candidates", label: "Candidates", chartable: true },
  { key: "exclusively_bad_count", label: "Excl. bad", chartable: true },
  { key: "exclusively_bad_pct", label: "% bad cand.", chartable: true },
  { key: "leaf_correct_pct", label: "Leaf good %", chartable: true },
  { key: "prob_good_pct", label: "P(good) %", chartable: true, highlight: true },
  { key: "prob_bad_pct", label: "P(bad) %", chartable: true, highlight: true },
];

let DATA = { runs: [], trees: {}, tree_summaries: [], models: [], taus: [] };
let currentRun = null;
let currentNodes = [];
let leafMap = new Map();
let statusMap = new Map();
let statsMap = new Map();
let nodeById = new Map();
let zoomBehavior = null;
let svg = null;
let gRoot = null;
let selectedNodeId = null;
let sortKey = "prompt_preview";
let sortAsc = true;

function fmtVal(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? v : Math.round(v * 1000) / 1000;
  return v;
}

function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach(panel => {
    panel.classList.toggle("active", panel.id === name + "Panel");
  });
  if (name === "trees" && currentRun) {
    setTimeout(() => loadRun(currentRun.tree_key), 50);
  }
}

function filteredSummaries() {
  const model = document.getElementById("modelFilter")?.value || "";
  const tau = document.getElementById("tauFilter")?.value || "";
  return (DATA.tree_summaries || []).filter(row => {
    if (model && row.model_id !== model) return false;
    if (tau && String(row.tau) !== tau) return false;
    return true;
  });
}

function chartRows() {
  return filteredSummaries().map(row => ({
    ...row,
    label: row.prompt_preview,
    model_label: MODEL_LABELS[row.model_id] || row.model_id,
  }));
}

function updateFilterHint() {
  const rows = filteredSummaries();
  document.getElementById("filterHint").textContent = `${rows.length} tree${rows.length === 1 ? "" : "s"} · click a row to open in Tree Explorer`;
}

function renderGlobalStats() {
  const filtered = filteredSummaries();
  const totalBad = filtered.reduce((s, r) => s + (r.exclusively_bad_count || 0), 0);
  const totalMassGood = filtered.reduce((s, r) => s + (r.prob_good || 0), 0);
  const totalMassBad = filtered.reduce((s, r) => s + (r.prob_bad || 0), 0);
  const totalMass = totalMassGood + totalMassBad;
  document.getElementById("globalStats").innerHTML = `
    <div class="stat"><strong>${filtered.length}</strong><span class="muted">trees</span></div>
    <div class="stat"><strong>${totalBad}</strong><span class="muted">excl. bad nodes</span></div>
    <div class="stat"><strong>${totalMass ? Math.round(1000 * totalMassGood / totalMass) / 10 : "—"}%</strong><span class="muted">P(good) mass</span></div>
    <div class="stat"><strong>${totalMass ? Math.round(1000 * totalMassBad / totalMass) / 10 : "—"}%</strong><span class="muted">P(bad) mass</span></div>
  `;
}

function populateFilters() {
  const modelSel = document.getElementById("modelFilter");
  modelSel.innerHTML = '<option value="">All models</option>' +
    (DATA.models || []).map(mid => `<option value="${mid}">${MODEL_LABELS[mid] || mid}</option>`).join("");

  const tauSel = document.getElementById("tauFilter");
  tauSel.innerHTML = '<option value="">All τ</option>' +
    (DATA.taus || []).map(t => `<option value="${t}">${t}</option>`).join("");

  const chartable = TABLE_COLUMNS.filter(col => col.chartable);
  const metricSel = document.getElementById("metricSelect");
  metricSel.innerHTML = chartable.map(col => `<option value="${col.key}">${col.label}</option>`).join("");
  metricSel.value = "exclusively_bad_pct";

  const scatterOpts = chartable.map(col => `<option value="${col.key}">${col.label}</option>`).join("");
  document.getElementById("scatterX").innerHTML = scatterOpts;
  document.getElementById("scatterY").innerHTML = scatterOpts;
  document.getElementById("scatterX").value = "prob_good_pct";
  document.getElementById("scatterY").value = "prob_bad_pct";
}

function sortRows(rows) {
  return rows.sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av === bv) return 0;
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    if (typeof av === "string") return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortAsc ? av - bv : bv - av;
  });
}

function renderSummaryTable(rows) {
  const sorted = sortRows([...rows]);
  const thead = document.querySelector("#summaryTable thead");
  const tbody = document.querySelector("#summaryTable tbody");
  thead.innerHTML = "<tr>" + TABLE_COLUMNS.map(col => {
    const arrow = sortKey === col.key ? (sortAsc ? " ▲" : " ▼") : "";
    const cls = col.highlight ? ' class="highlight"' : "";
    return `<th data-key="${col.key}"${cls} style="text-align:${col.align || "right"}">${col.label}${arrow}</th>`;
  }).join("") + "</tr>";
  tbody.innerHTML = sorted.map(row => {
    const cells = TABLE_COLUMNS.map(col => {
      let val = row[col.key];
      if (col.key === "model_id") val = MODEL_LABELS[val] || val;
      return `<td data-key="${col.key}" class="${col.highlight ? "highlight" : ""}" style="text-align:${col.align || "right"}">${fmtVal(val)}</td>`;
    }).join("");
    return `<tr class="clickable" data-tree-key="${row.tree_key}">${cells}</tr>`;
  }).join("");
  thead.querySelectorAll("th").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (sortKey === key) sortAsc = !sortAsc;
      else { sortKey = key; sortAsc = key === "prompt_preview"; }
      renderSummary();
    });
  });
  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => {
      const treeKey = tr.dataset.treeKey;
      document.getElementById("runSelect").value = treeKey;
      loadRun(treeKey);
      switchTab("trees");
    });
  });
}

function renderBarChart(rows, metricKey) {
  const svgEl = d3.select("#barChart");
  svgEl.selectAll("*").remove();
  const width = svgEl.node().clientWidth || 600;
  const height = 280;
  const margin = { top: 12, right: 12, bottom: 80, left: 48 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const g = svgEl.attr("viewBox", `0 0 ${width} ${height}`).append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);
  const data = rows.filter(r => r[metricKey] !== null && r[metricKey] !== undefined);
  if (!data.length) {
    document.getElementById("barChartTitle").textContent = "Generate a tree to see charts";
    return;
  }
  const x = d3.scaleBand().domain(data.map(d => d.label)).range([0, innerW]).padding(0.2);
  const minVal = Math.min(0, d3.min(data, d => d[metricKey]) || 0);
  const maxVal = d3.max(data, d => d[metricKey]) || 1;
  const y = d3.scaleLinear().domain([minVal, maxVal]).nice().range([innerH, 0]);
  g.append("g").attr("transform", `translate(0,${innerH})`)
    .call(d3.axisBottom(x)).selectAll("text")
    .attr("transform", "rotate(-35)").style("text-anchor", "end")
    .attr("fill", "#9aa0a6").attr("font-size", "10px");
  g.append("g").call(d3.axisLeft(y).ticks(5)).attr("color", "#555");
  g.selectAll(".bar").data(data).join("rect")
    .attr("x", d => x(d.label)).attr("y", d => y(Math.max(d[metricKey], 0)))
    .attr("width", x.bandwidth()).attr("height", d => Math.abs(y(d[metricKey]) - y(0)))
    .attr("fill", "var(--accent)").attr("rx", 3);
  document.getElementById("barChartTitle").textContent =
    document.getElementById("metricSelect").selectedOptions[0].textContent + " by tree";
}

function renderScatterChart(rows, xKey, yKey) {
  const svgEl = d3.select("#scatterChart");
  svgEl.selectAll("*").remove();
  const width = svgEl.node().clientWidth || 600;
  const height = 280;
  const margin = { top: 12, right: 12, bottom: 48, left: 52 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const g = svgEl.attr("viewBox", `0 0 ${width} ${height}`).append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);
  const data = rows.filter(r => r[xKey] != null && r[yKey] != null);
  const xLabel = document.getElementById("scatterX").selectedOptions[0].textContent;
  const yLabel = document.getElementById("scatterY").selectedOptions[0].textContent;
  if (!data.length) {
    document.getElementById("scatterChartTitle").textContent = "Generate a tree to see scatter";
    return;
  }
  const x = d3.scaleLinear().domain(d3.extent(data, d => d[xKey])).nice().range([0, innerW]);
  const y = d3.scaleLinear().domain(d3.extent(data, d => d[yKey])).nice().range([innerH, 0]);
  g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6)).attr("color", "#555");
  g.append("g").call(d3.axisLeft(y).ticks(6)).attr("color", "#555");
  const tooltip = document.getElementById("tooltip");
  g.selectAll(".dot").data(data).join("circle")
    .attr("cx", d => x(d[xKey])).attr("cy", d => y(d[yKey])).attr("r", 6)
    .attr("fill", "var(--accent)").attr("fill-opacity", 0.85).attr("stroke", "#fff")
    .on("mouseenter", (event, d) => {
      tooltip.innerHTML = `<div><strong>${escapeHtml(d.label)}</strong></div>
        <div>${escapeHtml(xLabel)}: ${d[xKey]}</div><div>${escapeHtml(yLabel)}: ${d[yKey]}</div>`;
      tooltip.style.display = "block";
    })
    .on("mousemove", event => {
      tooltip.style.left = (event.clientX + 14) + "px";
      tooltip.style.top = (event.clientY + 14) + "px";
    })
    .on("mouseleave", () => { tooltip.style.display = "none"; });
  document.getElementById("scatterChartTitle").textContent = `${yLabel} vs ${xLabel}`;
}

function renderSummary() {
  updateFilterHint();
  const rows = chartRows();
  renderSummaryTable(rows);
  renderBarChart(rows, document.getElementById("metricSelect").value);
  renderScatterChart(rows, document.getElementById("scatterX").value, document.getElementById("scatterY").value);
  renderGlobalStats();
}

function displayToken(token) {
  if (!token) return "⟨prompt⟩";
  if (token === "\n") return "\\n";
  if (token === "\t") return "\\t";
  if (token === " ") return "·";
  return token;
}

function fmtProb(p) {
  if (p >= 0.0001) return p.toFixed(4);
  return p.toExponential(2);
}

function escapeHtml(text) {
  return String(text).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function leafDescendants(nodes, nodeId) {
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  function walk(id) {
    const node = byId[id];
    if (!node.c || !node.c.length) return [id];
    return node.c.flatMap(walk);
  }
  return walk(nodeId);
}

function treePositions(nodes, rootId = "root") {
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  function leafIds(nodeId) {
    const node = byId[nodeId];
    if (!node.c || !node.c.length) return [nodeId];
    return node.c.flatMap(leafIds);
  }
  const leafOrder = leafIds(rootId);
  const leafX = Object.fromEntries(leafOrder.map((id, i) => [id, i]));
  const positions = {};
  function assign(nodeId) {
    const node = byId[nodeId];
    let x;
    if (!node.c || !node.c.length) x = leafX[nodeId];
    else x = node.c.reduce((sum, cid) => sum + assign(cid), 0) / node.c.length;
    positions[nodeId] = { x, y: -node.d };
    return x;
  }
  assign(rootId);
  return positions;
}

function buildEdges(nodes, minProb) {
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const edges = [];
  for (const node of nodes) {
    for (const childId of node.c || []) {
      const child = byId[childId];
      if (!child || child.p < minProb) continue;
      edges.push({ source: node.id, target: childId, prob: child.p, token: child.tok || "?" });
    }
  }
  return edges;
}

function visibleNodeIds(edges) {
  const ids = new Set(["root"]);
  edges.forEach(e => { ids.add(e.source); ids.add(e.target); });
  return ids;
}

function nodeCategory(nodeId) {
  const stats = statsMap.get(nodeId);
  const status = statusMap.get(nodeId);
  const leaf = leafMap.get(nodeId);
  if (stats?.color_class === "exclusively_bad" || status === "exclusively_bad") return "bad";
  if (stats?.color_class === "mostly_bad") return "mostly_bad";
  if (stats?.color_class === "exclusively_good") return "good";
  if (leaf) return leaf.answer_correct ? "good_leaf" : "bad_leaf";
  return "other";
}

function isHighlightedNode(nodeId) {
  const cat = nodeCategory(nodeId);
  const showBad = document.getElementById("showBad").checked;
  const showGood = document.getElementById("showGood").checked;
  if (cat === "bad" || cat === "mostly_bad" || cat === "bad_leaf") return showBad;
  if (cat === "good" || cat === "good_leaf") return showGood;
  return true;
}

function nodeFill(node) {
  const stats = statsMap.get(node.id);
  if (stats) {
    if (stats.color_class === "exclusively_bad") return "var(--bad-node)";
    if (stats.color_class === "exclusively_good") return "var(--good-node)";
    if (stats.color_class === "mostly_bad") return "var(--mostly-bad)";
    return "var(--internal)";
  }
  if (!node.c || !node.c.length) {
    const leaf = leafMap.get(node.id);
    if (leaf?.answer_correct) return "var(--good-node)";
    return "var(--bad-node)";
  }
  if (node.id === "root") return "var(--accent)";
  return "var(--internal)";
}

function nodeStroke(node) {
  if (node.id === selectedNodeId) return "#fff";
  const cat = nodeCategory(node.id);
  if (cat === "bad") return "#ff8a80";
  if (cat === "mostly_bad") return "#ffe082";
  if (cat === "good" || cat === "good_leaf") return "#a5d6a7";
  if (cat === "bad_leaf") return "#ef9a9a";
  return "#222";
}

function nodeRadius(node) {
  if (node.id === selectedNodeId) return 9;
  const cat = nodeCategory(node.id);
  if (cat === "bad") return 8;
  if (cat === "mostly_bad") return 7;
  if (cat === "good") return 7;
  if (!node.c || !node.c.length) return 5;
  return 4;
}

function pillClassForStats(stats) {
  if (!stats) return "";
  if (stats.color_class === "exclusively_bad") return "bad-node";
  if (stats.color_class === "mostly_bad") return "mostly-bad";
  if (stats.color_class === "exclusively_good") return "good-node";
  return "ditched";
}

function renderRunMeta(run) {
  const row = (DATA.tree_summaries || []).find(r => r.tree_key === run.tree_key) || {};
  const modeLabel = run.answer_mode ? ANSWER_MODE_LABELS[run.answer_mode] || run.answer_mode : null;
  const answersLine = run.expected_answers
    ? `<div class="muted">Answers: ${escapeHtml(run.expected_answers)}${modeLabel ? ` · ${modeLabel}` : ""}</div>`
    : "";
  document.getElementById("runMeta").innerHTML = [
    `<div><strong>${MODEL_LABELS[run.model_id] || run.model_id}</strong> · τ=${run.tau}</div>`,
    `<div>${escapeHtml(row.prompt_preview || run.prompt || "")}</div>`,
    answersLine,
  ].join("");
  document.getElementById("statGrid").innerHTML = `
    <div class="stat"><strong style="color:var(--bad-node)">${row.exclusively_bad_count ?? 0}</strong><span class="muted">excl. bad</span></div>
    <div class="stat"><strong style="color:var(--good-node)">${row.exclusively_good_count ?? 0}</strong><span class="muted">100% good</span></div>
    <div class="stat"><strong>${row.leaf_correct_pct ?? "—"}%</strong><span class="muted">good leaves</span></div>
  `;
}

function countExpectedAnswers(raw) {
  if (!raw || !raw.trim()) return 0;
  return raw.replace(/\n/g, ",").split(",").map(s => s.trim()).filter(Boolean).length;
}

function updateAnswerModeVisibility() {
  const raw = document.getElementById("expectedAnswers").value.trim();
  const label = document.getElementById("answerModeLabel");
  label.style.display = raw ? "grid" : "none";
}

function updateRescoreModeVisibility() {
  const raw = document.getElementById("rescoreAnswers").value.trim();
  const label = document.getElementById("rescoreModeLabel");
  label.style.display = raw ? "grid" : "block";
}

function syncRescoreForm(run) {
  const answersInput = document.getElementById("rescoreAnswers");
  const modeSelect = document.getElementById("rescoreModeSelect");
  answersInput.value = run?.expected_answers || "";
  modeSelect.value = run?.answer_mode || "or";
  updateRescoreModeVisibility();
}

const ANSWER_MODE_LABELS = {
  or: "OR (any)",
  and: "AND (all)",
  xor: "XOR (exactly one)",
  categories: "categories",
};

function renderAnswerMatches(leaf) {
  if (!leaf?.answer_matches || !Object.keys(leaf.answer_matches).length) return "";
  return `<div style="margin-top:8px">${Object.entries(leaf.answer_matches).map(([term, hit]) =>
    `<span class="pill ${hit ? "yes" : "no"}">${escapeHtml(term)}: ${hit ? "yes" : "no"}</span>`
  ).join(" ")}</div>`;
}

function nodeReasoningText(node, leaf) {
  const parts = [];
  if (node?.suffix) parts.push(node.suffix);
  if (leaf?.completion_text) parts.push(leaf.completion_text);
  return parts.join("");
}

function renderReasoningSection(node, leaf) {
  const text = nodeReasoningText(node, leaf);
  if (!text) {
    if (node?.id === "root") {
      return '<p class="muted">Root — click a child node to view reasoning along that path.</p>';
    }
    return '<p class="muted">No reasoning text stored for this node.</p>';
  }
  return `
    <div class="reasoning-section">
      <h4 class="reasoning-label">Full reasoning</h4>
      <pre class="reasoning-block">${escapeHtml(text)}</pre>
    </div>`;
}

function renderLeafCard(leaf) {
  const node = nodeById.get(leaf.leaf_id);
  const reasoning = nodeReasoningText(node, leaf);
  return `
    <div class="card leaf-card ${leaf.answer_correct ? "correct" : "incorrect"}">
      <div class="muted">${leaf.leaf_id} · p=${fmtProb(leaf.path_prob)}</div>
      ${reasoning ? `<pre class="reasoning-block compact">${escapeHtml(reasoning)}</pre>` : ""}
      <div class="answer" style="margin-top:8px">${leaf.answer_text ? escapeHtml(leaf.answer_text) : "<em class='muted'>no answer</em>"}</div>
      <div style="margin-top:8px">
        <span class="pill ${leaf.answer_correct ? "yes" : "no"}">${leaf.answer_correct ? "good" : "bad"}</span>
        ${leaf.reasoning_complete ? '<span class="pill yes">complete</span>' : '<span class="pill no">incomplete</span>'}
      </div>
      ${renderAnswerMatches(leaf)}
    </div>`;
}

function renderDetailPanel(nodeId) {
  const panel = document.getElementById("detailPanel");
  const title = document.getElementById("detailTitle");
  if (!nodeId) {
    title.textContent = "Selected node";
    panel.innerHTML = '<p class="muted">Click a node in the graph or a table row.</p>';
    return;
  }
  const status = statusMap.get(nodeId);
  const stats = statsMap.get(nodeId);
  const leaf = leafMap.get(nodeId);
  const node = nodeById.get(nodeId);
  const leaves = leafDescendants(currentNodes, nodeId).map(id => leafMap.get(id)).filter(Boolean)
    .sort((a, b) => b.path_prob - a.path_prob);
  const descendantCards = leaf ? leaves.filter(item => item.leaf_id !== nodeId) : leaves;
  title.textContent = `Node ${nodeId}`;
  panel.innerHTML = `
    <div class="card">
      <div class="mono muted">${node ? displayToken(node.tok) : ""} · depth ${node?.d ?? "?"} · p=${node ? fmtProb(node.p) : "?"}</div>
      ${stats ? `<div style="margin-top:6px"><span class="pill ${pillClassForStats(stats)}">${stats.bad_pct_display}% bad · ${stats.n_good}/${stats.n_leaves} good</span></div>` : ""}
      ${status ? `<div style="margin-top:6px"><span class="pill ${status === "exclusively_bad" ? "bad-node" : "ditched"}">${status}</span></div>` : ""}
      ${renderReasoningSection(node, leaf)}
      ${leaf?.answer_text ? `<div style="margin-top:10px"><h4 class="reasoning-label">Answer</h4><div class="answer">${escapeHtml(leaf.answer_text)}</div></div>` : ""}
      ${leaf ? renderAnswerMatches(leaf) : ""}
    </div>
    ${descendantCards.length ? `<h4 class="reasoning-label" style="margin:0 0 8px">Descendant leaves (${descendantCards.length})</h4>` : ""}
    ${descendantCards.length ? descendantCards.map(renderLeafCard).join("") : ""}`;
}

function selectNode(nodeId) {
  selectedNodeId = nodeId;
  renderDetailPanel(nodeId);
  if (currentRun) {
    drawGraph(currentNodes, parseFloat(document.getElementById("probFilter").value));
  }
}

function drawGraph(nodes, minProb) {
  currentNodes = nodes;
  nodeById = new Map(nodes.map(n => [n.id, n]));
  const dimOther = document.getElementById("dimOther").checked;
  const positions = treePositions(nodes);
  const edges = buildEdges(nodes, minProb);
  const visible = visibleNodeIds(edges);
  const graphNodes = nodes.filter(n => visible.has(n.id)).map(n => ({ ...n, ...positions[n.id] }));
  if (!graphNodes.length) return;
  const xs = graphNodes.map(n => n.x);
  const ys = graphNodes.map(n => n.y);
  const padX = Math.max((Math.max(...xs) - Math.min(...xs)) * 0.08, 1);
  const padY = Math.max((Math.max(...ys) - Math.min(...ys)) * 0.08, 1);
  const xScale = d => (d - Math.min(...xs) + padX) * 28;
  const yScale = d => (d - Math.min(...ys) + padY) * 60;
  graphNodes.forEach(n => { n.sx = xScale(n.x); n.sy = yScale(n.y); });
  const byId = Object.fromEntries(graphNodes.map(n => [n.id, n]));
  const links = edges.filter(e => byId[e.source] && byId[e.target]).map(e => ({
    ...e, sx: byId[e.source].sx, sy: byId[e.source].sy, tx: byId[e.target].sx, ty: byId[e.target].sy,
  }));
  const maxProb = Math.max(...links.map(l => l.prob), 1e-9);
  gRoot.selectAll("*").remove();
  gRoot.selectAll(".link").data(links, d => `${d.source}-${d.target}`).join("line")
    .attr("x1", d => d.sx).attr("y1", d => d.sy).attr("x2", d => d.tx).attr("y2", d => d.ty)
    .attr("stroke", d => {
      const cat = nodeCategory(d.target);
      if (cat === "bad") return "var(--bad-node)";
      if (cat === "mostly_bad") return "var(--mostly-bad)";
      if (cat === "good" || cat === "good_leaf") return "var(--good-node)";
      if (cat === "bad_leaf") return "var(--bad)";
      return "#555";
    })
    .attr("stroke-width", d => 0.8 + 5 * (d.prob / maxProb))
    .attr("stroke-opacity", d => isHighlightedNode(d.target) ? 0.85 : 0.08)
    .attr("stroke-linecap", "round");
  const node = gRoot.selectAll(".node").data(graphNodes, d => d.id).join("g")
    .attr("transform", d => `translate(${d.sx},${d.sy})`)
    .attr("opacity", d => {
      if (!isHighlightedNode(d.id)) return 0.08;
      const cat = nodeCategory(d.id);
      const polar = ["bad", "mostly_bad", "good", "good_leaf", "bad_leaf"].includes(cat);
      if (dimOther && !polar) return 0.18;
      return 1;
    })
    .style("cursor", "pointer")
    .on("click", (_, d) => selectNode(d.id));
  node.append("circle").attr("r", d => nodeRadius(d)).attr("fill", d => nodeFill(d))
    .attr("stroke", d => nodeStroke(d)).attr("stroke-width", d => d.id === selectedNodeId ? 2.5 : 1);
  node.append("text").attr("dy", d => (d.c && d.c.length ? -12 : 14))
    .attr("text-anchor", "middle").attr("fill", "#ccc").attr("font-size", "10px")
    .attr("font-family", "ui-monospace, monospace")
    .text(d => {
      if (d.id === "root") return "prompt";
      const tok = displayToken(d.tok);
      return tok.length > 12 ? tok.slice(0, 10) + "…" : tok;
    });
  const tooltip = document.getElementById("tooltip");
  node.on("mouseenter", (event, d) => {
    const stats = statsMap.get(d.id);
    const leaf = leafMap.get(d.id);
    let html = `<div class="tok">${escapeHtml(displayToken(d.id === "root" ? null : d.tok))}</div>`;
    html += `<div>depth ${d.d} · p=${fmtProb(d.p)}</div>`;
    if (stats) html += `<div><span class="pill ${pillClassForStats(stats)}">${stats.bad_pct_display}% bad</span></div>`;
    if (leaf) html += `<div><span class="pill ${leaf.answer_correct ? "yes" : "no"}">${leaf.answer_correct ? "good" : "bad"}</span></div>`;
    tooltip.innerHTML = html;
    tooltip.style.display = "block";
  }).on("mousemove", event => {
    tooltip.style.left = (event.clientX + 14) + "px";
    tooltip.style.top = (event.clientY + 14) + "px";
  }).on("mouseleave", () => { tooltip.style.display = "none"; });
  fitToScreen();
}

function fitToScreen() {
  if (!gRoot.node()) return;
  const bounds = gRoot.node().getBBox();
  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;
  if (!bounds.width || !bounds.height) return;
  const midX = bounds.x + bounds.width / 2;
  const midY = bounds.y + bounds.height / 2;
  const scale = 0.9 / Math.max(bounds.width / width, bounds.height / height);
  svg.transition().duration(400).call(
    zoomBehavior.transform,
    d3.zoomIdentity.translate(width / 2, height / 2).scale(scale).translate(-midX, -midY)
  );
}

function loadRun(treeKey) {
  const run = DATA.runs.find(r => r.tree_key === treeKey);
  const nodes = DATA.trees[treeKey];
  if (!run || !nodes) return;
  currentRun = run;
  selectedNodeId = null;
  statusMap = new Map(Object.entries(DATA.node_status[treeKey] || {}));
  statsMap = new Map(Object.entries(DATA.node_stats[treeKey] || {}));
  leafMap = new Map(Object.entries(DATA.leaf_completions[treeKey] || {}));
  syncRescoreForm(run);
  renderRunMeta(run);
  renderDetailPanel(null);
  drawGraph(nodes, parseFloat(document.getElementById("probFilter").value));
}

function refreshRunSelect() {
  const runSelect = document.getElementById("runSelect");
  const prev = runSelect.value;
  runSelect.innerHTML = "";
  DATA.runs.forEach(run => {
    const opt = document.createElement("option");
    opt.value = run.tree_key;
    const row = (DATA.tree_summaries || []).find(r => r.tree_key === run.tree_key) || {};
    opt.textContent = `${row.prompt_preview || "tree"} · ${MODEL_LABELS[run.model_id] || run.model_id} · τ=${run.tau}`;
    runSelect.appendChild(opt);
  });
  if (DATA.runs.length) {
    runSelect.value = prev && DATA.runs.some(r => r.tree_key === prev) ? prev : DATA.runs[0].tree_key;
  }
}

function applyData(data) {
  DATA = data;
  populateFilters();
  refreshRunSelect();
  renderSummary();
  document.getElementById("source").textContent =
    `${DATA.runs.length} tree${DATA.runs.length === 1 ? "" : "s"} loaded · Summary for tables/charts · Tree Explorer for node view`;
  if (DATA.runs.length) loadRun(document.getElementById("runSelect").value);
}

async function fetchState() {
  const response = await fetch("/api/state");
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function fetchModels() {
  const response = await fetch("/api/models");
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function setGenerateStatus(message, isError = false) {
  const el = document.getElementById("generateStatus");
  el.textContent = message;
  el.className = isError ? "status-line error" : "status-line muted";
}

function setProgressVisible(visible) {
  document.getElementById("progressWrap").classList.toggle("active", visible);
}

function updateProgressBar(data) {
  const fill = document.getElementById("progressFill");
  const stage = document.getElementById("progressStage");
  const percent = document.getElementById("progressPercent");
  const label = data.stage ? data.stage.replace(/_/g, " ") : "working";
  const detail = data.message || "";
  const nodes = data.nodes != null ? ` · ${data.nodes} nodes` : "";
  const leaves = data.leaves != null ? ` · ${data.leaves} leaves` : "";
  stage.textContent = `${label}: ${detail}${nodes}${leaves}`;

  let pct = data.percent;
  if (pct == null) {
    if (data.stage === "build_tree") pct = null;
    else if (data.status === "completed") pct = 100;
    else if (data.current > 0 && data.total) pct = 100 * data.current / data.total;
    else {
      const stagePct = {
        load_hf: 5,
        find_root: 10,
        build_tree: 45,
        load_vllm: 55,
        complete_leaves: 90,
        analyze: 98,
      };
      pct = stagePct[data.stage] ?? 15;
    }
  }

  if (pct == null) {
    fill.style.width = "100%";
    fill.style.opacity = "0.45";
    percent.textContent = "…";
  } else {
    fill.style.opacity = "1";
    fill.style.width = `${Math.max(2, Math.min(100, pct))}%`;
    percent.textContent = `${Math.round(pct)}%`;
  }
}

async function fetchProgress() {
  const response = await fetch("/api/progress");
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForGeneration() {
  while (true) {
    const data = await fetchProgress();
    updateProgressBar(data);
    if (!data.active) {
      if (data.status === "failed") {
        throw new Error(data.error || "Generation failed");
      }
      if (data.payload) return data.payload;
      if (data.status === "completed") {
        return fetchState();
      }
      throw new Error("Generation ended without a result");
    }
    await sleep(500);
  }
}

async function generateTree() {
  const prompt = document.getElementById("promptInput").value.trim();
  const modelId = document.getElementById("modelSelect").value;
  const tau = parseFloat(document.getElementById("tauSelect").value);
  const expectedAnswers = document.getElementById("expectedAnswers").value.trim();
  const answerMode = document.getElementById("answerModeSelect").value;
  if (!prompt) {
    setGenerateStatus("Enter a prompt first.", true);
    return;
  }
  const btn = document.getElementById("generateBtn");
  btn.disabled = true;
  setProgressVisible(true);
  updateProgressBar({ stage: "starting", message: "Submitting…", percent: 0 });
  setGenerateStatus("Building τ-tree and completing leaves…");
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        model_id: modelId,
        tau,
        expected_answers: expectedAnswers || null,
        answer_mode: answerMode,
      }),
    });
    const started = await response.json();
    if (!response.ok) throw new Error(started.detail || response.statusText);

    const payload = await waitForGeneration();
    applyData(payload);
    setGenerateStatus(`Done · ${payload.runs.length} tree(s) in session.`);
    switchTab("trees");
  } catch (err) {
    setGenerateStatus(`Error: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    setProgressVisible(false);
  }
}

async function rescoreCurrentTree() {
  if (!currentRun) {
    setGenerateStatus("Select a tree first.", true);
    return;
  }
  const expectedAnswers = document.getElementById("rescoreAnswers").value.trim();
  const answerMode = document.getElementById("rescoreModeSelect").value;
  const btn = document.getElementById("rescoreBtn");
  btn.disabled = true;
  try {
    const response = await fetch(
      `/api/runs/${encodeURIComponent(currentRun.tree_key)}/rescore`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          expected_answers: expectedAnswers || null,
          answer_mode: answerMode,
        }),
      }
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    const treeKey = currentRun.tree_key;
    applyData(payload);
    document.getElementById("runSelect").value = treeKey;
    loadRun(treeKey);
    setGenerateStatus("Rescored tree with updated answers.");
  } catch (err) {
    setGenerateStatus(`Rescore failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

async function clearTrees() {
  if (!DATA.runs.length) return;
  if (!confirm("Clear all generated trees from this session?")) return;
  const response = await fetch("/api/clear", { method: "POST" });
  const payload = await response.json();
  applyData(payload);
  setGenerateStatus("Cleared all trees.");
}

async function saveCurrentTree() {
  const treeKey = document.getElementById("runSelect").value || currentRun?.tree_key;
  if (!treeKey) {
    setGenerateStatus("No tree selected to save.", true);
    return;
  }
  const btn = document.getElementById("saveTreeBtn");
  btn.disabled = true;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(treeKey)}/save`, { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || response.statusText);
    setGenerateStatus(`Saved tree to ${result.filename}`);
    await refreshSavedSelect();
  } catch (err) {
    setGenerateStatus(`Save failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

async function saveAllTrees() {
  if (!DATA.runs.length) {
    setGenerateStatus("No trees in session to save.", true);
    return;
  }
  const btn = document.getElementById("saveAllBtn");
  btn.disabled = true;
  try {
    const response = await fetch("/api/save-all", { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || response.statusText);
    setGenerateStatus(`Saved ${result.count} tree(s) to disk.`);
    await refreshSavedSelect();
  } catch (err) {
    setGenerateStatus(`Save failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

function savedOptionHtml(saved) {
  return saved.map(item => {
    const label = `${item.prompt_preview || item.tree_key} · ${MODEL_LABELS[item.model_id] || item.model_id} · τ=${item.tau}`;
    return `<option value="${item.filename}">${escapeHtml(label)}</option>`;
  }).join("");
}

async function refreshSavedSelect() {
  const selects = [
    document.getElementById("savedSelect"),
    document.getElementById("savedSelectSummary"),
  ].filter(Boolean);
  const prev = selects[0]?.value || "";
  try {
    const response = await fetch("/api/saved");
    if (!response.ok) return;
    const { saved } = await response.json();
    const options = '<option value="">Load saved…</option>' + savedOptionHtml(saved);
    selects.forEach(sel => {
      sel.innerHTML = options;
      if (prev) sel.value = prev;
    });
    const hint = document.getElementById("filterHint");
    if (hint && !DATA.runs.length && saved.length) {
      hint.textContent = `No trees in session · ${saved.length} saved on disk — use Load or Load all saved above`;
    }
  } catch (_) {
    // ignore list failures on startup
  }
}

async function loadSavedTreeFromSelect(selectId) {
  const filename = document.getElementById(selectId).value;
  if (!filename) {
    setGenerateStatus("Pick a saved tree to load.", true);
    return;
  }
  const response = await fetch("/api/saved/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || response.statusText);
  applyData(result.payload);
  setGenerateStatus(`Loaded saved tree (${result.filename}).`);
  switchTab("trees");
}

async function loadSavedTree() {
  const btn = document.getElementById("loadSavedBtn");
  btn.disabled = true;
  try {
    await loadSavedTreeFromSelect("savedSelect");
  } catch (err) {
    setGenerateStatus(`Load failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

async function loadAllSavedTrees() {
  const btn = document.getElementById("loadAllSavedBtn");
  btn.disabled = true;
  try {
    const response = await fetch("/api/saved");
    if (!response.ok) throw new Error("Could not list saved trees");
    const { saved } = await response.json();
    if (!saved.length) {
      setGenerateStatus("No saved trees on disk.", true);
      return;
    }
    for (const item of saved) {
      const loadResp = await fetch("/api/saved/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: item.filename }),
      });
      const result = await loadResp.json();
      if (!loadResp.ok) throw new Error(result.detail || loadResp.statusText);
      applyData(result.payload);
    }
    setGenerateStatus(`Loaded ${saved.length} saved tree(s).`);
    switchTab("summary");
  } catch (err) {
    setGenerateStatus(`Load failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

function init() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  svg = d3.select("#graph");
  gRoot = svg.append("g");
  zoomBehavior = d3.zoom().scaleExtent([0.05, 8]).on("zoom", event => gRoot.attr("transform", event.transform));
  svg.call(zoomBehavior);
  document.getElementById("fitBtn").addEventListener("click", fitToScreen);
  document.getElementById("resetBtn").addEventListener("click", () => {
    svg.transition().duration(300).call(zoomBehavior.transform, d3.zoomIdentity);
  });
  ["modelFilter", "tauFilter", "metricSelect", "scatterX", "scatterY"].forEach(id => {
    document.getElementById(id).addEventListener("change", () => renderSummary());
  });
  document.getElementById("runSelect").addEventListener("change", () => loadRun(document.getElementById("runSelect").value));
  document.getElementById("probFilter").addEventListener("change", () => {
    if (currentRun) loadRun(currentRun.tree_key);
  });
  ["showBad", "showGood", "dimOther"].forEach(id => {
    document.getElementById(id).addEventListener("change", () => {
      if (currentRun) drawGraph(currentNodes, parseFloat(document.getElementById("probFilter").value));
    });
  });
  document.getElementById("generateBtn").addEventListener("click", generateTree);
  document.getElementById("expectedAnswers").addEventListener("input", updateAnswerModeVisibility);
  document.getElementById("rescoreAnswers").addEventListener("input", updateRescoreModeVisibility);
  document.getElementById("rescoreBtn").addEventListener("click", rescoreCurrentTree);
  document.getElementById("saveAllBtn").addEventListener("click", saveAllTrees);
  document.getElementById("saveTreeBtn").addEventListener("click", saveCurrentTree);
  document.getElementById("loadSavedBtn").addEventListener("click", loadSavedTree);
  document.getElementById("loadSavedSummaryBtn").addEventListener("click", async () => {
    const btn = document.getElementById("loadSavedSummaryBtn");
    btn.disabled = true;
    try {
      await loadSavedTreeFromSelect("savedSelectSummary");
    } catch (err) {
      setGenerateStatus(`Load failed: ${err.message}`, true);
    } finally {
      btn.disabled = false;
    }
  });
  document.getElementById("loadAllSavedBtn").addEventListener("click", loadAllSavedTrees);
  document.getElementById("clearBtn").addEventListener("click", clearTrees);
}

function enableViewOnlyMode() {
  const banner = document.createElement("div");
  banner.className = "view-only-banner";
  banner.setAttribute("role", "status");
  banner.innerHTML = "<strong>View only.</strong> Tree generation requires running <code>serve_dashboard.py</code> locally with a GPU.";
  banner.style.cssText = "margin-top:12px;padding:10px 14px;border:1px solid #5c4a1a;background:#2a2414;color:#f0d78c;border-radius:8px;font-size:13px";
  const header = document.querySelector("header");
  const generatePanel = document.querySelector(".generate-panel");
  if (header && generatePanel) header.insertBefore(banner, generatePanel);
  ["promptInput", "expectedAnswers", "answerModeSelect", "modelSelect", "tauSelect",
    "generateBtn", "saveAllBtn", "clearBtn", "saveTreeBtn", "loadSavedBtn",
    "loadAllSavedBtn", "loadSavedSummaryBtn", "rescoreBtn"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = true;
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  init();
  if (VIEW_ONLY) {
    enableViewOnlyMode();
    try {
      const response = await fetch("./trees_payload.json");
      if (!response.ok) throw new Error(response.statusText);
      const data = await response.json();
      applyData(data);
      const first = data.runs?.[0];
      if (first) {
        document.getElementById("promptInput").value = first.prompt || "";
        document.getElementById("expectedAnswers").value = first.expected_answers || "";
      }
      document.getElementById("source").textContent =
        `View-only demo · ${data.runs.length} tree(s) loaded`;
    } catch (err) {
      document.getElementById("source").innerHTML =
        `<span class="error">Failed to load trees_payload.json: ${escapeHtml(err.message)}</span>`;
    }
    return;
  }
  try {
    const { models } = await fetchModels();
    const modelSelect = document.getElementById("modelSelect");
    modelSelect.innerHTML = models.map(m =>
      `<option value="${m.id}">${MODEL_LABELS[m.id] || m.id}</option>`
    ).join("");
    const data = await fetchState();
    if (!data.runs?.length) {
      const savedResp = await fetch("/api/saved");
      if (savedResp.ok) {
        const { saved } = await savedResp.json();
        if (saved?.length) {
          for (const item of saved) {
            const loadResp = await fetch("/api/saved/load", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ filename: item.filename }),
            });
            const result = await loadResp.json();
            if (loadResp.ok && result.payload) {
              applyData(result.payload);
            }
          }
          setGenerateStatus(`Auto-loaded ${saved.length} saved tree(s).`);
        } else {
          applyData(data);
        }
      } else {
        applyData(data);
      }
    } else {
      applyData(data);
    }
    await refreshSavedSelect();
    updateAnswerModeVisibility();
  } catch (err) {
    document.getElementById("source").innerHTML = `<span class="error">Failed to connect to API: ${escapeHtml(err.message)}</span>`;
  }
});
