window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { utils } = TD;

TD.TreeGraph = class {
  constructor(svgSelector, tooltipSelector) {
    this.svg = d3.select(svgSelector);
    this.gRoot = this.svg.append("g");
    this.tooltip = document.querySelector(tooltipSelector);
    this.zoomBehavior = d3.zoom().scaleExtent([0.05, 8]).on("zoom", event => {
      this.gRoot.attr("transform", event.transform);
    });
    this.svg.call(this.zoomBehavior);
    this.currentNodes = [];
    this.leafMap = new Map();
    this.statusMap = new Map();
    this.statsMap = new Map();
    this.expansionMap = new Map();
    this.selectedNodeId = null;
    this.onSelect = null;
    this.options = {};
  }

  setOptions(options) {
    this.options = { ...this.options, ...options };
  }

  leafDescendants(nodes, nodeId) {
    const byId = new Map(nodes.map(node => [node.id, node]));
    const walk = id => {
      const node = byId.get(id);
      const children = node?.c || node?.child_ids || [];
      if (!children.length) return [id];
      return children.flatMap(walk);
    };
    return walk(nodeId);
  }

  treePositions(nodes, rootId = "root") {
    const byId = new Map(nodes.map(node => [node.id, node]));
    const depthGroups = new Map();
    const assign = (id, depth = 0) => {
      if (!depthGroups.has(depth)) depthGroups.set(depth, []);
      depthGroups.get(depth).push(id);
      const children = byId.get(id)?.c || byId.get(id)?.child_ids || [];
      children.forEach(childId => assign(childId, depth + 1));
    };
    assign(rootId);
    const positions = new Map();
    const xGap = 90;
    const yGap = 70;
    depthGroups.forEach((ids, depth) => {
      const totalWidth = (ids.length - 1) * xGap;
      ids.forEach((id, index) => {
        positions.set(id, { x: index * xGap - totalWidth / 2, y: depth * yGap });
      });
    });
    return positions;
  }

  buildEdges(nodes, minProb) {
    const byId = new Map(nodes.map(node => [node.id, node]));
    const edges = [];
    nodes.forEach(node => {
      const children = node.c || node.child_ids || [];
      children.forEach(childId => {
        const child = byId.get(childId);
        const prob = child?.p ?? child?.prob ?? 0;
        if (prob >= minProb) edges.push({ source: node.id, target: childId, prob });
      });
    });
    return edges;
  }

  nodeCategory(nodeId) {
    const stats = this.statsMap.get(nodeId);
    if (!stats) return "internal";
    if (stats.bad_pct >= 1) return "exclusively_bad";
    if (stats.bad_pct <= 0) return "exclusively_good";
    if (stats.bad_pct > 0.75) return "mostly_bad";
    return "mixed";
  }

  nodeFill(node) {
    const cat = this.nodeCategory(node.id);
    const opts = this.options;
    if (opts.dimOther && opts.highlightSet && !opts.highlightSet.has(node.id)) return "#2a2d36";
    if (cat === "exclusively_bad" && opts.showBad !== false) return "var(--bad-node)";
    if (cat === "mostly_bad" && opts.showBad !== false) return "var(--mostly-bad)";
    if (cat === "exclusively_good" && opts.showGood !== false) return "var(--good-node)";
    return "var(--internal)";
  }

  pillClassForStats(stats) {
    if (!stats) return "ditched";
    if (stats.bad_pct >= 1) return "bad-node";
    if (stats.bad_pct > 0.75) return "mostly-bad";
    if (stats.bad_pct <= 0) return "good-node";
    return "ditched";
  }

  async draw(nodes, minProb = 0) {
    this.currentNodes = nodes;
    const positions = this.treePositions(nodes);
    const edges = this.buildEdges(nodes, minProb);
    const visible = new Set();
    edges.forEach(edge => { visible.add(edge.source); visible.add(edge.target); });
    if (!visible.size && nodes.length) visible.add(nodes[0].id);

    const nodeData = nodes.filter(node => visible.has(node.id)).map(node => ({
      ...node,
      x: positions.get(node.id)?.x || 0,
      y: positions.get(node.id)?.y || 0,
    }));

    const link = this.gRoot.selectAll("line.edge").data(edges, d => `${d.source}-${d.target}`);
    link.exit().remove();
    link.enter().append("line").attr("class", "edge")
      .merge(link)
      .attr("x1", d => positions.get(d.source)?.x || 0)
      .attr("y1", d => positions.get(d.source)?.y || 0)
      .attr("x2", d => positions.get(d.target)?.x || 0)
      .attr("y2", d => positions.get(d.target)?.y || 0)
      .attr("stroke", "#4a5068")
      .attr("stroke-width", d => 1 + 6 * d.prob);

    const circles = this.gRoot.selectAll("circle.node").data(nodeData, d => d.id);
    circles.exit().remove();
    const enter = circles.enter().append("circle").attr("class", "node");
    enter.merge(circles)
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("r", d => (d.c || d.child_ids || []).length ? 7 : 5)
      .attr("fill", d => this.nodeFill(d))
      .attr("stroke", d => d.id === this.selectedNodeId ? "var(--accent)" : "#111")
      .attr("stroke-width", d => d.id === this.selectedNodeId ? 2.5 : 1)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        this.selectedNodeId = d.id;
        this.draw(nodes, minProb);
        if (this.onSelect) this.onSelect(d.id);
      })
      .on("mouseover", (event, d) => this.showTooltip(event, d))
      .on("mousemove", event => {
        if (this.tooltip) {
          this.tooltip.style.left = `${event.clientX + 14}px`;
          this.tooltip.style.top = `${event.clientY + 14}px`;
        }
      })
      .on("mouseleave", () => { if (this.tooltip) this.tooltip.style.display = "none"; });
  }

  showTooltip(event, node) {
    if (!this.tooltip) return;
    const leaf = this.leafMap.get(node.id);
    const stats = this.statsMap.get(node.id);
    const status = this.statusMap.get(node.id);
    let html = `<div class="tok">${utils.escapeHtml(utils.displayToken(node.t || node.token))}</div>`;
    if (node.p != null) html += `<div>p=${utils.fmtProb(node.p)}</div>`;
    if (stats) {
      html += `<div><span class="pill ${this.pillClassForStats(stats)}">${stats.bad_pct_display || Math.round(stats.bad_pct * 100)}% bad</span></div>`;
    }
    if (status) html += `<div><span class="pill ${status === "exclusively_bad" ? "bad-node" : "ditched"}">${status}</span></div>`;
    if (leaf) html += `<div style="margin-top:6px">${utils.escapeHtml(leaf.answer_text || leaf.completion_preview || "(no answer)")}</div>`;
    this.tooltip.innerHTML = html;
    this.tooltip.style.display = "block";
    this.tooltip.style.left = `${event.clientX + 14}px`;
    this.tooltip.style.top = `${event.clientY + 14}px`;
  }

  fitToScreen() {
    const bounds = this.gRoot.node()?.getBBox();
    const width = this.svg.node().clientWidth;
    const height = this.svg.node().clientHeight;
    if (!bounds?.width || !bounds?.height) return;
    const midX = bounds.x + bounds.width / 2;
    const midY = bounds.y + bounds.height / 2;
    const scale = 0.9 / Math.max(bounds.width / width, bounds.height / height);
    this.svg.transition().duration(400).call(
      this.zoomBehavior.transform,
      d3.zoomIdentity.translate(width / 2, height / 2).scale(scale).translate(-midX, -midY)
    );
  }

  resetZoom() {
    this.svg.transition().duration(300).call(this.zoomBehavior.transform, d3.zoomIdentity);
  }

  setTreeData({ nodes, leafMap, statusMap, statsMap, expansionMap }) {
    this.currentNodes = nodes || [];
    this.leafMap = leafMap || new Map();
    this.statusMap = statusMap || new Map();
    this.statsMap = statsMap || new Map();
    this.expansionMap = expansionMap || new Map();
  }
};
})(TreeDashboard);
