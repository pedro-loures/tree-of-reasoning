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
    this.incomingTokens = new Map();
    this.layoutPositions = new Map();
    this.selectedNodeId = null;
    this.dimFocusNodeId = null;
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

  buildParentMap(nodes = this.currentNodes) {
    const parent = new Map();
    for (const node of nodes) {
      const children = node.c || node.child_ids || [];
      children.forEach(childId => parent.set(childId, node.id));
    }
    return parent;
  }

  ancestorIds(nodeId, nodes = this.currentNodes) {
    const parent = this.buildParentMap(nodes);
    const ids = [];
    let cur = nodeId;
    while (cur) {
      ids.push(cur);
      cur = parent.get(cur);
    }
    return ids;
  }

  subtreeNodeIds(nodeId, nodes = this.currentNodes) {
    const byId = new Map(nodes.map(node => [node.id, node]));
    const walk = id => {
      const node = byId.get(id);
      if (!node) return [id];
      const children = node.c || node.child_ids || [];
      return [id, ...children.flatMap(walk)];
    };
    return walk(nodeId);
  }

  buildPathAndSubtreeSet(nodeId, nodes = this.currentNodes) {
    return new Set([
      ...this.ancestorIds(nodeId, nodes),
      ...this.subtreeNodeIds(nodeId, nodes),
    ]);
  }

  effectiveHighlightSet() {
    if (this.dimFocusNodeId) {
      return this.buildPathAndSubtreeSet(this.dimFocusNodeId);
    }
    const preset = this.options.highlightSet;
    if (this.options.dimOther && preset?.size) return preset;
    return null;
  }

  clearDimFocus() {
    this.dimFocusNodeId = null;
    this.selectedNodeId = null;
  }

  setDimFocus(nodeId) {
    this.dimFocusNodeId = nodeId;
    this.selectedNodeId = nodeId;
  }

  handleNodeClick(nodeId, nodes, minProb) {
    if (this.options.dimOther) {
      if (this.dimFocusNodeId === nodeId) {
        this.clearDimFocus();
      } else {
        this.setDimFocus(nodeId);
      }
    } else {
      this.selectedNodeId = nodeId;
    }
    void this.draw(nodes, minProb);
    if (this.onSelect) this.onSelect(this.selectedNodeId);
  }

  buildIncomingTokenMap(nodes) {
    const incoming = new Map();
    for (const node of nodes) {
      const children = node.c || node.child_ids || [];
      const tokens = node.ct || node.child_tokens || [];
      children.forEach((childId, index) => {
        if (tokens[index] != null) incoming.set(childId, tokens[index]);
      });
    }
    return incoming;
  }

  nodeToken(node) {
    if (!node || node.id === "root") return null;
    const direct = node.t ?? node.token ?? node.tok;
    if (direct != null && direct !== "") return direct;
    return this.incomingTokens.get(node.id) ?? null;
  }

  traditionalTreePositions(nodes, rootId = "root") {
    const byId = new Map(nodes.map(node => [node.id, node]));
    const leafIds = nodeId => {
      const node = byId.get(nodeId);
      const children = node?.c || node?.child_ids || [];
      if (!children.length) return [nodeId];
      return children.flatMap(leafIds);
    };
    const leafOrder = leafIds(rootId);
    const leafX = new Map(leafOrder.map((id, index) => [id, index]));
    const positions = new Map();
    const assign = nodeId => {
      const node = byId.get(nodeId);
      const children = node?.c || node?.child_ids || [];
      let x;
      if (!children.length) x = leafX.get(nodeId) ?? 0;
      else x = children.reduce((sum, childId) => sum + assign(childId), 0) / children.length;
      const depth = node?.d ?? node?.depth ?? 0;
      positions.set(nodeId, { x, y: -depth });
      return x;
    };
    assign(rootId);
    return positions;
  }

  computeLayout(nodes) {
    const raw = this.traditionalTreePositions(nodes);
    const rawNodes = nodes
      .filter(node => raw.has(node.id))
      .map(node => ({ id: node.id, x: raw.get(node.id).x, y: raw.get(node.id).y }));
    if (!rawNodes.length) return { positions: new Map() };

    const xs = rawNodes.map(node => node.x);
    const ys = rawNodes.map(node => node.y);
    const padX = Math.max((Math.max(...xs) - Math.min(...xs)) * 0.08, 1);
    const padY = Math.max((Math.max(...ys) - Math.min(...ys)) * 0.08, 1);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const xScale = x => (x - minX + padX) * 28;
    const yScale = y => (y - minY + padY) * 60;

    const positions = new Map();
    for (const node of rawNodes) {
      positions.set(node.id, { x: xScale(node.x), y: yScale(node.y) });
    }
    return { positions };
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

  isLeafNode(node) {
    return !(node.c || node.child_ids || []).length;
  }

  leafIsGood(leaf) {
    if (!leaf) return false;
    if (this.options.isLeafGood) return this.options.isLeafGood(leaf);
    return Boolean(leaf.answer_correct);
  }

  leafIsBad(leaf) {
    if (!leaf) return false;
    if (this.options.isLeafBad) return this.options.isLeafBad(leaf);
    return leaf.answer_correct === false;
  }

  nodeCategory(nodeId) {
    const stats = this.statsMap.get(nodeId);
    if (stats?.color_class) return stats.color_class;
    const status = this.statusMap.get(nodeId);
    if (status === "exclusively_bad") return "exclusively_bad";
    const leaf = this.leafMap.get(nodeId);
    if (leaf) {
      if (this.leafIsGood(leaf)) return "exclusively_good";
      if (this.leafIsBad(leaf)) return "exclusively_bad";
    }
    if (!stats) return "internal";
    if (stats.bad_pct >= 1) return "exclusively_bad";
    if (stats.bad_pct <= 0) return "exclusively_good";
    if (stats.bad_pct > 0.75) return "mostly_bad";
    return "mixed";
  }

  nodeFill(node) {
    const opts = this.options;
    const highlight = this.effectiveHighlightSet();
    if (opts.dimOther && highlight && !highlight.has(node.id)) return "#2a2d36";

    const stats = this.statsMap.get(node.id);
    if (stats) {
      if (stats.color_class === "exclusively_bad" && opts.showBad !== false) return "var(--bad-node)";
      if (stats.color_class === "mostly_bad" && opts.showBad !== false) return "var(--mostly-bad)";
      if (stats.color_class === "exclusively_good" && opts.showGood !== false) return "var(--good-node)";
    }

    if (this.isLeafNode(node)) {
      const leaf = this.leafMap.get(node.id);
      if (leaf) {
        if (this.leafIsGood(leaf) && opts.showGood !== false) return "var(--good-node)";
        if (this.leafIsBad(leaf) && opts.showBad !== false) return "var(--bad-node)";
      }
    }

    if (node.id === "root") return "var(--accent)";
    return "var(--internal)";
  }

  edgeColor(targetId) {
    const cat = this.nodeCategory(targetId);
    if (cat === "exclusively_bad") return "var(--bad-node)";
    if (cat === "mostly_bad") return "var(--mostly-bad)";
    if (cat === "exclusively_good") return "var(--good-node)";
    return "#555";
  }

  nodeLabel(node) {
    if (node.id === "root") return "prompt";
    const tok = utils.displayToken(this.nodeToken(node));
    const shortTok = tok.length > 12 ? `${tok.slice(0, 10)}…` : tok;
    const prob = Number(node.p ?? node.prob ?? 0).toFixed(2);
    return `${shortTok} ${prob}`;
  }

  nodeRadius(node) {
    if (node.id === this.selectedNodeId) return 9;
    if (this.expansionMap.has(node.id)) return 8;
    const cat = this.nodeCategory(node.id);
    if (cat === "exclusively_bad") return 8;
    if (cat === "mostly_bad" || cat === "exclusively_good") return 7;
    if (this.isLeafNode(node)) return 5;
    return 6;
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
    this.incomingTokens = this.buildIncomingTokenMap(nodes);
    const edges = this.buildEdges(nodes, minProb);
    const visible = new Set(["root"]);
    edges.forEach(edge => { visible.add(edge.source); visible.add(edge.target); });
    if (!visible.size && nodes.length) visible.add(nodes[0].id);

    const layout = this.computeLayout(nodes.filter(node => visible.has(node.id)));
    this.layoutPositions = layout.positions;

    const nodeData = nodes
      .filter(node => visible.has(node.id))
      .map(node => ({
        ...node,
        x: layout.positions.get(node.id)?.x || 0,
        y: layout.positions.get(node.id)?.y || 0,
      }));

    const maxProb = Math.max(...edges.map(edge => edge.prob), 1e-9);
    const link = this.gRoot.selectAll("line.edge").data(edges, d => `${d.source}-${d.target}`);
    link.exit().remove();
    link.enter().append("line").attr("class", "edge")
      .merge(link)
      .attr("x1", d => layout.positions.get(d.source)?.x || 0)
      .attr("y1", d => layout.positions.get(d.source)?.y || 0)
      .attr("x2", d => layout.positions.get(d.target)?.x || 0)
      .attr("y2", d => layout.positions.get(d.target)?.y || 0)
      .attr("stroke", d => this.edgeColor(d.target))
      .attr("stroke-width", d => 0.8 + 5 * (d.prob / maxProb))
      .attr("stroke-opacity", d => {
        const opts = this.options;
        const highlight = this.effectiveHighlightSet();
        if (opts.dimOther && highlight?.size) {
          return highlight.has(d.source) && highlight.has(d.target) ? 0.85 : 0.12;
        }
        return 0.85;
      })
      .attr("stroke-linecap", "round");

    const circles = this.gRoot.selectAll("circle.node").data(nodeData, d => d.id);
    circles.exit().remove();
    const enter = circles.enter().append("circle").attr("class", "node");
    const merged = enter.merge(circles);
    merged
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("r", d => this.nodeRadius(d))
      .attr("fill", d => this.nodeFill(d))
      .attr("stroke", d => {
        if (d.id === this.selectedNodeId) return "var(--accent)";
        if (this.expansionMap.has(d.id)) return "var(--expanded)";
        const cat = this.nodeCategory(d.id);
        if (cat === "exclusively_bad") return "#ff8a80";
        if (cat === "mostly_bad") return "#ffe082";
        if (cat === "exclusively_good") return "#a5d6a7";
        return "#222";
      })
      .attr("stroke-width", d => {
        if (d.id === this.selectedNodeId) return 2.5;
        if (this.expansionMap.has(d.id)) return 2;
        return this.nodeCategory(d.id) === "exclusively_bad" ? 2 : 1;
      })
      .attr("opacity", d => {
        const opts = this.options;
        const highlight = this.effectiveHighlightSet();
        if (opts.dimOther && highlight?.size && !highlight.has(d.id)) return 0.15;
        return 1;
      })
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        this.handleNodeClick(d.id, nodes, minProb);
      })
      .on("mouseover", (event, d) => this.showTooltip(event, d))
      .on("mousemove", event => {
        if (this.tooltip) {
          this.tooltip.style.left = `${event.clientX + 14}px`;
          this.tooltip.style.top = `${event.clientY + 14}px`;
        }
      })
      .on("mouseleave", () => { if (this.tooltip) this.tooltip.style.display = "none"; });

    const labels = this.gRoot.selectAll("text.node-label").data(nodeData, d => d.id);
    labels.exit().remove();
    labels.enter().append("text").attr("class", "node-label")
      .merge(labels)
      .attr("x", d => d.x)
      .attr("y", d => d.y + (this.isLeafNode(d) ? 14 : -12))
      .attr("text-anchor", "middle")
      .attr("fill", "#ccc")
      .attr("font-size", "10px")
      .attr("font-family", "ui-monospace, monospace")
      .attr("opacity", d => {
        const opts = this.options;
        const highlight = this.effectiveHighlightSet();
        if (opts.dimOther && highlight?.size && !highlight.has(d.id)) return 0.15;
        return 1;
      })
      .text(d => this.nodeLabel(d));
  }

  focusNode(nodeId, nodes, minProb = 0) {
    if (this.options.dimOther) this.setDimFocus(nodeId);
    else this.selectedNodeId = nodeId;
    const pos = this.layoutPositions.get(nodeId) || this.computeLayout(nodes).positions.get(nodeId);
    if (!pos) return;
    const width = this.svg.node().clientWidth;
    const height = this.svg.node().clientHeight;
    const scale = 1.4;
    this.svg.transition().duration(400).call(
      this.zoomBehavior.transform,
      d3.zoomIdentity.translate(width / 2, height / 2).scale(scale).translate(-pos.x, -pos.y)
    );
    this.draw(nodes, minProb);
  }

  showTooltip(event, node) {
    if (!this.tooltip) return;
    const leaf = this.leafMap.get(node.id);
    const stats = this.statsMap.get(node.id);
    const status = this.statusMap.get(node.id);
    let html = `<div class="tok">${utils.escapeHtml(utils.displayToken(this.nodeToken(node)))}</div>`;
    html += `<div>id: ${utils.escapeHtml(node.id)} · depth ${node.d ?? "?"}</div>`;
    if (node.p != null) html += `<div>p=${Number(node.p).toFixed(2)}</div>`;
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
    this.clearDimFocus();
  }
};
})(TreeDashboard);
