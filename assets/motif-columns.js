window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { utils } = TD;

TD.motifLenKey = function(length) {
  return `motif_len_${length}`;
};

TD.collectMotifLengths = function(rows) {
  const lengths = new Set();
  for (const row of rows || []) {
    const byLen = row.motif_hits_by_length || {};
    for (const len of Object.keys(byLen)) lengths.add(Number(len));
  }
  return [...lengths].sort((a, b) => a - b);
};

TD.buildMotifColumns = function(lengths, { chartable = false } = {}) {
  return lengths.map(length => ({
    key: TD.motifLenKey(length),
    label: String(length),
    title: `Branch hits for motifs of length ${length}`,
    chartable,
    motifLength: length,
  }));
};

TD.expandMotifHits = function(row) {
  const expanded = { ...row };
  const byLen = row.motif_hits_by_length || {};
  for (const [length, hits] of Object.entries(byLen)) {
    expanded[TD.motifLenKey(length)] = hits;
  }
  return expanded;
};

TD.tableColumnsWithMotifs = function(baseColumns, rows, { chartable = false } = {}) {
  const base = (baseColumns || []).filter(col => !col.motifLength && !String(col.key || "").startsWith("motif_len_"));
  const motifCols = TD.buildMotifColumns(TD.collectMotifLengths(rows), { chartable });
  return motifCols.length ? [...base, ...motifCols] : base;
};

TD.groupColumnLabel = function(dimension) {
  const labels = {
    whole: "Group",
    by_region: "Region",
    by_country: "Country",
    by_prefix_length: "Prefix",
    by_model: "Model",
    by_instruction_variant: "Prompt",
  };
  return labels[dimension] || "Group";
};

function columnHeaderLabel(col, dimension) {
  if (col.key === "label") return TD.groupColumnLabel(dimension);
  return col.label;
}

function headerCell(col, dimension, { rowspan = null } = {}) {
  const label = columnHeaderLabel(col, dimension);
  const rowspanAttr = rowspan ? ` rowspan="${rowspan}"` : "";
  return `<th data-key="${col.key}" class="${col.highlight ? "highlight" : ""}${col.freezeDefault ? " frozen" : ""}" style="text-align:${col.align || "right"}" title="${utils.escapeHtml(col.title || "")}"${rowspanAttr}>${utils.escapeHtml(label)}</th>`;
}

TD.buildSummaryTableHead = function(columns, dimension = "whole") {
  const motifCols = columns.filter(col => col.motifLength);
  const baseCols = columns.filter(col => !col.motifLength);
  if (!motifCols.length) {
    return `<tr>${columns.map(col => headerCell(col, dimension)).join("")}</tr>`;
  }
  const top = [
    ...baseCols.map(col => headerCell(col, dimension, { rowspan: 2 })),
    `<th colspan="${motifCols.length}" class="motif-group-header" style="text-align:center">Motif hits</th>`,
  ].join("");
  const bottom = motifCols.map(col => headerCell(col, dimension)).join("");
  return `<tr>${top}</tr><tr>${bottom}</tr>`;
};

TD.renderSummaryTableBody = function(columns, rows) {
  return rows.map(row => `<tr class="clickable" data-group-key="${utils.escapeHtml(String(row.key ?? row.tree_key ?? ""))}">${columns.map(col =>
    `<td data-key="${col.key}" class="${col.highlight ? "highlight" : ""}${col.freezeDefault ? " frozen" : ""}" style="text-align:${col.align || "right"}">${utils.escapeHtml(String(utils.fmtVal(row[col.key])))}</td>`).join("")}</tr>`).join("");
};

TD.applyFrozenColumns = function(table, frozenKeys = new Set(["label"])) {
  if (!table) return;
  table.querySelectorAll(".frozen, .frozen-last").forEach(cell => {
    cell.classList.remove("frozen", "frozen-last");
    cell.style.left = "";
    cell.style.zIndex = "";
  });

  const keys = [...frozenKeys];
  let left = 0;
  keys.forEach((key, index) => {
    const header = table.querySelector(`thead th[data-key="${key}"]`);
    const cells = table.querySelectorAll(`tbody td[data-key="${key}"]`);
    if (!header && !cells.length) return;
    const isLast = index === keys.length - 1;
    const apply = cell => {
      cell.classList.add("frozen");
      if (isLast) cell.classList.add("frozen-last");
      cell.style.left = `${left}px`;
      cell.style.zIndex = cell.tagName === "TH" ? String(4 + index) : String(2 + index);
    };
    if (header) apply(header);
    cells.forEach(apply);
    left += header?.offsetWidth || cells[0]?.offsetWidth || 0;
  });
};
})(TreeDashboard);
