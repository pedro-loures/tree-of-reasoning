window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { utils } = TD;

function pearsonCorrelation(data, xKey, yKey) {
  const n = data.length;
  if (n < 2) return { r: null, p: null, n };
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;
  let sumY2 = 0;
  for (const row of data) {
    const x = Number(row[xKey]);
    const y = Number(row[yKey]);
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumX2 += x * x;
    sumY2 += y * y;
  }
  const num = n * sumXY - sumX * sumY;
  const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  if (!den) return { r: null, p: null, n };
  const r = num / den;
  return { r, p: correlationPValue(r, n), n };
}

function linearRegression(data, xKey, yKey) {
  const n = data.length;
  if (n < 2) return null;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;
  for (const row of data) {
    const x = Number(row[xKey]);
    const y = Number(row[yKey]);
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumX2 += x * x;
  }
  const denom = n * sumX2 - sumX * sumX;
  if (!denom) return null;
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  return { slope, intercept };
}

function correlationPValue(r, n) {
  if (n < 3 || r === null || !Number.isFinite(r)) return null;
  const absR = Math.min(Math.abs(r), 1 - 1e-15);
  const t = absR * Math.sqrt((n - 2) / (1 - absR * absR));
  return 2 * (1 - studentTCDF(t, n - 2));
}

function studentTCDF(t, df) {
  const x = df / (df + t * t);
  return 1 - 0.5 * regularizedIncompleteBeta(x, df / 2, 0.5);
}

function regularizedIncompleteBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lnBeta = logGamma(a) + logGamma(b) - logGamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lnBeta) / a;
  if (x < (a + 1) / (a + b + 2)) {
    return front * betaCF(x, a, b);
  }
  return 1 - (Math.exp(Math.log(1 - x) * b + Math.log(x) * a - lnBeta) / b) * betaCF(1 - x, b, a);
}

function betaCF(x, a, b) {
  const maxIter = 200;
  const eps = 3e-7;
  let am = 1;
  let bm = 1;
  let az = 1;
  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let bz = 1 - qab * x / qap;
  for (let m = 1; m <= maxIter; m += 1) {
    const em = m;
    let tem = em + em;
    let d = em * (b - em) * x / ((qam + tem) * (a + tem));
    am = 1 + d * am;
    bm = 1 + d * bm;
    d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem));
    az = 1 + d * az;
    bz = 1 + d * bz;
    const old = az / bz;
    am = 1 / am;
    bm = 1 / bm;
    az *= am;
    bz *= bm;
    if (Math.abs(old - az / bz) < eps * Math.abs(az / bz)) {
      return az / bz;
    }
  }
  return az / bz;
}

const logGammaCache = new Map();
function logGamma(z) {
  const key = z.toFixed(6);
  if (logGammaCache.has(key)) return logGammaCache.get(key);
  const g = 7;
  const coef = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.984369578019571e-4, 1.18559624731541e-5,
  ];
  if (z < 0.5) {
    const value = Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
    logGammaCache.set(key, value);
    return value;
  }
  z -= 1;
  let x = coef[0];
  for (let i = 1; i < g + 2; i += 1) x += coef[i] / (z + i);
  const t = z + g + 0.5;
  const value = 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  logGammaCache.set(key, value);
  return value;
}

function formatPValue(p) {
  if (p === null || p === undefined || !Number.isFinite(p)) return "—";
  if (p < 0.001) return "<0.001";
  if (p < 0.01) return p.toFixed(3);
  return p.toFixed(2);
}

function formatCorrelation(r) {
  if (r === null || r === undefined || !Number.isFinite(r)) return "—";
  return (Math.round(r * 1000) / 1000).toFixed(3);
}

TD.scatterStats = function(data, xKey, yKey) {
  return {
    ...pearsonCorrelation(data, xKey, yKey),
    regression: linearRegression(data, xKey, yKey),
  };
};

TD.formatScatterStats = function(stats) {
  if (!stats || stats.r === null) return "";
  const sig = stats.p != null && stats.p < 0.05 ? "*" : "";
  return `r=${formatCorrelation(stats.r)}${sig}, p=${formatPValue(stats.p)}, n=${stats.n}`;
};

TD.SCATTER_NUMERIC_COLUMNS = [
  { key: "max_depth", label: "Depth" },
  { key: "total_nodes", label: "Nodes" },
  { key: "leaf_count", label: "τ-leaves" },
  { key: "max_breadth", label: "Max breadth" },
  { key: "mean_breadth_by_depth", label: "Breadth/depth" },
  { key: "breadth_warning_count", label: "Breadth warns" },
  { key: "mass_above_tau", label: "Mass above τ" },
  { key: "exclusively_bad_count", label: "Excl. bad" },
  { key: "leaf_correct_pct", label: "Leaf good %" },
  { key: "prob_good_pct", label: "P(good) %" },
  { key: "prob_bad_pct", label: "P(bad) %" },
  { key: "prob_other_pct", label: "P(other) %" },
  { key: "tau", label: "τ" },
];

TD.renderBarChart = function(svgSelector, titleSelector, rows, metricKey, metricLabel) {
  const svgEl = d3.select(svgSelector);
  svgEl.selectAll("*").remove();
  const width = svgEl.node()?.clientWidth || 600;
  const height = 280;
  const margin = { top: 12, right: 12, bottom: 80, left: 48 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const g = svgEl.attr("viewBox", `0 0 ${width} ${height}`).append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);
  const data = rows.filter(r => r[metricKey] !== null && r[metricKey] !== undefined);
  const titleEl = document.querySelector(titleSelector);
  if (!data.length) {
    if (titleEl) titleEl.textContent = "No data for selected metric";
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
  if (titleEl) titleEl.textContent = `${metricLabel || metricKey} by group`;
};

TD.renderScatterChart = function(svgSelector, titleSelector, rows, xKey, yKey, xLabel, yLabel) {
  const svgEl = d3.select(svgSelector);
  svgEl.selectAll("*").remove();
  const width = svgEl.node()?.clientWidth || 600;
  const height = 280;
  const margin = { top: 28, right: 12, bottom: 48, left: 52 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const g = svgEl.attr("viewBox", `0 0 ${width} ${height}`).append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);
  const data = rows.filter(r => r[xKey] != null && r[yKey] != null);
  const titleEl = document.querySelector(titleSelector);
  if (!data.length) {
    if (titleEl) titleEl.textContent = "No data for scatter";
    return;
  }
  const x = d3.scaleLinear().domain(d3.extent(data, d => Number(d[xKey]))).nice().range([0, innerW]);
  const y = d3.scaleLinear().domain(d3.extent(data, d => Number(d[yKey]))).nice().range([innerH, 0]);
  g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6)).attr("color", "#555");
  g.append("g").call(d3.axisLeft(y).ticks(6)).attr("color", "#555");
  g.append("text")
    .attr("x", innerW / 2).attr("y", innerH + 36)
    .attr("text-anchor", "middle").attr("fill", "#9aa0a6").attr("font-size", "11px")
    .text(xLabel || xKey);
  g.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -innerH / 2).attr("y", -40)
    .attr("text-anchor", "middle").attr("fill", "#9aa0a6").attr("font-size", "11px")
    .text(yLabel || yKey);

  const stats = TD.scatterStats(data, xKey, yKey);
  const reg = stats.regression;
  if (reg && data.length >= 2) {
    const [x0, x1] = x.domain();
    const line = d3.line()
      .x(v => x(v))
      .y(v => y(reg.slope * v + reg.intercept));
    g.append("path")
      .attr("d", line([x0, x1]))
      .attr("fill", "none")
      .attr("stroke", "#f4a261")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "6,4");
  }

  g.selectAll(".dot").data(data).join("circle")
    .attr("cx", d => x(Number(d[xKey]))).attr("cy", d => y(Number(d[yKey]))).attr("r", 6)
    .attr("fill", "var(--accent)").attr("fill-opacity", 0.85).attr("stroke", "#fff")
    .append("title").text(d => d.label || d.prompt_preview || d.country_name || "");

  const statsText = TD.formatScatterStats(stats);
  if (statsText) {
    g.append("text")
      .attr("x", innerW).attr("y", -10)
      .attr("text-anchor", "end")
      .attr("fill", "#cfe0ff")
      .attr("font-size", "11px")
      .text(statsText);
  }

  if (titleEl) {
    titleEl.textContent = statsText
      ? `${yLabel || yKey} vs ${xLabel || xKey} · ${statsText}`
      : `${yLabel || yKey} vs ${xLabel || xKey}`;
  }
};

TD.renderHorizontalBarChart = function(svgSelector, titleSelector, items, valueKey, labelKey) {
  const svgEl = d3.select(svgSelector);
  svgEl.selectAll("*").remove();
  const width = svgEl.node()?.clientWidth || 400;
  const height = Math.max(160, items.length * 28 + 40);
  const margin = { top: 12, right: 12, bottom: 12, left: 140 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const g = svgEl.attr("viewBox", `0 0 ${width} ${height}`).append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);
  const titleEl = document.querySelector(titleSelector);
  if (!items.length) {
    if (titleEl) titleEl.textContent = "No candidate data";
    return;
  }
  const y = d3.scaleBand().domain(items.map(d => d[labelKey])).range([0, innerH]).padding(0.2);
  const x = d3.scaleLinear().domain([0, d3.max(items, d => d[valueKey]) || 1]).nice().range([0, innerW]);
  g.append("g").call(d3.axisLeft(y)).attr("color", "#9aa0a6");
  g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(5)).attr("color", "#555");
  g.selectAll(".bar").data(items).join("rect")
    .attr("y", d => y(d[labelKey])).attr("x", 0)
    .attr("width", d => x(d[valueKey])).attr("height", y.bandwidth())
    .attr("fill", "var(--accent)").attr("rx", 2);
  if (titleEl) titleEl.textContent = titleEl.dataset.defaultTitle || "Candidate mention probs";
};
})(TreeDashboard);
