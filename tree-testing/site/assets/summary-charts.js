window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const { utils } = TD;

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
  const margin = { top: 12, right: 12, bottom: 48, left: 52 };
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
  const x = d3.scaleLinear().domain(d3.extent(data, d => d[xKey])).nice().range([0, innerW]);
  const y = d3.scaleLinear().domain(d3.extent(data, d => d[yKey])).nice().range([innerH, 0]);
  g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6)).attr("color", "#555");
  g.append("g").call(d3.axisLeft(y).ticks(6)).attr("color", "#555");
  g.selectAll(".dot").data(data).join("circle")
    .attr("cx", d => x(d[xKey])).attr("cy", d => y(d[yKey])).attr("r", 6)
    .attr("fill", "var(--accent)").attr("fill-opacity", 0.85).attr("stroke", "#fff")
    .append("title").text(d => d.label);
  if (titleEl) titleEl.textContent = `${yLabel || yKey} vs ${xLabel || xKey}`;
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
