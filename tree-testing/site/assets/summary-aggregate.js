window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
function emptyBucket() {
  return {
    trees: 0, bad_nodes_analyzed: 0, max_depth: 0, total_nodes: 0, leaf_count: 0,
    mass_above_tau: 0, max_breadth: 0, mean_breadth_by_depth: 0, breadth_warnings: 0,
    reasoning_tokens: 0, mean_entropy: 0, mean_logprob: 0, exclusively_bad: 0, ditched: 0,
    candidates: 0, probe_leaves: 0, leaf_correct: 0, prob_mass_total: 0, prob_good: 0,
    prob_bad: 0, prob_other: 0, greedy_correct: 0, top_1_correct: 0, top_k_correct: 0,
    _breadth_depth_n: 0, _mass_n: 0, _entropy_n: 0, _logprob_n: 0, _bad_metric_n: 0,
    nodes_expanded: 0, nodes_ditched_after: 0, tau_star_sum: 0, _tau_star_n: 0, _expansion_n: 0,
  };
}

function probMassForSummaryRow(row) {
  return {
    totalMass: Number(row.prob_mass_total) || 0,
    probGood: Number(row.prob_good) || 0,
    probBad: Number(row.prob_bad) || 0,
    probOther: Number(row.prob_other) || 0,
  };
}

function addToBucket(bucket, row) {
  bucket.trees += 1;
  if (row.max_depth != null) bucket.max_depth += row.max_depth;
  if (row.total_nodes != null) bucket.total_nodes += row.total_nodes;
  if (row.leaf_count != null) bucket.leaf_count += row.leaf_count;
  if (row.mass_above_tau != null) { bucket.mass_above_tau += row.mass_above_tau; bucket._mass_n += 1; }
  if (row.max_breadth != null) bucket.max_breadth += row.max_breadth;
  if (row.mean_breadth_by_depth != null) {
    bucket.mean_breadth_by_depth += row.mean_breadth_by_depth;
    bucket._breadth_depth_n += 1;
  }
  if (row.breadth_warning_count != null) bucket.breadth_warnings += row.breadth_warning_count;
  if (row.reasoning_token_count != null) bucket.reasoning_tokens += row.reasoning_token_count;
  if (row.mean_entropy_reasoning != null) { bucket.mean_entropy += row.mean_entropy_reasoning; bucket._entropy_n += 1; }
  if (row.mean_logprob_selected != null) { bucket.mean_logprob += row.mean_logprob_selected; bucket._logprob_n += 1; }
  bucket.greedy_correct += row.greedy_correct ? 1 : 0;
  bucket.top_1_correct += row.top_1_correct ? 1 : 0;
  bucket.top_k_correct += row.top_k_any_correct ? 1 : 0;
  if (row.has_bad_nodes) {
    bucket.bad_nodes_analyzed += 1;
    bucket._bad_metric_n += 1;
    bucket.exclusively_bad += row.exclusively_bad_count || 0;
    bucket.ditched += row.ditched_count || 0;
    bucket.candidates += row.total_candidates || 0;
    bucket.probe_leaves += row.total_leaves || 0;
    bucket.leaf_correct += row.leaf_correct || 0;
    const masses = probMassForSummaryRow(row);
    bucket.prob_mass_total += masses.totalMass;
    bucket.prob_good += masses.probGood;
    bucket.prob_bad += masses.probBad;
    bucket.prob_other += masses.probOther;
  }
  if (row.nodes_expanded != null) {
    bucket._expansion_n += 1;
    bucket.nodes_expanded += row.nodes_expanded || 0;
    bucket.nodes_ditched_after += row.nodes_ditched_after || 0;
    if (row.avg_tau_star != null) { bucket.tau_star_sum += row.avg_tau_star; bucket._tau_star_n += 1; }
  }
}

function finalizeBucket(key, label, bucket) {
  const trees = bucket.trees || 1;
  const badN = bucket._bad_metric_n || 0;
  const probeLeaves = bucket.probe_leaves || 0;
  const round2 = v => (v == null ? null : Math.round(v * 100) / 100);
  const round1 = v => (v == null ? null : Math.round(v * 10) / 10);
  return {
    key, label, trees: bucket.trees, bad_nodes_analyzed: bucket.bad_nodes_analyzed,
    avg_max_depth: round2(bucket.max_depth / trees),
    avg_total_nodes: round2(bucket.total_nodes / trees),
    avg_leaf_count: round2(bucket.leaf_count / trees),
    avg_mass_above_tau: bucket._mass_n ? round2(bucket.mass_above_tau / bucket._mass_n) : null,
    avg_exclusively_bad: badN ? round2(bucket.exclusively_bad / badN) : null,
    avg_ditched: badN ? round2(bucket.ditched / badN) : null,
    leaf_accuracy_pct: probeLeaves ? round1(100 * bucket.leaf_correct / probeLeaves) : null,
    prob_good_pct: bucket.prob_mass_total ? round1(100 * bucket.prob_good / bucket.prob_mass_total) : null,
    prob_bad_pct: bucket.prob_mass_total ? round1(100 * bucket.prob_bad / bucket.prob_mass_total) : null,
    prob_other_pct: bucket.prob_mass_total ? round1(100 * bucket.prob_other / bucket.prob_mass_total) : null,
    greedy_accuracy_pct: round1(100 * bucket.greedy_correct / trees),
    top_1_accuracy_pct: round1(100 * bucket.top_1_correct / trees),
    top_k_accuracy_pct: round1(100 * bucket.top_k_correct / trees),
    nodes_expanded: bucket._expansion_n ? round2(bucket.nodes_expanded / bucket._expansion_n) : null,
    avg_tau_star: bucket._tau_star_n ? round2(bucket.tau_star_sum / bucket._tau_star_n) : null,
  };
}

function groupMeta(row, dimension) {
  if (dimension === "whole") return { key: "all", label: "All (filtered)" };
  if (dimension === "by_region") return { key: row.region_id, label: row.region_label || row.region_id };
  if (dimension === "by_country") return { key: row.country_id, label: row.country_name || row.country_id };
  if (dimension === "by_prefix_length") return { key: String(row.prefix_length), label: `prefix ${row.prefix_length}` };
  if (dimension === "by_model") return { key: row.model_id, label: row.model_id };
  if (dimension === "by_instruction_variant") return { key: row.instruction_variant, label: row.instruction_variant };
  return { key: row.tree_key, label: row.prompt_preview || row.tree_key };
}

TD.aggregateSummaries = function(rows, dimension) {
  if (!rows.length) return [];
  if (dimension === "whole") {
    const bucket = emptyBucket();
    rows.forEach(row => addToBucket(bucket, row));
    return [finalizeBucket("all", "All (filtered)", bucket)];
  }
  const buckets = new Map();
  const labels = new Map();
  for (const row of rows) {
    const meta = groupMeta(row, dimension);
    if (!buckets.has(meta.key)) buckets.set(meta.key, emptyBucket());
    labels.set(meta.key, meta.label);
    addToBucket(buckets.get(meta.key), row);
  }
  return [...buckets.entries()]
    .map(([key, bucket]) => finalizeBucket(key, labels.get(key), bucket))
    .sort((a, b) => {
      if (dimension === "by_prefix_length") return Number(a.key) - Number(b.key);
      return String(a.label).localeCompare(String(b.label));
    });
};

TD.chartableLabel = function(row) {
  return row.label || row.country_name || row.prompt_preview || row.instruction_variant || row.tree_key || "—";
};
})(TreeDashboard);
