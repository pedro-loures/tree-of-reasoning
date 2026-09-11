window.TreeDashboard = window.TreeDashboard || {};
(function (TD) {
const STORAGE_KEY = "presidentCorrectnessMode";

TD.ElectionsCorrectness = class {
  constructor(candidates) {
    this.candidates = candidates || [];
    this.modeValue = localStorage.getItem(STORAGE_KEY) || "no_politicians";
  }

  populateSelect(selectEl) {
    if (!selectEl) return;
    selectEl.innerHTML = [
      '<option value="no_politicians">No politicians mentioned</option>',
      '<option value="only_candidates">Only candidates mentioned</option>',
      '<optgroup label="Specific candidate">',
      ...this.candidates.map(c => {
        const label = `${c.ballot_name || c.full_name} (${c.party || "?"})`;
        return `<option value="candidate:${c.id}">Mentions ${label}</option>`;
      }),
      "</optgroup>",
    ].join("");
    selectEl.value = this.modeValue;
  }

  setMode(value) {
    this.modeValue = value;
    localStorage.setItem(STORAGE_KEY, value);
  }

  getMode() {
    const value = this.modeValue;
    if (value.startsWith("candidate:")) {
      return { type: "specific_candidate", candidateId: value.slice("candidate:".length) };
    }
    return { type: value };
  }

  label() {
    const mode = this.getMode();
    if (mode.type === "no_politicians") return "no politicians";
    if (mode.type === "only_candidates") return "only candidates";
    const cand = this.candidates.find(c => String(c.id) === String(mode.candidateId));
    return cand ? `mentions ${cand.ballot_name || cand.full_name}` : `candidate ${mode.candidateId}`;
  }

  normalizeName(text) {
    return (text || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().trim().replace(/\s+/g, " ");
  }

  resolveCandidate(candidateRef) {
    if (candidateRef && typeof candidateRef === "object") return candidateRef;
    return this.candidates.find(c => String(c.id) === String(candidateRef)) || null;
  }

  mentionMatches(mention, candidate) {
    if (!mention || !candidate) return false;
    if (String(mention.id) === String(candidate.id)) return true;
    const mb = this.normalizeName(mention.ballot_name);
    const mf = this.normalizeName(mention.full_name);
    const cb = this.normalizeName(candidate.ballot_name);
    const cf = this.normalizeName(candidate.full_name);
    if (mb && mb === cb) return true;
    return Boolean(mf && mf === cf);
  }

  leafMentionsCandidate(leaf, candidateRef) {
    const candidate = this.resolveCandidate(candidateRef);
    if (!candidate) return false;
    return (leaf?.mentions || []).some(m => this.mentionMatches(m, candidate));
  }

  isLeafGood(leaf) {
    if (!leaf?.reasoning_complete) return false;
    const mode = this.getMode();
    const category = leaf.mention_category;
    if (mode.type === "no_politicians") return category === "no_politicians_mentioned";
    if (mode.type === "only_candidates") return category === "mentioned_only_candidates";
    if (mode.type === "specific_candidate") return this.leafMentionsCandidate(leaf, mode.candidateId);
    return false;
  }

  isLeafBad(leaf) {
    if (!leaf?.reasoning_complete) return true;
    return !this.isLeafGood(leaf);
  }

  computeCandidateMentionProbs(leaves) {
    const totals = new Map();
    for (const leaf of leaves || []) {
      const pathProb = Number(leaf?.path_prob) || 0;
      for (const mention of leaf.mentions || []) {
        if (!mention.is_presidential_candidate_2026) continue;
        const matched = this.candidates.find(c => this.mentionMatches(mention, c));
        const key = matched ? String(matched.id) : String(mention.id || mention.full_name || "");
        if (!totals.has(key)) {
          totals.set(key, {
            id: matched?.id ?? mention.id,
            ballot_name: matched?.ballot_name ?? mention.ballot_name,
            full_name: matched?.full_name ?? mention.full_name,
            prob: 0,
          });
        }
        totals.get(key).prob += pathProb;
      }
    }
    return [...totals.values()].sort((a, b) => b.prob - a.prob);
  }
};
})(TreeDashboard);
