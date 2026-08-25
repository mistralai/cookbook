// ---- Path configuration ----
let customBasePath = localStorage.getItem('comparisonDataBasePath') || '';

function togglePathConfig() {
  const body = document.getElementById('pathConfigBody');
  body.classList.toggle('expanded');
}
function updateBasePath() {
  customBasePath = document.getElementById('dataBasePath').value.trim();
  localStorage.setItem('comparisonDataBasePath', customBasePath);
  document.getElementById('currentBasePath').textContent = customBasePath || '(using original paths)';
}
function resolveFilePath(originalPath) {
  if (!originalPath) return '';
  if (!customBasePath || !originalBasePath) return originalPath;
  if (originalPath.startsWith(originalBasePath)) {
    return customBasePath + originalPath.slice(originalBasePath.length);
  }
  return originalPath;
}
document.addEventListener('DOMContentLoaded', () => {
  if (customBasePath) {
    document.getElementById('dataBasePath').value = customBasePath;
    document.getElementById('currentBasePath').textContent = customBasePath;
  }
  populateMetricSelector();
});

// ---- Metric selector ----
let currentFilter = 'all';

function populateMetricSelector() {
  const select = document.getElementById('metricSelect');
  if (!select) return;
  // Discover all metric names across all results
  const names = new Set();
  comparisonData.forEach(r => {
    (r.pipeline_a.all_metrics || []).forEach(m => names.add(m.metric_name));
    (r.pipeline_b.all_metrics || []).forEach(m => names.add(m.metric_name));
  });
  // Sort with current primary first, then alphabetical
  const sorted = [...names].sort((a, b) => {
    if (a === comparisonMetric) return -1;
    if (b === comparisonMetric) return 1;
    return a.localeCompare(b);
  });
  select.innerHTML = '';
  sorted.forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = metricDisplayNames[name] || name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    if (name === comparisonMetric) opt.selected = true;
    select.appendChild(opt);
  });
}

function getMetricValue(allMetrics, metricName) {
  if (!allMetrics) return null;
  const m = allMetrics.find(m => m.metric_name === metricName);
  return m ? m.value : null;
}

function switchMetric(newMetric) {
  comparisonMetric = newMetric;
  metricDisplayName = metricDisplayNames[newMetric] || newMetric;

  // Recompute categories and metric values for each row
  const counts = { total: comparisonData.length, a_better: 0, b_better: 0, tie: 0, both_bad: 0 };

  comparisonData.forEach((r, i) => {
    const vA = getMetricValue(r.pipeline_a.all_metrics, newMetric);
    const vB = getMetricValue(r.pipeline_b.all_metrics, newMetric);

    // Update metric_value on the data so detail panes use it
    r.pipeline_a.metric_value = vA;
    r.pipeline_b.metric_value = vB;

    // Recompute category
    let cat;
    if (vA != null && vB != null) {
      cat = vA > vB ? 'a_better' : vB > vA ? 'b_better' : 'tie';
    } else if (vA == null && vB == null) {
      cat = 'both_bad';
    } else if (vA == null) {
      cat = 'b_better';
    } else {
      cat = 'a_better';
    }
    r.category = cat;
    counts[cat] = (counts[cat] || 0) + 1;

    // Update DOM for this row
    const row = document.querySelector(`.result-row[data-index="${i}"]`);
    if (!row) return;
    row.dataset.category = cat;

    const cols = row.querySelector('.row-summary');
    if (!cols) return;
    const metricCols = cols.querySelectorAll('.col-metric');
    const deltaCol = cols.querySelector('.col-delta');
    const catCol = cols.querySelector('.col-category');

    if (metricCols[0]) metricCols[0].innerHTML = fmtMetric(vA);
    if (metricCols[1]) metricCols[1].innerHTML = fmtMetric(vB);
    if (deltaCol) deltaCol.innerHTML = fmtDelta(vA, vB);
    if (catCol) {
      const catLabels = {
        a_better: pipelineAName + ' Better',
        b_better: pipelineBName + ' Better',
        tie: 'Tie',
        both_bad: 'Both Bad',
      };
      catCol.innerHTML = `<span class="badge badge-${cat.replace('_', '-')}">${esc(catLabels[cat] || cat)}</span>`;
    }

    // If detail pane is open, rebuild it
    const detail = document.getElementById('detail-' + i);
    if (detail && detail.classList.contains('open')) {
      detail.innerHTML = buildDetailContent(i);
      initDetailInteractions(i);
    }
  });

  // Update stat cards
  const statCards = document.querySelectorAll('.stat-card');
  statCards.forEach(card => {
    const filter = card.dataset.filter;
    const valueEl = card.querySelector('.stat-value');
    if (!valueEl) return;
    if (filter === 'all') valueEl.textContent = counts.total;
    else if (counts[filter] !== undefined) valueEl.textContent = counts[filter];
  });

  // Update filter bar
  document.querySelectorAll('.filter-btn').forEach(btn => {
    const filter = btn.dataset.filter;
    if (filter === 'all') btn.textContent = `All (${counts.total})`;
    else if (filter === 'a_better') btn.textContent = `${pipelineAName} Better (${counts.a_better})`;
    else if (filter === 'b_better') btn.textContent = `${pipelineBName} Better (${counts.b_better})`;
    else if (filter === 'tie') btn.textContent = `Tie (${counts.tie})`;
    else if (filter === 'both_bad') btn.textContent = `Both Bad (${counts.both_bad})`;
  });

  // Re-apply current filter
  applyFilter(currentFilter);
}

function fmtMetric(val) {
  if (val === null || val === undefined) return '<span class="na">N/A</span>';
  const pct = val * 100;
  return `<span class="metric-val ${metricColorClass(val)}">${pct.toFixed(1)}%</span>`;
}

function fmtDelta(a, b) {
  if (a === null || a === undefined || b === null || b === undefined) return '<span class="na">&mdash;</span>';
  const d = (a - b) * 100;
  const sign = d > 0 ? '+' : '';
  const cls = d > 0 ? 'delta-pos' : d < 0 ? 'delta-neg' : 'delta-zero';
  return `<span class="${cls}">${sign}${d.toFixed(1)}pp</span>`;
}

// ---- Filter ----
function applyFilter(filter) {
  currentFilter = filter;
  // Update filter buttons
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  const matchBtn = document.querySelector(`.filter-btn[data-filter="${filter}"]`);
  if (matchBtn) matchBtn.classList.add('active');
  // Update stat cards
  document.querySelectorAll('.stat-card').forEach(c => c.classList.remove('active'));
  const matchCard = document.querySelector(`.stat-card[data-filter="${filter}"]`);
  if (matchCard) matchCard.classList.add('active');
  // Filter rows
  document.querySelectorAll('.result-row').forEach(row => {
    row.style.display = (filter === 'all' || row.dataset.category === filter) ? '' : 'none';
  });
}

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => applyFilter(btn.dataset.filter));
});

function filterFromCard(card) {
  applyFilter(card.dataset.filter);
}

// ---- Expand / collapse rows ----
const openRows = new Set();

function toggleRow(index) {
  const detail = document.getElementById('detail-' + index);
  const icon = document.getElementById('icon-' + index);
  if (openRows.has(index)) {
    detail.classList.remove('open');
    icon.classList.remove('expanded');
    openRows.delete(index);
  } else {
    // Build detail content if empty
    if (!detail.innerHTML.trim()) {
      detail.innerHTML = buildDetailContent(index);
      initDetailInteractions(index);
    }
    detail.classList.add('open');
    icon.classList.add('expanded');
    openRows.add(index);
  }
}

// ---- Build detail content ----
function buildDetailContent(index) {
  const r = comparisonData[index];
  if (!r) return '<p>No data</p>';

  if (productType === 'layout_detection') {
    return buildLayoutDetectionDetail(r, index);
  }

  let html = '<div class="detail-two-col">';

  // --- Left column: input preview (sticky) ---
  html += '<div class="detail-left">';
  html += buildInputPanel(r, index);
  html += buildMetricPills(r);
  html += '</div>';

  // --- Right column: output + metrics + rules ---
  html += '<div class="detail-right">';
  html += buildOutputTab(r, index);

  // Collapsible full metrics
  const fullMetrics = buildFullMetricsSection(r);
  if (fullMetrics) {
    html += `<details class="collapsible-section"><summary>All Metrics</summary>${fullMetrics}</details>`;
  }

  // Collapsible rules
  const rulesHtml = buildRulesSection(r);
  if (rulesHtml) {
    html += `<details class="collapsible-section"><summary>${rulesHtml.summary}</summary>${rulesHtml.body}</details>`;
  }

  html += '</div>';
  html += '</div>';

  return html;
}

// Compact metric pills for the left column
function buildMetricPills(r) {
  const metricsA = r.pipeline_a.all_metrics || [];
  const metricsB = r.pipeline_b.all_metrics || [];
  const statsA = r.pipeline_a.all_stats || [];
  const statsB = r.pipeline_b.all_stats || [];
  if (metricsA.length === 0 && metricsB.length === 0 && statsA.length === 0 && statsB.length === 0) return '';

  // Collect all metric names, primary first
  const names = new Set();
  metricsA.forEach(m => names.add(m.metric_name));
  metricsB.forEach(m => names.add(m.metric_name));
  const sorted = [...names].sort((a, b) => {
    if (a === comparisonMetric) return -1;
    if (b === comparisonMetric) return 1;
    return a.localeCompare(b);
  });

  let html = '<div class="metric-pills">';
  sorted.forEach(name => {
    const mA = metricsA.find(m => m.metric_name === name);
    const mB = metricsB.find(m => m.metric_name === name);
    const vA = mA ? mA.value : null;
    const vB = mB ? mB.value : null;
    const displayName = metricDisplayNames[name] || name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const isPrimary = name === comparisonMetric;

    const fmtVal = v => v === null || v === undefined ? 'N/A' : (v * 100).toFixed(1) + '%';
    let deltaHtml = '';
    if (vA != null && vB != null) {
      const d = (vA - vB) * 100;
      const sign = d > 0 ? '+' : '';
      const cls = d > 0 ? 'delta-pos' : d < 0 ? 'delta-neg' : 'delta-zero';
      deltaHtml = `<span class="pill-delta ${cls}">${sign}${d.toFixed(1)}</span>`;
    }

    html += `<div class="pill${isPrimary ? ' pill-primary' : ''}">`;
    html += `<span class="pill-label">${esc(displayName)}${tooltipIcon(name)}</span>`;
    html += `<span class="pill-values"><span class="${metricColorClass(vA)}">${fmtVal(vA)}</span> / <span class="${metricColorClass(vB)}">${fmtVal(vB)}</span>${deltaHtml}</span>`;
    html += '</div>';
  });

  // Stats pills (raw value + unit, not percentage)
  const statNames = new Set();
  statsA.forEach(s => statNames.add(s.name));
  statsB.forEach(s => statNames.add(s.name));
  [...statNames].sort().forEach(name => {
    const sA = statsA.find(s => s.name === name);
    const sB = statsB.find(s => s.name === name);
    const vA = sA ? sA.value : null;
    const vB = sB ? sB.value : null;
    const unit = (sA || sB || {}).unit || '';
    const displayName = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const fmtStat = v => v === null || v === undefined ? 'N/A' : v.toFixed(0) + unit;
    let deltaHtml = '';
    if (vA != null && vB != null) {
      const d = vA - vB;
      const sign = d > 0 ? '+' : '';
      const cls = d > 0 ? 'delta-neg' : d < 0 ? 'delta-pos' : 'delta-zero';
      deltaHtml = `<span class="pill-delta ${cls}">${sign}${d.toFixed(0)}${unit}</span>`;
    }
    html += `<div class="pill">`;
    html += `<span class="pill-label">${esc(displayName)}${tooltipIcon(name)}</span>`;
    html += `<span class="pill-values">${fmtStat(vA)} / ${fmtStat(vB)}${deltaHtml}</span>`;
    html += '</div>';
  });

  html += '</div>';
  return html;
}

// Input panel for left column
function buildInputPanel(r, index) {
  return buildInputTab(r, index);
}

// Full metrics table (for collapsible section)
function buildFullMetricsSection(r) {
  const metricsA = r.pipeline_a.all_metrics || [];
  const metricsB = r.pipeline_b.all_metrics || [];
  const statsA = r.pipeline_a.all_stats || [];
  const statsB = r.pipeline_b.all_stats || [];
  const metricNames = new Set();
  metricsA.forEach(m => metricNames.add(m.metric_name));
  metricsB.forEach(m => metricNames.add(m.metric_name));
  const statNames = new Set();
  statsA.forEach(s => statNames.add(s.name));
  statsB.forEach(s => statNames.add(s.name));
  if (metricNames.size === 0 && statNames.size === 0) return '';

  const sorted = [...metricNames].sort((a, b) => {
    if (a === comparisonMetric) return -1;
    if (b === comparisonMetric) return 1;
    return a.localeCompare(b);
  });

  let html = `<table class="metrics-table"><thead><tr><th>Metric</th><th style="text-align:center">${esc(pipelineAName)}</th><th style="text-align:center">${esc(pipelineBName)}</th><th style="text-align:center">Delta</th></tr></thead><tbody>`;

  sorted.forEach(name => {
    const mA = metricsA.find(m => m.metric_name === name);
    const mB = metricsB.find(m => m.metric_name === name);
    const vA = mA ? mA.value : null;
    const vB = mB ? mB.value : null;
    const displayName = metricDisplayNames[name] || name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const fmtVal = v => v === null || v === undefined ? '<span class="na">N/A</span>' : `<span class="${metricColorClass(v)}">${(v * 100).toFixed(1)}%</span>`;
    const delta = (vA != null && vB != null)
      ? (() => { const d = (vA - vB) * 100; const sign = d > 0 ? '+' : ''; const cls = d > 0 ? 'delta-pos' : d < 0 ? 'delta-neg' : 'delta-zero'; return `<span class="${cls}">${sign}${d.toFixed(1)}pp</span>`; })()
      : '<span class="na">&mdash;</span>';
    const isPrimary = name === comparisonMetric ? ' style="font-weight:600;"' : '';
    html += `<tr${isPrimary}><td class="metric-name">${esc(displayName)}${tooltipIcon(name)}</td><td class="val-cell">${fmtVal(vA)}</td><td class="val-cell">${fmtVal(vB)}</td><td class="val-cell">${delta}</td></tr>`;
  });

  html += '</tbody></table>';

  // Stats table (raw value + unit)
  if (statNames.size > 0) {
    html += `<table class="metrics-table" style="margin-top:10px"><thead><tr><th>Stat</th><th style="text-align:center">${esc(pipelineAName)}</th><th style="text-align:center">${esc(pipelineBName)}</th><th style="text-align:center">Delta</th></tr></thead><tbody>`;
    [...statNames].sort().forEach(name => {
      const sA = statsA.find(s => s.name === name);
      const sB = statsB.find(s => s.name === name);
      const vA = sA ? sA.value : null;
      const vB = sB ? sB.value : null;
      const unit = (sA || sB || {}).unit || '';
      const displayName = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      const fmtStat = v => v === null || v === undefined ? '<span class="na">N/A</span>' : v.toFixed(0) + unit;
      const delta = (vA != null && vB != null)
        ? (() => { const d = vA - vB; const sign = d > 0 ? '+' : ''; const cls = d > 0 ? 'delta-neg' : d < 0 ? 'delta-pos' : 'delta-zero'; return `<span class="${cls}">${sign}${d.toFixed(0)}${unit}</span>`; })()
        : '<span class="na">&mdash;</span>';
      html += `<tr><td class="metric-name">${esc(displayName)}</td><td class="val-cell">${fmtStat(vA)}</td><td class="val-cell">${fmtStat(vB)}</td><td class="val-cell">${delta}</td></tr>`;
    });
    html += '</tbody></table>';
  }

  return html;
}

// Normalize parse + extract rule_results for the comparison Rules table.
// Parse rules use type/name/page/passed/explanation; extract evidence rules use
// mode/field_path/value_pass/reason (and bbox_iou_pass is sometimes an IoU float).
function rulePass(rule) {
  if (!rule) return undefined;
  if (typeof rule.passed === 'boolean') return rule.passed;
  for (const key of ['value_pass', 'element_pass', 'page_pass', 'bbox_covered_pass', 'loc_pass', 'attr_pass']) {
    if (typeof rule[key] === 'boolean') return rule[key];
  }
  if (typeof rule.bbox_iou_pass === 'boolean') return rule.bbox_iou_pass;
  if (rule.bbox_iou_pass === 0 || rule.bbox_iou_pass === 1) return !!rule.bbox_iou_pass;
  return undefined;
}

function ruleType(rule) {
  return (rule && (rule.type || rule.mode)) || '';
}

function ruleName(rule) {
  return (rule && (rule.name || rule.field_path || rule.path || rule.id)) || '';
}

function rulePage(rule) {
  return (rule && rule.page != null && rule.page !== '') ? rule.page : '';
}

function ruleExplanation(rule) {
  return (rule && (rule.explanation || rule.message || rule.reason)) || '';
}

/** Short labels for narrow pipeline columns; full names stay in title. */
function shortPipelineLabels(nameA, nameB) {
  const a = String(nameA || '');
  const b = String(nameB || '');
  if (!a || !b || a === b) return [a || 'A', b || 'B'];
  let i = 0;
  const lim = Math.min(a.length, b.length);
  while (i < lim && a[i] === b[i]) i++;
  // Prefer cutting on a separator so labels stay readable.
  while (i > 0 && !/[-_./]/.test(a[i - 1])) i--;
  const shortA = (i > 0 ? a.slice(i) : a).replace(/^[-_./]+/, '');
  const shortB = (i > 0 ? b.slice(i) : b).replace(/^[-_./]+/, '');
  if (!shortA || !shortB) return [a, b];
  return [shortA, shortB];
}

// Rules section (returns {summary, body} or null)
function buildRulesSection(r) {
  const metricsA = r.pipeline_a.all_metrics || [];
  const metricsB = r.pipeline_b.all_metrics || [];

  function getRules(metrics) {
    for (const m of metrics) {
      if (m.metric_name === 'rule_pass_rate' && m.metadata && m.metadata.rule_results) return m.metadata.rule_results;
    }
    for (const m of metrics) {
      if (m.metadata && m.metadata.rule_results) return m.metadata.rule_results;
    }
    return [];
  }

  const rulesA = getRules(metricsA);
  const rulesB = getRules(metricsB);
  if (rulesA.length === 0 && rulesB.length === 0) return null;

  const maxLen = Math.max(rulesA.length, rulesB.length);
  let diffCount = 0;
  for (let i = 0; i < maxLen; i++) {
    const pA = rulePass(rulesA[i]);
    const pB = rulePass(rulesB[i]);
    if (pA !== undefined && pB !== undefined && pA !== pB) diffCount++;
  }

  // Extract evidence rules only expose page_pass/page_qualified — no page number.
  // Omit the Page column when neither side has any page values (parse rules do).
  let showPage = false;
  for (let i = 0; i < maxLen; i++) {
    if (rulePage(rulesA[i]) !== '' || rulePage(rulesB[i]) !== '') {
      showPage = true;
      break;
    }
  }
  const colCount = showPage ? 7 : 6;

  // Plain text only: summary is display:flex with gap, so inner tags become
  // separate flex items and look like extra spaces around the count.
  const summary = diffCount > 0
    ? `Rules (${maxLen} total, ${diffCount} differ)`
    : `Rules (${maxLen} total)`;
  const [labelA, labelB] = shortPipelineLabels(pipelineAName, pipelineBName);

  let body = `<table class="rule-table"><thead><tr>` +
    `<th class="rule-idx">#</th><th class="rule-type">Type</th><th class="rule-name">Name</th>` +
    (showPage ? `<th class="rule-page">Page</th>` : '') +
    `<th class="rule-pipeline" title="${esc(pipelineAName)}"><span>${esc(labelA)}</span></th>` +
    `<th class="rule-pipeline" title="${esc(pipelineBName)}"><span>${esc(labelB)}</span></th>` +
    `<th class="rule-details"></th></tr></thead><tbody>`;

  for (let i = 0; i < maxLen; i++) {
    const rA = rulesA[i] || {};
    const rB = rulesB[i] || {};
    const passA = rulePass(rA);
    const passB = rulePass(rB);
    let rowClass = '';
    if (passA !== undefined && passB !== undefined) {
      rowClass = passA === passB ? (passA ? 'rule-same-pass' : 'rule-same-fail') : 'rule-diff-row';
    }
    const type = ruleType(rA) || ruleType(rB);
    const name = ruleName(rA) || ruleName(rB);
    const page = rulePage(rA) || rulePage(rB);
    const fmtPass = v => v === true ? '<span class="rule-pass">PASS</span>' : v === false ? '<span class="rule-fail">FAIL</span>' : '<span class="na">—</span>';
    const explA = ruleExplanation(rA);
    const explB = ruleExplanation(rB);
    const hasExpl = explA || explB;
    body += `<tr class="${rowClass}"><td class="rule-idx">${i + 1}</td><td class="rule-type" title="${esc(type)}">${esc(type)}</td>` +
      `<td class="rule-name" title="${esc(name)}">${esc(name)}</td>` +
      (showPage ? `<td class="rule-page">${page || ''}</td>` : '') +
      `<td class="rule-pipeline">${fmtPass(passA)}</td>` +
      `<td class="rule-pipeline">${fmtPass(passB)}</td>` +
      `<td class="rule-details">${hasExpl ? '<span class="rule-expander" onclick="toggleRuleExpl(this)">details</span>' : ''}</td></tr>`;
    if (hasExpl) {
      body += `<tr class="${rowClass}" style="display:none;" data-expl-row="1"><td colspan="${colCount}" style="padding:0.25rem 0.6rem;"><div style="font-size:0.75rem;color:var(--muted);">`;
      if (explA) body += `<div><strong>${esc(pipelineAName)}:</strong> ${esc(explA)}</div>`;
      if (explB) body += `<div><strong>${esc(pipelineBName)}:</strong> ${esc(explB)}</div>`;
      body += '</div></td></tr>';
    }
  }
  body += '</tbody></table>';

  return { summary, body };
}

function toggleRuleExpl(el) {
  const tr = el.closest('tr');
  const next = tr.nextElementSibling;
  if (next && next.dataset.explRow) {
    next.style.display = next.style.display === 'none' ? '' : 'none';
  }
}

// ---- Output tab ----
function buildOutputTab(r, index) {
  if (productType === 'parse') {
    return buildParseOutputTab(r, index);
  } else if (productType === 'extract') {
    return buildExtractOutputTab(r, index);
  }
  return '<p class="na" style="padding:1rem;">No output comparison available for this product type.</p>';
}

function buildParseOutputTab(r, index) {
  const outA = r.pipeline_a.output || '';
  const outB = r.pipeline_b.output || '';

  let html = '';

  // View mode toggle
  html += `<div style="display:flex;gap:0.5rem;margin-bottom:0.75rem;">`;
  html += `<div class="view-toggle">`;
  html += `<button class="active" onclick="setParseView(${index}, 'rendered', this)">Rendered</button>`;
  html += `<button onclick="setParseView(${index}, 'raw', this)">Raw</button>`;
  html += `<button onclick="setParseView(${index}, 'diff', this)">Diff</button>`;
  html += `</div></div>`;

  // Side-by-side rendered
  html += `<div id="parse-rendered-${index}" class="output-grid">`;
  html += `<div class="output-panel"><div class="output-panel-header">${esc(pipelineAName)}</div>`;
  html += `<div class="output-panel-body md-view" id="md-a-${index}">${outA ? marked.parse(outA) : '<em>No output</em>'}</div></div>`;
  html += `<div class="output-panel"><div class="output-panel-header">${esc(pipelineBName)}</div>`;
  html += `<div class="output-panel-body md-view" id="md-b-${index}">${outB ? marked.parse(outB) : '<em>No output</em>'}</div></div>`;
  html += `</div>`;

  // Side-by-side raw
  html += `<div id="parse-raw-${index}" class="output-grid" style="display:none;">`;
  html += `<div class="output-panel"><div class="output-panel-header">${esc(pipelineAName)}</div>`;
  html += `<div class="output-panel-body json-view">${esc(outA) || '<em>No output</em>'}</div></div>`;
  html += `<div class="output-panel"><div class="output-panel-header">${esc(pipelineBName)}</div>`;
  html += `<div class="output-panel-body json-view">${esc(outB) || '<em>No output</em>'}</div></div>`;
  html += `</div>`;

  // Diff view
  html += `<div id="parse-diff-${index}" style="display:none;">`;
  html += `<div class="diff-container"><div class="diff-header">Unified Diff</div>`;
  html += `<div class="diff-body">${computeLineDiff(outA, outB)}</div></div>`;
  html += `</div>`;

  return html;
}

function setParseView(index, mode, btn) {
  const toggle = btn.closest('.view-toggle');
  toggle.querySelectorAll('button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  ['rendered', 'raw', 'diff'].forEach(m => {
    const el = document.getElementById(`parse-${m}-${index}`);
    if (el) el.style.display = m === mode ? '' : 'none';
  });
  if (mode === 'rendered' || mode === 'raw') {
    setupSyncScroll(index, mode);
  }
}

function formatJsonOutput(value) {
  if (value == null || value === '') return '';
  if (typeof value === 'string') {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch (e) {
      return value;
    }
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch (e) {
    return String(value);
  }
}

function highlightJson(value) {
  const formatted = esc(formatJsonOutput(value));
  return formatted.replace(
    /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?/g,
    function (match, quoted, colon) {
      if (quoted && colon) {
        return '<span class="json-key">' + quoted + '</span>' + colon;
      }
      if (quoted) {
        return '<span class="json-string">' + quoted + '</span>';
      }
      if (match === 'true' || match === 'false') {
        return '<span class="json-boolean">' + match + '</span>';
      }
      if (match === 'null') {
        return '<span class="json-null">' + match + '</span>';
      }
      return '<span class="json-number">' + match + '</span>';
    }
  );
}

function buildExtractOutputPanel(panelId, label, output) {
  const jsonText = formatJsonOutput(output);
  let html = `<div class="output-panel" id="${panelId}">`;
  html += `<div class="output-panel-header"><span>${esc(label)}</span>`;
  if (jsonText) {
    html += `<button class="output-copy-btn" onclick="event.stopPropagation();copyOutput('${panelId}', this)">Copy</button>`;
  }
  html += `</div>`;
  if (jsonText) {
    html += `<div class="output-raw" style="display:none">${esc(jsonText)}</div>`;
    html += `<div class="output-panel-body output-json">${highlightJson(jsonText)}</div>`;
  } else {
    html += `<div class="output-panel-body"><em>No output</em></div>`;
  }
  html += `</div>`;
  return html;
}

function copyOutput(panelId, btn) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  const rawEl = panel.querySelector('.output-raw');
  if (!rawEl) return;
  const text = rawEl.textContent || rawEl.innerText;
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
  });
}

function buildExtractOutputTab(r, index) {
  const outA = r.pipeline_a.output;
  const outB = r.pipeline_b.output;
  const jsonA = formatJsonOutput(outA);
  const jsonB = formatJsonOutput(outB);

  let html = `<div style="display:flex;gap:0.5rem;margin-bottom:0.75rem;">`;
  html += `<div class="view-toggle">`;
  html += `<button class="active" onclick="setExtractView(${index}, 'side', this)">Side by Side</button>`;
  html += `<button onclick="setExtractView(${index}, 'diff', this)">Diff</button>`;
  html += `</div></div>`;

  // Side by side JSON (highlighted like the detailed report)
  html += `<div id="extract-side-${index}" class="output-grid">`;
  html += buildExtractOutputPanel(`extract-out-a-${index}`, pipelineAName, outA);
  html += buildExtractOutputPanel(`extract-out-b-${index}`, pipelineBName, outB);
  html += `</div>`;

  // Diff
  html += `<div id="extract-diff-${index}" style="display:none;">`;
  html += `<div class="diff-container"><div class="diff-header">Unified Diff</div>`;
  html += `<div class="diff-body">${computeLineDiff(jsonA, jsonB)}</div></div>`;
  html += `</div>`;

  return html;
}

function setExtractView(index, mode, btn) {
  const toggle = btn.closest('.view-toggle');
  toggle.querySelectorAll('button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  ['side', 'diff'].forEach(m => {
    const el = document.getElementById(`extract-${m}-${index}`);
    if (el) el.style.display = m === mode ? '' : 'none';
  });
  if (mode === 'side') {
    setupSyncScroll(index, 'side');
  }
}

// ---- Input tab ----
function buildInputTab(r, index) {
  const inputFile = r.input_file;
  const inputFileRel = r.input_file_rel;
  const dataUrl = r.input_data_url;
  const citationsA = (r.pipeline_a && r.pipeline_a.field_citations) || [];
  const citationsB = (r.pipeline_b && r.pipeline_b.field_citations) || [];
  const gtAnnotations = r.gt_annotations || [];
  const hasCitations = citationsA.length > 0 || citationsB.length > 0 || gtAnnotations.length > 0;

  if (!inputFile && !dataUrl) {
    return '<p class="na" style="padding:1rem;">No input file available.</p>';
  }

  const ext = inputFile ? inputFile.split('.').pop().toLowerCase() : '';
  const isPdf = ext === 'pdf';

  let html = '<div class="input-preview">';

  if (inputFile) {
    html += `<p class="file-path">${esc(inputFile)}</p>`;
  }

  if (dataUrl && !isPdf) {
    // Embedded image
    html += `<img src="${dataUrl}" />`;
  } else if (isPdf) {
    // Resolve PDF URL: use pdfBaseUrl + relative path, like the detailed report
    let pdfSrc = '';
    if (pdfBaseUrl && inputFileRel) {
      const base = pdfBaseUrl.endsWith('/') ? pdfBaseUrl.slice(0, -1) : pdfBaseUrl;
      const rel = inputFileRel.startsWith('/') ? inputFileRel.slice(1) : inputFileRel;
      pdfSrc = base + '/' + rel;
    } else if (inputFile) {
      // Never feed absolute filesystem paths to PDF.js — the browser would
      // resolve them against the local server origin and 404.
      const resolved = resolveFilePath(inputFile);
      if (resolved && !resolved.startsWith('/') && !/^[A-Za-z]:[\\/]/.test(resolved)) {
        pdfSrc = resolved;
      }
    }
    if (pdfSrc) {
      const viewerId = 'pdfviewer-' + index;
      if (hasCitations) {
        window[`extractCitations_${index}`] = {
          gt: gtAnnotations,
          predA: citationsA,
          predB: citationsB,
          showGt: gtAnnotations.length > 0,
          showA: true,
          showB: true,
        };
        html += `<div class="bbox-controls">`;
        if (gtAnnotations.length > 0) {
          html += `<button class="bbox-btn active" onclick="toggleExtractCitationLayer(${index}, 'GT', this)">Ground truth</button>`;
        }
        html += `<button class="bbox-btn active" onclick="toggleExtractCitationLayer(${index}, 'A', this)">${esc(pipelineAName)}</button>`;
        html += `<button class="bbox-btn active" onclick="toggleExtractCitationLayer(${index}, 'B', this)">${esc(pipelineBName)}</button>`;
        html += `</div>`;
      }
      html += `<div id="${viewerId}" class="pdfjs-viewer" data-pdf-src="${esc(pdfSrc)}" data-pdf-pending="true" data-compare-index="${index}" style="max-height:700px;overflow-y:auto;border:1px solid var(--border);border-radius:8px;padding:8px;background:#f5f5f4;"><p class="na">Loading PDF...</p></div>`;
      if (hasCitations) {
        html += `<div class="bbox-legend">`;
        if (gtAnnotations.length > 0) {
          html += `<div class="bbox-legend-item"><span class="bbox-swatch gt"></span><span>Ground truth</span></div>`;
        }
        html += `<div class="bbox-legend-item"><span class="bbox-swatch pred-a"></span><span>${esc(pipelineAName)}</span></div>`;
        html += `<div class="bbox-legend-item"><span class="bbox-swatch pred-b"></span><span>${esc(pipelineBName)}</span></div>`;
        html += `</div>`;
      }
    } else {
      html += `<p class="na">PDF preview not available — no base URL configured.</p>`;
    }
  } else if (inputFile) {
    if (['png', 'jpg', 'jpeg', 'gif'].includes(ext)) {
      // Resolve image URL via file server, same as PDFs
      let imgSrc = '';
      if (pdfBaseUrl && inputFileRel) {
        const base = pdfBaseUrl.endsWith('/') ? pdfBaseUrl.slice(0, -1) : pdfBaseUrl;
        const rel = inputFileRel.startsWith('/') ? inputFileRel.slice(1) : inputFileRel;
        imgSrc = base + '/' + rel;
      } else {
        imgSrc = resolveFilePath(inputFile);
        if (imgSrc && (imgSrc.startsWith('/') || /^[A-Za-z]:[\\/]/.test(imgSrc))) {
          imgSrc = '';
        }
      }
      html += `<img src="${esc(imgSrc)}" />`;
    } else {
      html += `<p class="na">Preview not available for .${esc(ext)} files</p>`;
    }
  }

  html += '</div>';
  return html;
}

function toggleExtractCitationLayer(index, which, btn) {
  const state = window[`extractCitations_${index}`];
  if (!state) return;
  if (which === 'GT') state.showGt = !state.showGt;
  if (which === 'A') state.showA = !state.showA;
  if (which === 'B') state.showB = !state.showB;
  btn.classList.toggle('active');
  drawExtractCitationOverlays(index);
}

function drawExtractCitationOverlays(index) {
  const state = window[`extractCitations_${index}`];
  const viewer = document.getElementById('pdfviewer-' + index);
  if (!state || !viewer || typeof BboxOverlay === 'undefined') return;

  viewer.querySelectorAll('.bbox-overlay-stack').forEach(stack => {
    const page = Number(stack.getAttribute('data-page'));
    const pdfCanvas = stack.querySelector('canvas.pdf-page-canvas');
    const overlay = stack.querySelector('canvas.bbox-overlay-canvas');
    if (!pdfCanvas || !overlay) return;
    BboxOverlay.syncCanvasToCanvas(overlay, pdfCanvas);
    const ctx = overlay.getContext('2d');
    BboxOverlay.clearCanvas(ctx, overlay);
    BboxOverlay.drawExtractComparisonOverlay(
      ctx,
      overlay.width,
      overlay.height,
      state.gt || [],
      state.predA,
      state.predB,
      { page: page, showGt: !!state.showGt, showA: state.showA, showB: state.showB }
    );
  });
}

// ---- Layout detection detail ----
function buildLayoutDetectionDetail(r, index) {
  const predA = r.pipeline_a.predictions || [];
  const predB = r.pipeline_b.predictions || [];
  const gt = r.gt_annotations || [];
  const imgPath = r.input_data_url || resolveFilePath(r.input_file || '');

  // Store in window for drawing
  window[`predA_${index}`] = predA;
  window[`predB_${index}`] = predB;
  window[`gt_${index}`] = gt;
  window[`bboxState_${index}`] = { showA: true, showB: true, showGT: true };

  const metricA = r.pipeline_a.metric_value != null ? (r.pipeline_a.metric_value * 100).toFixed(1) + '%' : 'N/A';
  const metricB = r.pipeline_b.metric_value != null ? (r.pipeline_b.metric_value * 100).toFixed(1) + '%' : 'N/A';

  let html = '';

  // Metric pills at top
  html += buildMetricPills(r);

  // 3-column grid
  html += `<div class="layoutdet-grid">`;

  // Panel A
  html += `<div class="layoutdet-panel"><div class="layoutdet-panel-header"><span>${esc(pipelineAName)} (${metricA})</span>`;
  html += `<button class="bbox-btn active" onclick="toggleLayoutBbox(${index}, 'A', this)">Bboxes</button>`;
  html += `</div><div class="layoutdet-panel-body"><img id="ld-img-a-${index}" src="${imgPath}" onload="drawLayoutPanel(${index}, 'A')" /><canvas id="ld-canvas-a-${index}" class="bbox-overlay-canvas"></canvas></div></div>`;

  // Panel B
  html += `<div class="layoutdet-panel"><div class="layoutdet-panel-header"><span>${esc(pipelineBName)} (${metricB})</span>`;
  html += `<button class="bbox-btn active" onclick="toggleLayoutBbox(${index}, 'B', this)">Bboxes</button>`;
  html += `</div><div class="layoutdet-panel-body"><img id="ld-img-b-${index}" src="${imgPath}" onload="drawLayoutPanel(${index}, 'B')" /><canvas id="ld-canvas-b-${index}" class="bbox-overlay-canvas"></canvas></div></div>`;

  // GT Panel
  html += `<div class="layoutdet-panel"><div class="layoutdet-panel-header"><span>Ground Truth</span>`;
  html += `<span style="display:flex;gap:4px;">`;
  html += `<button class="bbox-btn active" onclick="toggleLayoutOverlay(${index}, 'GT', this)">GT</button>`;
  html += `<button class="bbox-btn active" onclick="toggleLayoutOverlay(${index}, 'A', this)">A</button>`;
  html += `<button class="bbox-btn active" onclick="toggleLayoutOverlay(${index}, 'B', this)">B</button>`;
  html += `</span></div><div class="layoutdet-panel-body"><img id="ld-img-gt-${index}" src="${imgPath}" onload="drawLayoutOverlay(${index})" /><canvas id="ld-canvas-gt-${index}" class="bbox-overlay-canvas"></canvas></div></div>`;

  html += `</div>`;

  // Legend
  html += `<div class="bbox-legend">`;
  html += `<div class="bbox-legend-item"><div class="bbox-swatch gt"></div><span>Ground Truth</span></div>`;
  html += `<div class="bbox-legend-item"><div class="bbox-swatch pred-a"></div><span>${esc(pipelineAName)}</span></div>`;
  html += `<div class="bbox-legend-item"><div class="bbox-swatch pred-b"></div><span>${esc(pipelineBName)}</span></div>`;
  html += `</div>`;

  return html;
}

function drawLayoutPanel(index, which) {
  const canvas = document.getElementById(`ld-canvas-${which.toLowerCase()}-${index}`);
  const img = document.getElementById(`ld-img-${which.toLowerCase()}-${index}`);
  const preds = window[`pred${which}_${index}`];
  const state = window[`bboxState_${index}`];
  if (!canvas || !img || !window.BboxOverlay) return;
  const ctx = canvas.getContext('2d');
  const dims = BboxOverlay.syncCanvasToImage(canvas, img);
  if (!dims) return;
  BboxOverlay.clearCanvas(ctx, canvas);
  if (!state[`show${which}`] || !preds) return;
  BboxOverlay.drawLayoutPredictions(ctx, preds, dims.scale, BboxOverlay.LAYOUTDET_COLORS);
}

function drawLayoutOverlay(index) {
  const canvas = document.getElementById(`ld-canvas-gt-${index}`);
  const img = document.getElementById(`ld-img-gt-${index}`);
  if (!canvas || !img || !window.BboxOverlay) return;
  const ctx = canvas.getContext('2d');
  const dims = BboxOverlay.syncCanvasToImage(canvas, img);
  if (!dims) return;
  BboxOverlay.clearCanvas(ctx, canvas);
  const state = window[`bboxState_${index}`];
  BboxOverlay.drawLayoutComparisonOverlay(
    ctx,
    window[`gt_${index}`] || [],
    window[`predA_${index}`] || [],
    window[`predB_${index}`] || [],
    dims.scale,
    { showGT: state.showGT, showA: state.showA, showB: state.showB }
  );
}

function toggleLayoutBbox(index, which, btn) {
  const state = window[`bboxState_${index}`];
  state[`show${which}`] = !state[`show${which}`];
  btn.classList.toggle('active', state[`show${which}`]);
  drawLayoutPanel(index, which);
}

function toggleLayoutOverlay(index, which, btn) {
  const state = window[`bboxState_${index}`];
  state[`show${which}`] = !state[`show${which}`];
  btn.classList.toggle('active', state[`show${which}`]);
  drawLayoutOverlay(index);
  // Also redraw individual panels if A or B toggled
  if (which === 'A' || which === 'B') {
    drawLayoutPanel(index, which);
  }
}

// ---- Diff computation (simple line diff) ----
function computeLineDiff(textA, textB) {
  if (!textA && !textB) return '<div class="diff-line diff-ctx">(both empty)</div>';
  // Real newlines — not the two-char sequence \n left over from Python strings.
  const linesA = (textA || '').split('\n');
  const linesB = (textB || '').split('\n');

  // Simple LCS-based diff
  const m = linesA.length, n = linesB.length;

  // For very large files, fall back to simple comparison
  if (m + n > 4000) {
    let html = '';
    const maxLen = Math.max(m, n);
    for (let i = 0; i < maxLen; i++) {
      const a = i < m ? linesA[i] : undefined;
      const b = i < n ? linesB[i] : undefined;
      if (a === b) {
        html += `<div class="diff-line diff-ctx"> ${esc(a)}</div>`;
      } else {
        if (a !== undefined) html += `<div class="diff-line diff-del">-${esc(a)}</div>`;
        if (b !== undefined) html += `<div class="diff-line diff-add">+${esc(b)}</div>`;
      }
    }
    return html;
  }

  // Build LCS table
  const dp = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = linesA[i - 1] === linesB[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  // Backtrack
  const ops = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && linesA[i - 1] === linesB[j - 1]) {
      ops.push({ type: 'ctx', line: linesA[i - 1] });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      ops.push({ type: 'add', line: linesB[j - 1] });
      j--;
    } else {
      ops.push({ type: 'del', line: linesA[i - 1] });
      i--;
    }
  }
  ops.reverse();

  // Render with context collapsing
  let html = '';
  let ctxCount = 0;
  const CTX_LIMIT = 3;

  ops.forEach((op, idx) => {
    if (op.type === 'ctx') {
      // Show context lines near changes
      const nearChange = ops.slice(Math.max(0, idx - CTX_LIMIT), idx).some(o => o.type !== 'ctx')
        || ops.slice(idx + 1, idx + CTX_LIMIT + 1).some(o => o.type !== 'ctx');
      if (nearChange) {
        if (ctxCount > 0) {
          html += `<div class="diff-line diff-hunk">@@ ${ctxCount} unchanged lines @@</div>`;
          ctxCount = 0;
        }
        html += `<div class="diff-line diff-ctx"> ${esc(op.line)}</div>`;
      } else {
        ctxCount++;
      }
    } else {
      if (ctxCount > 0) {
        html += `<div class="diff-line diff-hunk">@@ ${ctxCount} unchanged lines @@</div>`;
        ctxCount = 0;
      }
      const cls = op.type === 'add' ? 'diff-add' : 'diff-del';
      const prefix = op.type === 'add' ? '+' : '-';
      html += `<div class="diff-line ${cls}">${prefix}${esc(op.line)}</div>`;
    }
  });

  if (ctxCount > 0) {
    html += `<div class="diff-line diff-hunk">@@ ${ctxCount} unchanged lines @@</div>`;
  }

  return html || '<div class="diff-line diff-ctx">(no differences)</div>';
}

// ---- Sync scroll ----
function setupSyncScroll(index, mode) {
  const container =
    document.getElementById(`parse-${mode}-${index}`) ||
    document.getElementById(`extract-${mode}-${index}`);
  if (!container) return;
  const panels = container.querySelectorAll('.output-panel-body');
  if (panels.length < 2) return;

  let syncing = false;
  panels.forEach((pane, i) => {
    pane.addEventListener('scroll', () => {
      if (syncing) return;
      syncing = true;
      const other = panels[1 - i];
      const pct = pane.scrollTop / (pane.scrollHeight - pane.clientHeight || 1);
      other.scrollTop = pct * (other.scrollHeight - other.clientHeight);
      setTimeout(() => { syncing = false; }, 10);
    });
  });
}

// ---- Detail interactions initialization ----
function initDetailInteractions(index) {
  // Setup sync scroll for the default side-by-side / rendered view
  if (productType === 'parse') {
    setTimeout(() => setupSyncScroll(index, 'rendered'), 100);
  } else if (productType === 'extract') {
    setTimeout(() => setupSyncScroll(index, 'side'), 100);
  }
  // Render any pending PDF viewers
  const detail = document.getElementById('detail-' + index);
  if (detail) {
    detail.querySelectorAll('[data-pdf-pending]').forEach(el => {
      el.removeAttribute('data-pdf-pending');
      renderEmbeddedPdf(el.id);
    });
  }
}

// ---- Utilities ----
function metricColorClass(v) {
  if (v === null || v === undefined) return 'metric-na';
  if (v >= 0.9) return 'metric-high';
  if (v >= 0.7) return 'metric-mid';
  if (v >= 0.5) return 'metric-low';
  return 'metric-bad';
}

function esc(str) {
  if (str === null || str === undefined) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

// __REPORT_DEPS__


// ---- PDF.js rendering ----
const pdfJsCdnList = [
  {
    src: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.min.js',
    worker: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js'
  },
  {
    src: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.4.120/build/pdf.min.js',
    worker: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.4.120/build/pdf.worker.min.js'
  },
];
let pdfJsLoadPromise = null;

function ensurePdfJsLoaded() {
  if (window.pdfjsLib) return Promise.resolve(true);
  if (pdfJsLoadPromise) return pdfJsLoadPromise;
  pdfJsLoadPromise = new Promise(resolve => {
    let idx = 0;
    const tryLoad = () => {
      if (idx >= pdfJsCdnList.length) { resolve(false); return; }
      const { src, worker } = pdfJsCdnList[idx++];
      const s = document.createElement('script');
      s.src = src; s.async = true;
      s.onload = () => {
        if (window.pdfjsLib) {
          pdfjsLib.GlobalWorkerOptions.workerSrc = worker;
          resolve(true);
        } else tryLoad();
      };
      s.onerror = tryLoad;
      document.head.appendChild(s);
    };
    tryLoad();
  });
  return pdfJsLoadPromise;
}

async function renderEmbeddedPdf(viewerId) {
  const viewer = document.getElementById(viewerId);
  if (!viewer) return;
  const pdfSrc = viewer.getAttribute('data-pdf-src');
  if (!pdfSrc) { viewer.innerHTML = '<p class="na">No PDF data</p>'; return; }
  const compareIndexAttr = viewer.getAttribute('data-compare-index');
  const compareIndex = compareIndexAttr != null && compareIndexAttr !== '' ? Number(compareIndexAttr) : null;
  const hasOverlay = compareIndex != null && window[`extractCitations_${compareIndex}`];

  const ready = await ensurePdfJsLoaded();
  if (!ready) { viewer.innerHTML = '<p class="na">Failed to load PDF.js</p>'; return; }

  try {
    const loadingTask = pdfjsLib.getDocument(pdfSrc);
    const pdfDoc = await loadingTask.promise;
    viewer.innerHTML = '';

    for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
      const page = await pdfDoc.getPage(pageNum);
      const containerWidth = viewer.clientWidth - 16;
      const unscaledViewport = page.getViewport({ scale: 1 });
      const scale = containerWidth / unscaledViewport.width;
      const viewport = page.getViewport({ scale });

      const canvas = document.createElement('canvas');
      canvas.className = 'pdf-page-canvas';
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      canvas.style.display = 'block';
      canvas.style.borderRadius = '4px';

      if (hasOverlay) {
        const stack = document.createElement('div');
        stack.className = 'bbox-overlay-stack';
        stack.setAttribute('data-page', String(pageNum));
        stack.style.display = 'block';
        stack.style.marginBottom = '8px';
        stack.appendChild(canvas);

        const overlay = document.createElement('canvas');
        overlay.className = 'bbox-overlay-canvas';
        stack.appendChild(overlay);
        viewer.appendChild(stack);
      } else {
        canvas.style.marginBottom = '8px';
        viewer.appendChild(canvas);
      }

      await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;
    }

    if (hasOverlay) {
      drawExtractCitationOverlays(compareIndex);
    }
  } catch (e) {
    viewer.innerHTML = `<p class="na">Failed to render PDF: ${esc(e.message || '')}</p>`;
  }
}
