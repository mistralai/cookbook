(function () {
  "use strict";

  // ─── State ───
  var state = {
    currentMetric: '',
    sortMode: 'score_desc',  // score_desc, score_asc, alpha
    searchQuery: '',
    activeTags: [],
    rangeMin: 0.0,
    rangeMax: 1.0,
    currentPage: 1,
    perPage: 50,
    filtered: [],
    expandedId: null
  };

  var metricKeys = Object.keys(DATA.metricNames);
  // Prefer the headline / tag-table ordering (e.g. Value F1), not avg-sorted aggMetrics[0].
  if (DATA.defaultMetric && DATA.metricNames[DATA.defaultMetric]) {
    state.currentMetric = DATA.defaultMetric;
  } else if (DATA.aggMetrics.length > 0) {
    state.currentMetric = DATA.aggMetrics[0].name;
  } else if (metricKeys.length > 0) {
    state.currentMetric = metricKeys[0];
  }

  // ─── Helpers ───
  function esc(s) {
    if (s == null) return '';
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }

  // __REPORT_DEPS__

  // Toggle between raw/rendered markdown view
  window.toggleOutputView = function (panelId, mode) {
    var panel = document.getElementById(panelId);
    if (!panel) return;
    var rawEl = panel.querySelector('.output-raw-view');
    var renderedEl = panel.querySelector('.output-rendered');
    var btns = panel.querySelectorAll('.output-view-btn');
    btns.forEach(function (b) { b.classList.toggle('active', b.getAttribute('data-mode') === mode); });
    if (mode === 'rendered') {
      rawEl.style.display = 'none';
      renderedEl.style.display = 'block';
    } else {
      rawEl.style.display = 'block';
      renderedEl.style.display = 'none';
    }
  };

  window.copyOutput = function (panelId, btn) {
    var panel = document.getElementById(panelId);
    if (!panel) return;
    // Prefer plain-text copy source (JSON); fall back to markdown raw view.
    var rawEl = panel.querySelector('.output-raw') || panel.querySelector('.output-raw-view');
    if (!rawEl) return;
    var text = rawEl.textContent || rawEl.innerText;
    navigator.clipboard.writeText(text).then(function () {
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(function () { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
    });
  };

  function highlightJson(text) {
    // Highlight the payload verbatim. It arrives already pretty-printed at
    // indent 2 from the report builder, and a JSON.parse/stringify round-trip
    // here would hoist integer-like keys to the front — making the highlighted
    // view disagree with the plain-text copy source below it.
    var formatted = esc(text || '');
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

  function buildOutputPanel(panelId, label, output, html, outputFormat) {
    var isJson = outputFormat === 'json';
    var isMarkdown = outputFormat === 'markdown';
    var htmlOut = '<div class="output-panel" id="' + panelId + '">';
    htmlOut += '<div class="output-panel-header"><span>' + esc(label) + '</span>';
    if (output) {
      if (isMarkdown) {
        htmlOut += '<div class="output-view-toggle">' +
          '<button class="output-view-btn active" data-mode="rendered" onclick="event.stopPropagation();toggleOutputView(\'' + panelId + '\',\'rendered\')">Rendered</button>' +
          '<button class="output-view-btn" data-mode="raw" onclick="event.stopPropagation();toggleOutputView(\'' + panelId + '\',\'raw\')">Raw</button>' +
          '</div>';
      }
      htmlOut += '<button class="output-copy-btn" onclick="event.stopPropagation();copyOutput(\'' + panelId + '\',this)">Copy</button>';
    }
    htmlOut += '</div>';
    if (output) {
      if (isJson) {
        // Plain text for Copy; highlighted view for display.
        htmlOut += '<div class="output-raw" style="display:none">' + esc(output) + '</div>';
        htmlOut += '<div class="output-panel-body output-json">' + highlightJson(output) + '</div>';
      } else if (isMarkdown) {
        htmlOut += '<div class="output-panel-body output-raw-view" style="display:none">' + esc(output) + '</div>';
        htmlOut += '<div class="output-panel-body rendered-md output-rendered">' + (html || esc(output)) + '</div>';
      } else {
        htmlOut += '<div class="output-panel-body output-raw-view">' + esc(output) + '</div>';
      }
    } else {
      htmlOut += '<div class="output-panel-body"><span class="output-empty">No ' + esc(label.toLowerCase()) + ' available</span></div>';
    }
    htmlOut += '</div>';
    return htmlOut;
  }

  function scoreColor(v) {
    if (v >= 0.9) return 'emerald';
    if (v >= 0.7) return 'amber';
    return 'red';
  }

  function fmt(v, decimals) {
    if (v == null || isNaN(v)) return '-';
    return Number(v).toFixed(decimals !== undefined ? decimals : 4);
  }

  function fmtStat(v, unit) {
    if (v == null || isNaN(v)) return '-';
    var n = Number(v);
    if (Math.abs(n) >= 1000) return n.toFixed(0) + (unit || '');
    if (Math.abs(n) >= 1) return n.toFixed(1) + (unit || '');
    return n.toFixed(3) + (unit || '');
  }

  function debounce(fn, ms) {
    var timer;
    return function () {
      var args = arguments, ctx = this;
      clearTimeout(timer);
      timer = setTimeout(function () { fn.apply(ctx, args); }, ms);
    };
  }

  function getExampleScore(ex) {
    if (!state.currentMetric) return null;
    var v = ex.metrics[state.currentMetric];
    return (v !== undefined && v !== null) ? v : null;
  }

  // ─── Render summary cards ───
  function renderSummary() {
    var el = document.getElementById('summary-cards');
    var s = DATA.summary;
    el.innerHTML =
      '<div class="summary-card"><div class="label">Total Examples</div><div class="big-number">' + s.total + '</div></div>' +
      '<div class="summary-card card-success"><div class="label">Successful</div><div class="big-number">' + s.successful + '</div></div>' +
      '<div class="summary-card card-failed"><div class="label">Failed</div><div class="big-number">' + s.failed + '</div></div>' +
      '<div class="summary-card card-skipped"><div class="label">Skipped</div><div class="big-number">' + s.skipped + '</div></div>';
  }

  // ─── Render aggregate metrics ───
  function renderAggMetrics() {
    var el = document.getElementById('agg-metrics');
    if (!DATA.aggMetrics.length) { el.style.display = 'none'; return; }
    var html = '<h2 class="section-title">Aggregate Metrics</h2><div class="metrics-grid">';
    for (var i = 0; i < DATA.aggMetrics.length; i++) {
      var m = DATA.aggMetrics[i];
      var c = scoreColor(m.avg);
      html += '<div class="metric-card">' +
        '<div class="metric-label">' + esc(m.displayName) + tooltipIcon(m.name) + '</div>' +
        '<div class="metric-avg color-' + c + '">' + fmt(m.avg) + '</div>' +
        '<div class="metric-bar-track"><div class="metric-bar-fill bar-' + c + '" style="width:' + (m.avg * 100).toFixed(1) + '%"></div></div>' +
        '<div class="metric-range"><span>Min: ' + fmt(m.min) + '</span><span>Max: ' + fmt(m.max) + '</span></div>' +
        '</div>';
    }
    html += '</div>';
    el.innerHTML = html;
  }

  // ─── Render aggregate stats ───
  function renderAggStats() {
    var el = document.getElementById('agg-stats');
    if (!DATA.aggStats.length) { el.style.display = 'none'; return; }
    var html = '<button class="stats-toggle" id="stats-toggle">' +
      '<span class="chevron">&#9654;</span> Operational Statistics</button>' +
      '<div class="stats-body" id="stats-body"><div class="stats-grid">';
    for (var i = 0; i < DATA.aggStats.length; i++) {
      var s = DATA.aggStats[i];
      var u = s.unit || '';
      html += '<div class="stat-card">' +
        '<div class="stat-label">' + esc(s.displayName) + (u ? ' (' + esc(u) + ')' : '') + '</div>' +
        '<div class="stat-avg">' + fmtStat(s.avg, u) + '</div>' +
        '<div class="stat-detail">' +
        '<span>Min: ' + fmtStat(s.min, u) + '</span>' +
        '<span>Max: ' + fmtStat(s.max, u) + '</span>' +
        '<span>P50: ' + fmtStat(s.p50, u) + '</span>' +
        '<span>P95: ' + fmtStat(s.p95, u) + '</span>' +
        '<span>P99: ' + fmtStat(s.p99, u) + '</span>' +
        '<span>Total: ' + fmtStat(s.total, u) + '</span>' +
        '<span>Count: ' + s.count + '</span>' +
        '</div></div>';
    }
    html += '</div></div>';
    el.innerHTML = html;

    document.getElementById('stats-toggle').addEventListener('click', function () {
      this.classList.toggle('open');
      document.getElementById('stats-body').classList.toggle('open');
    });
  }

  // ─── Render tag metrics ───
  function renderTagMetrics() {
    var el = document.getElementById('tag-metrics');
    var tagKeys = Object.keys(DATA.tagMetrics);
    if (!tagKeys.length) { el.style.display = 'none'; return; }
    tagKeys.sort();

    // Union of metric rows across tags, preserving first-seen (headline) order.
    var metricOrder = [];
    var displayByName = {};
    var byTag = {};
    for (var t = 0; t < tagKeys.length; t++) {
      var tag = tagKeys[t];
      var tm = DATA.tagMetrics[tag];
      var rows = (tm && Array.isArray(tm.metrics)) ? tm.metrics : [];
      var map = {};
      for (var mk = 0; mk < rows.length; mk++) {
        var m = rows[mk];
        map[m.name] = m;
        if (!displayByName[m.name]) {
          displayByName[m.name] = m.displayName;
          metricOrder.push(m.name);
        }
      }
      byTag[tag] = {
        exampleCount: (tm && tm.exampleCount != null) ? tm.exampleCount : null,
        metrics: map
      };
    }

    var html = '<h2 class="section-title">Metrics by Tag</h2>' +
      '<div class="tag-metrics-scroll"><table class="tag-metrics-table"><thead><tr>' +
      '<th class="tag-metric-col">Metric</th>';
    for (t = 0; t < tagKeys.length; t++) {
      tag = tagKeys[t];
      html += '<th class="tag-col" title="' + esc(tag) + '"><div class="tag-col-label">' +
        esc(tag) + '</div></th>';
    }
    html += '</tr></thead><tbody>';

    // Example counts live in the table body so headers stay tag names only.
    html += '<tr class="tag-count-row"><th class="tag-metric-col" scope="row">' +
      '<span class="tag-metric-label">Documents</span></th>';
    for (t = 0; t < tagKeys.length; t++) {
      tag = tagKeys[t];
      var nEx = byTag[tag].exampleCount;
      html += '<td class="tag-val tag-count">' +
        (nEx != null ? esc(String(nEx)) : '—') + '</td>';
    }
    html += '</tr>';

    for (var mi = 0; mi < metricOrder.length; mi++) {
      var mName = metricOrder[mi];
      html += '<tr><th class="tag-metric-col" scope="row">' +
        '<span class="tag-metric-label">' + esc(displayByName[mName]) + tooltipIcon(mName) +
        '</span></th>';
      for (t = 0; t < tagKeys.length; t++) {
        tag = tagKeys[t];
        var cell = byTag[tag].metrics[mName];
        if (!cell) {
          html += '<td class="tag-val missing">—</td>';
          continue;
        }
        var rangeTitle = tag + '\nAvg: ' + fmt(cell.avg) +
          '\nMin: ' + fmt(cell.min) + '\nMax: ' + fmt(cell.max);
        html += '<td class="tag-val color-' + scoreColor(cell.avg) + '" title="' +
          esc(rangeTitle) + '">' + fmt(cell.avg) + '</td>';
      }
      html += '</tr>';
    }
    html += '</tbody></table></div>';
    el.innerHTML = html;
  }

  // ─── Controls ───
  function renderControls() {
    var el = document.getElementById('controls');
    var html = '';

    // Metric selector
    html += '<div class="control-group"><label>Metric</label><select id="metric-select">';
    for (var i = 0; i < metricKeys.length; i++) {
      var k = metricKeys[i];
      var sel = (k === state.currentMetric) ? ' selected' : '';
      html += '<option value="' + esc(k) + '"' + sel + '>' + esc(DATA.metricNames[k]) + '</option>';
    }
    html += '</select></div>';

    // Sort
    html += '<div class="control-group"><label>Sort</label><select id="sort-select">' +
      '<option value="score_desc"' + (state.sortMode === 'score_desc' ? ' selected' : '') + '>Score &#x2193;</option>' +
      '<option value="score_asc"' + (state.sortMode === 'score_asc' ? ' selected' : '') + '>Score &#x2191;</option>' +
      '<option value="alpha"' + (state.sortMode === 'alpha' ? ' selected' : '') + '>Name A-Z</option>' +
      '</select></div>';

    // Score range
    html += '<div class="control-group range-group"><label>Score Range</label>' +
      '<span class="range-label" id="range-min-label">' + state.rangeMin.toFixed(1) + '</span>' +
      '<input type="range" id="range-min" min="0" max="1" step="0.05" value="' + state.rangeMin + '">' +
      '<input type="range" id="range-max" min="0" max="1" step="0.05" value="' + state.rangeMax + '">' +
      '<span class="range-label" id="range-max-label">' + state.rangeMax.toFixed(1) + '</span>' +
      '</div>';

    // Search
    html += '<div class="control-group"><label>Search</label>' +
      '<input type="text" id="search-input" placeholder="Filter by test ID..." value="' + esc(state.searchQuery) + '"></div>';

    el.innerHTML = html;

    // Tag pills
    var tagEl = document.getElementById('tag-filters');
    if (DATA.tags.length) {
      var thtml = '';
      for (var t = 0; t < DATA.tags.length; t++) {
        var active = state.activeTags.indexOf(DATA.tags[t]) >= 0 ? ' active' : '';
        thtml += '<span class="tag-pill' + active + '" data-tag="' + esc(DATA.tags[t]) + '">' + esc(DATA.tags[t]) + '</span>';
      }
      tagEl.innerHTML = thtml;
      tagEl.style.display = '';
    } else {
      tagEl.style.display = 'none';
    }

    // Event listeners
    document.getElementById('metric-select').addEventListener('change', function () {
      state.currentMetric = this.value;
      state.currentPage = 1;
      applyFiltersAndRender();
    });
    document.getElementById('sort-select').addEventListener('change', function () {
      state.sortMode = this.value;
      state.currentPage = 1;
      applyFiltersAndRender();
    });
    document.getElementById('range-min').addEventListener('input', function () {
      state.rangeMin = parseFloat(this.value);
      if (state.rangeMin > state.rangeMax) { state.rangeMin = state.rangeMax; this.value = state.rangeMin; }
      document.getElementById('range-min-label').textContent = state.rangeMin.toFixed(1);
    });
    document.getElementById('range-min').addEventListener('change', function () {
      state.currentPage = 1;
      applyFiltersAndRender();
    });
    document.getElementById('range-max').addEventListener('input', function () {
      state.rangeMax = parseFloat(this.value);
      if (state.rangeMax < state.rangeMin) { state.rangeMax = state.rangeMin; this.value = state.rangeMax; }
      document.getElementById('range-max-label').textContent = state.rangeMax.toFixed(1);
    });
    document.getElementById('range-max').addEventListener('change', function () {
      state.currentPage = 1;
      applyFiltersAndRender();
    });
    document.getElementById('search-input').addEventListener('input', debounce(function () {
      state.searchQuery = this.value.toLowerCase();
      state.currentPage = 1;
      applyFiltersAndRender();
    }, 300));

    tagEl.addEventListener('click', function (e) {
      var pill = e.target.closest('.tag-pill');
      if (!pill) return;
      var tag = pill.getAttribute('data-tag');
      var idx = state.activeTags.indexOf(tag);
      if (idx >= 0) {
        state.activeTags.splice(idx, 1);
        pill.classList.remove('active');
      } else {
        state.activeTags.push(tag);
        pill.classList.add('active');
      }
      state.currentPage = 1;
      applyFiltersAndRender();
    });
  }

  // ─── Filter + sort ───
  function applyFilters() {
    var results = [];
    for (var i = 0; i < DATA.examples.length; i++) {
      var ex = DATA.examples[i];
      // Search filter
      if (state.searchQuery && ex.id.toLowerCase().indexOf(state.searchQuery) < 0) continue;
      // Tag filter
      if (state.activeTags.length > 0) {
        var hasTag = false;
        for (var t = 0; t < state.activeTags.length; t++) {
          if (ex.tags.indexOf(state.activeTags[t]) >= 0) { hasTag = true; break; }
        }
        if (!hasTag) continue;
      }
      // Score range filter
      var score = getExampleScore(ex);
      if (score !== null) {
        if (score < state.rangeMin || score > state.rangeMax) continue;
      }
      results.push(ex);
    }

    // Sort
    results.sort(function (a, b) {
      if (state.sortMode === 'alpha') {
        return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
      }
      var sa = getExampleScore(a);
      var sb = getExampleScore(b);
      // Nulls go last
      if (sa === null && sb === null) return 0;
      if (sa === null) return 1;
      if (sb === null) return -1;
      if (state.sortMode === 'score_desc') return sb - sa;
      return sa - sb;
    });

    state.filtered = results;
  }

  function applyFiltersAndRender() {
    applyFilters();
    state.expandedId = null;
    renderResultsCount();
    renderTable();
    renderPagination();
  }

  // ─── Results count ───
  function renderResultsCount() {
    var el = document.getElementById('results-count');
    el.textContent = state.filtered.length + ' of ' + DATA.examples.length + ' examples';
  }

  // ─── Table ───
  function renderTable() {
    var tbody = document.getElementById('examples-tbody');
    var frag = document.createDocumentFragment();
    var start = (state.currentPage - 1) * state.perPage;
    var end = Math.min(start + state.perPage, state.filtered.length);

    // Clear
    tbody.innerHTML = '';

    for (var i = start; i < end; i++) {
      var ex = state.filtered[i];
      var score = getExampleScore(ex);
      var tr = document.createElement('tr');
      tr.setAttribute('data-id', ex.id);
      if (ex.id === state.expandedId) tr.className = 'selected';

      // Status
      var td0 = document.createElement('td');
      td0.innerHTML = '<span class="status-dot ' + (ex.success ? 'ok' : 'fail') + '"></span>';
      tr.appendChild(td0);

      // Test ID
      var td1 = document.createElement('td');
      td1.textContent = ex.id;
      td1.style.fontFamily = 'var(--font-mono)';
      td1.style.fontSize = '0.8rem';
      td1.style.wordBreak = 'break-all';
      tr.appendChild(td1);

      // Score
      var td2 = document.createElement('td');
      if (score !== null) {
        var c = scoreColor(score);
        td2.innerHTML = '<div class="score-cell">' +
          '<div class="score-bar-track"><div class="score-bar-fill bar-' + c + '" style="width:' + (score * 100).toFixed(1) + '%"></div></div>' +
          '<span class="score-value color-' + c + '">' + fmt(score) + '</span></div>';
      } else {
        td2.innerHTML = '<span style="color:var(--muted-light);font-size:0.8rem">-</span>';
      }
      tr.appendChild(td2);

      // Tags
      var td3 = document.createElement('td');
      td3.className = 'col-tags-cell';
      if (ex.tags.length) {
        var badges = '';
        for (var t = 0; t < ex.tags.length; t++) {
          badges += '<span class="tag-badge">' + esc(ex.tags[t]) + '</span>';
        }
        td3.innerHTML = '<div class="tag-badges">' + badges + '</div>';
      }
      tr.appendChild(td3);

      frag.appendChild(tr);

      // Detail panel if expanded
      if (ex.id === state.expandedId) {
        var detailTr = document.createElement('tr');
        var detailTd = document.createElement('td');
        detailTd.colSpan = 4;
        detailTd.style.padding = '0';
        // Before buildDetailPanel — it renders the layer buttons from this state.
        setGroundingExample(ex);
        detailTd.innerHTML = buildDetailPanel(ex);
        detailTr.appendChild(detailTd);
        frag.appendChild(detailTr);
      }
    }

    tbody.appendChild(frag);

    // Row click handler (delegated)
    tbody.onclick = function (e) {
      var tr = e.target.closest('tr[data-id]');
      if (!tr) return;
      // Don't toggle if clicking inside detail panel
      if (e.target.closest('.detail-panel')) return;
      var id = tr.getAttribute('data-id');
      if (state.expandedId === id) {
        state.expandedId = null;
      } else {
        state.expandedId = id;
      }
      renderTable();
      initPdfViewerIfNeeded();
    };
  }

  // ─── Detail panel ───
  function buildGroundingSection(ex) {
    var g = ex.grounding || { gt: [], pred: [], rules: [] };
    var rules = g.rules || [];
    var fieldSet = {};
    rules.forEach(function (rule) { if (rule.fieldPath) fieldSet[rule.fieldPath] = true; });
    (g.gt || []).forEach(function (item) { if (item.fieldPath) fieldSet[item.fieldPath] = true; });
    (g.pred || []).forEach(function (item) { if (item.fieldPath) fieldSet[item.fieldPath] = true; });
    var fields = Object.keys(fieldSet).sort();

    var html = '<div class="detail-section">' +
      '<div class="detail-section-title">Grounding Annotations (' + fields.length + ' fields)</div>' +
      '<table class="detail-table grounding-table"><thead><tr>' +
      '<th>Field</th><th>Value</th><th>Page</th><th>BBox</th><th>IoU</th>' +
      '</tr></thead><tbody>';

    for (var i = 0; i < fields.length; i++) {
      var fieldPath = fields[i];
      var rule = null;
      for (var ri = 0; ri < rules.length; ri++) {
        if (rules[ri].fieldPath === fieldPath) { rule = rules[ri]; break; }
      }
      var valueBadge = groundingBadge(rule ? rule.valuePass : null);
      var pageBadge = groundingBadge(rule ? rule.pagePass : null);
      // bboxIouPass is the value-gated IoU float (e.g. 0.93), not a pass flag —
      // bboxCoveredPass is the boolean, and the IoU is shown in the next column.
      var bboxBadge = groundingBadge(rule ? rule.bboxCoveredPass : null);
      var iouText = (rule && rule.bboxIou != null) ? fmt(rule.bboxIou, 3) : '—';
      html += '<tr data-field-path="' + esc(fieldPath).replace(/"/g, '&quot;') + '">' +
        '<td class="mono">' + esc(fieldPath) + '</td>' +
        '<td>' + valueBadge + '</td>' +
        '<td>' + pageBadge + '</td>' +
        '<td>' + bboxBadge + '</td>' +
        '<td class="mono">' + iouText + '</td></tr>';
    }

    html += '</tbody></table></div>';
    return html;
  }

  function groundingBadge(passed) {
    if (passed === true || passed === 1) {
      return '<span class="grounding-badge pass">Pass</span>';
    }
    if (passed === false || passed === 0) {
      return '<span class="grounding-badge fail">Fail</span>';
    }
    return '<span class="grounding-badge na">N/A</span>';
  }

  function buildDetailPanel(ex) {
    var html = '<div class="detail-panel">';
    html += '<div class="detail-header"><div class="detail-title">' + esc(ex.id) + '</div>' +
      '<button class="detail-close" onclick="event.stopPropagation();closeDetail()">Close</button></div>';

    // Job IDs
    if (ex.parseJobId || ex.jobId) {
      html += '<div style="font-size:0.8rem;color:var(--muted);margin-bottom:12px">';
      if (ex.parseJobId) {
        html += '<span><strong>Parse Job ID:</strong> <span style="font-family:var(--font-mono);font-size:0.78rem">' + esc(ex.parseJobId) + '</span></span>';
      }
      if (ex.jobId && ex.jobId !== ex.parseJobId) {
        if (ex.parseJobId) html += '<span style="margin:0 10px;color:var(--border)">|</span>';
        html += '<span><strong>Job ID:</strong> <span style="font-family:var(--font-mono);font-size:0.78rem">' + esc(ex.jobId) + '</span></span>';
      }
      html += '</div>';
    }

    if (ex.parseJobLogsUrl || ex.parseJobLogsLocalPath || ex.parseJobLogsHtmlPath) {
      html += '<div style="font-size:0.8rem;color:var(--muted);margin:-4px 0 12px 0">';
      html += '<strong>Parse Job Logs:</strong> ';
      var links = [];
      if (ex.parseJobLogsUrl) {
        links.push('<a href="' + esc(ex.parseJobLogsUrl) + '" target="_blank" rel="noopener noreferrer">jobLogs.json (presigned)</a>');
      }
      if (ex.parseJobLogsLocalPath) {
        links.push('<a href="' + esc(ex.parseJobLogsLocalPath) + '" target="_blank" rel="noopener noreferrer">jobLogs.json (local)</a>');
      }
      if (ex.parseJobLogsHtmlPath) {
        links.push('<a href="' + esc(ex.parseJobLogsHtmlPath) + '" target="_blank" rel="noopener noreferrer">Pretty Viewer</a>');
      }
      html += links.join(' <span style="color:var(--border);margin:0 8px;">|</span> ');
      html += '</div>';
    }

    // Error
    if (ex.error) {
      html += '<div class="detail-error">' + esc(ex.error) + '</div>';
    }

    // All metrics (collapsible, with inline sub-collapsible details per metric)
    var metricEntries = Object.keys(ex.metrics);
    if (metricEntries.length) {
      var metricSummary = metricEntries.length + ' metric' + (metricEntries.length > 1 ? 's' : '');
      html += '<div class="detail-collapsible">' +
        '<button class="detail-collapsible-toggle" onclick="event.stopPropagation();toggleCollapsible(this)">' +
        '<span class="chevron">&#9654;</span> Metrics (' + metricSummary + ')</button>' +
        '<div class="detail-collapsible-body">';
      metricEntries.sort();
      for (var i = 0; i < metricEntries.length; i++) {
        var mName = metricEntries[i];
        var mVal = ex.metrics[mName];
        var c = scoreColor(mVal);
        var ruleStr = '';
        if (ex.ruleDetails[mName]) {
          var rd = ex.ruleDetails[mName];
          ruleStr = ' <span class="mono" style="margin-left:8px;font-size:0.78rem;color:var(--muted)">' + rd.passed + '/' + rd.total + ' rules</span>';
        }
        var hasDetails = ex.metricDetails && ex.metricDetails[mName] && ex.metricDetails[mName].length > 0;
        if (hasDetails) {
          html += '<div class="detail-collapsible" style="margin:0;border:none;border-bottom:1px solid var(--border)">' +
            '<button class="detail-collapsible-toggle" style="padding:5px 14px;font-size:0.82rem" onclick="event.stopPropagation();toggleCollapsible(this)">' +
            '<span class="chevron">&#9654;</span> ' +
            '<span style="display:inline-flex;align-items:center;min-width:220px">' + esc(DATA.metricNames[mName] || mName) + tooltipIcon(mName) + '</span>' +
            '<span class="mono color-' + c + '" style="margin-left:8px">' + fmt(mVal) + '</span>' +
            ruleStr + '</button>' +
            '<div class="detail-collapsible-body" style="padding:2px 14px 8px 32px;background:var(--bg)">';
          var lines = ex.metricDetails[mName];
          // Check if lines use [SECTION:...] markers for sub-collapsibles
          var hasSections = false;
          for (var si = 0; si < lines.length; si++) {
            if (lines[si].indexOf('[SECTION:') === 0) { hasSections = true; break; }
          }
          if (hasSections) {
            var inSection = false;
            for (var li = 0; li < lines.length; li++) {
              var ln = lines[li];
              var secMatch = ln.match(/^\[SECTION:(.+)\]$/);
              if (secMatch) {
                if (inSection) html += '</div></div></div>';
                html += '<div class="detail-collapsible" style="margin:2px 0;border:none;border-bottom:1px solid var(--border)">' +
                  '<button class="detail-collapsible-toggle" style="padding:3px 10px;font-size:0.78rem" onclick="event.stopPropagation();toggleCollapsible(this)">' +
                  '<span class="chevron">&#9654;</span> ' + esc(secMatch[1]) + '</button>' +
                  '<div class="detail-collapsible-body" style="padding:2px 10px 6px 28px;background:var(--bg)">' +
                  '<div style="font-size:0.75rem;line-height:1.5;font-family:var(--font-mono);white-space:pre-wrap;color:var(--text)">';
                inSection = true;
              } else {
                html += esc(ln) + '\n';
              }
            }
            if (inSection) html += '</div></div></div>';
          } else {
            html += '<div style="font-size:0.78rem;line-height:1.6;font-family:var(--font-mono);white-space:pre-wrap;color:var(--text)">';
            for (var li = 0; li < lines.length; li++) {
              html += esc(lines[li]) + '\n';
            }
            html += '</div>';
          }
          html += '</div></div>';
        } else {
          html += '<div style="padding:5px 14px;font-size:0.82rem;border-bottom:1px solid var(--border)">' +
            '<span style="display:inline-block;width:18px"></span>' +
            '<span style="display:inline-flex;align-items:center;min-width:220px">' + esc(DATA.metricNames[mName] || mName) + tooltipIcon(mName) + '</span>' +
            '<span class="mono color-' + c + '" style="margin-left:8px">' + fmt(mVal) + '</span>' +
            ruleStr + '</div>';
        }
      }
      html += '</div></div>';
    }

    // Rule results (collapsible)
    var ruleMetrics = Object.keys(ex.ruleResults);
    if (ruleMetrics.length) {
      var totalRules = 0;
      for (var r = 0; r < ruleMetrics.length; r++) { totalRules += ex.ruleResults[ruleMetrics[r]].length; }
      if (totalRules > 0) {
        html += '<div class="detail-collapsible">' +
          '<button class="detail-collapsible-toggle" onclick="event.stopPropagation();toggleCollapsible(this)">' +
          '<span class="chevron">&#9654;</span> Rule Results (' + totalRules + ' rules)</button>' +
          '<div class="detail-collapsible-body">';
        for (var r = 0; r < ruleMetrics.length; r++) {
          var rmName = ruleMetrics[r];
          var rules = ex.ruleResults[rmName];
          if (!rules.length) continue;
          html += '<div style="padding:8px 14px 4px;font-size:0.75rem;font-weight:600;color:var(--muted)">' + esc(DATA.metricNames[rmName] || rmName) + '</div>';
          html += '<table class="detail-table"><thead><tr><th>Type</th><th>ID</th><th>Status</th><th>Message</th></tr></thead><tbody>';
          for (var ri = 0; ri < rules.length; ri++) {
            var rule = rules[ri];
            var badge = rule.passed
              ? '<span class="rule-pass">Pass</span>'
              : '<span class="rule-fail">Fail</span>';
            html += '<tr><td>' + esc(rule.type) + '</td><td class="mono">' + esc(rule.id) + '</td>' +
              '<td>' + badge + '</td><td>' + esc(rule.message) + '</td></tr>';
          }
          html += '</tbody></table>';
        }
        html += '</div></div>';
      }
    }

    // Stats (collapsible)
    var statEntries = Object.keys(ex.stats);
    if (statEntries.length) {
      html += '<div class="detail-collapsible">' +
        '<button class="detail-collapsible-toggle" onclick="event.stopPropagation();toggleCollapsible(this)">' +
        '<span class="chevron">&#9654;</span> Operational Stats</button>' +
        '<div class="detail-collapsible-body">' +
        '<table class="detail-table"><thead><tr><th>Stat</th><th>Value</th></tr></thead><tbody>';
      statEntries.sort();
      for (var si = 0; si < statEntries.length; si++) {
        var sName = statEntries[si];
        html += '<tr><td>' + esc(sName.replace(/_/g, ' ')) + '</td><td class="mono">' + fmt(ex.stats[sName], 2) + '</td></tr>';
      }
      html += '</tbody></table></div></div>';
    }

    // PDF viewer with grounding overlays
    var productType = ex.productType || 'parse';
    var hasGrounding = ex.grounding && (
      (ex.grounding.gt && ex.grounding.gt.length) ||
      (ex.grounding.pred && ex.grounding.pred.length) ||
      (ex.grounding.rules && ex.grounding.rules.length)
    );
    if (hasGrounding) {
      html += buildGroundingSection(ex);
    }
    html += '<div class="detail-section pdf-viewer-section">' +
      '<div class="detail-section-title">PDF Viewer' + (hasGrounding ? ' + Grounding' : '') + '</div>' +
      '<div class="pdf-url-bar">' +
      '<label>Base URL</label>' +
      '<input type="text" id="pdf-base-url" value="' + esc(getPdfBaseUrl()) + '" placeholder="http://localhost:8080/data">' +
      '<button onclick="event.stopPropagation();savePdfBaseUrl();loadPdf(\'' + esc(ex.id) + '\',\'' + esc(productType) + '\')">Load PDF</button>' +
      '</div>';
    if (hasGrounding) {
      // Render the pressed state from groundingState so a re-render of the same
      // example keeps the buttons in step with the overlay.
      var on = function (flag) { return flag ? ' active' : ''; };
      html += '<div class="bbox-controls">' +
        '<button class="bbox-btn' + on(groundingState.showGt) + '" id="grounding-toggle-gt" onclick="event.stopPropagation();toggleGroundingLayer(\'gt\', this)">Ground truth</button>' +
        '<button class="bbox-btn' + on(groundingState.showPred) + '" id="grounding-toggle-pred" onclick="event.stopPropagation();toggleGroundingLayer(\'pred\', this)">Predicted</button>' +
        '<button class="bbox-btn' + on(groundingState.selectedOnly) + '" id="grounding-toggle-selected" onclick="event.stopPropagation();toggleGroundingLayer(\'selectedOnly\', this)">Selected field only</button>' +
        '</div>';
    }
    html += '<div class="pdf-canvas-wrap" id="pdf-canvas-wrap"><div class="pdf-placeholder">Set base path and click Load PDF</div></div>';
    if (hasGrounding) {
      html += '<div class="bbox-legend">' +
        '<div class="bbox-legend-item"><span class="bbox-swatch gt"></span><span>Ground truth evidence</span></div>' +
        '<div class="bbox-legend-item"><span class="bbox-swatch pred"></span><span>Predicted citation</span></div>' +
        '<div class="bbox-legend-item"><span class="bbox-swatch selected-gt"></span><span>Ground truth evidence (selected field)</span></div>' +
        '<div class="bbox-legend-item"><span class="bbox-swatch selected-pred"></span><span>Predicted citation (selected field)</span></div>' +
        '</div>';
    }
    html += '<div class="pdf-nav" id="pdf-nav" style="display:none">' +
      '<button id="pdf-prev" onclick="event.stopPropagation();pdfPrev()">Prev</button>' +
      '<span id="pdf-page-info">-</span>' +
      '<button id="pdf-next" onclick="event.stopPropagation();pdfNext()">Next</button>' +
      '<span style="margin-left:12px;border-left:1px solid var(--border);padding-left:12px"></span>' +
      '<button onclick="event.stopPropagation();pdfZoomOut()">−</button>' +
      '<span id="pdf-zoom-info">150%</span>' +
      '<button onclick="event.stopPropagation();pdfZoomIn()">+</button>' +
      '</div></div>';

    // Predicted / Expected output side-by-side
    html += '<div class="output-columns">';

    var predId = 'output-pred-' + ex.id.replace(/[^a-zA-Z0-9]/g, '_');
    html += buildOutputPanel(
      predId,
      'Predicted Output',
      ex.predictedOutput,
      ex.predictedHtml,
      ex.predictedOutputFormat || ''
    );

    var expId = 'output-exp-' + ex.id.replace(/[^a-zA-Z0-9]/g, '_');
    html += buildOutputPanel(
      expId,
      'Expected Output',
      ex.expectedOutput,
      ex.expectedHtml,
      ex.expectedOutputFormat || ''
    );

    html += '</div>';

    html += '</div>';
    return html;
  }

  // Close detail (global)
  window.closeDetail = function () {
    state.expandedId = null;
    renderTable();
  };

  // Toggle collapsible sections
  window.toggleCollapsible = function (btn) {
    btn.classList.toggle('open');
    var body = btn.nextElementSibling;
    if (body) body.classList.toggle('open');
  };

  // ─── PDF viewer + grounding overlays ───
  var pdfState = { doc: null, page: 1, total: 0, scale: 1.5 };
  var groundingState = {
    example: null,
    selectedField: null,
    showGt: true,
    showPred: true,
    selectedOnly: false,
  };

  // Called on every renderTable() pass for the expanded row, so only reset the
  // layer toggles when the example actually changes — otherwise typing in the
  // search box would silently undo them.
  window.setGroundingExample = function (ex) {
    var previousId = groundingState.example ? groundingState.example.id : null;
    var nextId = ex ? ex.id : null;
    groundingState.example = ex || null;
    if (previousId === nextId) return;
    groundingState.selectedField = null;
    groundingState.showGt = true;
    groundingState.showPred = true;
    groundingState.selectedOnly = false;
  };

  function selectGroundingField(fieldPath) {
    groundingState.selectedField = fieldPath || null;
    var rows = document.querySelectorAll('.grounding-table tbody tr');
    for (var i = 0; i < rows.length; i++) {
      rows[i].classList.toggle('selected', rows[i].getAttribute('data-field-path') === fieldPath);
    }
    if (!groundingState.example || !groundingState.example.grounding) {
      drawGroundingOverlay();
      return;
    }
    var targetPage = null;
    var g = groundingState.example.grounding;
    var collections = [g.gt || [], g.pred || []];
    for (var ci = 0; ci < collections.length; ci++) {
      for (var ai = 0; ai < collections[ci].length; ai++) {
        var ann = collections[ci][ai];
        if (ann.fieldPath === fieldPath && ann.page != null) {
          targetPage = ann.page;
          break;
        }
      }
      if (targetPage != null) break;
    }
    if (targetPage != null && pdfState.doc && targetPage !== pdfState.page) {
      pdfState.page = targetPage;
      renderPdfPage();
      return;
    }
    drawGroundingOverlay();
  }

  // Field selection via data-field-path (avoid embedding paths in inline JS handlers).
  document.addEventListener('click', function (e) {
    var row = e.target.closest ? e.target.closest('.grounding-table tbody tr[data-field-path]') : null;
    if (!row) return;
    selectGroundingField(row.getAttribute('data-field-path'));
  });

  window.toggleGroundingLayer = function (layer, btn) {
    if (layer === 'gt') {
      groundingState.showGt = !groundingState.showGt;
      btn.classList.toggle('active', groundingState.showGt);
    } else if (layer === 'pred') {
      groundingState.showPred = !groundingState.showPred;
      btn.classList.toggle('active', groundingState.showPred);
    } else if (layer === 'selectedOnly') {
      groundingState.selectedOnly = !groundingState.selectedOnly;
      btn.classList.toggle('active', groundingState.selectedOnly);
    }
    drawGroundingOverlay();
  };

  function drawGroundingOverlay() {
    var overlay = document.getElementById('pdf-overlay-canvas');
    var pdfCanvas = document.getElementById('pdf-render-canvas');
    if (!overlay || !pdfCanvas || !window.BboxOverlay) return;
    var ctx = overlay.getContext('2d');
    var dims = BboxOverlay.syncCanvasToCanvas(overlay, pdfCanvas);
    if (!dims) return;
    BboxOverlay.clearCanvas(ctx, overlay);

    var ex = groundingState.example;
    if (!ex || !ex.grounding) return;

    BboxOverlay.drawExtractGroundingOverlay(
      ctx,
      dims.width,
      dims.height,
      ex.grounding.gt || [],
      ex.grounding.pred || [],
      {
        page: pdfState.page,
        selectedField: groundingState.selectedField,
        selectedOnly: groundingState.selectedOnly,
        showGt: groundingState.showGt,
        showPred: groundingState.showPred,
      }
    );
  }

  window.getPdfBaseUrl = function () {
    try { return DATA.pdfBaseUrl || localStorage.getItem('bench_pdf_base_url') || ''; } catch (e) { return DATA.pdfBaseUrl || ''; }
  };
  window.savePdfBaseUrl = function () {
    var input = document.getElementById('pdf-base-url');
    if (input) {
      try { localStorage.setItem('bench_pdf_base_url', input.value); } catch (e) { }
    }
  };

  window.loadPdf = function (testId, productType) {
    var baseUrl = document.getElementById('pdf-base-url').value.replace(/\/+$/, '');
    if (!baseUrl) return;
    savePdfBaseUrl();
    // Strip overlapping path segments between baseUrl and testId to avoid duplication
    // e.g. baseUrl="http://host/data/tables/v1" + testId="tables/v1/file" -> ".../data/tables/v1/file.pdf"
    var relPath = testId;
    var baseParts = baseUrl.replace(/^https?:\/\/[^\/]*/i, '').split('/').filter(Boolean);
    var idParts = testId.split('/');
    for (var overlap = Math.min(baseParts.length, idParts.length); overlap > 0; overlap--) {
      var baseTail = baseParts.slice(baseParts.length - overlap).join('/');
      var idHead = idParts.slice(0, overlap).join('/');
      if (baseTail === idHead) {
        relPath = idParts.slice(overlap).join('/');
        break;
      }
    }
    var url;
    if (/^https?:\/\//i.test(baseUrl)) {
      url = baseUrl + '/' + relPath.split('/').map(function (p) { return encodeURIComponent(p); }).join('/') + '.pdf';
    } else {
      url = baseUrl + '/' + relPath + '.pdf';
    }
    var wrap = document.getElementById('pdf-canvas-wrap');
    wrap.innerHTML = '<div class="pdf-placeholder">Loading PDF...</div>';
    document.getElementById('pdf-nav').style.display = 'none';

    if (typeof pdfjsLib === 'undefined') {
      // Load PDF.js from CDN
      var script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.min.js';
      script.onload = function () {
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js';
        doLoadPdf(url);
      };
      script.onerror = function () {
        wrap.innerHTML = '<div class="pdf-placeholder">Failed to load PDF.js library</div>';
      };
      document.head.appendChild(script);
    } else {
      doLoadPdf(url);
    }
  };

  function doLoadPdf(url) {
    var wrap = document.getElementById('pdf-canvas-wrap');
    pdfjsLib.getDocument(url).promise.then(function (doc) {
      pdfState.doc = doc;
      pdfState.total = doc.numPages;
      pdfState.page = 1;
      renderPdfPage();
      document.getElementById('pdf-nav').style.display = 'flex';
    }).catch(function (err) {
      wrap.innerHTML = '<div class="pdf-placeholder">Failed to load PDF: ' + esc(String(err)) + '</div>';
    });
  }

  function renderPdfPage() {
    if (!pdfState.doc) return;
    pdfState.doc.getPage(pdfState.page).then(function (page) {
      var viewport = page.getViewport({ scale: pdfState.scale });
      var wrap = document.getElementById('pdf-canvas-wrap');
      wrap.innerHTML = '';
      var stack = document.createElement('div');
      stack.className = 'bbox-overlay-stack';
      stack.id = 'pdf-page-stack';

      var canvas = document.createElement('canvas');
      canvas.id = 'pdf-render-canvas';
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      stack.appendChild(canvas);

      var overlay = document.createElement('canvas');
      overlay.id = 'pdf-overlay-canvas';
      overlay.className = 'bbox-overlay-canvas';
      overlay.width = viewport.width;
      overlay.height = viewport.height;
      stack.appendChild(overlay);

      wrap.appendChild(stack);
      drawGroundingOverlay();
      var renderTask = page.render({ canvasContext: canvas.getContext('2d'), viewport: viewport });
      var renderDone = renderTask && renderTask.promise ? renderTask.promise : Promise.resolve();
      renderDone.then(function () {
        drawGroundingOverlay();
      }).catch(function () {
        drawGroundingOverlay();
      });
      document.getElementById('pdf-page-info').textContent = pdfState.page + ' / ' + pdfState.total;
      document.getElementById('pdf-zoom-info').textContent = Math.round(pdfState.scale * 100) + '%';
      document.getElementById('pdf-prev').disabled = pdfState.page <= 1;
      document.getElementById('pdf-next').disabled = pdfState.page >= pdfState.total;
    });
  }

  window.pdfPrev = function () {
    if (pdfState.page > 1) { pdfState.page--; renderPdfPage(); }
  };
  window.pdfNext = function () {
    if (pdfState.page < pdfState.total) { pdfState.page++; renderPdfPage(); }
  };
  window.pdfZoomIn = function () {
    pdfState.scale = Math.min(pdfState.scale + 0.25, 5.0);
    renderPdfPage();
  };
  window.pdfZoomOut = function () {
    pdfState.scale = Math.max(pdfState.scale - 0.25, 0.5);
    renderPdfPage();
  };

  function initPdfViewerIfNeeded() {
    // Restore saved base URL into input if present
    var input = document.getElementById('pdf-base-url');
    if (input) {
      var saved = getPdfBaseUrl();
      if (saved && !input.value) input.value = saved;
      // Auto-load if we have a base URL and a test is expanded
      if (input.value && state.expandedId) {
        var ex = null;
        for (var i = 0; i < DATA.examples.length; i++) {
          if (DATA.examples[i].id === state.expandedId) { ex = DATA.examples[i]; break; }
        }
        if (ex) {
          setGroundingExample(ex);
          loadPdf(ex.id, ex.productType || 'parse');
        }
      }
    }
  }

  // ─── Pagination ───
  function totalPages() {
    return Math.max(1, Math.ceil(state.filtered.length / state.perPage));
  }

  function renderPagination() {
    var el = document.getElementById('pagination');
    var tp = totalPages();
    if (tp <= 1) { el.innerHTML = ''; return; }

    var html = '';
    html += '<button ' + (state.currentPage <= 1 ? 'disabled' : '') + ' data-page="' + (state.currentPage - 1) + '">&laquo; Prev</button>';

    // Page numbers (show max 7 around current)
    var startP = Math.max(1, state.currentPage - 3);
    var endP = Math.min(tp, startP + 6);
    startP = Math.max(1, endP - 6);

    if (startP > 1) {
      html += '<button data-page="1">1</button>';
      if (startP > 2) html += '<span class="page-info">...</span>';
    }
    for (var p = startP; p <= endP; p++) {
      html += '<button data-page="' + p + '"' + (p === state.currentPage ? ' class="active"' : '') + '>' + p + '</button>';
    }
    if (endP < tp) {
      if (endP < tp - 1) html += '<span class="page-info">...</span>';
      html += '<button data-page="' + tp + '">' + tp + '</button>';
    }

    html += '<button ' + (state.currentPage >= tp ? 'disabled' : '') + ' data-page="' + (state.currentPage + 1) + '">Next &raquo;</button>';
    html += '<span class="page-info">Page ' + state.currentPage + ' of ' + tp + '</span>';

    el.innerHTML = html;

    el.onclick = function (e) {
      var btn = e.target.closest('button[data-page]');
      if (!btn || btn.disabled) return;
      state.currentPage = parseInt(btn.getAttribute('data-page'));
      state.expandedId = null;
      renderTable();
      renderPagination();
      // Scroll to table
      document.getElementById('examples-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
  }

  // ─── Init ───
  function init() {
    renderSummary();
    renderAggMetrics();
    renderAggStats();
    renderTagMetrics();
    renderControls();
    applyFilters();
    renderResultsCount();
    renderTable();
    renderPagination();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
