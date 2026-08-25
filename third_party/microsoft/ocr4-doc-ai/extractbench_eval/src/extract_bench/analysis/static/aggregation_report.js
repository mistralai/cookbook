(function () {
  function colorClass(rate) {
    if (rate >= 80) return 'emerald';
    if (rate >= 50) return 'amber';
    return 'red';
  }

  function pct(val, d) {
    d = d !== undefined ? d : 1;
    return val.toFixed(d) + '%';
  }

  // Cost formatters. Null means no priced document in the pool — show an em
  // dash rather than a misleading $0.00.
  function usd(v) {
    if (v == null) return '—';
    return v < 1 ? '$' + v.toFixed(4) : '$' + v.toFixed(2);
  }
  function cents(v) {
    if (v == null) return '—';
    return (v * 100).toFixed(2) + '¢';
  }

  function esc(s) {
    if (s == null) return '';
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }

  // __REPORT_DEPS__


  // Divider entry between the paper's headline metrics and the rest.
  var SEPARATOR = '__separator__';

  // ─── State: one metric for every category ───
  var selectedMetric = DATA.defaultMetric;

  // ─── Header ───
  var titleText = DATA.pipelineName
    ? DATA.pipelineName.replace(/_/g, ' ').replace(/\b\w/g, function (c) { return c.toUpperCase(); })
    : 'Evaluation Report';
  document.getElementById('report-title').textContent = titleText;
  document.getElementById('subtitle').textContent = 'Generated: ' + DATA.generatedAt;

  // ─── Helpers ───
  // null (not 0) when this split does not report the selected metric — a zero
  // would read as a real score and drag the average down with it.
  function getSelectedValue(cat) {
    for (var i = 0; i < cat.metrics.length; i++) {
      if (cat.metrics[i].name === selectedMetric) {
        var v = cat.metrics[i].value;
        return (v === null || v === undefined) ? null : v * 100;
      }
    }
    return null;
  }

  function computeAvgScore() {
    var sum = 0, count = 0;
    DATA.categories.forEach(function (cat) {
      var v = getSelectedValue(cat);
      if (v === null) return;
      sum += v;
      count++;
    });
    return count > 0 ? sum / count : null;
  }

  // ─── Render summary cards ───
  function renderSummary() {
    var sc = document.getElementById('summary-cards');
    sc.innerHTML = '';
    var avg = computeAvgScore();
    var hasAvg = avg !== null;
    var ac = hasAvg ? colorClass(avg) : 'muted';

    // Total Files card
    var d1 = document.createElement('div');
    d1.className = 'summary-card';
    d1.innerHTML = '<div class="label">TOTAL FILES</div><div class="big-number">' + DATA.totalFiles + '</div>';
    sc.appendChild(d1);

    // Failed card. These documents are scored zero rather than dropped, so the
    // average beside this card already carries the penalty — the count is what
    // separates "extracted badly" from "never returned an answer".
    var failed = DATA.totalFailed || 0;
    var dF = document.createElement('div');
    dF.className = 'summary-card';
    dF.innerHTML = '<div class="label">FAILED</div>'
      + '<div class="big-number color-' + (failed ? 'red' : 'emerald') + '">'
      + failed + ' / ' + DATA.totalFiles + '</div>'
      + '<div class="sub-note">'
      + (DATA.totalFiles ? pct(100 * failed / DATA.totalFiles) + ' of documents' : 'no documents')
      + '</div>';
    sc.appendChild(dF);

    // Avg Score card
    var d2 = document.createElement('div');
    d2.className = 'summary-card';
    d2.innerHTML = '<div class="label">AVG SCORE</div><div class="big-number color-' + ac + '">'
      + (hasAvg ? pct(avg) : '—') + '</div>';
    sc.appendChild(d2);

    // Cost cards. Both figures pool every document once across splits rather
    // than averaging the per-split means.
    var oc = DATA.overallCost || {};
    if (oc.hasCost) {
      var d3 = document.createElement('div');
      d3.className = 'summary-card';
      d3.innerHTML = '<div class="label">TOTAL COST</div><div class="big-number">' + usd(oc.totalUsd) + '</div>'
        + '<div class="sub-note">' + oc.documents + ' doc' + (oc.documents === 1 ? '' : 's') + '</div>';
      sc.appendChild(d3);

      var d4 = document.createElement('div');
      d4.className = 'summary-card';
      d4.innerHTML = '<div class="label">COST / PAGE</div>'
        + '<div class="big-number">' + cents(oc.meanPerPageUsd) + '</div>'
        + '<div class="sub-note">document-level mean</div>';
      sc.appendChild(d4);
    }
  }

  // ─── Render category cards ───
  function renderCategories() {
    var grid = document.getElementById('categories-grid');
    grid.innerHTML = '';
    // Set grid columns to match number of categories
    var numCats = DATA.categories.length;
    grid.style.gridTemplateColumns = 'repeat(' + numCats + ', 1fr)';

    DATA.categories.forEach(function (cat) {
      var card = document.createElement('div');
      card.className = 'category-card';

      var selMetric = selectedMetric;
      var mainVal = getSelectedValue(cat);
      var hasVal = mainVal !== null;
      var c = hasVal ? colorClass(mainVal) : 'muted';

      var html = '<h3>' + esc(cat.displayName) + ' <span class="file-count">(' + cat.files + ' files)</span></h3>';
      html += '<div class="main-score color-' + c + '">' + (hasVal ? pct(mainVal) : '—') + '</div>';
      html += '<div class="progress-bar-track"><div class="progress-bar-fill bar-' + c
        + '" style="width:' + (hasVal ? Math.min(mainVal, 100) : 0) + '%"></div></div>';

      // Metric list: selected metric first, then the rest in original order
      // (headline block, divider, everything else).
      var sorted = [];
      var rest = [];
      for (var j = 0; j < cat.metrics.length; j++) {
        if (cat.metrics[j].name === selMetric) {
          sorted.unshift(cat.metrics[j]);
        } else {
          rest.push(cat.metrics[j]);
        }
      }
      sorted = sorted.concat(rest);

      for (var k = 0; k < sorted.length; k++) {
        var sm = sorted[k];
        if (sm.name === SEPARATOR) {
          html += '<div class="metric-divider"></div>';
          continue;
        }
        var mc = colorClass(sm.value * 100);
        var selClass = sm.name === selMetric ? ' selected' : '';
        html += '<div class="metric-row' + selClass + '">';
        html += '<span class="metric-name">' + esc(sm.displayName) + '</span>' + tooltipIcon(sm.name);
        html += '<span class="metric-value color-' + mc + '">' + pct(sm.value * 100) + '</span>';
        html += '</div>';
      }

      // Per-split cost: per-page is the comparable figure so it leads, with the
      // split total beneath it as context.
      var cc = cat.cost || {};
      if (cc.hasCost) {
        html += '<div class="cost-block">';
        html += '<span class="cost-page">' + cents(cc.meanPerPageUsd)
          + '<span class="cost-unit"> / page</span></span>';
        html += '<span class="cost-total">' + usd(cc.totalUsd) + ' total</span>';
        html += '</div>';
      }

      card.innerHTML = html;

      // Click on card navigates to detailed report (but not when clicking dropdown)
      card.addEventListener('click', function (e) {
        if (e.target.tagName === 'SELECT' || e.target.tagName === 'OPTION') return;
        if (e.target.closest && e.target.closest('.metric-hint')) return;
        window.location.href = cat.name + '/_evaluation_report_detailed.html';
      });

      grid.appendChild(card);
    });
  }

  // ─── The one metric selector, above the cards it governs ───
  function renderMetricSelector() {
    var select = document.getElementById('metric-select');
    if (!select) return;
    var html = '';
    var metrics = DATA.unifiedMetrics || [];
    for (var i = 0; i < metrics.length; i++) {
      var m = metrics[i];
      // SEPARATOR is a divider, rendered disabled so it can never be selected.
      if (m.name === SEPARATOR) {
        html += '<option disabled>──────────</option>';
        continue;
      }
      var selected = m.name === selectedMetric ? ' selected' : '';
      html += '<option value="' + esc(m.name) + '"' + selected + '>' + esc(m.displayName) + '</option>';
    }
    select.innerHTML = html;
    select.addEventListener('change', function (e) {
      selectedMetric = e.target.value;
      renderCategories();
      renderSummary();
    });
  }

  renderMetricSelector();
  renderSummary();
  renderCategories();
})();
