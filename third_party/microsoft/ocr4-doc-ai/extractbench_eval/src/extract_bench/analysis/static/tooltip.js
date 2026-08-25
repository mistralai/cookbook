// ─── Metric tooltip system (fixed-position, never clipped) ───
var _tipEl = null;
var _tipHideTimer = null;
function _ensureTip() {
  if (_tipEl) return _tipEl;
  _tipEl = document.createElement('div');
  _tipEl.id = 'metric-tooltip';
  document.body.appendChild(_tipEl);
  return _tipEl;
}
function _getTooltipText(metricKey) {
  var tips = (typeof DATA !== 'undefined' && DATA.metricTooltips)
    ? DATA.metricTooltips
    : (typeof metricTooltips !== 'undefined' ? metricTooltips : {});
  var text = tips[metricKey] || '';
  if (!text) {
    if (metricKey.indexOf('field_accuracy_') === 0) {
      var field = metricKey.slice(15).replace(/_/g, ' ');
      text = 'JSON subset match accuracy for the \u201c' + field + '\u201d field.';
    } else if (metricKey.indexOf('rule_') === 0 && metricKey.lastIndexOf('_pass_rate') === metricKey.length - 10) {
      var ruleType = metricKey.slice(5, metricKey.length - 10).replace(/_/g, ' ');
      text = 'Fraction of \u201c' + ruleType + '\u201d rules that pass.';
    } else if (metricKey.indexOf('f1_') === 0) {
      var cls = metricKey.slice(3).replace(/_/g, ' ');
      text = 'F1 score for the \u201c' + cls + '\u201d layout class: 2\u00d7P\u00d7R/(P+R).';
    } else if (metricKey.indexOf('precision_') === 0) {
      var cls2 = metricKey.slice(10).replace(/_/g, ' ');
      text = 'Precision for the \u201c' + cls2 + '\u201d layout class: TP/(TP+FP).';
    } else if (metricKey.indexOf('recall_') === 0) {
      var cls3 = metricKey.slice(7).replace(/_/g, ' ');
      text = 'Recall for the \u201c' + cls3 + '\u201d layout class: TP/(TP+FN).';
    }
  }
  return text;
}
function _showTip(icon) {
  clearTimeout(_tipHideTimer);
  var key = icon.getAttribute('data-metric');
  var text = _getTooltipText(key);
  if (!text) return;
  var tip = _ensureTip();
  tip.textContent = text;
  // Reset: position offscreen to measure, remove classes
  tip.className = '';
  tip.style.cssText = 'position:fixed;display:block;visibility:hidden;top:-9999px;left:-9999px';
  // Measure after text is set
  var iconRect = icon.getBoundingClientRect();
  var tipW = tip.offsetWidth;
  var tipH = tip.offsetHeight;
  var gap = 8;
  // Default: above
  var top = iconRect.top - tipH - gap;
  var arrowDir = 'arrow-bottom';
  // If not enough room above, go below
  if (top < 4) {
    top = iconRect.bottom + gap;
    arrowDir = 'arrow-top';
  }
  // Horizontal: center on icon, but clamp to viewport
  var left = iconRect.left + iconRect.width / 2 - tipW / 2;
  var maxLeft = window.innerWidth - tipW - 8;
  if (left < 8) left = 8;
  if (left > maxLeft) left = maxLeft;
  // Arrow position relative to tooltip
  var arrowLeft = (iconRect.left + iconRect.width / 2 - left);
  arrowLeft = Math.max(12, Math.min(arrowLeft, tipW - 12));
  // Apply final position — clear inline styles so CSS classes take effect
  tip.style.cssText = '';
  tip.style.top = top + 'px';
  tip.style.left = left + 'px';
  tip.style.setProperty('--arrow-left', arrowLeft + 'px');
  tip.className = arrowDir + ' visible';
}
function _hideTip() {
  _tipHideTimer = setTimeout(function () {
    if (_tipEl) { _tipEl.className = ''; }
  }, 80);
}
// Attach listeners via event delegation (mouseover/mouseout bubble, mouseenter/mouseleave do NOT)
var _currentHint = null;
document.addEventListener('mouseover', function (e) {
  var icon = e.target.closest ? e.target.closest('.metric-hint') : null;
  if (icon && icon !== _currentHint) {
    _currentHint = icon;
    _showTip(icon);
  } else if (!icon && _currentHint) {
    _currentHint = null;
    _hideTip();
  }
});
document.addEventListener('mouseout', function (e) {
  if (!_currentHint) return;
  var related = e.relatedTarget;
  if (!related || !(related.closest && related.closest('.metric-hint') === _currentHint)) {
    _currentHint = null;
    _hideTip();
  }
});

function tooltipIcon(metricKey) {
  var text = _getTooltipText(metricKey);
  if (!text) return '';
  return '<span class="metric-hint" data-metric="' + esc(metricKey) + '"></span>';
}
