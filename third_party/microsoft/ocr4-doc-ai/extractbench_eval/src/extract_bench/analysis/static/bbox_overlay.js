// Shared bbox overlay renderer (layout-det + extract reports)
window.BboxOverlay = (function () {
  var STYLES = {
    gt: { stroke: '#4CAF50', dash: [5, 3], lineWidth: 2 },
    gtExtract: { stroke: '#059669', dash: [5, 3], lineWidth: 2, fill: 'rgba(5, 150, 105, 0.12)' },
    pred: { stroke: '#2196F3', dash: [], lineWidth: 2 },
    predExtract: { stroke: '#2563eb', dash: [], lineWidth: 2, fill: 'rgba(37, 99, 235, 0.12)' },
    predAlt: { stroke: '#9C27B0', dash: [3, 2], lineWidth: 2, fill: 'rgba(156, 39, 176, 0.12)' },
    selectedGt: { stroke: '#ea580c', dash: [], lineWidth: 3, fill: 'rgba(234, 88, 12, 0.18)' },
    selectedPred: { stroke: '#eab308', dash: [], lineWidth: 3, fill: 'rgba(234, 179, 8, 0.22)' },
    selectedPredAlt: { stroke: '#c026d3', dash: [], lineWidth: 3, fill: 'rgba(192, 38, 211, 0.22)' },
  };

  var LAYOUTDET_COLORS = {
    'Caption': '#E91E63', 'Footnote': '#9C27B0', 'Formula': '#673AB7',
    'List-item': '#3F51B5', 'Page-footer': '#2196F3', 'Page-header': '#00BCD4',
    'Picture': '#4CAF50', 'Section-header': '#FF9800', 'Table': '#FF5722',
    'Text': '#795548', 'Title': '#F44336'
  };

  function toRect(bbox, width, height, format) {
    if (!bbox || bbox.length < 4) return null;
    if (format === 'xyxy') {
      return {
        x: bbox[0] * width,
        y: bbox[1] * height,
        w: (bbox[2] - bbox[0]) * width,
        h: (bbox[3] - bbox[1]) * height,
      };
    }
    return {
      x: bbox[0] * width,
      y: bbox[1] * height,
      w: bbox[2] * width,
      h: bbox[3] * height,
    };
  }

  function drawRect(ctx, rect, style) {
    if (!rect || rect.w <= 0 || rect.h <= 0) return;
    ctx.save();
    ctx.setLineDash(style.dash || []);
    ctx.lineWidth = style.lineWidth || 2;
    ctx.strokeStyle = style.stroke || '#2563eb';
    if (style.fill) {
      ctx.fillStyle = style.fill;
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    }
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    if (style.label) {
      ctx.font = (style.font || '11px var(--font-body, Arial)');
      var tw = ctx.measureText(style.label).width;
      ctx.fillStyle = style.stroke;
      ctx.fillRect(rect.x, Math.max(0, rect.y - 14), tw + 6, 14);
      ctx.fillStyle = '#fff';
      ctx.fillText(style.label, rect.x + 3, Math.max(10, rect.y - 3));
    }
    ctx.restore();
  }

  function syncCanvasToImage(overlayCanvas, img) {
    if (!overlayCanvas || !img) return null;
    overlayCanvas.width = img.clientWidth;
    overlayCanvas.height = img.clientHeight;
    return {
      width: overlayCanvas.width,
      height: overlayCanvas.height,
      scale: img.clientWidth / img.naturalWidth,
    };
  }

  function syncCanvasToCanvas(overlayCanvas, baseCanvas) {
    if (!overlayCanvas || !baseCanvas) return null;
    overlayCanvas.width = baseCanvas.width;
    overlayCanvas.height = baseCanvas.height;
    return { width: overlayCanvas.width, height: overlayCanvas.height, scale: 1 };
  }

  function clearCanvas(ctx, canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function drawBoxes(ctx, items, width, height, options) {
    var format = (options && options.format) || 'coco';
    var getBbox = (options && options.getBbox) || function (item) { return item.bbox; };
    var getStyle = options && options.getStyle;
    var filter = options && options.filter;
    items.forEach(function (item) {
      if (filter && !filter(item)) return;
      var bbox = getBbox(item);
      var rect = toRect(bbox, width, height, format);
      var style = getStyle ? getStyle(item) : (options.style || STYLES.pred);
      drawRect(ctx, rect, style);
    });
  }

  function drawLayoutPredictions(ctx, preds, scale, colors) {
    if (!preds) return;
    var palette = colors || LAYOUTDET_COLORS;
    preds.forEach(function (p) {
      if (!p || !p.bbox) return;
      var bbox = p.bbox.map(function (v) { return v * scale; });
      var color = palette[p.class] || '#999';
      var rect = {
        x: bbox[0],
        y: bbox[1],
        w: bbox[2] - bbox[0],
        h: bbox[3] - bbox[1],
      };
      drawRect(ctx, rect, {
        stroke: color,
        fill: color + '1a',
        lineWidth: 2,
        label: p.score != null ? (p.class + ' (' + (p.score * 100).toFixed(0) + '%)') : p.class,
        font: '10px Arial',
      });
    });
  }

  function drawLayoutComparisonOverlay(ctx, gt, predA, predB, scale, visibility) {
    if (visibility.showGT && gt && gt.length) {
      // Layout-det GT is COCO [x, y, w, h]; the callback converts to corners,
      // so the format must be 'xyxy' or toRect would read x2/y2 as w/h.
      drawBoxes(ctx, gt, 1, 1, {
        format: 'xyxy',
        getBbox: function (item) {
          return [
            item.bbox[0] * scale,
            item.bbox[1] * scale,
            (item.bbox[0] + item.bbox[2]) * scale,
            (item.bbox[1] + item.bbox[3]) * scale,
          ];
        },
        style: STYLES.gt,
      });
    }
    if (visibility.showA && predA && predA.length) {
      drawBoxes(ctx, predA, 1, 1, {
        format: 'xyxy',
        getBbox: function (item) {
          return item.bbox.map(function (v) { return v * scale; });
        },
        style: STYLES.pred,
      });
    }
    if (visibility.showB && predB && predB.length) {
      drawBoxes(ctx, predB, 1, 1, {
        format: 'xyxy',
        getBbox: function (item) {
          return item.bbox.map(function (v) { return v * scale; });
        },
        style: STYLES.predAlt,
      });
    }
  }

  function styleForExtractItem(item, layerStyle, selectedField, layer) {
    var isSelected = selectedField && item.fieldPath === selectedField;
    if (isSelected) {
      var selectedStyle = STYLES.selectedPred;
      if (layer === 'gt') selectedStyle = STYLES.selectedGt;
      else if (layer === 'predB') selectedStyle = STYLES.selectedPredAlt;
      return Object.assign({}, selectedStyle, { label: item.fieldPath });
    }
    return layerStyle;
  }

  function drawExtractGroundingOverlay(ctx, width, height, gt, pred, options) {
    var page = options.page;
    var selectedField = options.selectedField;
    var selectedOnly = options.selectedOnly;
    var showGt = options.showGt !== false;
    var showPred = options.showPred !== false;

    function pageFilter(item) {
      if (Number(item.page) !== Number(page) || !item.bbox) return false;
      if (selectedOnly && item.fieldPath !== selectedField) return false;
      return true;
    }

    if (showPred) {
      drawBoxes(ctx, pred || [], width, height, {
        format: 'coco',
        filter: pageFilter,
        getStyle: function (item) {
          return styleForExtractItem(item, STYLES.predExtract, selectedField, 'pred');
        },
      });
    }
    if (showGt) {
      drawBoxes(ctx, gt || [], width, height, {
        format: 'coco',
        filter: pageFilter,
        getStyle: function (item) {
          return styleForExtractItem(item, STYLES.gtExtract, selectedField, 'gt');
        },
      });
    }
  }

  function drawExtractComparisonOverlay(ctx, width, height, gt, predA, predB, options) {
    var page = options.page;
    var selectedField = options.selectedField;
    var selectedOnly = options.selectedOnly;
    var showGt = options.showGt === true;
    var showA = options.showA !== false;
    var showB = options.showB !== false;

    function pageFilter(item) {
      if (Number(item.page) !== Number(page) || !item.bbox) return false;
      if (selectedOnly && item.fieldPath !== selectedField) return false;
      return true;
    }

    // Draw A then B so B (purple) sits above when boxes overlap; GT last if shown.
    if (showA) {
      drawBoxes(ctx, predA || [], width, height, {
        format: 'coco',
        filter: pageFilter,
        getStyle: function (item) {
          return styleForExtractItem(item, STYLES.predExtract, selectedField, 'pred');
        },
      });
    }
    if (showB) {
      drawBoxes(ctx, predB || [], width, height, {
        format: 'coco',
        filter: pageFilter,
        getStyle: function (item) {
          return styleForExtractItem(item, STYLES.predAlt, selectedField, 'predB');
        },
      });
    }
    if (showGt) {
      drawBoxes(ctx, gt || [], width, height, {
        format: 'coco',
        filter: pageFilter,
        getStyle: function (item) {
          return styleForExtractItem(item, STYLES.gtExtract, selectedField, 'gt');
        },
      });
    }
  }

  return {
    STYLES: STYLES,
    LAYOUTDET_COLORS: LAYOUTDET_COLORS,
    toRect: toRect,
    drawRect: drawRect,
    drawBoxes: drawBoxes,
    clearCanvas: clearCanvas,
    syncCanvasToImage: syncCanvasToImage,
    syncCanvasToCanvas: syncCanvasToCanvas,
    drawLayoutPredictions: drawLayoutPredictions,
    drawLayoutComparisonOverlay: drawLayoutComparisonOverlay,
    drawExtractGroundingOverlay: drawExtractGroundingOverlay,
    drawExtractComparisonOverlay: drawExtractComparisonOverlay,
  };
})();
