/**
 * Fullscreen viewer + canvas annotations for pathology tiles.
 */
(function () {
  const STORAGE_KEY = "pathassist_annotations_v1";
  const COLORS = {
    red: "#e11d48",
    yellow: "#ea580c",
    teal: "#0284c7",
    green: "#16a34a",
    white: "#f5f5f4",
  };
  const DEFAULT_INK = 0.72;
  const MIN_INK = 0.25;
  const MAX_INK = 1;
  const WEIGHTS = { s: 0.75, m: 1, l: 1.85 };

  let caseId = null;
  let tool = "pan";
  let color = "red";
  let strokeOpacity = DEFAULT_INK;
  let lineWeight = WEIGHTS.m;
  let drawing = false;
  let currentStroke = null;
  let startPoint = null;
  let onResize = null;
  let delegationBound = false;
  let boundCanvas = null;

  function loadStore() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    } catch {
      return {};
    }
  }

  function saveStore(store) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  }

  function getAnnotations(id) {
    const store = loadStore();
    return store[id] || [];
  }

  function setAnnotations(id, items) {
    const store = loadStore();
    store[id] = items;
    saveStore(store);
  }

  function $(sel, root = document) {
    return root.querySelector(sel);
  }

  function frameEl() {
    return $("#tile-frame");
  }

  function canvasEl() {
    return $("#annotation-canvas");
  }

  function rgba(name, alpha) {
    const hex = COLORS[name] || COLORS.red;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function itemOpacity(item) {
    if (typeof item?.opacity === "number") return item.opacity;
    return DEFAULT_INK;
  }

  function syncCanvas() {
    const frame = frameEl();
    const canvas = canvasEl();
    if (!frame || !canvas) return;
    const w = frame.offsetWidth;
    const h = frame.offsetHeight;
    if (w < 1 || h < 1) return;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    redraw();
  }

  function norm(x, y, w, h) {
    return { x: x / w, y: y / h };
  }

  function denorm(p, w, h) {
    return { x: p.x * w, y: p.y * h };
  }

  function itemWeight(item) {
    if (typeof item?.weight === "number") return item.weight;
    return WEIGHTS.m;
  }

  function strokeWidth(w, weight) {
    return Math.max(2.5, (w / 88) * weight);
  }

  function drawItem(ctx, item, w, h) {
    const alpha = itemOpacity(item);
    const stroke = rgba(item.color, alpha);
    const fill = rgba(item.color, alpha);
    const wt = itemWeight(item);
    const lineW = strokeWidth(w, wt);
    ctx.lineWidth = lineW;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    function contrastStroke(drawPath) {
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.45)";
      ctx.lineWidth = lineW + 1.75;
      drawPath();
      ctx.stroke();
      ctx.restore();
      ctx.strokeStyle = stroke;
      ctx.fillStyle = fill;
      ctx.lineWidth = lineW;
      drawPath();
      ctx.stroke();
    }

    if (item.type === "stroke") {
      if (item.points.length < 2) return;
      contrastStroke(() => {
        ctx.beginPath();
        const p0 = denorm(item.points[0], w, h);
        ctx.moveTo(p0.x, p0.y);
        for (let i = 1; i < item.points.length; i++) {
          const p = denorm(item.points[i], w, h);
          ctx.lineTo(p.x, p.y);
        }
      });
      return;
    }

    if (item.type === "circle") {
      const c = denorm(item.center, w, h);
      const r = item.radius * Math.min(w, h);
      contrastStroke(() => {
        ctx.beginPath();
        ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
      });
      return;
    }

    if (item.type === "rect") {
      const x = item.x * w;
      const y = item.y * h;
      const rw = item.w * w;
      const rh = item.h * h;
      contrastStroke(() => {
        ctx.beginPath();
        ctx.rect(x, y, rw, rh);
      });
      return;
    }

    if (item.type === "arrow") {
      const a = denorm(item.from, w, h);
      const b = denorm(item.to, w, h);
      contrastStroke(() => {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
      });
      const angle = Math.atan2(b.y - a.y, b.x - a.x);
      const head = Math.max(8, w / 32) * (wt / WEIGHTS.m);
      ctx.save();
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.moveTo(b.x, b.y);
      ctx.lineTo(b.x - head * Math.cos(angle - 0.4), b.y - head * Math.sin(angle - 0.4));
      ctx.lineTo(b.x - head * Math.cos(angle + 0.4), b.y - head * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      return;
    }

    if (item.type === "line") {
      const a = denorm(item.from, w, h);
      const b = denorm(item.to, w, h);
      contrastStroke(() => {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
      });
      return;
    }

    if (item.type === "text" && item.text) {
      const p = denorm(item.at, w, h);
      const size = Math.max(11, w / 38);
      ctx.font = `600 ${size}px "Plus Jakarta Sans", system-ui, sans-serif`;
      ctx.textBaseline = "top";
      ctx.lineWidth = Math.max(2, w / 180);
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.strokeText(item.text, p.x + 1, p.y + 1);
      ctx.fillStyle = rgba(item.color, alpha);
      ctx.fillText(item.text, p.x, p.y);
      return;
    }

    if (item.type === "pin") {
      const p = denorm(item.at, w, h);
      const arm = Math.max(5, w / 80) * (wt / WEIGHTS.m);
      const dot = Math.max(2, w / 150) * (wt / WEIGHTS.m);
      ctx.lineWidth = Math.max(2, w / 140) * (wt / WEIGHTS.m);
      contrastStroke(() => {
        ctx.beginPath();
        ctx.moveTo(p.x - arm, p.y);
        ctx.lineTo(p.x + arm, p.y);
        ctx.moveTo(p.x, p.y - arm);
        ctx.lineTo(p.x, p.y + arm);
      });
      ctx.beginPath();
      ctx.arc(p.x, p.y, dot, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function redraw() {
    const canvas = canvasEl();
    if (!canvas || !caseId) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    for (const item of getAnnotations(caseId)) {
      drawItem(ctx, item, w, h);
    }
  }

  function pointerPos(evt) {
    const canvas = canvasEl();
    const rect = canvas.getBoundingClientRect();
    // The canvas may be visually scaled by the viewer's zoom transform, so map
    // the pointer back into intrinsic canvas pixels for correct, aligned marks.
    const sx = rect.width ? canvas.width / rect.width : 1;
    const sy = rect.height ? canvas.height / rect.height : 1;
    return {
      x: (evt.clientX - rect.left) * sx,
      y: (evt.clientY - rect.top) * sy,
      w: canvas.width,
      h: canvas.height,
    };
  }

  function annotationPayload(extra) {
    return { color, opacity: strokeOpacity, weight: lineWeight, ...extra };
  }

  function pushAnnotation(item) {
    const items = getAnnotations(caseId).concat([item]);
    setAnnotations(caseId, items);
    redraw();
  }

  function onPointerDown(evt) {
    if (tool === "pan" || !caseId) return;
    evt.preventDefault();
    canvasEl()?.setPointerCapture(evt.pointerId);
    const { x, y, w, h } = pointerPos(evt);
    const p = norm(x, y, w, h);
    drawing = true;
    if (tool === "pen") {
      currentStroke = annotationPayload({ type: "stroke", points: [p] });
    } else if (tool === "circle" || tool === "arrow" || tool === "rect" || tool === "line") {
      startPoint = p;
    } else if (tool === "text") {
      const label = window.prompt("Label for this region:", "");
      if (label?.trim()) {
        pushAnnotation(annotationPayload({ type: "text", at: p, text: label.trim().slice(0, 80) }));
      }
      drawing = false;
    } else if (tool === "pin") {
      pushAnnotation(annotationPayload({ type: "pin", at: p }));
      drawing = false;
    }
  }

  function onPointerMove(evt) {
    if (!drawing || !caseId) return;
    const { x, y, w, h } = pointerPos(evt);
    const p = norm(x, y, w, h);
    if (tool === "pen" && currentStroke) {
      currentStroke.points.push(p);
      redraw();
      const ctx = canvasEl().getContext("2d");
      drawItem(ctx, currentStroke, w, h);
    }
  }

  function onPointerUp(evt) {
    if (!drawing || !caseId) return;
    const { x, y, w, h } = pointerPos(evt);
    const p = norm(x, y, w, h);
    if (tool === "pen" && currentStroke) {
      if (currentStroke.points.length > 1) pushAnnotation(currentStroke);
      else redraw();
      currentStroke = null;
    } else if (tool === "circle" && startPoint) {
      const dx = (p.x - startPoint.x) * w;
      const dy = (p.y - startPoint.y) * h;
      const r = Math.sqrt(dx * dx + dy * dy) / Math.min(w, h);
      if (r > 0.01) {
        pushAnnotation(annotationPayload({ type: "circle", center: startPoint, radius: r }));
      } else redraw();
      startPoint = null;
    } else if (tool === "rect" && startPoint) {
      const x = Math.min(startPoint.x, p.x);
      const y = Math.min(startPoint.y, p.y);
      const rw = Math.abs(p.x - startPoint.x);
      const rh = Math.abs(p.y - startPoint.y);
      if (rw > 0.008 && rh > 0.008) {
        pushAnnotation(annotationPayload({ type: "rect", x, y, w: rw, h: rh }));
      } else redraw();
      startPoint = null;
    } else if (tool === "arrow" && startPoint) {
      const dx = p.x - startPoint.x;
      const dy = p.y - startPoint.y;
      if (dx * dx + dy * dy > 0.0004) {
        pushAnnotation(annotationPayload({ type: "arrow", from: startPoint, to: p }));
      } else redraw();
      startPoint = null;
    } else if (tool === "line" && startPoint) {
      const dx = p.x - startPoint.x;
      const dy = p.y - startPoint.y;
      if (dx * dx + dy * dy > 0.0004) {
        pushAnnotation(annotationPayload({ type: "line", from: startPoint, to: p }));
      } else redraw();
      startPoint = null;
    }
    drawing = false;
  }

  function setTool(next) {
    tool = next;
    document.querySelectorAll(".ann-tool, .fs-ann-tool").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tool === next);
    });
    const canvas = canvasEl();
    if (canvas) {
      canvas.classList.toggle("drawing", next !== "pan");
    }
  }

  function setLineWeight(next) {
    lineWeight = WEIGHTS[next] || WEIGHTS.m;
    document.querySelectorAll(".ann-weight, .fs-ann-weight").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.weight === next);
    });
    // Weight applies to new marks only — do not redraw existing annotations.
  }

  function syncWeightButtons() {
    const key = Object.entries(WEIGHTS).find(([, v]) => v === lineWeight)?.[0] || "m";
    setLineWeight(key);
  }

  function updateColorSwatches() {
    document.querySelectorAll(".ann-color").forEach((btn) => {
      const name = btn.dataset.color;
      const hex = COLORS[name] || COLORS.red;
      btn.style.setProperty("--swatch", hex);
      btn.style.setProperty("--ink-alpha", String(strokeOpacity));
    });
    const triggerSwatch = document.getElementById("ann-ink-swatch");
    if (triggerSwatch) {
      const hex = COLORS[color] || COLORS.red;
      triggerSwatch.style.setProperty("--swatch", hex);
      triggerSwatch.style.background = hex;
      triggerSwatch.dataset.color = color;
    }
  }

  function setColor(next) {
    color = next;
    document.querySelectorAll(".ann-color").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.color === next);
    });
    updateColorSwatches();
  }

  function setStrokeOpacity(value) {
    const pct = Number(value);
    strokeOpacity = Math.min(MAX_INK, Math.max(MIN_INK, pct / 100));
    updateColorSwatches();
    syncOpacityLabels();
  }

  function syncOpacityLabels() {
    const pct = Math.round(strokeOpacity * 100);
    document.querySelectorAll(".ann-opacity-val").forEach((el) => {
      el.textContent = `${pct}%`;
    });
  }

  function syncOpacitySliders() {
    const pct = Math.round(strokeOpacity * 100);
    const el = document.getElementById("ann-opacity");
    if (el) el.value = String(pct);
    syncOpacityLabels();
  }

  function undo() {
    if (!caseId) return;
    const items = getAnnotations(caseId);
    if (!items.length) {
      if (typeof window.toast === "function") window.toast("Nothing to undo");
      return;
    }
    setAnnotations(caseId, items.slice(0, -1));
    redraw();
  }

  function askConfirm(message) {
    const dialog = $("#confirm-dialog");
    if (!dialog) return Promise.resolve(false);

    const msgEl = $("#confirm-message");
    if (msgEl) msgEl.textContent = message;

    return new Promise((resolve) => {
      const onCancel = () => {
        dialog.close();
        cleanup();
        resolve(false);
      };
      const onOk = () => {
        dialog.close();
        cleanup();
        resolve(true);
      };
      const cleanup = () => {
        dialog.removeEventListener("close", onClose);
        $("#confirm-cancel")?.removeEventListener("click", onCancel);
        $("#confirm-ok")?.removeEventListener("click", onOk);
      };
      const onClose = () => {
        cleanup();
        resolve(false);
      };

      $("#confirm-cancel")?.addEventListener("click", onCancel);
      $("#confirm-ok")?.addEventListener("click", onOk);
      dialog.addEventListener("close", onClose, { once: true });
      dialog.showModal();
    });
  }

  async function clearAll() {
    if (!caseId) return;
    const count = getAnnotations(caseId).length;
    if (!count) {
      if (typeof window.toast === "function") window.toast("No annotations on this slide");
      return;
    }
    const ok = await askConfirm(
      `Remove ${count} mark${count === 1 ? "" : "s"} on this slide? This cannot be undone.`
    );
    if (!ok) return;
    setAnnotations(caseId, []);
    redraw();
    if (typeof window.toast === "function") window.toast("Annotations cleared");
  }

  function bindCanvas() {
    const canvas = canvasEl();
    if (!canvas || canvas === boundCanvas) return;
    boundCanvas = canvas;
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
  }

  function bindAnnotationUi() {
    if (!delegationBound) {
      delegationBound = true;
      document.addEventListener("click", (e) => {
        const toolBtn = e.target.closest(".ann-tool, .fs-ann-tool");
        if (toolBtn?.dataset.tool) {
          setTool(toolBtn.dataset.tool);
          return;
        }
        const colorBtn = e.target.closest(".ann-color");
        if (colorBtn?.dataset.color) {
          setColor(colorBtn.dataset.color);
          return;
        }
        const weightBtn = e.target.closest(".ann-weight, .fs-ann-weight");
        if (weightBtn?.dataset.weight) {
          setLineWeight(weightBtn.dataset.weight);
          return;
        }
        if (e.target.closest("#ann-undo, #fs-ann-undo")) undo();
        if (e.target.closest("#ann-clear, #fs-ann-clear")) clearAll();
      });
      document.addEventListener("input", (e) => {
        if (e.target?.id === "ann-opacity") {
          setStrokeOpacity(e.target.value);
          syncOpacitySliders();
        }
      });
    }
    bindCanvas();
    setTool(tool);
    setColor(color);
    syncOpacitySliders();
    syncWeightButtons();
    updateColorSwatches();
  }

  function init(id, resizeCb) {
    caseId = id;
    onResize = resizeCb;
    bindAnnotationUi();
    setTool("pan");
    setColor("red");
    setStrokeOpacity(DEFAULT_INK * 100);
    syncOpacitySliders();
    syncWeightButtons();
    updateColorSwatches();
    syncCanvas();
    redraw();
  }

  function destroy() {
    caseId = null;
    boundCanvas = null;
  }

  window.PathAssistViewer = {
    init,
    destroy,
    syncCanvas,
    getAnnotations,
    undo,
    clearAll,
  };
})();
