/**
 * PathAssist Workstation — clinical case review with visual explainability.
 */
const API = "";
const state = {
  view: "dashboard",
  samples: [],
  cases: [],
  organs: [],
  defaultOrgan: "lymph_node",
  selectedOrganId: null,
  pendingUpload: null,
  mission: null,
  model: null,
  health: null,
  selectedCaseId: null,
  sampleFilter: "all",
  activeCase: null,
  viewerLayer: "source",
  overlayOpacity: 0.55,
  viewerFocus: false,
  viewerZoom: 1,
  viewerPan: { x: 0, y: 0 },
  validationResult: null,
  validationFilter: { bucket: "all", outcome: "all", search: "" },
  validationThresholds: {
    detection_threshold: 0.25,
    metastasis_threshold: 0.55,
    min_review_score: 0.15,
  },
  validationThresholdSource: "yaml",
  validationTuneTimer: null,
  sidebarCollapsed: false,
  caseRailCollapsed: false,
};

const ASSET_V = "28";
const VIEWER_ZOOM_MIN = 1;
const VIEWER_ZOOM_MAX = 10;

function viewerPanelEl() {
  return $("#panel-viewer");
}

function isViewerFullscreen() {
  const panel = viewerPanelEl();
  return Boolean(panel && document.fullscreenElement === panel);
}

function isViewerExpanded() {
  return isViewerFullscreen() || state.viewerFocus;
}

function updateZoomUi() {
  const label = $("#zoom-label");
  const canvas = $("#viewer-canvas");
  if (label) label.textContent = `${Math.round((state.viewerZoom || 1) * 100)}%`;
  const zoomed = (state.viewerZoom || 1) > 1.001;
  canvas?.classList.toggle("is-zoomed", zoomed);
}

/**
 * Transform-based pan/zoom. The tile stays at its "fit" size; we visually scale
 * and translate it with a CSS transform (transform-origin: center) inside a
 * fixed, overflow-hidden viewport. This is immune to page scroll and to the
 * various `height/overflow !important` layout rules, and matches how real WSI
 * viewers navigate. Pan offset lives in state.viewerPan (viewport px, relative
 * to the viewport centre).
 */
function viewerViewport() {
  const canvas = $("#viewer-canvas");
  if (!canvas) return null;
  // The stage is the actual square clip region (it can be narrower than the
  // full-width canvas), so all pan/zoom math is relative to it.
  const stage = canvas.querySelector(".viewer-stage") || canvas;
  const rect = stage.getBoundingClientRect();
  return { canvas, stage, rect, w: rect.width, h: rect.height };
}

function clampViewerPan(zoom, base) {
  const vp = viewerViewport();
  if (!vp || !base?._fitW) return;
  const scaledW = base._fitW * zoom;
  const scaledH = base._fitH * zoom;
  const maxX = Math.max(0, (scaledW - vp.w) / 2);
  const maxY = Math.max(0, (scaledH - vp.h) / 2);
  state.viewerPan.x = Math.min(maxX, Math.max(-maxX, state.viewerPan.x));
  state.viewerPan.y = Math.min(maxY, Math.max(-maxY, state.viewerPan.y));
}

function applyViewerZoom(base, overlay) {
  if (!base?._fitW) return;
  const zoom = state.viewerZoom || 1;
  const zoomed = zoom > 1.001;
  const canvas = $("#viewer-canvas");
  const frame = $("#tile-frame");

  // Images + frame stay pinned at fit size; the transform does the scaling.
  for (const el of [base, overlay]) {
    if (!el) continue;
    el.style.width = `${base._fitW}px`;
    el.style.height = `${base._fitH}px`;
    el.style.maxWidth = "none";
    el.style.maxHeight = "none";
  }

  clampViewerPan(zoom, base);
  if (frame) {
    frame.style.transformOrigin = "center center";
    frame.style.transform =
      `translate(${state.viewerPan.x}px, ${state.viewerPan.y}px) scale(${zoom})`;
    frame.style.willChange = zoomed ? "transform" : "";
  }
  canvas?.classList.toggle("is-zoomed", zoomed);
  updateZoomUi();
  window.PathAssistViewer?.syncCanvas();
}

function setViewerZoom(zoom, base, overlay) {
  state.viewerZoom = Math.min(VIEWER_ZOOM_MAX, Math.max(VIEWER_ZOOM_MIN, zoom));
  applyViewerZoom(base, overlay);
}

/**
 * Zoom toward a screen point (clientX/clientY). Keeps the content point under
 * the cursor fixed. When client coords are omitted, zooms toward the centre.
 */
function zoomViewerAt(factor, clientX, clientY, base, overlay) {
  const vp = viewerViewport();
  if (!vp || !base?._fitW) return;
  const oldZoom = state.viewerZoom || 1;
  const newZoom = Math.min(VIEWER_ZOOM_MAX, Math.max(VIEWER_ZOOM_MIN, oldZoom * factor));
  if (Math.abs(newZoom - oldZoom) < 0.0001) return;

  // Pointer position relative to the viewport centre.
  const ux = (clientX != null ? clientX - vp.rect.left : vp.w / 2) - vp.w / 2;
  const uy = (clientY != null ? clientY - vp.rect.top : vp.h / 2) - vp.h / 2;
  // Content point (frame-local, relative to its centre) currently under cursor.
  const fx = (ux - state.viewerPan.x) / oldZoom;
  const fy = (uy - state.viewerPan.y) / oldZoom;

  state.viewerZoom = newZoom;
  state.viewerPan.x = ux - fx * newZoom;
  state.viewerPan.y = uy - fy * newZoom;
  applyViewerZoom(base, overlay);
}

function zoomViewerBy(direction, base, overlay) {
  zoomViewerAt(direction > 0 ? 1.2 : 1 / 1.2, null, null, base, overlay);
}

function resetViewerZoom(base, overlay) {
  state.viewerZoom = 1;
  state.viewerPan = { x: 0, y: 0 };
  applyViewerZoom(base, overlay);
}

function activePanTool() {
  return document.querySelector(".ann-tool.active, .fs-ann-tool.active")?.dataset.tool === "pan";
}

function canViewerPan(base) {
  if ((state.viewerZoom || 1) <= 1.001 || !base?._fitW) return false;
  const vp = viewerViewport();
  if (!vp) return false;
  const zoom = state.viewerZoom || 1;
  return base._fitW * zoom > vp.w + 1 || base._fitH * zoom > vp.h + 1;
}

function shouldPanPointer(e, base) {
  if (!canViewerPan(base)) return false;
  if (e.button === 1) return true;
  if (e.button !== 0) return false;
  if (e.target.closest(".annotation-canvas.drawing")) return false;
  return activePanTool();
}

/** True only when the pointer is over the rendered slide pixels (#viewer-base). */
function pointerOverViewerImage(clientX, clientY) {
  const base = $("#viewer-base");
  if (!base) return false;
  const r = base.getBoundingClientRect();
  if (r.width < 1 || r.height < 1) return false;
  return clientX > r.left && clientX < r.right && clientY > r.top && clientY < r.bottom;
}

function bindViewerZoom(base, overlay) {
  state._viewerBase = base;
  state._viewerOverlay = overlay;

  if (bindViewerZoom._bound) return;
  bindViewerZoom._bound = true;

  document.addEventListener("click", (e) => {
    const b = state._viewerBase || $("#viewer-base");
    const o = state._viewerOverlay || $("#viewer-overlay");
    if (e.target.closest("#btn-zoom-in")) zoomViewerAt(1.2, null, null, b, o);
    if (e.target.closest("#btn-zoom-out")) zoomViewerAt(1 / 1.2, null, null, b, o);
    if (e.target.closest("#btn-zoom-reset")) resetViewerZoom(b, o);
  });

  // Document-level so handlers survive case re-renders; zoom only on slide pixels.
  document.addEventListener("wheel", (e) => {
    if (!pointerOverViewerImage(e.clientX, e.clientY)) return;
    const b = $("#viewer-base");
    const o = $("#viewer-overlay");
    if (!b?._fitW) return;
    e.preventDefault();
    zoomViewerAt(e.deltaY < 0 ? 1.2 : 1 / 1.2, e.clientX, e.clientY, b, o);
  }, { passive: false });

  let panOrigin = null;
  document.addEventListener("pointerdown", (e) => {
    const canvas = $("#viewer-canvas");
    if (!canvas?.contains(e.target)) return;
    const b = $("#viewer-base");
    if (!shouldPanPointer(e, b)) return;
    e.preventDefault();
    panOrigin = { px: e.clientX, py: e.clientY, ox: state.viewerPan.x, oy: state.viewerPan.y };
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add("is-panning");
  });
  document.addEventListener("pointermove", (e) => {
    if (!panOrigin) return;
    const b = $("#viewer-base");
    const o = $("#viewer-overlay");
    state.viewerPan.x = panOrigin.ox + (e.clientX - panOrigin.px);
    state.viewerPan.y = panOrigin.oy + (e.clientY - panOrigin.py);
    applyViewerZoom(b, o);
  });
  const endPan = (e) => {
    if (!panOrigin) return;
    panOrigin = null;
    $("#viewer-canvas")?.classList.remove("is-panning");
    try { $("#viewer-canvas")?.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
  };
  document.addEventListener("pointerup", endPan);
  document.addEventListener("pointercancel", endPan);
}

function updateViewerFocusUi() {
  const on = isViewerExpanded();
  const panel = viewerPanelEl();
  const mobileFs = on && isReviewMobileLayout();
  $(".review-panels")?.classList.toggle("viewer-focus", on);
  panel?.classList.toggle("is-viewer-expanded", on);
  panel?.classList.toggle("is-fs-markup", on);
  document.body.classList.toggle("mobile-viewer-focus", mobileFs);
  const markupDock = $("#viewer-markup-dock");
  if (markupDock) markupDock.hidden = !on;
  const btn = $("#btn-fullscreen");
  if (btn) {
    btn.textContent = on ? "Exit fullscreen" : "Fullscreen";
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  }
}

async function showReportPanel() {
  if (isViewerFullscreen()) {
    try { await document.exitFullscreen(); } catch (_) { /* ignore */ }
  } else if (state.viewerFocus) {
    setViewerFocus(false);
  }
  window.requestAnimationFrame(() => {
    const dock = $("#report-dock");
    const report = $("#report-box");
    if (!report) return;

    const mobile = document.body.classList.contains("review-mobile")
      || document.body.classList.contains("compact-viewport");

    if (mobile) {
      const scroller = $(".content");
      const target = dock || report;
      if (scroller) {
        const r = target.getBoundingClientRect();
        const s = scroller.getBoundingClientRect();
        scroller.scrollTo({
          top: scroller.scrollTop + r.top - s.top - 20,
          behavior: "smooth",
        });
      } else {
        target.scrollIntoView({ block: "center", behavior: "smooth" });
      }
      return;
    }

    dock?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    report.focus({ preventScroll: true });
  });
}

function onFullscreenChange() {
  state.viewerFocus = isViewerFullscreen();
  updateViewerFocusUi();
  const base = $("#viewer-base");
  const overlay = $("#viewer-overlay");
  if (base && viewerCaseData) {
    applyViewerLayer(state.viewerLayer || "source", viewerCaseData);
  } else if (base) {
    fitTileToViewer(base, overlay);
  }
  window.PathAssistViewer?.syncCanvas();
}

function setViewerFocus(on) {
  state.viewerFocus = on;
  updateViewerFocusUi();
  const base = $("#viewer-base");
  const overlay = $("#viewer-overlay");
  if (base) fitTileToViewer(base, overlay);
  window.PathAssistViewer?.syncCanvas();
}

async function toggleViewerFocus() {
  const panel = viewerPanelEl();
  if (!panel) return;
  if (isReviewMobileLayout()) {
    setViewerFocus(!state.viewerFocus);
    return;
  }
  if (typeof panel.requestFullscreen === "function") {
    try {
      if (!document.fullscreenElement) {
        await panel.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
      return;
    } catch (_) {
      /* fallback below */
    }
  }
  setViewerFocus(!state.viewerFocus);
}

function isReviewMobileLayout() {
  if (window.matchMedia("(max-width: 768px)").matches) return true;
  if (window.matchMedia("(max-height: 640px)").matches) return true;
  if (window.matchMedia("(max-height: 500px) and (orientation: landscape)").matches) return true;
  const workspace = document.querySelector(".review-workspace");
  return Boolean(workspace && workspace.clientWidth < 720);
}

function isCompactViewport() {
  return isReviewMobileLayout();
}

function syncCompactLayout() {
  const mobile = isReviewMobileLayout();
  document.body.classList.toggle("compact-viewport", mobile);
  document.body.classList.toggle("review-mobile", mobile);
  document.body.classList.toggle("review-narrow", Boolean(
    document.querySelector(".review-body")?.clientWidth < 560,
  ));
  if (mobile && state.caseRailCollapsed) {
    applyCaseRailCollapsed(false);
  }
}

function setupIngestPanel() {
  if (setupIngestPanel._bound) return;
  setupIngestPanel._bound = true;
  $("#ingest-body")?.addEventListener("click", (e) => e.stopPropagation());
}

function organById(id) {
  return state.organs.find((o) => o.id === id);
}

function organReady(id) {
  const o = organById(id);
  return o?.model_ready === true;
}

function getSelectedOrganId() {
  const sel = $("#organ-select");
  return sel?.value || state.selectedOrganId || state.defaultOrgan;
}

function organBadgeHtml(organName, opts = {}) {
  const cls = opts.mismatch ? "mismatch" : opts.warn ? "warn" : "";
  const icon = opts.mismatch ? "⚠ " : opts.ready === false ? "○ " : "● ";
  return `<span class="organ-badge ${cls}">${icon}${esc(organName || "—")}</span>`;
}

function updateOrganContextBar(caseData = null) {
  const bar = $("#organ-context-bar");
  const badge = $("#organ-context-badge");
  const hint = $("#organ-model-hint");
  if (!bar || !badge) return;

  const organId = caseData?.organ_id || getSelectedOrganId();
  const organ = organById(organId);
  const name = caseData?.organ_name || organ?.name || "—";
  const mismatch = caseData?.metadata_mismatch;
  const ready = caseData ? true : organ?.model_ready;

  bar.hidden = !organId;
  badge.innerHTML = organBadgeHtml(name, { mismatch, ready: ready !== false });
  if (hint) {
    if (caseData?.organ_task) {
      hint.textContent = caseData.organ_task;
    } else if (organ?.task) {
      hint.textContent = organ.model_ready ? organ.task : "Model not trained";
    } else {
      hint.textContent = "";
    }
  }
}

function readyOrganList() {
  return state.organs.filter((o) => o.model_ready);
}

function pickInitialOrganId(activeOrganId) {
  const ready = readyOrganList();
  if (!ready.length) return null;
  for (const id of [activeOrganId, state.selectedOrganId, state.defaultOrgan]) {
    if (id && ready.some((o) => o.id === id)) return id;
  }
  return ready[0].id;
}

async function fetchSamplesForOrgan(organId) {
  if (!organId) return [];
  try {
    return await api(`/api/samples?organ=${encodeURIComponent(organId)}`);
  } catch {
    return [];
  }
}

// Only trained models appear in the selector.
function populateOrganSelect(selectEl, selectedId) {
  if (!selectEl) return;
  const ready = readyOrganList();
  if (!ready.length) {
    selectEl.innerHTML = `<option value="" disabled selected>No trained models</option>`;
    selectEl.value = "";
    updateUploadZoneState();
    return;
  }
  selectEl.innerHTML = ready
    .map((o) => `<option value="${esc(o.id)}" ${o.id === selectedId ? "selected" : ""}>${esc(o.name)}</option>`)
    .join("");
  selectEl.value = ready.some((o) => o.id === selectedId) ? selectedId : ready[0].id;
  updateUploadZoneState();
}

function updateUploadZoneState() {
  const zone = $("#upload-zone");
  const organId = getSelectedOrganId();
  const ready = organReady(organId);
  if (!zone) return;
  const enabled = Boolean(organId && ready);
  zone.setAttribute("aria-disabled", enabled ? "false" : "true");
  zone.title = enabled
    ? "Upload a histology tile for the selected organ"
    : organId && !ready
      ? "Train a model for this organ before uploading"
      : "Select an organ before uploading";
}

function setupOrganSelectors() {
  const main = $("#organ-select");
  const side = $("#organ-sidebar-select");

  populateOrganSelect(main, state.selectedOrganId);
  if (side) {
    $("#organ-sidebar-block").hidden = false;
    populateOrganSelect(side, state.selectedOrganId);
    side.addEventListener("change", () => selectOrganModel(side.value));
  }
  main?.addEventListener("change", () => selectOrganModel(main.value));

  syncOrganSelectors();
  updateUploadZoneState();
}

function syncOrganSelectors() {
  populateOrganSelect($("#organ-select"), state.selectedOrganId);
  populateOrganSelect($("#organ-sidebar-select"), state.selectedOrganId);
  updateOrganContextBar();
  updateUploadZoneState();
}

// Selecting a model re-scopes the whole workstation (samples, cases, worklist,
// validation, dashboard) to that organ so nothing from another model is shown.
async function selectOrganModel(organId) {
  if (!organId || organId === state.selectedOrganId) return;
  if (!organReady(organId)) {
    syncOrganSelectors();
    return;
  }
  state.selectedOrganId = organId;
  state.selectedCaseId = null;
  state.activeCase = null;
  syncOrganSelectors();

  const organ = organById(organId);
  showLoading(true, `Loading ${organ?.name || organId}…`);
  try {
    try {
      await api("/api/datasets/set-active", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ organ_id: organId }),
      });
    } catch {
      // No imported benchmark yet — upload still works.
    }
    state.model = await api(`/api/model?organ=${encodeURIComponent(organId)}`).catch(() => null);
    state.samples = await fetchSamplesForOrgan(organId);
    state.cases = await api("/api/cases").catch(() => []);
    updateStatusFooter();
    updateOrganContextBar();
    renderDashboard();
    renderSamples();
    if (state.view === "worklist") renderWorklist();
    if (state.view === "validation") await renderValidation();
    if (state.view === "review") showReviewPlaceholder();
    if (!state.samples.length) {
      toast(`${organ?.name || organId}: upload a tile or import holdout from Dataset Hub`);
    } else {
      toast(`Model: ${organ?.name || organId}`);
    }
  } finally {
    showLoading(false);
  }
}

function showReviewPlaceholder() {
  const body = $("#review-body");
  if (body) {
    body.innerHTML = `
      <div class="empty-state empty-review">
        <div class="empty-icon" aria-hidden="true">◎</div>
        <h3>Select a case</h3>
        <p>Pick a tile from the list or upload histology to review triage, overlays, and sign-off.</p>
      </div>`;
  }
}

function setupOrganMismatchDialog() {
  const dlg = $("#organ-mismatch-dialog");
  $("#organ-mismatch-cancel")?.addEventListener("click", () => {
    state.pendingUpload = null;
    dlg?.close();
  });
  $("#organ-mismatch-confirm")?.addEventListener("click", async () => {
    const pending = state.pendingUpload;
    dlg?.close();
    state.pendingUpload = null;
    if (!pending) return;
    if (pending.type === "sample") {
      showLoading(true, `Analyzing ${pending.caseId}…`);
      try {
        const fd = new FormData();
        fd.append("organ", pending.organ);
        fd.append("confirm_mismatch", "true");
        const data = await analyzeWithOrgan(`/api/analyze/sample/${pending.caseId}`, { body: fd });
        const idx = state.samples.findIndex((s) => s.case_id === pending.caseId);
        if (idx >= 0) state.samples[idx] = { ...state.samples[idx], ...data, ready: true };
        state.cases = await api("/api/cases");
        updateOrganContextBar(data);
        renderCaseReview(data);
      } catch (e) {
        toast(e.message);
      } finally {
        showLoading(false);
      }
      return;
    }
    if (pending.file) await uploadFile(pending.file, { confirmMismatch: true, organ: pending.organ });
  });
}

async function analyzeWithOrgan(path, opts = {}) {
  const organ = opts.organ || getSelectedOrganId();
  const method = opts.method || "POST";
  const isForm = opts.body instanceof FormData;
  if (isForm) {
    opts.body.set("organ", organ);
    if (opts.confirmMismatch) opts.body.set("confirm_mismatch", "true");
  }
  const res = await fetch(API + path, { method, body: opts.body, headers: opts.headers });
  if (res.status === 409) {
    const detail = await res.json().catch(() => ({}));
    const msg = detail.detail?.message || detail.message || "Organ metadata mismatch";
    return { mismatch: true, detail: detail.detail || detail, message: msg };
  }
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const VIEW_META = {
  dashboard: { title: "Overview", sub: "System status and model performance" },
  review: { title: "Case Review", sub: "Triage, explainability, and pathologist sign-off" },
  worklist: { title: "Worklist", sub: "Priority-ranked cases awaiting review" },
  validation: { title: "Validation", sub: "Production recall-first metrics on holdout benchmark tiles" },
  datasets: { title: "Dataset Hub", sub: "Search public datasets, import holdout tiles, activate benchmark set" },
  insights: { title: "Insights", sub: "Dataset similarity and model behaviour" },
};

function renderInkPicker() {
  const colorDefs = [
    ["red", "Concern", "Red"],
    ["yellow", "Highlight", "Amber"],
    ["teal", "Reference", "Blue"],
    ["green", "Approved", "Green"],
    ["white", "Light ink", "White", " ann-color-light"],
  ];
  const swatches = colorDefs
    .map(([k, title, aria, extra = ""]) =>
      `<button type="button" class="ann-color${k === "red" ? " active" : ""}${extra}" data-color="${k}" title="${title}" aria-label="${aria} ink"></button>`,
    )
    .join("");
  return `
    <div class="ann-ink-picker" id="ann-ink-picker">
      <button type="button" class="ann-ink-trigger" id="ann-ink-trigger" aria-haspopup="listbox" aria-expanded="false" aria-controls="ann-ink-menu" title="Ink color & opacity">
        <span class="ann-ink-swatch" id="ann-ink-swatch" data-color="red" aria-hidden="true"></span>
        <span class="ann-ink-label">Color</span>
        <svg class="ann-ink-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>
      </button>
      <div class="ann-ink-menu" id="ann-ink-menu" role="listbox" aria-label="Ink color">
        <div class="ann-ink-menu-head">
          <span class="ann-ink-menu-title">Ink color</span>
          <button type="button" class="ann-ink-close" id="ann-ink-close" aria-label="Close color picker">Done</button>
        </div>
        <div class="ann-ink-menu-colors">${swatches}</div>
        <div class="ann-ink-menu-opacity opacity-wrap ann-opacity">
          <span>Opacity</span>
          <input type="range" id="ann-opacity" min="25" max="100" value="72" aria-label="Ink opacity" />
          <span class="ann-opacity-val">72%</span>
        </div>
      </div>
    </div>`;
}

function setupInkPicker() {
  if (setupInkPicker._bound) return;
  setupInkPicker._bound = true;

  const openMenu = () => {
    const menu = $("#ann-ink-menu");
    const trigger = $("#ann-ink-trigger");
    if (!menu || !trigger) return;
    menu.classList.add("is-open");
    menu.hidden = false;
    trigger.setAttribute("aria-expanded", "true");
  };

  const closeMenu = () => {
    const menu = $("#ann-ink-menu");
    const trigger = $("#ann-ink-trigger");
    if (!menu || !menu.classList.contains("is-open")) return;
    menu.classList.remove("is-open");
    menu.hidden = true;
    trigger?.setAttribute("aria-expanded", "false");
  };

  document.addEventListener("click", (e) => {
    const trigger = $("#ann-ink-trigger");
    const menu = $("#ann-ink-menu");
    if (!trigger || !menu) return;

    if (e.target.closest("#ann-ink-close")) {
      e.preventDefault();
      e.stopPropagation();
      closeMenu();
      return;
    }

    if (e.target.closest("#ann-ink-trigger")) {
      e.preventDefault();
      e.stopPropagation();
      if (menu.classList.contains("is-open")) closeMenu();
      else openMenu();
      return;
    }

    if (e.target.closest(".ann-color")) {
      closeMenu();
      return;
    }

    if (e.target.closest("#ann-ink-picker")) return;

    closeMenu();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeMenu();
  });
}

function annotationToolbar(prefix = "", opts = {}) {
  const expanded = Boolean(opts.expanded);
  const basic = Boolean(opts.basic);
  const p = prefix ? `${prefix}-` : "";
  const cls = prefix ? "fs-ann-tool" : "ann-tool";
  const wcls = prefix ? "fs-ann-weight" : "ann-weight";
  const allTools = [
    ["pan", "Pan — move around zoomed slide", "Pan"],
    ["pen", "Freehand ink", "Pen"],
    ["circle", "Circle a region of interest", "Circle"],
    ["rect", "Rectangle ROI", "Box"],
    ["arrow", "Arrow pointer", "Arrow"],
    ["line", "Straight line / measure", "Line"],
    ["pin", "Pin a landmark", "Pin"],
    ["text", "Text label", "Text"],
  ];
  const toolKeys = basic ? ["pan", "pen", "circle"] : allTools.map(([k]) => k);
  const tools = allTools
    .filter(([k]) => toolKeys.includes(k))
    .map(([k, title, label]) =>
      `<button type="button" class="${cls}${k === "pan" ? " active" : ""}" data-tool="${k}" title="${title}">${label}</button>`,
    )
    .join("");
  const weights = `
      <button type="button" class="${wcls}" data-weight="s" title="Thin stroke">S</button>
      <button type="button" class="${wcls} active" data-weight="m" title="Medium stroke">M</button>
      <button type="button" class="${wcls}" data-weight="l" title="Thick stroke">L</button>`;
  const actions = basic
    ? `<button type="button" class="btn btn-ghost btn-sm ann-action" id="${p}ann-undo">Undo</button>`
    : `
      <button type="button" class="btn btn-ghost btn-sm ann-action" id="${p}ann-undo">Undo</button>
      <button type="button" class="btn btn-ghost btn-sm ann-action" id="${p}ann-clear">Clear all</button>`;

  if (expanded) {
    return `
    <div class="annotate-group annotate-group--expanded" aria-label="Markup tools">
      <div class="annotate-row">
        <span class="annotate-row-label">Draw</span>
        <div class="annotate-row-body">${tools}</div>
      </div>
      <div class="annotate-row">
        <span class="annotate-row-label">Style</span>
        <div class="annotate-row-body annotate-style-row">
          <span class="annotate-inline-label">Weight</span>${weights}
          <span class="ann-sep" aria-hidden="true"></span>
          ${actions}
        </div>
      </div>
    </div>`;
  }

  const groupCls = basic ? "annotate-group annotate-group--basic" : "annotate-group";
  const weightBlock = basic
    ? ""
    : `<span class="ann-sep" aria-hidden="true"></span>
      <span class="annotate-inline-label">Wt</span>${weights}`;

  return `
    <div class="${groupCls}" aria-label="Annotation tools">
      ${tools}
      <span class="ann-sep" aria-hidden="true"></span>
      ${weightBlock}
      ${weightBlock ? `<span class="ann-sep" aria-hidden="true"></span>` : ""}
      ${actions}
    </div>`;
}

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

function showLoading(on, msg = "Running pipeline…") {
  $("#loading-msg").textContent = msg;
  $("#loading").classList.toggle("show", on);
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 3200);
}
window.toast = toast;

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function fmtPct(v) {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function priorityBadge(p) {
  const map = {
    URGENT: "badge-urgent",
    HIGH: "badge-high",
    ROUTINE: "badge-routine",
    REVIEW_QC: "badge-qc",
  };
  return `<span class="badge ${map[p] || "badge-routine"}">${p || "—"}</span>`;
}

function predBadge(predicted, correct) {
  const cls =
    predicted === "metastasis" ? "badge-meta" :
    predicted === "borderline" ? "badge-borderline" : "badge-normal";
  let extra = "";
  if (correct === true) extra = ' <span style="color:var(--ok)">✓</span>';
  if (correct === false) extra = ' <span style="color:var(--danger)">✗</span>';
  return `<span class="badge ${cls}">${predicted}</span>${extra}`;
}

function verdictClass(priority) {
  if (priority === "URGENT") return "verdict-urgent";
  if (priority === "HIGH") return "verdict-high";
  if (priority === "REVIEW_QC") return "verdict-qc";
  return "verdict-routine";
}

function clinicalVerdict(data) {
  const p = data.predicted;
  if (p === "metastasis") {
    return {
      headline: "Suspicious for metastasis",
      detail: "Model recommends urgent pathologist review. Verify suspicious regions in Grad-CAM overlay.",
    };
  }
  if (p === "borderline") {
    return {
      headline: "Borderline — review required",
      detail: "Score near decision threshold. Examine uncertainty map and layer activations before sign-off.",
    };
  }
  return {
    headline: "Likely benign appearance",
    detail: "Low malignancy score. Confirm QC passed and no review flags before routine sign-off.",
  };
}

function scoreGauge(score) {
  const s = Math.min(1, Math.max(0, score || 0));
  const pct = Math.round(s * 100);
  const radius = 36;
  const circ = 2 * Math.PI * radius;
  const offset = circ * (1 - s);
  const cls = s >= 0.55 ? "high" : s >= 0.25 ? "mid" : "low";
  const tier = s >= 0.55 ? "High" : s >= 0.25 ? "Elevated" : "Low";
  const hint = s >= 0.55 ? "Prioritize review" : s >= 0.25 ? "Careful review" : "Routine triage";
  return `
    <div class="score-gauge score-gauge--${cls}" role="img" aria-label="AI malignancy score ${pct} percent, ${tier} risk">
      <div class="score-gauge-ring" aria-hidden="true">
        <svg viewBox="0 0 88 88">
          <circle class="score-gauge-bg" cx="44" cy="44" r="${radius}" fill="none" stroke-width="6"/>
          <circle class="score-gauge-arc" cx="44" cy="44" r="${radius}" fill="none" stroke-width="6"
            transform="rotate(-90 44 44)"
            stroke-dasharray="${circ.toFixed(2)}" stroke-dashoffset="${offset.toFixed(2)}"/>
        </svg>
        <div class="score-gauge-center">
          <span class="score-gauge-value">${pct}</span><span class="score-gauge-unit">%</span>
        </div>
      </div>
      <div class="score-gauge-meta">
        <strong class="score-gauge-tier">${tier}</strong>
        <span class="score-gauge-caption">${hint}</span>
      </div>
    </div>`;
}

function explainNarrative(data) {
  const exp = data.nn_explanation;
  const score = data.case_score;
  const pred = data.predicted;
  if (!exp?.layers?.length) {
    return "Neural explainability requires the ensemble model. Run analysis with a trained checkpoint to see Grad-CAM and per-layer activation maps.";
  }
  const early = exp.layers[0];
  const late = exp.layers[exp.layers.length - 1];
  const gc = exp.grad_cam || {};
  const focus = gc.max > 0.6 ? "focal" : "diffuse";
  return `The ${exp.backbone || "model"} assigned ${(score * 100).toFixed(0)}% malignancy (${pred}). `
    + `Grad-CAM shows ${focus} attention (peak ${(gc.max * 100).toFixed(0)}%). `
    + `Early "${early.name}" (${early.stage}) — ${early.description || "edges and tissue boundaries"}. `
    + `Late "${late.name}" (${late.stage}) — ${late.description || "semantic regions"} `
    + `(peak salience ${(late.max_activation * 100).toFixed(0)}%).`;
}

const LAYER_HINTS = {
  source: "Unmodified H&E slide from the scanner — all AI overlays removed",
  gradcam: "Grad-CAM — where the model focused (blended on slide)",
  heatmap: "Malignancy heatmap blended on the slide",
  uncertainty: "Model uncertainty — brighter areas are less confident",
};

let viewerCaseData = null;

function clearViewerOverlay(overlay) {
  if (!overlay) return;
  overlay.hidden = true;
  overlay.classList.add("is-off");
  overlay.style.opacity = "0";
  overlay.removeAttribute("src");
}

function applyViewerLayer(layerId, data = viewerCaseData) {
  const base = $("#viewer-base");
  const overlay = $("#viewer-overlay");
  const slider = $("#opacity-slider");
  if (!base || !data) return;

  state.viewerLayer = layerId;
  const isOriginal = layerId === "source";

  $$(".layer-card").forEach((c) => {
    const on = c.dataset.layer === layerId;
    c.classList.toggle("active", on);
    c.setAttribute("aria-selected", on ? "true" : "false");
  });
  $$(".layer-pill").forEach((p) => {
    const on = p.dataset.layer === layerId;
    p.classList.toggle("active", on);
    p.setAttribute("aria-pressed", on ? "true" : "false");
  });

  $("#blend-wrap")?.classList.toggle("is-hidden", isOriginal);
  $("#blend-wrap-fs")?.classList.toggle("is-hidden", isOriginal);

  base.src = cacheBust(resolveLayerUrl(data, "source"));

  if (isOriginal) {
    clearViewerOverlay(overlay);
    resetViewerZoom(base, overlay);
    fitTileToViewer(base, overlay);
    return;
  }

  overlay.classList.remove("is-off");
  const overlayUrl = resolveLayerUrl(data, layerId);
  if (overlayUrl) {
    overlay.hidden = false;
    overlay.src = cacheBust(overlayUrl);
    overlay.style.opacity = String(state.overlayOpacity);
    if (slider) slider.value = String(Math.round(state.overlayOpacity * 100));
    overlay.onload = () => fitTileToViewer(base, overlay);
  } else {
    clearViewerOverlay(overlay);
    toast("Overlay not available for this layer");
  }
  fitTileToViewer(base, overlay);
}

function setupViewerLayers() {
  if (setupViewerLayers._bound) return;
  setupViewerLayers._bound = true;

  document.addEventListener("click", (e) => {
    const panel = $("#panel-viewer");
    if (!panel?.contains(e.target) || !viewerCaseData) return;
    const card = e.target.closest(".layer-card, .layer-pill");
    if (card?.dataset.layer) applyViewerLayer(card.dataset.layer);
  });

  document.addEventListener("input", (e) => {
    if (e.target.id !== "opacity-slider" && e.target.id !== "opacity-slider-fs") return;
    state.overlayOpacity = Number(e.target.value) / 100;
    const overlay = $("#viewer-overlay");
    if (overlay && !overlay.hidden && !overlay.classList.contains("is-off")) {
      overlay.style.opacity = String(state.overlayOpacity);
    }
    const main = $("#opacity-slider");
    const fs = $("#opacity-slider-fs");
    const val = String(e.target.value);
    if (main && main !== e.target) main.value = val;
    if (fs && fs !== e.target) fs.value = val;
  });
}

function buildLayerCatalog(data) {
  const paths = data.paths || {};
  const exp = data.nn_explanation;
  const expPaths = exp?.paths || {};
  const layers = [
    { id: "source", label: "Original", title: LAYER_HINTS.source, thumb: paths.source },
  ];
  if (expPaths.gradcam) {
    layers.push({
      id: "gradcam",
      label: "Grad-CAM",
      title: LAYER_HINTS.gradcam,
      thumb: expPaths.gradcam,
    });
  }
  if (paths.heatmap) {
    layers.push({
      id: "heatmap",
      label: "Heatmap",
      title: LAYER_HINTS.heatmap,
      thumb: paths.heatmap_thumb || paths.heatmap,
    });
  }
  if (paths.uncertainty) {
    layers.push({
      id: "uncertainty",
      label: "Uncertainty",
      title: LAYER_HINTS.uncertainty,
      thumb: paths.uncertainty_thumb || paths.uncertainty,
    });
  }
  for (const layer of exp?.layers || []) {
    const overlayKey = `activation_${layer.name}`;
    const thumbKey = `heatmap_${layer.name}`;
    if (expPaths[overlayKey]) {
      layers.push({
        id: overlayKey,
        label: layer.name,
        title: `${layer.stage || "Layer"} activation — ${layer.name}`,
        thumb: expPaths[thumbKey] || expPaths[overlayKey],
        stage: layer.stage,
      });
    }
  }
  return layers;
}

function renderFsLayerBar(data) {
  const layers = buildLayerCatalog(data);
  const active = state.viewerLayer || "source";
  const blendHidden = active === "source";
  if (!layers.length) return "";
  return `
    <div class="toolbar-cluster toolbar-cluster-fs-layers" id="toolbar-fs-layers" aria-label="Overlay layers">
      <span class="toolbar-section-label">View</span>
      <div class="toolbar-cluster-body fs-layer-bar">
        ${layers.map((layer) => `
          <button type="button" class="layer-pill ${layer.id === active ? "active" : ""}" data-layer="${layer.id}" title="${esc(layer.title)}" aria-pressed="${layer.id === active}">
            ${esc(layer.label)}
          </button>`).join("")}
        <div class="blend-control opacity-wrap blend-control--inline ${blendHidden ? "is-hidden" : ""}" id="blend-wrap-fs">
          <span>Blend</span>
          <input type="range" id="opacity-slider-fs" min="20" max="80" value="55" aria-label="Overlay blend strength" />
        </div>
      </div>
    </div>`;
}

function renderExplainDock(data) {
  const layers = buildLayerCatalog(data);
  const active = state.viewerLayer || "source";
  const blendHidden = active === "source";
  if (!layers.length) {
    return `<div class="explain-dock" id="explain-dock"><div class="explain-empty">No overlays for this case</div></div>`;
  }
  return `
    <div class="explain-dock" id="explain-dock" aria-label="Slide overlay maps">
      <div class="explain-dock-head">
        <h4>Overlays — select layer</h4>
        <div class="blend-control opacity-wrap ${blendHidden ? "is-hidden" : ""}" id="blend-wrap">
          <span>Blend</span>
          <input type="range" id="opacity-slider" min="20" max="80" value="55" aria-label="Overlay blend strength" />
        </div>
      </div>
      <div class="layer-strip" role="listbox" aria-label="Available overlay maps">
        ${layers.map((layer) => `
          <button type="button" class="layer-card ${layer.id === active ? "active" : ""}" data-layer="${layer.id}" title="${esc(layer.title)}" role="option" aria-selected="${layer.id === active}">
            ${layer.thumb ? `<img src="${cacheBust(layer.thumb)}" alt="" loading="lazy" />` : `<span class="layer-card-fallback" aria-hidden="true">◎</span>`}
            <span class="layer-card-body">
              <span class="layer-card-name">${esc(layer.label)}</span>
              ${layer.stage ? `<span class="layer-card-stage">${esc(layer.stage)}</span>` : ""}
            </span>
          </button>`).join("")}
      </div>
    </div>`;
}

function isSquareTile(w, h) {
  if (!w || !h) return false;
  return Math.abs(w - h) / Math.max(w, h) < 0.08;
}

function fitTileToViewer(base, overlay) {
  const apply = () => {
    const w = base.naturalWidth;
    const h = base.naturalHeight;
    if (!w || !h) return;
    const canvas = base.closest(".viewer-canvas");
    const stage = base.closest(".viewer-stage") || canvas;
    if (!stage) return;

    const panelW = canvas?.parentElement?.clientWidth || canvas?.clientWidth || w;
    const expanded = isViewerExpanded();
    const nativeFs = isViewerFullscreen();
    const panel = canvas?.closest(".panel-viewer");
    const mobileFs = document.body.classList.contains("mobile-viewer-focus");
    const viewportH = mobileFs
      ? (window.visualViewport?.height ?? window.innerHeight)
      : (panel?.clientHeight ?? window.innerHeight);
    const chromeH = ["#viewer-toolbar", "#viewer-markup-dock", "#toolbar-fs-layers", ".viewer-meta"]
      .reduce((sum, sel) => {
        const el = panel?.querySelector(sel);
        if (!el || el.hidden || getComputedStyle(el).display === "none") return sum;
        return sum + el.offsetHeight;
      }, 0);
    const maxW = Math.max(64, (canvas?.clientWidth || stage.clientWidth) - 16);
    let maxH = Math.max(64, (canvas?.clientHeight || stage.clientHeight) - 16);
    if (expanded && panel) {
      maxH = Math.max(120, viewportH - chromeH - (mobileFs ? 12 : 24));
    }
    const square = isSquareTile(w, h);

    let scale;
    if (square) {
      const dockReserve = expanded ? chromeH + (mobileFs ? 16 : 40) : 340;
      const topChrome = expanded ? chromeH + (mobileFs ? 8 : 16) : 120;
      const byWidth = Math.max(200, Math.min(panelW - 16, window.innerWidth - 32) * 0.9);
      const byHeight = Math.max(200, (expanded ? viewportH : window.innerHeight) - topChrome - (expanded ? (mobileFs ? 8 : 24) : dockReserve));
      const maxSide = expanded
        ? Math.min(maxW, maxH, window.innerWidth - (mobileFs ? 24 : 48))
        : Math.min(byWidth, byHeight);
      scale = maxSide / Math.max(w, h);
      if (w <= 512) scale = Math.min(scale, expanded ? 12 : 10);
      else scale = Math.min(scale, 1);
    } else {
      scale = Math.min(maxW / w, maxH / h);
      if (expanded) scale = Math.min(scale, nativeFs ? 12 : 8);
      else if (w <= 128) scale = Math.min(scale, 5);
      else scale = Math.min(scale, 1);
      if (w <= 128 && scale < 2 && maxW >= w * 2) scale = 2;
    }

    const dw = Math.max(1, Math.round(w * scale));
    const dh = Math.max(1, Math.round(h * scale));
    base._fitW = dw;
    base._fitH = dh;

    if (canvas) {
      canvas.classList.toggle("is-square-tile", square);
      const stageEl = canvas.querySelector(".viewer-stage");
      canvas._fitViewH = dh + 16;
      // The viewport box is a fixed square that fills the column; zoom/pan is
      // handled purely by the CSS transform, so layout never changes with zoom.
      canvas.style.height = square && !expanded ? `${dh + 16}px` : "";
      if (stageEl) {
        stageEl.style.width = square ? `${dw}px` : "";
        stageEl.style.height = square ? `${dh}px` : "";
      }
    }

    applyViewerZoom(base, overlay);
  };
  if (base.complete) apply();
  else base.onload = apply;
}

function setView(name) {
  state.view = name;
  document.body.classList.toggle("mode-review", name === "review");
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === name));
  const meta = VIEW_META[name] || { title: name, sub: "" };
  $("#page-title").textContent = meta.title;
  $("#page-subtitle").textContent = meta.sub;
  if (name === "review") {
    updateOrganContextBar(state.activeCase);
  } else if (state.activeCase?.organ_id) {
    updateOrganContextBar(state.activeCase);
  } else {
    updateOrganContextBar();
  }
  if (name === "worklist") renderWorklist();
  if (name === "datasets") renderDatasetHub();
  if (name === "validation") renderValidation();
  if (name === "insights") renderInsights();
  if (name === "review" && !isCompactViewport()) window.scrollTo(0, 0);
  if (!isCompactViewport()) applySidebarCollapsed(true);
  syncCompactLayout();
}

function syncPanelLayout() {
  const base = $("#viewer-base");
  const overlay = $("#viewer-overlay");
  if (base) fitTileToViewer(base, overlay);
  window.PathAssistViewer?.syncCanvas();
}

function setupViewportSync() {
  if (setupViewportSync._bound) return;
  setupViewportSync._bound = true;
  let timer;
  const onChange = () => {
    clearTimeout(timer);
    timer = setTimeout(syncPanelLayout, 80);
  };
  window.addEventListener("resize", onChange);
  window.visualViewport?.addEventListener("resize", onChange);
  window.visualViewport?.addEventListener("scroll", onChange);
}

function loadPanelPrefs() {
  try {
    const raw = localStorage.getItem("pathassist-panels");
    if (!raw) return;
    const prefs = JSON.parse(raw);
    if (typeof prefs.sidebarCollapsed === "boolean") state.sidebarCollapsed = prefs.sidebarCollapsed;
    if (typeof prefs.caseRailCollapsed === "boolean") state.caseRailCollapsed = prefs.caseRailCollapsed;
  } catch (_) { /* ignore corrupt prefs */ }
  document.body.classList.remove("clinical-rail-collapsed");
}

function savePanelPrefs() {
  try {
    localStorage.setItem("pathassist-panels", JSON.stringify({
      sidebarCollapsed: state.sidebarCollapsed,
      caseRailCollapsed: state.caseRailCollapsed,
    }));
  } catch (_) { /* storage unavailable */ }
}

function applySidebarCollapsed(collapsed) {
  state.sidebarCollapsed = collapsed;
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  const btn = $("#sidebar-collapse");
  if (btn) {
    btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    btn.setAttribute("aria-label", collapsed ? "Expand navigation" : "Collapse navigation");
    btn.title = collapsed ? "Expand navigation" : "Collapse navigation";
  }
  savePanelPrefs();
  syncPanelLayout();
  window.setTimeout(syncCompactLayout, 0);
  window.setTimeout(syncPanelLayout, 240);
}

function applyCaseRailCollapsed(collapsed) {
  if (isReviewMobileLayout()) collapsed = false;
  state.caseRailCollapsed = collapsed;
  document.body.classList.toggle("case-rail-collapsed", collapsed);
  const toggle = $("#case-rail-toggle");
  const actions = $("#case-rail-collapsed-actions");
  if (toggle) {
    toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    toggle.setAttribute("aria-label", collapsed ? "Show case list" : "Hide case list");
    toggle.title = collapsed ? "Show case list" : "Hide case list";
  }
  if (actions) actions.hidden = !collapsed;
  savePanelPrefs();
  syncPanelLayout();
  window.setTimeout(syncCompactLayout, 0);
  window.setTimeout(syncPanelLayout, 0);
  window.setTimeout(syncPanelLayout, 240);
}

function setupSidebarCollapse() {
  const btn = $("#sidebar-collapse");
  const brand = $("#brand-toggle");
  if (!btn) return;
  applySidebarCollapsed(state.sidebarCollapsed);
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (window.matchMedia("(max-width: 768px)").matches) return;
    applySidebarCollapsed(!state.sidebarCollapsed);
  });
  brand?.addEventListener("click", () => {
    if (window.matchMedia("(max-width: 768px)").matches) return;
    applySidebarCollapsed(!state.sidebarCollapsed);
  });
  brand?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (window.matchMedia("(max-width: 768px)").matches) return;
      applySidebarCollapsed(!state.sidebarCollapsed);
    }
  });
}

function setupCaseRailToggle() {
  if (setupCaseRailToggle._bound) {
    applyCaseRailCollapsed(state.caseRailCollapsed);
    return;
  }
  setupCaseRailToggle._bound = true;
  applyCaseRailCollapsed(state.caseRailCollapsed);
  document.addEventListener("click", (e) => {
    if (e.target.closest("#case-rail-toggle")) {
      e.stopPropagation();
      applyCaseRailCollapsed(!state.caseRailCollapsed);
      return;
    }
    if (e.target.closest("#case-rail-upload")) {
      applyCaseRailCollapsed(false);
      $("#file-input")?.click();
    }
  });
}

function setSidebarOpen(open) {
  document.body.classList.toggle("sidebar-open", open);
  document.body.style.overflow = open ? "hidden" : "";
  const toggle = $("#nav-toggle");
  const backdrop = $("#sidebar-backdrop");
  if (toggle) toggle.setAttribute("aria-expanded", open ? "true" : "false");
  if (backdrop) backdrop.hidden = !open;
}

function setupSidebarToggle() {
  const toggle = $("#nav-toggle");
  const backdrop = $("#sidebar-backdrop");
  if (!toggle) return;

  toggle.addEventListener("click", () => {
    setSidebarOpen(!document.body.classList.contains("sidebar-open"));
  });
  backdrop?.addEventListener("click", () => setSidebarOpen(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setSidebarOpen(false);
  });
  $$(".nav a").forEach((a) => {
    a.addEventListener("click", () => {
      if (window.matchMedia("(max-width: 768px)").matches) setSidebarOpen(false);
    });
  });
}

async function init() {
  loadPanelPrefs();
  setupSidebarToggle();
  setupSidebarCollapse();
  setupCaseRailToggle();
  setupIngestPanel();
  setupInkPicker();
  setupViewerLayers();
  setupViewportSync();
  document.addEventListener("fullscreenchange", onFullscreenChange);
  syncCompactLayout();
  window.addEventListener("resize", syncCompactLayout);
  const reviewWorkspace = document.querySelector(".review-workspace");
  if (reviewWorkspace && typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => syncCompactLayout());
    ro.observe(reviewWorkspace);
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isViewerExpanded()) {
      if (document.fullscreenElement) document.exitFullscreen();
      else setViewerFocus(false);
    }
  });
  $$(".nav a").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      setView(a.dataset.view);
    });
  });

  setupUpload();
  setupOrganMismatchDialog();
  $("#btn-warm").addEventListener("click", warmCache);
  $("#btn-full-test").addEventListener("click", runFullTest);
  $("#btn-export").addEventListener("click", exportCorrections);

  document.addEventListener("keydown", (e) => {
    if (state.view !== "review" || !state.activeCase) return;
    if (e.target.matches("input, textarea, select")) return;
    const num = parseInt(e.key, 10);
    if (num >= 1 && num <= 9) {
      const buttons = $$(".layer-card, .layer-pill", $("#review-body"));
      const btn = buttons[num - 1];
      if (btn) btn.click();
    }
  });

  try {
    const [health, mission, organsResp] = await Promise.all([
      api("/api/health"),
      api("/api/mission"),
      api("/api/organs"),
    ]);
    state.health = health;
    state.mission = mission;
    state.organs = organsResp.organs || [];
    state.defaultOrgan = organsResp.default_organ || "lymph_node";

    if (!readyOrganList().length) {
      setupOrganSelectors();
      updateStatusFooter();
      renderDashboard();
      renderSamples();
      toast("No trained models found — train a model to begin");
      return;
    }

    const active = await api("/api/datasets/active").catch(() => null);
    state.selectedOrganId = pickInitialOrganId(active?.organ_id);
    setupOrganSelectors();
    if (state.selectedOrganId && state.selectedOrganId !== active?.organ_id) {
      await api("/api/datasets/set-active", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ organ_id: state.selectedOrganId }),
      }).catch(() => null);
    }
    if (organReady(getSelectedOrganId())) {
      state.model = await api(`/api/model?organ=${encodeURIComponent(getSelectedOrganId())}`).catch(() => null);
    }
    state.samples = await fetchSamplesForOrgan(getSelectedOrganId());
    state.cases = await api("/api/cases").catch(() => []);
    updateStatusFooter();
    updateOrganContextBar();
    renderDashboard();
    renderSamples();
    if (state.samples.length && organReady(getSelectedOrganId()) && !state.samples.some((s) => s.ready)) {
      warmCache();
    }
  } catch (e) {
    toast("Failed to connect: " + e.message);
  }
}

function updateStatusFooter() {
  const h = state.health;
  const readyOrgans = state.organs.filter((o) => o.model_ready).length;
  const dot = readyOrgans > 0 ? "ok" : "warn";
  const organ = organById(getSelectedOrganId());
  $("#status-footer").innerHTML =
    `<span class="status-dot ${dot}"></span>` +
    `<span class="status-footer-text">` +
    `${readyOrgans}/${state.organs.length || 0} organ models ready` +
    (organ ? ` · ${esc(organ.name)}` : "") +
    ` · ${h?.cases_cached || 0} cases` +
    `</span>`;
}

function renderDashboard() {
  const h = state.health;
  const m = state.mission;
  const metrics = h?.metrics || state.model?.metrics || {};

  $("#dash-stats").innerHTML = `
    <div class="card"><h3>Ensemble accuracy</h3>
      <div class="value">${fmtPct(metrics.ensemble_acc)}</div>
      <div class="sub">Holdout validation</div></div>
    <div class="card"><h3>Test recall</h3>
      <div class="value">${fmtPct(metrics.test_recall)}</div>
      <div class="sub">Training checkpoint @ 0.5 · see Validation for production</div></div>
    <div class="card"><h3>Cases cached</h3>
      <div class="value">${h?.cases_cached || 0}</div>
      <div class="sub">Ready for review</div></div>
    <div class="card"><h3>Ensemble</h3>
      <div class="value">${state.model?.n_members || "—"}</div>
      <div class="sub">${state.model?.vote_mode || "weighted"} voting</div></div>
  `;

  if (m) {
    $("#dash-mission").innerHTML = `
      <div class="mission-hero">
        <h3>${esc(m.title)}</h3>
        <p>${esc(m.aim)}</p>
        <div class="pipeline">${m.pipeline.map((s) => `<span>${esc(s)}</span>`).join("")}</div>
        <div class="disclaimer">${esc(m.not)}</div>
      </div>`;
  }

  if (state.model?.members?.length) {
    $("#dash-model").innerHTML = `
      <div class="card">
        <h3>Ensemble composition</h3>
        <ul class="member-list">
          ${state.model.members.map((mem) =>
            `<li><span>${esc(mem.name)} <small style="color:var(--muted)">(${esc(mem.backbone)})</small></span>
             <span>${mem.val_acc != null ? (mem.val_acc * 100).toFixed(1) + "%" : "—"}</span></li>`
          ).join("")}
        </ul>
      </div>`;
  } else {
    $("#dash-model").innerHTML = `<div class="card"><h3>Model</h3><p class="sub">No checkpoint loaded — demo mode uses heuristic scorer.</p></div>`;
  }

  $("#dash-quick").innerHTML = `
    <div class="card">
      <h3>Quick actions</h3>
      <div style="display:flex;flex-direction:column;gap:0.5rem;margin-top:0.75rem">
        <button class="btn btn-primary" type="button" id="dash-go-review">Open Case Review</button>
        <button class="btn" type="button" id="dash-go-worklist">View Worklist</button>
        <button class="btn" type="button" id="dash-go-validation">Run validation</button>
      </div>
    </div>`;
  $("#dash-go-review")?.addEventListener("click", () => setView("review"));
  $("#dash-go-worklist")?.addEventListener("click", () => setView("worklist"));
  $("#dash-go-validation")?.addEventListener("click", () => { setView("validation"); renderValidation(); });
}

function caseRowTags(sample) {
  const tags = [];
  if (sample.label_name) {
    tags.push(`<span class="case-row-tag">${esc(sample.label_name)}</span>`);
  }
  if (sample.ready && sample.predicted) {
    const cls =
      sample.predicted === "metastasis" ? "pred-metastasis" :
      sample.predicted === "borderline" ? "pred-borderline" : "pred-normal";
    tags.push(`<span class="case-row-tag ${cls}">${esc(sample.predicted)}</span>`);
  }
  return tags.join("");
}

function renderSamples() {
  const grid = $("#sample-grid");
  if (!grid) return;

  const scrollTop = grid.scrollTop;

  const organId = getSelectedOrganId();
  const organSamples = state.samples.filter(
    (s) => (s.organ_id || organId) === organId,
  );

  const filtered = organSamples.filter((s) => {
    if (state.sampleFilter === "all") return true;
    if (state.sampleFilter === "metastasis") return s.label === 1;
    if (state.sampleFilter === "normal") return s.label === 0;
    return true;
  });

  if (!organSamples.length) {
    const organName = organById(organId)?.name || "this model";
    const ready = organReady(organId);
    grid.innerHTML = `<div class="empty-state" style="padding:1.25rem 0.75rem">${
      ready
        ? `No benchmark tiles for ${esc(organName)}. Import holdout from <strong>Dataset Hub</strong> or upload a tile.`
        : `No model trained for ${esc(organName)} yet.`
    }</div>`;
    return;
  }

  grid.innerHTML = filtered.map((s) => `
    <div class="case-row ${state.selectedCaseId === s.case_id ? "selected" : ""}" data-id="${s.case_id}">
      <img class="case-thumb" src="${s.source_image}" alt="" loading="lazy" />
      <div class="case-row-info">
        <div class="case-row-id">${esc(s.case_id)}</div>
        <div class="case-row-meta">${caseRowTags(s)}</div>
      </div>
      <span class="case-row-status">${s.ready ? (s.correct === false ? "✗" : "✓") : "…"}</span>
    </div>`
  ).join("");

  $$(".case-row", grid).forEach((row) => {
    row.addEventListener("click", () => selectCase(row.dataset.id));
  });

  $$(".filter-chip", $("#sample-filters")).forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.filter === state.sampleFilter);
    chip.onclick = () => {
      state.sampleFilter = chip.dataset.filter;
      renderSamples();
    };
  });

  grid.scrollTop = scrollTop;
  const selected = grid.querySelector(".case-row.selected");
  if (document.body.classList.contains("review-mobile")) {
    selected?.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
  } else {
    selected?.scrollIntoView({ block: "nearest", behavior: "instant" });
  }
}

async function selectCase(caseId) {
  state.selectedCaseId = caseId;
  setView("review");
  renderSamples();

  let data = state.samples.find((s) => s.case_id === caseId);
  const sampleOrgan = getSelectedOrganId() || data?.organ_id;

  if (!data?.ready) {
    showLoading(true, `Analyzing ${caseId}…`);
    try {
      const fd = new FormData();
      fd.append("organ", sampleOrgan);
      const result = await analyzeWithOrgan(`/api/analyze/sample/${caseId}`, { body: fd });
      if (result?.mismatch) {
        state.pendingUpload = { type: "sample", caseId, organ: sampleOrgan };
        $("#organ-mismatch-message").textContent = result.message;
        $("#organ-mismatch-dialog")?.showModal();
        return;
      }
      data = result;
      const idx = state.samples.findIndex((s) => s.case_id === caseId);
      if (idx >= 0) state.samples[idx] = { ...state.samples[idx], ...data, ready: true };
      state.cases = await api("/api/cases");
      updateStatusFooter();
    } finally {
      showLoading(false);
    }
  } else {
    data = await api(`/api/result/${caseId}`);
  }
  updateOrganContextBar(data);
  renderCaseReview(data);
}

function renderCaseReview(data) {
  state.activeCase = data;
  state.viewerFocus = false;
  state.viewerZoom = 1;
  state.viewerPan = { x: 0, y: 0 };
  state.viewerLayer = "source";

  const body = $("#review-body");
  const verdict = clinicalVerdict(data);
  const qc = data.qc || {};
  const flags = data.review_flags || [];
  const pct = Math.round((data.case_score || 0) * 100);

  const organBanner = data.organ_name ? `
    <div class="organ-banner ${data.metadata_mismatch ? "mismatch" : ""}" role="status">
      <strong>Organ:</strong> ${esc(data.organ_name)}
      <span>· ${esc(data.organ_task || data.organ_specialty || "")}</span>
      ${data.metadata_mismatch ? `<span class="flag-chip">Metadata mismatch — verify organ</span>` : ""}
      ${data.wsi_mode === "full_res" && data.wsi ? `<span class="flag-chip">WSI full-res · ${data.wsi.tiles_scored} tiles @ ${Number(data.wsi.level_mpp).toFixed(2)} µm/px</span>` : ""}
    </div>` : "";

  body.innerHTML = `
    ${organBanner}
    <div class="review-layout">
      <header class="case-bar ${verdictClass(data.priority)}">
        <div class="case-bar-score">${scoreGauge(data.case_score)}</div>
        <div class="case-bar-main">
          <div class="case-bar-head">
            <span class="case-bar-id">${esc(data.case_id)}</span>
            <span class="case-bar-verdict">${esc(verdict.headline)}</span>
          </div>
          <p class="case-bar-detail">${esc(verdict.detail)}</p>
        </div>
        <div class="case-bar-chips">
          ${priorityBadge(data.priority)}
          ${predBadge(data.predicted, data.correct)}
        </div>
      </header>

      <div class="review-panels" id="review-panels">
        <div class="panel-viewer" id="panel-viewer">
          <div class="viewer-toolbar" role="toolbar" aria-label="Image viewer controls">
            <div class="toolbar-cluster toolbar-cluster-zoom">
              <span class="toolbar-section-label">Zoom</span>
              <div class="toolbar-cluster-body zoom-controls" aria-label="Zoom controls">
                <button type="button" class="btn btn-ghost btn-sm zoom-btn" id="btn-zoom-out" title="Zoom out" aria-label="Zoom out">−</button>
                <span class="zoom-label" id="zoom-label">100%</span>
                <button type="button" class="btn btn-ghost btn-sm zoom-btn" id="btn-zoom-in" title="Zoom in" aria-label="Zoom in">+</button>
                <button type="button" class="btn btn-ghost btn-sm" id="btn-zoom-reset" title="Reset zoom">Fit</button>
              </div>
            </div>
            <div class="toolbar-cluster toolbar-cluster-actions">
              <button type="button" class="btn-toolbar-action btn-report" id="btn-report" title="Jump to draft report">Report</button>
              <button type="button" class="btn-toolbar-action btn-fullscreen" id="btn-fullscreen" aria-pressed="false">Fullscreen</button>
            </div>
            <div class="toolbar-cluster toolbar-cluster-markup">
              <span class="toolbar-section-label">Mark</span>
              <div class="toolbar-cluster-body">
                ${annotationToolbar("", { basic: true })}
              </div>
            </div>
            <div class="toolbar-cluster toolbar-cluster-ink">
              ${renderInkPicker()}
            </div>
            ${renderFsLayerBar(data)}
          </div>

          <div class="viewer-markup-dock" id="viewer-markup-dock" aria-label="Fullscreen markup palette">
            <div class="viewer-markup-head">
              <strong class="viewer-markup-title">Mark slide</strong>
              <span class="viewer-markup-hint">Pick a tool, then draw on the image</span>
            </div>
            ${annotationToolbar("fs", { expanded: true })}
          </div>

          <div class="viewer-canvas" id="viewer-canvas">
            <div class="viewer-stage">
              <div class="tile-frame" id="tile-frame">
                <img class="viewer-base" id="viewer-base" src="${cacheBust(data.paths?.source)}" alt="Histology" />
                <img class="viewer-overlay" id="viewer-overlay" src="" alt="" hidden />
                <canvas class="annotation-canvas" id="annotation-canvas"></canvas>
              </div>
            </div>
          </div>

          <div class="viewer-meta">
            <span>${data.tile_size}px tile · ${pct}% score</span>
            <span>${data.label_name ? `GT: ${data.label_name}` : ""}</span>
          </div>

          ${renderExplainDock(data)}

          <section class="report-dock" id="report-dock" aria-label="Draft report">
            <div class="report-dock-head">
              <h4>Draft report</h4>
              <span class="report-dock-hint">AI draft — review before sign-off</span>
            </div>
            <div class="report-box" id="report-box" tabindex="0">${esc(data.report || "—")}</div>
          </section>
        </div>

        <aside class="clinical-rail" id="clinical-rail" aria-label="Clinical panel">
          <div class="rail-content" id="clinical-rail-content">
          <div class="clinical-section">
            <h4>Assessment</h4>
            <div class="metric-grid">
              <div class="metric-cell"><label>Score</label><div class="val">${data.case_score}</div></div>
              <div class="metric-cell"><label>Grade</label><div class="val">${data.grade || "—"}</div></div>
              <div class="metric-cell"><label>Unc.</label><div class="val">${data.mean_uncertainty}</div></div>
              <div class="metric-cell"><label>Dis.</label><div class="val">${data.mean_disagreement}</div></div>
            </div>
            ${flags.length ? `<div class="qc-flags" style="margin-top:0.35rem">${flags.map((f) => `<span class="flag-chip">${esc(f)}</span>`).join("")}</div>` : ""}
            <p class="clinical-note">${esc(explainNarrative(data))}</p>
          </div>

          <div class="clinical-section">
            <h4>Quality control</h4>
            <p style="font-size:0.76rem">
              <strong style="color:${qc.passed ? "var(--ok)" : "var(--danger)"}">${qc.passed ? "Passed" : "Failed"}</strong>
              · Tissue ${((qc.tissue_coverage || 0) * 100).toFixed(0)}%
            </p>
            <div class="qc-flags">
              ${(qc.flags || []).length
                ? qc.flags.map((f) => `<span class="qc-flag fail">${esc(f)}</span>`).join("")
                : '<span class="qc-flag pass">OK</span>'}
            </div>
          </div>

          <div class="clinical-section">
            <h4>Pathologist sign-off</h4>
            ${data.review ? `<p style="font-size:0.72rem;margin-bottom:0.35rem"><strong>${esc(data.review.decision)}</strong> · ${esc(data.review.reviewer)}</p>` : ""}
            <form class="review-form" id="review-form" data-case="${esc(data.case_id)}">
              <input name="reviewer" placeholder="Reviewer ID" required />
              <select name="decision" required>
                <option value="">Decision…</option>
                <option value="approve">Approve</option>
                <option value="modify">Modify</option>
                <option value="reject">Reject</option>
              </select>
              <textarea name="note" placeholder="Clinical note"></textarea>
              <button type="submit" class="btn btn-primary btn-sm">Submit review</button>
            </form>
          </div>
          </div>
        </aside>
      </div>
    </div>`;

  bindViewer(data);
  setupInkPicker();
  setupReviewForm();
  document.body.classList.remove("clinical-rail-collapsed");
  bindViewerZoom($("#viewer-base"), $("#viewer-overlay"));
  $("#btn-fullscreen")?.addEventListener("click", toggleViewerFocus);
  $("#btn-report")?.addEventListener("click", showReportPanel);
  updateViewerFocusUi();
  syncCompactLayout();
}

function cacheBust(url) {
  if (!url) return "";
  return url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
}

function resolveLayerUrl(data, layerId) {
  if (layerId === "source") return data.paths?.source;
  if (layerId === "heatmap") return data.paths?.heatmap;
  if (layerId === "uncertainty") return data.paths?.uncertainty;
  if (layerId === "gradcam") return data.nn_explanation?.paths?.gradcam;
  return data.nn_explanation?.paths?.[layerId] || null;
}

function bindViewer(data) {
  viewerCaseData = data;
  const base = $("#viewer-base");
  const overlay = $("#viewer-overlay");
  const slider = $("#opacity-slider");

  if (slider) {
    state.overlayOpacity = Number(slider.value) / 100;
  }

  fitTileToViewer(base, overlay);
  applyViewerLayer("source", data);

  window.PathAssistViewer?.destroy();
  window.PathAssistViewer?.init(data.case_id, () => fitTileToViewer(base, overlay));
}

function setupReviewForm() {
  const form = $("#review-form");
  if (!form) return;
  form.onsubmit = async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    showLoading(true, "Recording decision…");
    try {
      await api("/api/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          case_id: form.dataset.case,
          reviewer: fd.get("reviewer"),
          decision: fd.get("decision"),
          note: fd.get("note") || "",
        }),
      });
      toast("Review recorded");
      if (state.selectedCaseId) selectCase(state.selectedCaseId);
    } catch (err) {
      toast(err.message);
    } finally {
      showLoading(false);
    }
  };
}

function setupUpload() {
  const zone = $("#upload-zone");
  const input = $("#file-input");
  if (!zone || !input) return;

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files.length) uploadFile(input.files[0]);
  });
}

async function uploadFile(file, opts = {}) {
  const organ = opts.organ || getSelectedOrganId();
  if (!organ) {
    toast("Select an organ before uploading");
    return;
  }
  if (!organReady(organ) && !opts.force) {
    toast("No trained model for this organ yet");
    return;
  }

  showLoading(true, `Analyzing upload (${organById(organ)?.name || organ})…`);
  const fd = new FormData();
  fd.append("file", file);
  fd.append("organ", organ);
  if (opts.confirmMismatch) fd.append("confirm_mismatch", "true");

  try {
    const data = await analyzeWithOrgan("/api/analyze/upload", { body: fd });
    if (data?.mismatch) {
      state.pendingUpload = { file, organ };
      $("#organ-mismatch-message").textContent = data.message;
      $("#organ-mismatch-dialog")?.showModal();
      return;
    }
    state.cases = await api("/api/cases");
    state.selectedCaseId = data.case_id;
    state.selectedOrganId = data.organ_id || organ;
    updateStatusFooter();
    updateOrganContextBar(data);
    setView("review");
    renderCaseReview(data);
    toast(`Analyzed ${data.case_id} · ${data.organ_name || organ}`);
  } catch (err) {
    toast(err.message);
  } finally {
    showLoading(false);
  }
}

async function warmCache() {
  showLoading(true, "Warming sample cache…");
  try {
    const r = await api(`/api/warm-cache?organ=${encodeURIComponent(getSelectedOrganId())}`, { method: "POST" });
    state.samples = await fetchSamplesForOrgan(getSelectedOrganId());
    state.cases = await api("/api/cases");
    updateStatusFooter();
    renderSamples();
    renderDashboard();
    toast(`Cached ${r.total_cached} cases (${r.warmed} new)`);
  } catch (e) {
    toast(e.message);
  } finally {
    showLoading(false);
  }
}

async function runFullTest() {
  showLoading(true, "Running benchmark…");
  try {
    const r = await api("/api/run-full-test", { method: "POST" });
    state.samples = await fetchSamplesForOrgan(getSelectedOrganId());
    state.cases = await api("/api/cases");
    updateStatusFooter();
    renderSamples();
    renderDashboard();
    setView("validation");
    await renderValidation();
    toast(`${r.correct}/${r.total} correct (${(r.accuracy * 100).toFixed(0)}%)`);
  } catch (e) {
    toast(e.message);
  } finally {
    showLoading(false);
  }
}

async function exportCorrections() {
  try {
    const r = await api("/api/export/corrections", { method: "POST" });
    window.open(r.download, "_blank");
    toast("Corrections exported");
  } catch (e) {
    toast(e.message);
  }
}

async function renderWorklist() {
  const el = $("#worklist-body");
  try {
    const all = await api("/api/worklist");
    const organId = getSelectedOrganId();
    const items = organId ? all.filter((c) => (c.organ_id || organId) === organId) : all;
    if (!items.length) {
      const organName = organById(organId)?.name || "this model";
      el.innerHTML = `<div class="empty-state">No cases for ${esc(organName)} yet — analyze a tile or warm the cache</div>`;
      return;
    }
    el.innerHTML = `
      <table class="data">
        <thead><tr>
          <th>Case</th><th>Organ</th><th>Priority</th><th>Score</th><th>Prediction</th>
          <th>Grade</th><th>QC</th><th>Review</th>
        </tr></thead>
        <tbody>
          ${items.map((c) => `
            <tr class="clickable" data-id="${esc(c.case_id)}">
              <td><code>${esc(c.case_id)}</code></td>
              <td>${organBadgeHtml(c.organ_name || c.organ_id || "—", { mismatch: c.metadata_mismatch })}</td>
              <td>${priorityBadge(c.priority)}</td>
              <td>${c.case_score}</td>
              <td>${predBadge(c.predicted, c.correct)}</td>
              <td>${c.grade || "—"}</td>
              <td>${c.qc?.passed ? "✓" : "⚠"}</td>
              <td>${c.review?.decision || "—"}</td>
            </tr>`).join("")}
        </tbody>
      </table>`;
    $$("tr.clickable", el).forEach((row) => {
      row.addEventListener("click", () => selectCase(row.dataset.id));
    });
  } catch (e) {
    el.innerHTML = `<div class="empty-state">${esc(e.message)}</div>`;
  }
}

const CONFUSION_LABELS = {
  tp: "True positive",
  tn: "True negative",
  fp: "False positive",
  fn: "False negative",
};

function validationConfusionCell(c) {
  if (c.confusion_cell) return c.confusion_cell;
  const predPos = c.predicted === "metastasis" || c.predicted === "borderline";
  const actualPos = Number(c.label) === 1;
  if (predPos && actualPos) return "tp";
  if (!predPos && !actualPos) return "tn";
  if (predPos && !actualPos) return "fp";
  return "fn";
}

function normalizeValidationCases(cases) {
  return (cases || []).map((c) => ({
    ...c,
    label: Number(c.label),
    correct: c.correct === true,
    confusion_cell: validationConfusionCell(c),
  }));
}

function filterValidationCases(cases) {
  const { bucket, outcome, search } = state.validationFilter;
  const q = search.trim().toLowerCase();
  return cases.filter((c) => {
    const cell = c.confusion_cell || validationConfusionCell(c);
    if (bucket !== "all" && cell !== bucket) return false;
    if (outcome === "correct" && !c.correct) return false;
    if (outcome === "incorrect" && c.correct) return false;
    if (!q) return true;
    const hay = [c.case_id, c.label_name, c.predicted, cell]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return hay.includes(q);
  });
}

function validationFilterSummary(total) {
  const { bucket, outcome, search } = state.validationFilter;
  const parts = [];
  if (bucket !== "all") parts.push(CONFUSION_LABELS[bucket] || bucket.toUpperCase());
  if (outcome === "correct") parts.push("correct only");
  if (outcome === "incorrect") parts.push("errors only");
  if (search.trim()) parts.push(`“${search.trim()}”`);
  if (!parts.length) return `Showing all ${total} cases`;
  return `Showing ${total} case${total === 1 ? "" : "s"} · ${parts.join(" · ")}`;
}

function renderValidationCaseRows(cases) {
  if (!cases.length) {
    return `<tr><td colspan="7" class="empty-row">No cases match the current filters</td></tr>`;
  }
  return cases.map((c) => {
    const bucket = c.confusion_cell || validationConfusionCell(c);
    const score = c.case_score != null ? Number(c.case_score).toFixed(3) : "—";
    const r05 = c.research_predicted || "—";
    return `<tr class="clickable val-case-row" data-id="${esc(c.case_id)}" data-bucket="${esc(bucket)}">
      <td>${esc(c.case_id)}</td>
      <td>${esc(c.label_name)}</td>
      <td>${score}</td>
      <td>${esc(c.predicted)}</td>
      <td>${esc(r05)}</td>
      <td><span class="cm-tag cm-${esc(bucket)}">${esc(bucket.toUpperCase())}</span></td>
      <td>${c.correct ? "✓" : "✗"}</td>
    </tr>`;
  }).join("");
}

function refreshValidationTable(root) {
  const v = state.validationResult;
  if (!v?.cases) return;
  const filtered = filterValidationCases(v.cases);
  const summary = $("#val-filter-summary", root);
  const tbody = $("#val-cases-body", root);
  if (summary) summary.textContent = validationFilterSummary(filtered.length);
  if (tbody) tbody.innerHTML = renderValidationCaseRows(filtered);
  $$(".cm-cell", root).forEach((cell) => {
    cell.classList.toggle("active", state.validationFilter.bucket === cell.dataset.bucket);
    cell.setAttribute("aria-pressed", state.validationFilter.bucket === cell.dataset.bucket);
  });
  $$(".val-filter-btn", root).forEach((btn) => {
    const key = btn.dataset.filter;
    const val = btn.dataset.value;
    const active = state.validationFilter[key] === val;
    btn.classList.toggle("active", active);
  });
}

function bindValidationInteractions(root) {
  if (root._validationBound) return;
  root._validationBound = true;

  const applyFilter = (patch) => {
    state.validationFilter = { ...state.validationFilter, ...patch };
    refreshValidationTable(root);
  };

  root.addEventListener("click", (e) => {
    const cell = e.target.closest(".cm-cell");
    if (cell) {
      const bucket = cell.dataset.bucket;
      const next = state.validationFilter.bucket === bucket ? "all" : bucket;
      applyFilter({ bucket: next });
      return;
    }
    const btn = e.target.closest(".val-filter-btn");
    if (btn) {
      const key = btn.dataset.filter;
      const val = btn.dataset.value;
      const current = state.validationFilter[key];
      applyFilter({ [key]: current === val ? "all" : val });
      return;
    }
    if (e.target.closest("#val-filter-clear")) {
      state.validationFilter = { bucket: "all", outcome: "all", search: "" };
      const search = $("#val-filter-search", root);
      if (search) search.value = "";
      refreshValidationTable(root);
      return;
    }
    const row = e.target.closest(".val-case-row");
    if (row?.dataset.id) selectCase(row.dataset.id);
  });

  root.addEventListener("input", (e) => {
    if (e.target.id === "val-filter-search") {
      applyFilter({ search: e.target.value });
    }
  });
}

function confusionCellHtml(bucket, count, active) {
  return `<button type="button" class="cell cm-cell ${bucket}${active ? " active" : ""}"
    data-bucket="${bucket}" aria-pressed="${active}" title="Filter: ${CONFUSION_LABELS[bucket]}">
    ${bucket.toUpperCase()} ${count}
  </button>`;
}

const VALIDATION_THRESHOLD_FIELDS = [
  {
    key: "detection_threshold",
    label: "Review from",
    hint: "Cases at or above this score are flagged (borderline or metastasis). Lower = fewer missed cancers.",
    step: 0.01,
  },
  {
    key: "metastasis_threshold",
    label: "Strong call",
    hint: "High-confidence metastasis call. Usually leave above review threshold.",
    step: 0.01,
  },
  {
    key: "min_review_score",
    label: "Low-score review",
    hint: "Still review scores in this band — catches faint tumor signal.",
    step: 0.01,
  },
];

function validationThresholdQuery(thr) {
  const params = new URLSearchParams();
  for (const { key } of VALIDATION_THRESHOLD_FIELDS) {
    if (thr[key] != null) params.set(key, String(thr[key]));
  }
  const q = params.toString();
  return q ? `?${q}` : "";
}

async function fetchValidationMetrics(thr = state.validationThresholds) {
  return api(`/api/validation${validationThresholdQuery(thr)}`);
}

function thresholdSourceLabel(source) {
  if (source === "session") return "Applied to Analyze + Validation";
  if (source === "preview") return "Preview (not saved)";
  return "YAML defaults";
}

function renderValidationThresholdControls(thr, defaults, source) {
  const customized = VALIDATION_THRESHOLD_FIELDS.some(
    ({ key }) => defaults && Math.abs((thr[key] ?? 0) - (defaults[key] ?? 0)) > 0.0001,
  );
  const sliders = VALIDATION_THRESHOLD_FIELDS.map(({ key, label, hint, step }) => {
    const val = thr[key] ?? defaults?.[key] ?? 0.25;
    const def = defaults?.[key];
    return `<div class="val-thr-control">
      <label for="val-thr-${key}">
        <span class="val-thr-label">${label}</span>
        <strong class="val-thr-value" data-thr-key="${key}">${Number(val).toFixed(2)}</strong>
        ${def != null ? `<small>YAML ${Number(def).toFixed(2)}</small>` : ""}
      </label>
      <input type="range" id="val-thr-${key}" class="val-thr-slider" data-thr-key="${key}"
        min="0.05" max="0.95" step="${step}" value="${val}" />
      <p class="val-thr-hint">${hint}</p>
    </div>`;
  }).join("");

  return `<div class="panel val-prod-banner val-tune-panel" style="margin-bottom:1.25rem">
    <div class="panel-head val-tune-head">
      <div>
        <h3>Threshold tuning</h3>
        <p>Drag sliders to trade recall vs false alarms on this benchmark — no re-inference.</p>
      </div>
      <div class="val-tune-actions">
        <button type="button" class="btn btn-ghost btn-sm" id="val-thr-reset">Reset to YAML</button>
        <button type="button" class="btn btn-primary btn-sm" id="val-thr-apply">Use for Analyze</button>
      </div>
    </div>
    <div class="val-threshold-tuners">${sliders}</div>
    <p class="val-config-src">
      ${customized ? `<span class="val-tune-badge">Custom</span>` : ""}
      <span id="val-thr-source">${esc(thresholdSourceLabel(source))}</span>
    </p>
  </div>`;
}

function renderValidationMetricsBlock(v) {
  const thr = v.thresholds || {};
  const r05 = v.research_0_5 || {};
  const fnDelta = (r05.fn || 0) - (v.fn || 0);
  return `<div class="panel val-prod-banner" style="margin-bottom:1.25rem">
    <div class="panel-head">
      <h3>Production operating point (recall-first)</h3>
      <p>Live triage rules — not the notebook’s fixed 0.5 cutoff</p>
    </div>
    <div class="val-threshold-grid">
      <div><span class="val-thr-label">Review from</span><strong>${thr.detection_threshold ?? "—"}</strong>
        <small>detection_threshold</small></div>
      <div><span class="val-thr-label">Strong call</span><strong>${thr.metastasis_threshold ?? "—"}</strong>
        <small>metastasis_threshold</small></div>
      <div><span class="val-thr-label">Low-score review</span><strong>${thr.min_review_score ?? "—"}</strong>
        <small>min_review_score</small></div>
      <div><span class="val-thr-label">Borderline</span><strong>counts as positive</strong>
        <small>for recall metrics</small></div>
    </div>
    ${v.config_source ? `<p class="val-config-src">Config: ${esc(v.config_source)}</p>` : ""}
    ${fnDelta > 0 ? `<p class="val-compare-note">At research 0.5 you would miss <strong>${fnDelta}</strong> more metastasis case${fnDelta === 1 ? "" : "s"} (FN ${r05.fn} vs ${v.fn}).</p>` : ""}
  </div>
  <div class="grid-4 val-metrics-grid" style="margin-bottom:1.25rem">
    <div class="card card-prod"><h3>Recall (production)</h3>
      <div class="value">${(v.recall * 100).toFixed(1)}%</div>
      <div class="sub">FN ${v.fn} · borderline = detected</div></div>
    <div class="card"><h3>Precision</h3>
      <div class="value">${(v.precision * 100).toFixed(1)}%</div>
      <div class="sub">FP ${v.fp}</div></div>
    <div class="card"><h3>Accuracy</h3>
      <div class="value">${(v.accuracy * 100).toFixed(1)}%</div>
      <div class="sub">${v.total} labeled tiles</div></div>
    <div class="card card-muted"><h3>Recall @ 0.5 (research)</h3>
      <div class="value">${((r05.recall || 0) * 100).toFixed(1)}%</div>
      <div class="sub">FN ${r05.fn ?? "—"} · notebook default</div></div>
  </div>
  <div class="grid-2">
    <div class="panel">
      <div class="panel-head">
        <h3>Confusion matrix</h3>
        <p>Production rules · click a cell to filter</p>
      </div>
      <div class="confusion" id="val-confusion" role="grid" aria-label="Confusion matrix">
        ${renderValidationConfusion(v)}
      </div>
    </div>
    <div class="panel">
      <div class="panel-head panel-head-row">
        <div>
          <h3>Per-case results</h3>
          <p id="val-filter-summary">${esc(validationFilterSummary(filterValidationCases(normalizeValidationCases(v.cases)).length))}</p>
        </div>
        <button type="button" class="btn btn-ghost btn-sm" id="val-filter-clear">Clear filters</button>
      </div>
      <div class="val-filter-bar">
        <input type="search" id="val-filter-search" class="val-filter-search"
          placeholder="Search case, label, prediction…" value="${esc(state.validationFilter.search)}" />
        <div class="val-filter-chips" aria-label="Confusion filters">
          <button type="button" class="val-filter-btn${state.validationFilter.bucket === "all" ? " active" : ""}"
            data-filter="bucket" data-value="all">All</button>
          <button type="button" class="val-filter-btn${state.validationFilter.bucket === "tp" ? " active" : ""}"
            data-filter="bucket" data-value="tp">TP</button>
          <button type="button" class="val-filter-btn${state.validationFilter.bucket === "tn" ? " active" : ""}"
            data-filter="bucket" data-value="tn">TN</button>
          <button type="button" class="val-filter-btn${state.validationFilter.bucket === "fp" ? " active" : ""}"
            data-filter="bucket" data-value="fp">FP</button>
          <button type="button" class="val-filter-btn${state.validationFilter.bucket === "fn" ? " active" : ""}"
            data-filter="bucket" data-value="fn">FN</button>
        </div>
        <div class="val-filter-chips" aria-label="Outcome filters">
          <button type="button" class="val-filter-btn${state.validationFilter.outcome === "all" ? " active" : ""}"
            data-filter="outcome" data-value="all">Any outcome</button>
          <button type="button" class="val-filter-btn${state.validationFilter.outcome === "correct" ? " active" : ""}"
            data-filter="outcome" data-value="correct">Correct</button>
          <button type="button" class="val-filter-btn${state.validationFilter.outcome === "incorrect" ? " active" : ""}"
            data-filter="outcome" data-value="incorrect">Errors</button>
        </div>
      </div>
      <div class="val-table-wrap">
        <table class="data">
          <thead><tr><th>Case</th><th>Label</th><th>Score</th><th>Pred</th><th>@0.5</th><th>Cell</th><th>OK</th></tr></thead>
          <tbody id="val-cases-body"></tbody>
        </table>
      </div>
    </div>
  </div>`;
}

function renderValidationConfusion(v) {
  const f = state.validationFilter;
  return `<div></div><div class="cell head">Pred +</div><div class="cell head">Pred −</div>
    <div class="cell head">Actual +</div>
    ${confusionCellHtml("tp", v.tp, f.bucket === "tp")}
    ${confusionCellHtml("fn", v.fn, f.bucket === "fn")}
    <div class="cell head">Actual −</div>
    ${confusionCellHtml("fp", v.fp, f.bucket === "fp")}
    ${confusionCellHtml("tn", v.tn, f.bucket === "tn")}`;
}

function updateValidationMetricsDom(root, v) {
  const metricsHost = root.querySelector("#val-metrics-host");
  if (metricsHost) metricsHost.innerHTML = renderValidationMetricsBlock(v);
  const cm = root.querySelector("#val-confusion");
  if (cm) cm.innerHTML = renderValidationConfusion(v);
  refreshValidationTable(root);
}

function bindValidationThresholdTuners(root) {
  const scheduleRefresh = () => {
    clearTimeout(state.validationTuneTimer);
    state.validationTuneTimer = setTimeout(async () => {
      try {
        const v = await fetchValidationMetrics(state.validationThresholds);
        if (!v.ready) return;
        state.validationResult = { ...v, cases: normalizeValidationCases(v.cases) };
        state.validationThresholdSource = v.threshold_source || "preview";
        updateValidationMetricsDom(root, v);
        const src = root.querySelector("#val-thr-source");
        if (src) src.textContent = thresholdSourceLabel(state.validationThresholdSource);
      } catch (e) {
        toast(e.message);
      }
    }, 200);
  };

  root.querySelectorAll(".val-thr-slider").forEach((slider) => {
    slider.addEventListener("input", () => {
      const key = slider.dataset.thrKey;
      const val = Number(slider.value);
      state.validationThresholds[key] = val;
      const label = root.querySelector(`.val-thr-value[data-thr-key="${key}"]`);
      if (label) label.textContent = val.toFixed(2);
      state.validationThresholdSource = "preview";
      const src = root.querySelector("#val-thr-source");
      if (src) src.textContent = thresholdSourceLabel("preview");
      scheduleRefresh();
    });
  });

  root.querySelector("#val-thr-reset")?.addEventListener("click", async () => {
    try {
      const session = await api("/api/session/triage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reset: true }),
      });
      state.validationThresholds = { ...session.defaults };
      state.validationThresholdSource = "yaml";
      await renderValidation();
      toast("Thresholds reset to YAML defaults");
    } catch (e) {
      toast(e.message);
    }
  });

  root.querySelector("#val-thr-apply")?.addEventListener("click", async () => {
    try {
      const session = await api("/api/session/triage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...state.validationThresholds }),
      });
      state.validationThresholds = { ...session.active };
      state.validationThresholdSource = "session";
      const src = root.querySelector("#val-thr-source");
      if (src) src.textContent = thresholdSourceLabel("session");
      toast("Thresholds applied to Analyze + Validation");
    } catch (e) {
      toast(e.message);
    }
  });
}

async function renderValidation() {
  const el = $("#validation-body");
  try {
    const session = await api("/api/session/triage").catch(() => null);
    if (session?.active) {
      state.validationThresholds = { ...session.active };
      state.validationThresholdSource = session.applied_to_analysis ? "session" : "yaml";
    }
    const v = await fetchValidationMetrics(state.validationThresholds);
    if (!v.ready) {
      el.innerHTML = `<div class="empty-state panel">${esc(v.message)}<br>
        <button class="btn btn-primary" style="margin-top:1rem" type="button" id="val-run">Run benchmark</button></div>`;
      $("#val-run")?.addEventListener("click", runFullTest);
      return;
    }
    state.validationResult = { ...v, cases: normalizeValidationCases(v.cases) };
    state.validationThresholdSource = v.threshold_source || state.validationThresholdSource;
    const defaults = v.yaml_defaults || session?.defaults || v.thresholds;
    if (v.thresholds) {
      state.validationThresholds = {
        detection_threshold: v.thresholds.detection_threshold,
        metastasis_threshold: v.thresholds.metastasis_threshold,
        min_review_score: v.thresholds.min_review_score,
      };
    }
    el.innerHTML = `
      ${renderValidationThresholdControls(
        state.validationThresholds,
        defaults,
        state.validationThresholdSource,
      )}
      <div id="val-metrics-host">${renderValidationMetricsBlock(v)}</div>`;
    bindValidationThresholdTuners(el);
    bindValidationInteractions(el);
    refreshValidationTable(el);
  } catch (e) {
    el.innerHTML = `<div class="empty-state">${esc(e.message)}</div>`;
  }
}

let hubFilter = "";
let hubImportJob = null;

function hubStatusBadge(ds) {
  if (ds.is_active) return `<span class="hub-badge active">Active benchmark</span>`;
  const st = ds.import_status;
  if (st?.sample_count) return `<span class="hub-badge ready">${st.sample_count} imported</span>`;
  if (ds.importable) return `<span class="hub-badge">Holdout import ready</span>`;
  return `<span class="hub-badge manual">Manual download</span>`;
}

async function pollImportJob(jobId, organId, onDone) {
  const maxTries = 120;
  for (let i = 0; i < maxTries; i += 1) {
    await new Promise((r) => setTimeout(r, 2000));
    const job = await api(`/api/datasets/import/${jobId}`);
    if (job.status === "running") continue;
    if (job.status === "error") throw new Error(job.error || "Import failed");
    onDone(job.result);
    return;
  }
  throw new Error("Import timed out");
}

async function onDatasetImported(organId, result) {
  const n = result?.sample_count ?? 0;
  if (!n) {
    toast(`Import finished but no tiles were saved for ${organId}`);
    await renderDatasetHub();
    return;
  }
  const organName = organById(organId)?.name || organId;
  toast(
    result?.cached
      ? `Loaded ${n} tiles for ${organName} (cached)`
      : `Imported ${n} holdout tiles for ${organName}`,
  );
  if (organReady(organId)) {
    if (state.selectedOrganId === organId) {
      state.samples = await fetchSamplesForOrgan(organId);
      renderSamples();
      updateOrganContextBar();
    } else {
      await selectOrganModel(organId);
    }
  } else {
    toast(`Train a ${organName} model to analyze these tiles`);
  }
  await renderDatasetHub();
}

async function startLocalDatasetImport(organId) {
  showLoading(true, "Loading tiles from data folder…");
  try {
    const result = await api(`/api/datasets/${organId}/import-local`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activate: true }),
    });
    await onDatasetImported(organId, result);
  } finally {
    showLoading(false);
  }
}

async function startDatasetImport(organId, count, opts = {}) {
  const force = Boolean(opts.force);
  showLoading(true, force ? `Re-downloading ${count} tiles…` : `Loading ${count} holdout tiles…`);
  try {
    const resp = await api(`/api/datasets/${organId}/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ count, activate: true, force }),
    });
    await pollImportJob(resp.job_id, organId, async (result) => {
      await onDatasetImported(organId, result);
    });
  } finally {
    showLoading(false);
  }
}

async function activateDataset(organId) {
  await api("/api/datasets/set-active", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ organ_id: organId }),
  });
  toast(`Active benchmark: ${organId}`);
  if (state.selectedOrganId !== organId) {
    await selectOrganModel(organId);
    return;
  }
  state.samples = await fetchSamplesForOrgan(organId);
  renderSamples();
  await renderDatasetHub();
}

async function renderDatasetHub() {
  const el = $("#datasets-body");
  try {
    const hub = await api("/api/datasets/hub");
    const datasets = hub.datasets || [];
    const q = hubFilter.trim().toLowerCase();
    const filtered = datasets.filter((d) => {
      if (!q) return true;
      const hay = [
        d.name,
        d.organ_id,
        d.task,
        d.notes,
        d.hf_dataset,
      ].filter(Boolean).join(" ").toLowerCase();
      return hay.includes(q);
    });

    el.innerHTML = `
      <div class="hub-toolbar panel">
        <div class="panel-head">
          <h3>Public pathology datasets</h3>
          <p>Import balanced holdout tiles (API) or load your own from <code>data/benchmarks/&lt;organ&gt;/benign|malignant/</code></p>
        </div>
        <div class="hub-toolbar-row">
          <input type="search" id="hub-search" class="hub-search" placeholder="Search organ, task, dataset…" value="${esc(hubFilter)}" />
          <span class="hub-active-label">Active: <strong>${esc(hub.active_organ || "legacy PCam (outputs/real_pcam)")}</strong></span>
        </div>
      </div>
      <div class="hub-grid">
        ${filtered.map((d) => {
          const links = [
            d.hf_url ? `<a href="${esc(d.hf_url)}" target="_blank" rel="noopener">Hugging Face</a>` : "",
            d.manual_url ? `<a href="${esc(d.manual_url)}" target="_blank" rel="noopener">Download page</a>` : "",
          ].filter(Boolean).join(" · ");
          const sim = d.composite_similarity != null
            ? `<div class="sim-bar"><div class="sim-bar-fill" style="width:${(d.composite_similarity * 100).toFixed(0)}%"></div></div>`
            : "";
          const split = d.hf_test_split || d.hf_split || "test";
          const labels = d.label_names
            ? `${esc(d.label_names["0"])} / ${esc(d.label_names["1"])}`
            : "—";
          return `
            <article class="panel hub-card" data-organ="${esc(d.organ_id)}">
              <div class="hub-card-head">
                <div>
                  <h3>${esc(d.name)}</h3>
                  <p class="dataset-meta">${esc(d.task || "")}</p>
                </div>
                ${hubStatusBadge(d)}
              </div>
              ${sim}
              <dl class="hub-meta">
                <div><dt>Holdout split</dt><dd>${esc(split)}</dd></div>
                <div><dt>Labels</dt><dd>${labels}</dd></div>
                <div><dt>Tile size</dt><dd>${esc(String(d.tile_size || "—"))} px</dd></div>
                <div><dt>Source</dt><dd>${links || esc(d.manual_url || "—")}</dd></div>
              </dl>
              <p class="hub-notes">${esc(d.notes || "")}</p>
              <p class="hub-notes hub-local-path">Local folder: <code>${esc(d.local_dir || "")}</code>${d.local_tile_count ? ` · ${d.local_tile_count} tile(s) ready` : " · empty"}</p>
              <div class="hub-actions">
                <button type="button" class="btn btn-ghost hub-local-import" data-organ="${esc(d.organ_id)}" ${d.local_tile_count ? "" : "disabled"} title="Load tiles from data/benchmarks (no API)">Use local folder</button>
                ${d.importable ? `
                  <label class="hub-count-label">Samples
                    <select class="hub-count" data-organ="${esc(d.organ_id)}">
                      <option value="10">10</option>
                      <option value="20" selected>20</option>
                      <option value="50">50</option>
                      <option value="100">100</option>
                    </select>
                  </label>
                  <button type="button" class="btn btn-primary hub-import" data-organ="${esc(d.organ_id)}">${d.import_status?.sample_count ? "Load / add tiles" : "Import holdout"}</button>
                ` : `<span class="hub-manual-hint">Manual tiles — download link only (folder import later)</span>`}
                ${d.import_status?.sample_count ? `
                  <button type="button" class="btn btn-ghost hub-activate" data-organ="${esc(d.organ_id)}" ${d.is_active ? "disabled" : ""}>
                    ${d.is_active ? "Active" : "Set active"}
                  </button>
                  <button type="button" class="btn btn-ghost hub-redownload" data-organ="${esc(d.organ_id)}" title="Re-fetch from the dataset API and overwrite the saved tiles">Re-download</button>
                ` : ""}
              </div>
            </article>`;
        }).join("")}
      </div>`;

    $("#hub-search")?.addEventListener("input", (e) => {
      hubFilter = e.target.value;
      renderDatasetHub();
    });
    $$(".hub-local-import", el).forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await startLocalDatasetImport(btn.dataset.organ);
        } catch (err) {
          toast(err.message);
        }
      });
    });
    $$(".hub-import", el).forEach((btn) => {
      btn.addEventListener("click", async () => {
        const organId = btn.dataset.organ;
        const sel = $(`.hub-count[data-organ="${organId}"]`, el);
        const count = parseInt(sel?.value || "20", 10);
        try {
          await startDatasetImport(organId, count);
        } catch (err) {
          toast(err.message);
        }
      });
    });
    $$(".hub-activate", el).forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await activateDataset(btn.dataset.organ);
        } catch (err) {
          toast(err.message);
        }
      });
    });
    $$(".hub-redownload", el).forEach((btn) => {
      btn.addEventListener("click", async () => {
        const organId = btn.dataset.organ;
        const sel = $(`.hub-count[data-organ="${organId}"]`, el);
        const count = parseInt(sel?.value || "20", 10);
        try {
          await startDatasetImport(organId, count, { force: true });
        } catch (err) {
          toast(err.message);
        }
      });
    });
  } catch (e) {
    el.innerHTML = `<div class="empty-state">${esc(e.message)}</div>`;
  }
}

async function renderInsights() {
  const el = $("#insights-body");
  try {
    const [catalog, corr] = await Promise.all([
      api("/api/datasets/catalog"),
      api("/api/correlation").catch(() => ({ ready: false })),
    ]);

    const datasets = catalog.datasets || [];
    el.innerHTML = `
      <div class="insight-grid">
        <div class="panel">
          <div class="panel-head">
            <h3>Dataset similarity to PatchCamelyon</h3>
            <p>Heuristic ranking for external validation and fine-tuning candidates</p>
          </div>
          ${datasets.slice(0, 8).map((d) => `
            <div class="dataset-row">
              <div>
                <div class="dataset-name">${esc(d.name)}</div>
                <div class="dataset-meta">${esc(d.organ)} · ${esc(d.task)}</div>
                <div class="sim-bar"><div class="sim-bar-fill" style="width:${(d.composite_similarity * 100).toFixed(0)}%"></div></div>
              </div>
              <div class="dataset-score">${(d.composite_similarity * 100).toFixed(0)}%</div>
            </div>`).join("")}
        </div>
        <div class="panel">
          <div class="panel-head">
            <h3>Similarity ↔ model behaviour</h3>
            <p>Correlation between stain similarity and explainability patterns</p>
          </div>
          ${corr.ready ? `
            <div class="corr-grid">
              ${Object.entries(corr.correlations || {}).map(([k, v]) => `
                <div class="corr-cell">
                  <label>${esc(k.replace(/_/g, " "))}</label>
                  <div class="val">${v != null ? Number(v).toFixed(2) : "—"}</div>
                </div>`).join("")}
            </div>
            <ul style="margin-top:1rem;padding-left:1.2rem;font-size:0.82rem;color:var(--text-secondary);line-height:1.6">
              ${(corr.interpretation || []).map((line) => `<li>${esc(line)}</li>`).join("")}
            </ul>
          ` : `<div class="explain-empty">${esc(corr.message || "Run scripts/dataset_correlation.py")}</div>`}
        </div>
      </div>`;
  } catch (e) {
    el.innerHTML = `<div class="empty-state">${esc(e.message)}</div>`;
  }
}

document.addEventListener("DOMContentLoaded", init);
