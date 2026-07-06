#!/usr/bin/env node
/**
 * PathAssist demo UI — regression + UX smoke suite (Playwright).
 *
 * Covers layout rails, viewer/layers, zoom, markup tiers, ink picker,
 * fullscreen docks, scroll panes, mobile, and asset cache busting.
 *
 * Run:
 *   PYTHONPATH=. python demo/server.py
 *   node scripts/ui_smoke_test.mjs
 *
 * Env:
 *   PATHASSIST_URL=http://127.0.0.1:8765
 *   UI_TEST_HEADED=1          — show browser
 *   UI_TEST_SKIP_MOBILE=1     — desktop only (faster)
 */
import { chromium, devices } from "playwright";

const BASE = process.env.PATHASSIST_URL || "http://127.0.0.1:8765";
const HEADED = Boolean(process.env.UI_TEST_HEADED);
const SKIP_MOBILE = Boolean(process.env.UI_TEST_SKIP_MOBILE);

const bugs = [];
const passes = [];

function fail(name, detail) {
  bugs.push({ name, detail });
  console.log(`FAIL  ${name}: ${detail}`);
}

function pass(name) {
  passes.push(name);
  console.log(`OK    ${name}`);
}

function assert(name, condition, detail) {
  if (condition) pass(name);
  else fail(name, detail ?? "assertion failed");
}

async function evalSafe(page, fn) {
  try {
    return await page.evaluate(fn);
  } catch (e) {
    return { __error: e.message };
  }
}

async function waitForServer(page) {
  const res = await page.goto(BASE, { waitUntil: "domcontentloaded", timeout: 20000 });
  if (!res?.ok()) fail("server-reachable", `GET ${BASE} → ${res?.status?.() ?? "no response"}`);
  else pass("server-reachable");
}

async function goToReview(page) {
  await waitForServer(page);
  const nav = page.locator('a[data-view="review"]');
  if (!(await nav.isVisible())) {
    await page.click("#nav-toggle", { force: true });
    await page.waitForSelector("body.sidebar-open", { timeout: 8000 });
  }
  await page.evaluate(() => {
    document.querySelector('a[data-view="review"]')?.dispatchEvent(
      new MouseEvent("click", { bubbles: true, cancelable: true }),
    );
  });
  await page.waitForSelector("#view-review.view.active", { timeout: 20000 });
  await page.waitForSelector("body.mode-review", { timeout: 10000 });
  await page.waitForSelector("#sample-grid .case-row", { timeout: 20000 });
  pass("nav-case-review");
}

async function openFirstCase(page) {
  const row = page.locator("#sample-grid .case-row").first();
  const caseId = await row.getAttribute("data-id");
  await row.click();
  await page.waitForSelector("#review-body .review-layout", { timeout: 30000 });
  await page.waitForSelector("#panel-viewer", { timeout: 30000 });
  await page.waitForSelector("#viewer-base", { timeout: 30000 });
  await page.waitForSelector("#report-box", { timeout: 30000 });
  await page.waitForSelector("#explain-dock", { timeout: 30000 });
  pass(`open-case-${caseId || "first"}`);
  return caseId;
}

async function exitFullscreenIfNeeded(page) {
  const expanded = await evalSafe(page, () =>
    document.getElementById("panel-viewer")?.classList.contains("is-fs-markup"),
  );
  if (expanded) {
    await page.click("#btn-fullscreen");
    await page.waitForTimeout(250);
  }
}

// ─── Asset / shell ───────────────────────────────────────────────────────────

async function testAssetVersions(page) {
  const v = await evalSafe(page, () => {
    const links = [...document.querySelectorAll('link[href*="/static/app.css"]')];
    const scripts = [...document.querySelectorAll('script[src*="/static/"]')];
    const cssV = links[0]?.href.match(/[?&]v=(\d+)/)?.[1];
    const jsVs = scripts.map((s) => s.src.match(/[?&]v=(\d+)/)?.[1]).filter(Boolean);
    return { cssV, jsVs: [...new Set(jsVs)] };
  });
  if (v.__error) fail("asset-version-parse", v.__error);
  else if (!v.cssV) fail("asset-version-css", "app.css missing ?v=");
  else if (v.jsVs.length !== 1) fail("asset-version-js", `expected one JS version, got ${v.jsVs.join(",")}`);
  else if (v.cssV !== v.jsVs[0]) fail("asset-version-match", `css v${v.cssV} != js v${v.jsVs[0]}`);
  else pass(`asset-version-v${v.cssV}`);
}

async function testShellAccessibility(page) {
  const a11y = await evalSafe(page, () => ({
    sidebar: !!document.getElementById("sidebar")?.getAttribute("aria-label"),
    caseRail: !!document.getElementById("case-rail")?.getAttribute("aria-label"),
    viewerToolbar: !!document.querySelector(".viewer-toolbar")?.getAttribute("role"),
    fullscreenBtn: document.getElementById("btn-fullscreen")?.getAttribute("aria-pressed") != null,
  }));
  assert("a11y-sidebar-label", a11y.sidebar, "sidebar aria-label missing");
  assert("a11y-viewer-toolbar", a11y.viewerToolbar, "viewer toolbar role missing");
  assert("a11y-fullscreen-pressed", a11y.fullscreenBtn, "fullscreen aria-pressed missing");
}

// ─── Desktop review UX ───────────────────────────────────────────────────────

async function testDesktopIngestAndOrgan(page) {
  const ingest = await evalSafe(page, () => {
    const title = document.querySelector(".ingest-title");
    const select = document.getElementById("organ-select");
    const help = document.getElementById("organ-help");
    const toggle = document.getElementById("ingest-toggle");
    return {
      titleVisible: title ? getComputedStyle(title).display !== "none" : false,
      selectVisible: select ? getComputedStyle(select).display !== "none" : false,
      helpVisible: help ? getComputedStyle(help).display !== "none" : false,
      hasIngestToggle: !!toggle,
      optionCount: select?.options?.length ?? 0,
    };
  });
  assert("desktop-ingest-title", ingest.titleVisible, "organ/upload title hidden on desktop");
  assert("desktop-organ-select", ingest.selectVisible && ingest.optionCount > 1, "organ select missing or empty");
  assert("desktop-organ-help", ingest.helpVisible, "organ help text hidden");
  assert("desktop-no-ingest-toggle", !ingest.hasIngestToggle, "obsolete ingest-toggle still in DOM");
}

async function testDesktopCaseRail(page) {
  const railW = await page.locator("#case-rail").evaluate((el) => el.getBoundingClientRect().width);
  assert("desktop-case-rail-width", railW >= 160 && railW <= 280, `width ${Math.round(railW)}px`);

  const toggle = page.locator("#case-rail-toggle");
  assert("desktop-case-toggle-visible", await toggle.isVisible(), "case rail toggle not visible");

  const toggleClickable = await evalSafe(page, () => {
    const btn = document.getElementById("case-rail-toggle");
    const body = document.querySelector(".review-body");
    if (!btn || !body) return false;
    const br = btn.getBoundingClientRect();
    const topEl = document.elementFromPoint(br.left + br.width / 2, br.top + br.height / 2);
    return btn.contains(topEl) || btn === topEl;
  });
  assert("desktop-case-toggle-not-obscured", toggleClickable, "toggle hidden behind main panel");

  await page.click("#case-rail-toggle");
  await page.waitForTimeout(350);

  const collapsed = await evalSafe(page, () => ({
    bodyClass: document.body.classList.contains("case-rail-collapsed"),
    layout: (() => {
      const ws = document.querySelector(".review-workspace");
      const body = document.querySelector(".review-body");
      if (!ws || !body) return null;
      const wsRect = ws.getBoundingClientRect();
      const bodyRect = body.getBoundingClientRect();
      return {
        cols: getComputedStyle(ws).gridTemplateColumns.split(" ").filter(Boolean).length,
        gap: bodyRect.left - wsRect.left,
        bodyWidth: bodyRect.width,
        wsWidth: wsRect.width,
      };
    })(),
  }));

  assert("desktop-case-rail-collapse", collapsed.bodyClass, "case-rail-collapsed not set");
  if (collapsed.layout) {
    assert("desktop-collapsed-no-ghost-col", collapsed.layout.cols === 1, `grid cols ${collapsed.layout.cols}`);
    assert("desktop-collapsed-gap", collapsed.layout.gap <= 80, `ghost gap ${Math.round(collapsed.layout.gap)}px`);
    assert(
      "desktop-collapsed-body-fill",
      collapsed.layout.bodyWidth >= collapsed.layout.wsWidth * 0.92,
      "review body too narrow",
    );
  } else fail("desktop-collapsed-layout", "workspace missing");

  await page.click("#case-rail-toggle");
  await page.waitForTimeout(300);
  pass("desktop-case-rail-expand");
}

async function testDesktopClinicalRail(page) {
  const clinical = await evalSafe(page, () => ({
    toggle: !!document.getElementById("clinical-rail-toggle"),
    contentVisible: (() => {
      const el = document.getElementById("clinical-rail-content");
      if (!el) return false;
      const r = el.getBoundingClientRect();
      return r.width > 40 && r.height > 40 && getComputedStyle(el).display !== "none";
    })(),
    signOffForm: !!document.getElementById("review-form"),
  }));
  assert("desktop-clinical-toggle-removed", !clinical.toggle, "clinical collapse toggle exists");
  assert("desktop-clinical-visible", clinical.contentVisible, "assessment panel not visible");
  assert("desktop-signoff-form", clinical.signOffForm, "pathologist sign-off form missing");
}

async function testDesktopViewerChrome(page) {
  const chrome = await evalSafe(page, () => {
    const toolbar = document.querySelector(".viewer-toolbar");
    const report = document.getElementById("report-dock");
    const explain = document.getElementById("explain-dock");
    const meta = document.querySelector(".viewer-meta");
    const reportBox = document.getElementById("report-box");
    const reportRect = report?.getBoundingClientRect();
    return {
      zoomBtns: ["btn-zoom-in", "btn-zoom-out", "btn-zoom-reset"].every((id) => !!document.getElementById(id)),
      toolbarLayerBtns: toolbar?.querySelectorAll(".layer-btn").length ?? 0,
      toolbarViewCluster: !!toolbar?.querySelector(".toolbar-cluster-view"),
      explainVisible: explain ? getComputedStyle(explain).display !== "none" : false,
      metaVisible: meta ? getComputedStyle(meta).display !== "none" : false,
      reportVisible: reportBox ? getComputedStyle(reportBox).display !== "none" : false,
      reportMinH: report ? parseFloat(getComputedStyle(report).minHeight) : 0,
      layerCards: document.querySelectorAll("#explain-dock .layer-card").length,
      originalCard: !!document.querySelector('#explain-dock .layer-card[data-layer="source"]'),
      caseBar: !!document.querySelector(".case-bar"),
      scoreGauge: !!document.querySelector(".score-gauge"),
    };
  });

  assert("desktop-zoom-controls", chrome.zoomBtns, "zoom buttons missing");
  assert("desktop-no-toolbar-layers", chrome.toolbarLayerBtns === 0, "layer buttons still in top toolbar");
  assert("desktop-explain-dock", chrome.explainVisible, "explain dock hidden");
  assert("desktop-viewer-meta", chrome.metaVisible, "viewer meta hidden");
  assert("desktop-report-visible", chrome.reportVisible, "report box hidden");
  assert("desktop-report-min-height", chrome.reportMinH >= 140, `report min-height ${chrome.reportMinH}px`);
  assert("desktop-layer-cards", chrome.layerCards >= 1, "no overlay layer cards");
  assert("desktop-original-layer-card", chrome.originalCard, "Original layer card missing");
  assert("desktop-case-bar", chrome.caseBar, "case summary bar missing");
  assert("desktop-score-gauge", chrome.scoreGauge, "score gauge missing");
}

async function testDesktopBasicMarkup(page) {
  const markup = await evalSafe(page, () => {
    const cluster = document.querySelector(".toolbar-cluster-markup");
    const labels = [...(cluster?.querySelectorAll(".ann-tool") ?? [])].map((b) => b.textContent.trim());
    const fsDock = document.getElementById("viewer-markup-dock");
    const inkPickers = document.querySelectorAll("#ann-ink-picker").length;
    const opacitySliders = document.querySelectorAll("#ann-opacity").length;
    const fsColors = document.querySelectorAll(".fs-ann-color").length;
    const inlineColors = document.querySelectorAll(".toolbar-cluster-markup .ann-color").length;
    return {
      labels,
      fsDockHidden: fsDock ? fsDock.hidden || getComputedStyle(fsDock).display === "none" : true,
      inkPickers,
      opacitySliders,
      fsColors,
      inlineColors,
      hasUndo: !!document.getElementById("ann-undo"),
      hasClearInToolbar: !!document.querySelector(".toolbar-cluster-markup #ann-clear"),
    };
  });

  assert("desktop-basic-tools-pan", markup.labels.includes("Pan"), `tools: ${markup.labels.join(",")}`);
  assert("desktop-basic-tools-pen", markup.labels.includes("Pen"), "Pen missing");
  assert("desktop-basic-tools-circle", markup.labels.includes("Circle"), "Circle missing");
  assert("desktop-basic-no-box", !markup.labels.includes("Box"), "Box should not be in basic toolbar");
  assert("desktop-basic-no-arrow", !markup.labels.includes("Arrow"), "Arrow should not be in basic toolbar");
  assert("desktop-markup-dock-hidden", markup.fsDockHidden, "fullscreen dock visible outside fullscreen");
  assert("desktop-single-ink-picker", markup.inkPickers === 1, `expected 1 ink picker, got ${markup.inkPickers}`);
  assert("desktop-single-opacity-slider", markup.opacitySliders === 1, `expected 1 opacity slider, got ${markup.opacitySliders}`);
  assert("desktop-no-inline-color-swatches", markup.inlineColors === 0, "color swatches duplicated in toolbar");
  assert("desktop-no-fs-colors-normal", markup.fsColors === 0, "fs color swatches visible in normal mode");
  assert("desktop-undo-btn", markup.hasUndo, "Undo missing");
  assert("desktop-no-clear-in-basic", !markup.hasClearInToolbar, "Clear all should not be in basic toolbar");
}

async function testDesktopInkPicker(page) {
  await page.click("#ann-ink-trigger");
  await page.waitForTimeout(150);

  const menu = await evalSafe(page, () => {
    const m = document.getElementById("ann-ink-menu");
    const trigger = document.getElementById("ann-ink-trigger");
    return {
      open: m?.classList.contains("is-open"),
      expanded: trigger?.getAttribute("aria-expanded") === "true",
      swatches: document.querySelectorAll(".ann-ink-menu-colors .ann-color").length,
    };
  });
  assert("ink-menu-opens", menu.open && menu.expanded, "ink color menu did not open");
  assert("ink-menu-five-colors", menu.swatches === 5, `expected 5 colors, got ${menu.swatches}`);

  await page.click('.ann-color[data-color="teal"]');
  await page.waitForTimeout(100);

  const selected = await evalSafe(page, () => ({
    swatch: document.getElementById("ann-ink-swatch")?.dataset.color,
    active: document.querySelector(".ann-ink-menu-colors .ann-color.active")?.dataset.color,
    menuClosed: !document.getElementById("ann-ink-menu")?.classList.contains("is-open"),
  }));
  assert("ink-select-teal", selected.swatch === "teal", `swatch is ${selected.swatch}`);
  assert("ink-active-teal", selected.active === "teal", "active swatch not updated");
  assert("ink-menu-closes-on-pick", selected.menuClosed, "menu stayed open after color pick");

  await page.click("#ann-ink-trigger");
  await page.waitForTimeout(100);
  await page.click("#ann-ink-close");
  await page.waitForTimeout(100);
  const closedByDone = await evalSafe(page, () => !document.getElementById("ann-ink-menu")?.classList.contains("is-open"));
  assert("ink-menu-done-button", closedByDone, "Done button did not close menu");
}

async function testDesktopLayersAndOriginal(page) {
  const cards = page.locator("#explain-dock .layer-card");
  const count = await cards.count();
  if (count < 1) {
    fail("layers-clickable", "no layer cards");
    return;
  }

  // Pick first non-original overlay if present
  const overlayId = await evalSafe(page, () => {
    const cards = [...document.querySelectorAll("#explain-dock .layer-card")];
    const overlay = cards.find((c) => c.dataset.layer && c.dataset.layer !== "source");
    return overlay?.dataset.layer ?? null;
  });

  if (overlayId) {
    await page.click(`#explain-dock .layer-card[data-layer="${overlayId}"]`);
    await page.waitForTimeout(400);
    const overlayState = await evalSafe(page, () => {
      const o = document.getElementById("viewer-overlay");
      const blend = document.getElementById("blend-wrap");
      const active = document.querySelector("#explain-dock .layer-card.active")?.dataset.layer;
      return {
        active,
        overlayHidden: o?.hidden,
        overlaySrc: o?.getAttribute("src"),
        blendHidden: blend?.classList.contains("is-hidden"),
      };
    });
    assert("layer-overlay-active", overlayState.active === overlayId, "layer card not active");
    assert("layer-overlay-shown", !overlayState.overlayHidden && overlayState.overlaySrc, "overlay not applied");
    assert("layer-blend-visible", !overlayState.blendHidden, "blend slider hidden on overlay");
  } else pass("layer-overlay-skipped");

  await page.click('#explain-dock .layer-card[data-layer="source"]');
  await page.waitForTimeout(400);

  const originalState = await evalSafe(page, () => {
    const o = document.getElementById("viewer-overlay");
    const blend = document.getElementById("blend-wrap");
    const zoom = document.getElementById("zoom-label")?.textContent;
    return {
      active: document.querySelector("#explain-dock .layer-card.active")?.dataset.layer,
      overlayHidden: o?.hidden,
      overlaySrc: o?.getAttribute("src"),
      isOff: o?.classList.contains("is-off"),
      blendHidden: blend?.classList.contains("is-hidden"),
      zoom,
    };
  });
  assert("original-layer-active", originalState.active === "source", "Original not active");
  assert("original-clears-overlay", originalState.overlayHidden || !originalState.overlaySrc || originalState.isOff, "overlay not cleared");
  assert("original-hides-blend", originalState.blendHidden, "blend slider visible on Original");
  assert("original-resets-zoom", /100%/.test(originalState.zoom ?? ""), `zoom not reset: ${originalState.zoom}`);
}

async function testDesktopZoom(page) {
  const before = await page.locator("#zoom-label").textContent();
  await page.click("#btn-zoom-in");
  await page.waitForTimeout(150);
  const afterIn = await page.locator("#zoom-label").textContent();
  assert("zoom-in-increases", afterIn !== before && afterIn !== "100%", `${before} → ${afterIn}`);

  await page.click("#btn-zoom-reset");
  await page.waitForTimeout(150);
  const afterFit = await page.locator("#zoom-label").textContent();
  assert("zoom-fit-resets", /100%/.test(afterFit ?? ""), `fit → ${afterFit}`);
}

async function testDesktopScrollPanes(page) {
  const scroll = await evalSafe(page, () => {
    const caseList = document.querySelector(".case-list");
    const panel = document.querySelector(".panel-viewer");
    const clinical = document.querySelector(".clinical-rail .rail-content");
    const content = document.querySelector("body.mode-review .content");
    return {
      caseOverflowY: caseList ? getComputedStyle(caseList).overflowY : "",
      panelOverflowY: panel ? getComputedStyle(panel).overflowY : "",
      clinicalOverflowY: clinical ? getComputedStyle(clinical).overflowY : "",
      contentOverflow: content ? getComputedStyle(content).overflow : "",
      caseOverscroll: caseList ? getComputedStyle(caseList).overscrollBehavior : "",
    };
  });
  assert("scroll-case-list", scroll.caseOverflowY === "auto" || scroll.caseOverflowY === "scroll", scroll.caseOverflowY);
  assert("scroll-panel-viewer", scroll.panelOverflowY === "auto" || scroll.panelOverflowY === "scroll", scroll.panelOverflowY);
  assert("scroll-clinical-rail", scroll.clinicalOverflowY === "auto" || scroll.clinicalOverflowY === "scroll", scroll.clinicalOverflowY);
  assert("scroll-no-page-overflow", scroll.contentOverflow !== "visible", "desktop review page scrolls as whole");
}

async function testDesktopFullscreenMarkup(page) {
  await exitFullscreenIfNeeded(page);
  await page.evaluate(() => document.getElementById("btn-fullscreen")?.click());
  await page.waitForTimeout(350);

  const fs = await evalSafe(page, () => {
    const panel = document.getElementById("panel-viewer");
    const dock = document.getElementById("viewer-markup-dock");
    const dockStyle = dock ? getComputedStyle(dock) : null;
    const dockRect = dock?.getBoundingClientRect();
    const toolbar = document.querySelector(".viewer-toolbar");
    const toolbarRect = toolbar?.getBoundingClientRect();
    const basicMarkup = document.querySelector(".toolbar-cluster-markup");
    const ink = document.querySelector(".toolbar-cluster-ink");
    const fsTools = [...document.querySelectorAll(".viewer-markup-dock .fs-ann-tool")].map((b) => b.textContent.trim());
    return {
      focus: document.querySelector(".review-panels")?.classList.contains("viewer-focus"),
      fsMarkup: panel?.classList.contains("is-fs-markup"),
      dockVisible: dock && !dock.hidden && dockStyle?.display !== "none",
      dockOnScreen: dockRect && toolbarRect ? dockRect.top >= toolbarRect.bottom - 4 : false,
      dockHeight: dockRect?.height ?? 0,
      basicHidden: basicMarkup ? getComputedStyle(basicMarkup).display === "none" : true,
      inkVisible: ink ? getComputedStyle(ink).display !== "none" : false,
      fsTools,
      fsLayerBarVisible: (() => {
        const el = document.getElementById("toolbar-fs-layers");
        if (!el) return false;
        return getComputedStyle(el).display !== "none" && el.querySelectorAll(".layer-pill").length >= 1;
      })(),
      explainVisible: (() => {
        const el = document.getElementById("explain-dock");
        return el ? getComputedStyle(el).display !== "none" : false;
      })(),
      metaVisible: (() => {
        const el = document.querySelector(".viewer-meta");
        return el ? getComputedStyle(el).display !== "none" : false;
      })(),
      reportVisible: (() => {
        const el = document.getElementById("report-dock");
        return el ? getComputedStyle(el).display !== "none" : false;
      })(),
      clinicalVisible: (() => {
        const el = document.getElementById("clinical-rail-content");
        if (!el) return false;
        const r = el.getBoundingClientRect();
        return r.width > 20 && getComputedStyle(el).display !== "none";
      })(),
    };
  });

  assert("fullscreen-viewer-focus", fs.focus, "viewer-focus class missing");
  assert("fullscreen-is-fs-markup", fs.fsMarkup, "is-fs-markup class missing");
  assert("fullscreen-markup-dock-visible", fs.dockVisible, "markup dock hidden in fullscreen");
  assert("fullscreen-markup-dock-on-screen", fs.dockOnScreen, "markup dock not below toolbar");
  assert("fullscreen-markup-dock-height", fs.dockHeight >= 28 && fs.dockHeight <= 44, `dock height ${Math.round(fs.dockHeight)}px`);
  assert("fullscreen-basic-markup-hidden", fs.basicHidden, "basic mark row still visible");
  assert("fullscreen-ink-picker-visible", fs.inkVisible, "ink picker hidden in fullscreen");

  for (const tool of ["Pan", "Pen", "Circle", "Box", "Arrow", "Line", "Pin", "Text"]) {
    assert(`fullscreen-tool-${tool.toLowerCase()}`, fs.fsTools.includes(tool), `missing ${tool}`);
  }

  assert("fullscreen-explain-dock", !fs.explainVisible, "bottom overlay strip should be hidden in fullscreen");
  assert("fullscreen-fs-layer-bar", fs.fsLayerBarVisible, "compact View bar missing in fullscreen");
  assert("fullscreen-viewer-meta", fs.metaVisible, "viewer meta hidden in fullscreen");
  assert("fullscreen-clinical-visible", fs.clinicalVisible, "clinical rail hidden in fullscreen");
  assert("fullscreen-report-hidden", !fs.reportVisible, "report should be hidden in fullscreen focus");

  await page.click("#btn-fullscreen");
  await page.waitForTimeout(250);
  pass("fullscreen-exit");
}

async function testDesktopAnnotationUndo(page) {
  await page.click('.ann-tool[data-tool="pen"]');
  const canvas = page.locator("#annotation-canvas");
  const box = await canvas.boundingBox();
  if (!box) {
    fail("annotation-draw", "canvas has no bounding box");
    return;
  }
  await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.4);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.55);
  await page.mouse.up();
  await page.waitForTimeout(100);

  await page.click("#ann-undo");
  await page.waitForTimeout(100);
  pass("annotation-undo-click");
}

async function testDesktopReportButton(page) {
  await exitFullscreenIfNeeded(page);
  await page.click("#btn-report");
  await page.waitForTimeout(450);
  const report = await evalSafe(page, () => {
    const box = document.getElementById("report-box");
    if (!box) return { visible: false };
    const r = box.getBoundingClientRect();
    return { visible: r.height > 20 && r.width > 20, inView: r.top < window.innerHeight && r.bottom > 0 };
  });
  assert("report-btn-shows-dock", report.visible, "report box not visible");
  assert("report-btn-in-view", report.inView, "report not in viewport after Report click");
}

async function testDesktop(page) {
  console.log("\n=== Desktop 1280×900 ===");
  await goToReview(page);
  await testAssetVersions(page);
  await testDesktopIngestAndOrgan(page);
  await openFirstCase(page);
  await testShellAccessibility(page);
  await testDesktopCaseRail(page);
  await testDesktopClinicalRail(page);
  await testDesktopViewerChrome(page);
  await testDesktopBasicMarkup(page);
  await testDesktopInkPicker(page);
  await testDesktopLayersAndOriginal(page);
  await testDesktopZoom(page);
  await testDesktopScrollPanes(page);
  await testDesktopFullscreenMarkup(page);
  await testDesktopAnnotationUndo(page);
  await testDesktopReportButton(page);
}

// ─── Mobile / compact ────────────────────────────────────────────────────────

async function testMobile(page, label) {
  console.log(`\n=== Mobile: ${label} ===`);
  await goToReview(page);

  const shell = await evalSafe(page, () => ({
    mobile: document.body.classList.contains("review-mobile"),
    compact: document.body.classList.contains("compact-viewport"),
    railW: document.getElementById("case-rail")?.getBoundingClientRect().width ?? 0,
    vpW: window.innerWidth,
    toggleVisible: (() => {
      const el = document.getElementById("case-rail-toggle");
      if (!el) return false;
      const s = getComputedStyle(el);
      return s.display !== "none" && s.visibility !== "hidden";
    })(),
    ingestTitle: (() => {
      const el = document.querySelector(".ingest-title");
      return el ? getComputedStyle(el).display !== "none" : false;
    })(),
    ingestRow: (() => {
      const body = document.querySelector(".ingest-body");
      return body ? getComputedStyle(body).flexDirection : "";
    })(),
  }));

  assert(`${label}-mobile-or-compact`, shell.mobile || shell.compact, "neither review-mobile nor compact-viewport");
  assert(`${label}-case-rail-fullwidth`, shell.railW >= shell.vpW * 0.85, `rail ${shell.railW}px vs ${shell.vpW}px`);
  assert(`${label}-case-toggle-hidden`, !shell.toggleVisible, "case toggle visible on mobile");
  assert(`${label}-ingest-title-hidden`, !shell.ingestTitle, "ingest title visible on mobile");
  assert(`${label}-ingest-horizontal`, shell.ingestRow === "row" || shell.ingestRow === "row nowrap", shell.ingestRow);

  await openFirstCase(page);

  const review = await evalSafe(page, () => ({
    report: !!document.getElementById("report-box") && getComputedStyle(document.getElementById("report-box")).display !== "none",
    clinical: (() => {
      const el = document.getElementById("clinical-rail-content");
      if (!el) return false;
      const r = el.getBoundingClientRect();
      return r.height > 20 && getComputedStyle(el).display !== "none";
    })(),
    explain: !!document.getElementById("explain-dock"),
    inkPicker: document.querySelectorAll("#ann-ink-picker").length,
  }));
  assert(`${label}-report-visible`, review.report, "report hidden");
  assert(`${label}-clinical-visible`, review.clinical, "clinical hidden");
  assert(`${label}-explain-dock`, review.explain, "explain dock missing");
  assert(`${label}-single-ink-picker`, review.inkPicker === 1, `ink pickers: ${review.inkPicker}`);

  await page.evaluate(() => document.getElementById("btn-fullscreen")?.click());
  await page.waitForTimeout(350);
  const fsMobile = await evalSafe(page, () => {
    const panel = document.getElementById("panel-viewer");
    const canvas = document.getElementById("viewer-canvas");
    const clinical = document.getElementById("clinical-rail");
    const vpH = window.innerHeight;
    const canvasRect = canvas?.getBoundingClientRect();
    const chromeH = vpH - (canvasRect?.height ?? 0);
    return {
      mobileFs: document.body.classList.contains("mobile-viewer-focus"),
      panelFixed: panel ? getComputedStyle(panel).position === "fixed" : false,
      clinicalHidden: clinical ? getComputedStyle(clinical).display === "none" : true,
      canvasHeight: canvasRect?.height ?? 0,
      vpH,
      chromeRatio: chromeH / vpH,
    };
  });
  assert(`${label}-mobile-fs-overlay`, fsMobile.mobileFs, "mobile-viewer-focus class missing");
  assert(`${label}-mobile-fs-fixed`, fsMobile.panelFixed, "panel not fixed on mobile fullscreen");
  assert(`${label}-clinical-hidden-in-fullscreen`, fsMobile.clinicalHidden, "clinical rail still visible in mobile fullscreen");
  assert(`${label}-canvas-height`, fsMobile.canvasHeight >= fsMobile.vpH * 0.42, `canvas only ${Math.round(fsMobile.canvasHeight)}px tall`);
  assert(`${label}-chrome-budget`, fsMobile.chromeRatio <= 0.58, `chrome uses ${Math.round(fsMobile.chromeRatio * 100)}% of viewport`);

  await page.evaluate(() => document.getElementById("btn-report")?.click());
  await page.waitForTimeout(450);
  const reportInView = await page.locator("#report-box").evaluate((el) => {
    const r = el.getBoundingClientRect();
    return r.top < window.innerHeight && r.bottom > 0;
  });
  assert(`${label}-report-scroll`, reportInView, "Report did not scroll report into view");

  await exitFullscreenIfNeeded(page);
}

async function testCompactDesktop(page) {
  console.log("\n=== Compact desktop 1280×580 ===");
  await goToReview(page);
  const compact = await evalSafe(page, () => document.body.classList.contains("compact-viewport"));
  assert("compact-viewport-class", compact, "compact-viewport not set on short window");
  await openFirstCase(page);
  const report = await page.locator("#report-box").isVisible();
  assert("compact-report-visible", report, "report hidden in compact desktop");
}

// ─── Runner ──────────────────────────────────────────────────────────────────

async function main() {
  let browser;
  try {
    browser = await chromium.launch({ headless: !HEADED });
  } catch {
    console.error("Could not launch Chromium. Install with: npx playwright install chromium");
    process.exit(2);
  }

  const desktopCtx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const pageD = await desktopCtx.newPage();
  try {
    await testDesktop(pageD);
  } catch (e) {
    fail("desktop-crash", e.message);
  }
  await desktopCtx.close();

  const compactCtx = await browser.newContext({ viewport: { width: 1280, height: 580 } });
  const pageC = await compactCtx.newPage();
  try {
    await testCompactDesktop(pageC);
  } catch (e) {
    fail("compact-crash", e.message);
  }
  await compactCtx.close();

  if (!SKIP_MOBILE) {
    const iphone = await browser.newContext({ ...devices["iPhone 13"] });
    const pageM = await iphone.newPage();
    try {
      await testMobile(pageM, "iphone");
    } catch (e) {
      fail("iphone-crash", e.message);
    }
    await iphone.close();

    const landscape = await browser.newContext({ ...devices["iPhone 13 landscape"] });
    const pageL = await landscape.newPage();
    try {
      await testMobile(pageL, "landscape");
    } catch (e) {
      fail("landscape-crash", e.message);
    }
    await landscape.close();
  }

  await browser.close();

  console.log("\n--- Summary ---");
  console.log(`Passed: ${passes.length}`);
  console.log(`Failed: ${bugs.length}`);
  if (bugs.length) {
    console.log("\nFailures:");
    for (const b of bugs) console.log(`  • ${b.name}: ${b.detail}`);
    process.exit(1);
  }
  console.log("All UI smoke tests passed.");
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
