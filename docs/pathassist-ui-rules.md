# PathAssist demo UI — rules, changelog & regressions

**Read this file before changing anything in `demo/static/` or review layout CSS.**

Living log for the pathologist workstation (`demo/server.py`, port **8765**).  
**Current asset version:** `v68` in `demo/static/index.html`  
**Last updated:** 2026-07-06

---

## Files & test

| Path | Role |
|------|------|
| `demo/static/index.html` | Shell; bump `?v=N` on css/js after changes |
| `demo/static/app.js` | Layout, viewer layers, zoom/pan, rails, case review |
| `demo/static/app.css` | All review/viewer/rail styling |
| `demo/static/viewer.js` | Annotations (pen/circle), canvas sync with zoom |
| `scripts/ui_smoke_test.mjs` | **Run after every UI change** — Playwright regression + UX suite (~70 checks) |

```bash
PYTHONPATH=. python demo/server.py   # if not running
node scripts/ui_smoke_test.mjs
# optional: UI_TEST_HEADED=1  UI_TEST_SKIP_MOBILE=1
```

### Smoke suite coverage

| Area | Checks |
|------|--------|
| Server + assets | reachable, css/js `?v=` match |
| Ingest / organ | title, select, help, no ingest-toggle |
| Case rail | width, toggle visible & clickable, collapse no ghost column |
| Clinical rail | always visible, no collapse toggle, sign-off form |
| Viewer chrome | zoom, explain-dock, meta, report min-height, case bar, score gauge |
| Layers | cards in explain-dock only (no toolbar layers), overlay + blend, Original clears overlay |
| Markup tiers | basic Pan/Pen/Circle only; fullscreen dock with 8 tools; ink picker ×1 |
| Ink picker | opens, 5 colors, selection updates swatch |
| Zoom | in + fit reset |
| Scroll panes | case list / panel / clinical independent overflow |
| Fullscreen | markup dock on-screen, ink picker, explain + meta, report hidden |
| Mobile / compact | full-width rail, hidden toggle/title, mobile FS overlay, canvas ≥42% viewport, report scroll |

---

## Do NOT (common regressions)

These have broken the UI before. Check each one before shipping.

### Viewer / zoom

1. **Do not** attach wheel zoom to the whole `.viewer-canvas` with unconditional `preventDefault`.  
   → Only zoom when pointer is over **`#viewer-base`** slide pixels (`pointerOverViewerImage`). Wheel handler is **document-level**; no canvas bounding-box gate.

2. **Do not** use one page-level scroll for the whole case review on desktop.  
   → **Split scroll panes:** outer shell locked (`100dvh`); `#sample-grid` / `.case-list` scrolls in the left rail; `.panel-viewer` scrolls the viewer + report; clinical rail has its own `.rail-content` scroll. Use `overscroll-behavior: contain` so wheels do not chain between panes.

3. **Do not** use scroll-based or element-resize zoom.  
   → Pan/zoom is **CSS transform** on `#tile-frame` (`translate` + `scale`). Images stay at fit size (`_fitW` / `_fitH`).

4. **Do not** set `height: auto !important` on `.viewer-canvas.is-square-tile` in ways that steal space from docks or break transform zoom (v44 was reverted for this).

5. **Do not** size square tiles without reserving ~340px below for toolbar + meta + explain + **report** (`fitTileToViewer` dock reserve).

6. **Do not** make “Original” only hide the overlay with `hidden=true`.  
   → Must call `clearViewerOverlay()` (remove `src`, add `.is-off`, opacity 0) **and** `resetViewerZoom()` + `fitTileToViewer()`.

7. **Do not** re-bind layer button listeners inside `bindViewer()` on every case render.  
   → Use delegated handlers in `setupViewerLayers()` (once at init).

### Fullscreen / viewer-focus

8. **Do not** duplicate overlay/layer pickers in the top toolbar — organ-adaptive layers live in `#explain-dock` below the image only (Original, Grad-CAM, Heatmap, Uncertainty, per-layer activations when present). Top toolbar: zoom, Report/Fullscreen, basic Mark row only.

9. **Do not** show the full markup palette on the normal case-review toolbar — basic only (Pan/Pen/Circle, red+yellow, ink, Undo). Full palette (Box/Arrow/Line/Pin/Text, all colors, weights) in **fullscreen** `#viewer-markup-dock` only.

9. **Do not** hide `.explain-dock` or `.viewer-meta` in `:fullscreen` or `.review-panels.viewer-focus`.  
   → User needs layer strip, blend slider, and meta in fullscreen. **Report** may stay hidden (toolbar **Report** exits focus + scrolls to draft).

7. **Do not** use native browser `:fullscreen` on phone — use `body.mobile-viewer-focus` fixed overlay instead (hides case/clinical chrome; image gets ≥42% viewport). Desktop keeps native FS + clinical hidden via `.viewer-focus`.

8. **Do not** use `height: 100% !important` on fullscreen square canvas if it pushes layer docks off-screen — use `flex: 1 1 auto` so toolbar + canvas + meta + explain-dock all fit.

### Rails / collapse

9. **Do not** use unscoped `.rail-content { display: none }` when collapsing the case rail — it hid the **clinical** panel too (v47 bug). Scope: `.case-rail .rail-content` only.

10. **Do not** use `body:not(.case-rail-collapsed) .rail-collapsed-strip` — hides **both** rails’ expand strips. Scope per rail (`.case-rail` / `.clinical-rail`).

11. **Do not** hide the Cases rail collapse tab behind `.review-body` — `.case-rail` needs `z-index` above the main column; tab uses `translate(50%, -50%)` on the rail edge.

12. **Do not** add a collapsible ingest header — organ select + upload stay always visible on desktop; use static `h3.ingest-title` only (hidden on mobile/compact horizontal bar).

13. **Do not** add a collapse control for the **clinical / Assessment** rail — it must stay always visible (no `#clinical-rail-toggle`, no `clinical-rail-collapsed` state).

14. **Do not** leave a ghost grid column when the **Cases** rail is collapsed — use a single-column grid, `review-body` on row 1 with `padding-left: var(--case-rail-w-collapsed)` for the overlay tab only.

### Layout / visibility

15. **Do not** hide the draft report without a way to reopen it (toolbar **Report** button + mobile: clinical cannot stay collapsed).

14. **Do not** cap `.report-dock` below ~160px min-height on desktop review — pathologists need readable draft text.

15. **Do not** forget to bump `app.css?v=`, `app.js?v=`, `viewer.js?v=` in `index.html` — users need hard refresh otherwise.

16. **Do not** change square tile sizing without checking report + explain + meta still visible below viewer on desktop review mode.

---

## DO (required patterns)

### Original / layers

- **Original** = source image only, overlay fully cleared, view reset to fit.
- Overlay layers: Grad-CAM, heatmap, uncertainty, `activation_*` from `nn_explanation.paths`.
- Blend slider hidden on Original; visible on overlay layers (`#blend-wrap.is-hidden`).
- Toolbar layer buttons and explain-dock `.layer-card` stay in sync via `applyViewerLayer()`.

### Zoom / pan

- Wheel on **tile only**; +/- buttons zoom to center; **Fit** resets.
- Pan tool (or middle mouse) when zoomed and content exceeds viewport.
- `viewerViewport()` uses `.viewer-stage` rect for math, not full canvas width.

### Square tiles

- `fitTileToViewer()`: column width × **0.85**, also cap by viewport minus **~340px dock reserve** (report + explain + meta + toolbar).

### Fullscreen

- Desktop: `.viewer-focus` hides clinical rail; explain-dock + meta visible; dark theme on toolbar/layer cards in `:fullscreen`.
- Report: hidden in focus; **Report** button exits focus and scrolls to `#report-box`.

### Rails

- Single edge tab on **Cases** rail only; clinical/Assessment rail has **no** collapse control.
- PathAssist brand click toggles sidebar; auto-collapse sidebar on nav (desktop).

### Split scroll (desktop review)

- Shell locked to viewport; **no** document-level scroll on case review.
- **Left:** `.case-list` / `#sample-grid` scrolls; ingest panel stays pinned above.
- **Center:** `.panel-viewer` scrolls (viewer, explain dock, report).
- **Right:** `.clinical-rail .rail-content` scrolls independently.
- `overscroll-behavior: contain` on each pane so wheel does not bleed between them.

---

## Changelog (recent)

| Version | Date | Change |
|---------|------|--------|
| **v68** | 2026-07-06 | Fullscreen markup dock: single compact scroll row, smaller tools/toolbar |
| **v67** | 2026-07-06 | Mobile fullscreen: fixed overlay (`mobile-viewer-focus`), no native FS; compact toolbar + scroll markup row; canvas sized to visualViewport |
| **v66** | 2026-07-06 | Ink picker Done/close; per-mark weight stored; FS compact View bar; image fit fix |
| **v64** | 2026-07-06 | Fullscreen markup dock fixed — sits under toolbar, always visible; canvas flex no longer clips it |
| **v63** | 2026-07-06 | Overlays only in explain-dock (organ-adaptive); top toolbar = zoom + basic mark; fullscreen full markup + Line/Text |
| **v62** | 2026-07-06 | Cases rail tab z-index fix; removed useless ingest collapse — static organ/upload header |
| **v61** | 2026-07-06 | Two-tier markup: basic toolbar on case review; full palette dock in fullscreen only |
| **v60** | 2026-07-06 | Redesigned Cases rail toggle — no rotated edge tab; vertical strip when collapsed |
| **v58** | 2026-07-06 | Score gauge redesign |
| **v57** | 2026-07-06 | Removed `#viewer-layer-hint` toolbar text |
| **v56** | 2026-07-06 | Case rail collapsed: no ghost column; review-body fills width |
| **v55** | 2026-07-06 | Clinical/Assessment rail always visible; toggle removed |
| **v54** | 2026-07-06 | Split scroll: case list vs panel-viewer vs clinical rail |
| **v53** | 2026-07-06 | Desktop review scroll restored; report dock taller (superseded by v54 split panes) |
| **v52** | 2026-07-06 | Wheel zoom: hit-test `#viewer-base` only; document-level listener (superseded v53 scroll/layout fixes) |
| **v51** | 2026-07-06 | Wheel zoom only when pointer over `#tile-frame` (superseded by v52) |
| **v50** | 2026-07-06 | Original: full overlay clear + reset zoom; fullscreen shows explain-dock + meta + dark styling; delegated layer handlers; activation layer buttons in toolbar |
| v49 | 2026-07-06 | Score gauge redesign; case bar grid; fluid rail widths |
| v48 | 2026-07-06 | Toolbar clusters (View / Zoom / Mark / Report / Fullscreen); layer hints |
| v47 | 2026-07-06 | **Fix:** case-rail collapse CSS scoped — clinical no longer hidden |
| v44 | — | **Reverted:** reserved-height layout — image too small, docks broken |
| v43 | 2026-07-06 | Transform-based pan/zoom (replaced broken scroll zoom) |
| v30 | 2026-07-06 | **Fix:** clinical expand strip selector scope; Playwright smoke test added |
| v29 | 2026-07-06 | Report toolbar button; mobile clinical always reachable |

*(Older entries: see git history / agent transcript `faf59c98`.)*

---

## Session summary (2026-07-06)

**User goals this session:** Fix broken **Original** layer button; restore **fullscreen** image controls (layers, blend, explain strip); fix **wheel scroll** stealing page scroll outside the image.

**What we changed:**

1. **`applyViewerLayer()`** — central layer switch; Original clears overlay and resets view.
2. **Fullscreen CSS** — explain-dock + viewer-meta visible again; report still via toolbar.
3. **`pointerOverViewerTile()`** — wheel zoom gated to image bounds only.

**What went wrong before (do not repeat):**

- Hiding explain-dock/meta in fullscreen removed layer controls user relied on.
- Original only toggled `hidden` on overlay — ghost blend / wrong layer state.
- Wheel listener on entire canvas blocked page scroll in letterbox padding.
- Broad collapse CSS selectors broke unrelated panels (clinical, expand strips).
- Scroll-based zoom distorted image; v44 height tricks shrank viewer too much.

**Before next UI edit:** read **Do NOT**, run smoke test, bump `?v=`.

---

## Checklist (copy for PRs / agent turns)

- [ ] Read this file
- [ ] Smallest diff; match existing patterns in `app.js` / `app.css`
- [ ] No regressions from **Do NOT** list
- [ ] Bump `index.html` cache version
- [ ] `node scripts/ui_smoke_test.mjs` passes
- [ ] Append row to **Changelog** table above
