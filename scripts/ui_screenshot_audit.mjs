/**
 * UI audit at real monitor sizes — no fullscreen-only layout.
 */
import { chromium } from "playwright";
import { mkdir } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const base = process.argv[2] || "http://127.0.0.1:8765";
const outDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../outputs/ui_audit");

const SIZES = [
  { name: "1920x1080", width: 1920, height: 1080 },
  { name: "1512x982_macbook", width: 1512, height: 982 },
  { name: "1440x900", width: 1440, height: 900 },
  { name: "1366x768", width: 1366, height: 768 },
];

async function auditSize(browser, size) {
  const page = await browser.newPage({ viewport: { width: size.width, height: size.height } });
  await page.goto(base, { waitUntil: "networkidle" });
  await page.click('a[data-view="review"]');
  await page.waitForTimeout(400);
  const row = page.locator(".case-row").first();
  if (!(await row.count())) {
    console.log(size.name, "no cases");
    await page.close();
    return;
  }
  await row.click();
  await page.waitForTimeout(2000);

  const dir = path.join(outDir, size.name);
  await mkdir(dir, { recursive: true });
  await page.screenshot({ path: path.join(dir, "review.png") });

  const clinical = await page.locator(".clinical-rail").boundingBox();
  const canvas = await page.locator(".viewer-canvas").boundingBox();
  const signoff = await page.locator("form#review-form").boundingBox();
  const inViewport = (box) =>
    box && box.y >= 0 && box.y + box.height <= size.height && box.width > 50;

  console.log(size.name, {
    clinicalVisible: inViewport(clinical),
    canvasH: canvas?.height,
    signoffVisible: inViewport(signoff),
    clinicalX: clinical?.x,
  });

  await page.locator('.layer-btn[data-layer="gradcam"]').click().catch(() => {});
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(dir, "gradcam.png") });

  await page.locator("#btn-viewer-size").click().catch(() => {});
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(dir, "larger_view.png") });
  const clinicalAfter = await page.locator(".clinical-rail").boundingBox();
  console.log(size.name, "after larger view, clinical still visible:", inViewport(clinicalAfter));

  await page.close();
}

async function main() {
  await mkdir(outDir, { recursive: true });
  const browser = await chromium.launch();
  for (const size of SIZES) {
    await auditSize(browser, size);
  }
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
