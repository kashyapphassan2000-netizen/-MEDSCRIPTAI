// Render docs/MedScriptAI_Lab_Manual.html to PDF with headless Chromium (Playwright).
// Usage (from the repo root): node docs/build/make_pdf.js
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const html = path.resolve(__dirname, '..', 'MedScriptAI_Lab_Manual.html');
  const pdf = path.resolve(__dirname, '..', 'MedScriptAI_Lab_Manual.pdf');
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + html, { waitUntil: 'load' });
  await page.pdf({
    path: pdf, format: 'A4', printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#888">MedScriptAI Lab Manual - page <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: { top: '14mm', bottom: '16mm', left: '12mm', right: '12mm' },
  });
  await browser.close();
  console.log('wrote', pdf);
})();
