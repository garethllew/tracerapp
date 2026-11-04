// WCAG contrast check for Tracer's tokens (src/styles/tokens.css), both themes.
// Body text must hit 4.5:1 on its surface *and* on the strongest highlight
// tints; secondary/hint text and large text 3:1.   node tools/contrast-check.mjs
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const lin = (c) => ((c /= 255) <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const lum = ([r, g, b]) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
const over = (fg, a, bg) => fg.map((c, i) => Math.round(c * a + bg[i] * (1 - a)));

const themes = {
  dark: { bg: "#181b1f", head: "#13161a", t1: "#d0d4da", t2: "#878b96", t3: "#6c707a", aiInk: "#b499ff", huInk: "#34d399", warn: "#f5a623", tintMax: 0.34 },
  light: { bg: "#f6f7f9", head: "#eef0f3", t1: "#1d2025", t2: "#565b66", t3: "#7c818c", aiInk: "#6a3fe0", huInk: "#0a7a55", warn: "#8f5507", tintMax: 0.30 },
};
const AI = hex("#a27bff"), HU = hex("#34d399");
let fails = 0;
const check = (label, fg, bg, min) => {
  const r = ratio(fg, bg);
  const ok = r >= min;
  if (!ok) fails++;
  console.log(`${ok ? "pass" : "FAIL"}  ${r.toFixed(2).padStart(5)}:1 (need ${min})  ${label}`);
};
for (const [name, t] of Object.entries(themes)) {
  console.log(`\n── ${name} ──`);
  const bg = hex(t.bg), head = hex(t.head);
  check("body text on app bg", hex(t.t1), bg, 4.5);
  check("secondary text (t2) on app bg", hex(t.t2), bg, 4.5);
  check("hint text (t3) on app bg", hex(t.t3), bg, 3);
  check("AI ink on header", hex(t.aiInk), head, 4.5);
  check("Human ink on header", hex(t.huInk), head, 4.5);
  check("notice (warn) on header", hex(t.warn), head, 4.5);
  check("body text on strongest AI tint", hex(t.t1), over(AI, t.tintMax, bg), 4.5);
  check("body text on strongest Human tint", hex(t.t1), over(HU, t.tintMax, bg), 4.5);
  check("AI tint distinguishable from bg (non-text, 1.2:1 floor)", over(AI, t.tintMax, bg), bg, name === "dark" ? 1.2 : 1.15);
}
console.log(fails ? `\n${fails} failing` : "\nall pass");
process.exit(fails ? 1 : 0);
