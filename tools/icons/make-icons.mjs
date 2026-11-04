// Generates Tracer's icon options (design/icons/*.svg) + a preview sheet.
//   node tools/icons/make-icons.mjs
//
// Construction follows Vixer's icon (vixerapp/src-tauri/icons): thick
// round-capped open rings with a solid dot, flat colour, transparent
// background, diagonal composition. Tracer's twist is the trail: a tracer
// round leaves a glowing streak, drawn here as stepped opacity (no blur, so it
// stays crisp at 16 px).
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = join(ROOT, "design/icons");
mkdirSync(OUT, { recursive: true });

const VIOLET = "#A27BFF";
const GREEN = "#34D399"; // Vixer Talk green = Human
const f = (n) => +n.toFixed(2);
const pt = (cx, cy, r, deg) => [f(cx + r * Math.cos((deg * Math.PI) / 180)), f(cy + r * Math.sin((deg * Math.PI) / 180))];

/** SVG arc path from a0 to a1 degrees, clockwise (screen coords). */
function arc(cx, cy, r, a0, a1) {
  const [x0, y0] = pt(cx, cy, r, a0);
  const [x1, y1] = pt(cx, cy, r, a1);
  const large = a1 - a0 > 180 ? 1 : 0;
  return `M${x0} ${y0}A${r} ${r} 0 ${large} 1 ${x1} ${y1}`;
}

/**
 * An arc that fades in from tail (a0) to head (a1): the trail. SVG has no
 * conic gradient, and abutting translucent segments show seams, so stack
 * `layers` translucent arcs that all END at the head and START progressively
 * later. Opacity builds up smoothly toward the head with no shared edges.
 * The head gets a solid round cap.
 */
function trailArc(cx, cy, r, a0, a1, width, color, layers = 40) {
  const alpha = 1 - Math.pow(0.04, 1 / layers); // reaches ~96% at the head
  const parts = [];
  for (let i = 0; i < layers; i++) {
    const s = a0 + ((a1 - a0) * Math.pow(i / layers, 1.15)) ; // slightly front-loaded fade
    parts.push(`<path d="${arc(cx, cy, r, s, a1)}" stroke="${color}" stroke-opacity="${f(alpha)}" stroke-width="${width}" stroke-linecap="butt" fill="none"/>`);
  }
  const [hx, hy] = pt(cx, cy, r, a1);
  // Solid head: a short full-opacity arc plus the round cap.
  parts.push(`<path d="${arc(cx, cy, r, a1 - 14, a1)}" stroke="${color}" stroke-width="${width}" stroke-linecap="butt" fill="none"/>`);
  parts.push(`<circle cx="${hx}" cy="${hy}" r="${width / 2}" fill="${color}"/>`);
  return `<g>${parts.join("")}</g>`;
}

const svg = (body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">${body}</svg>\n`;

const options = {
  // A. Orbit: one ring whose tail fades in toward the head, round in the middle.
  //    Closest to "a round leaving a glowing trail". Single colour.
  "a-orbit": svg(
    trailArc(128, 128, 82, 50, 370, 34, VIOLET) /* 320 deg: user choice */ +
      `<circle cx="128" cy="128" r="27" fill="${VIOLET}"/>`
  ),

  // B. Pair: Vixer's interlocking two-ring composition, recast as the app's
  //    two sides: Human (green) and AI (violet), the violet ring trailing.
  //    The most obviously "sibling of Vixer" option.
  "b-pair": svg(
    `<path d="${arc(104, 154, 56, 0, 270)}" stroke="${GREEN}" stroke-width="27" stroke-linecap="round" fill="none"/>` +
      `<circle cx="104" cy="154" r="18" fill="${GREEN}"/>` +
      trailArc(152, 102, 56, 180, 450, 27, VIOLET) +
      `<circle cx="152" cy="102" r="18" fill="${VIOLET}"/>`
  ),

  // C. Streak: the round itself with speed lines behind it, on Vixer's
  //    diagonal. The most literal "tracer"; reads well at 16 px.
  "c-streak": (() => {
    const head = [178, 78];
    const dir = [-Math.SQRT1_2, Math.SQRT1_2]; // toward lower-left
    const perp = [Math.SQRT1_2, Math.SQRT1_2];
    const lines = [
      { off: 0, start: 44, len: 150, w: 28, id: "g0" },
      { off: -40, start: 40, len: 92, w: 22, id: "g1" },
      { off: 40, start: 40, len: 92, w: 22, id: "g1b" },
    ];
    let defs = "<defs>";
    let body = "";
    for (const l of lines) {
      const x0 = f(head[0] + dir[0] * l.start + perp[0] * l.off);
      const y0 = f(head[1] + dir[1] * l.start + perp[1] * l.off);
      const x1 = f(x0 + dir[0] * l.len);
      const y1 = f(y0 + dir[1] * l.len);
      const peak = l.off === 0 ? 1 : 0.7;
      defs += `<linearGradient id="${l.id}" gradientUnits="userSpaceOnUse" x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}"><stop offset="0" stop-color="${VIOLET}" stop-opacity="${peak}"/><stop offset="1" stop-color="${VIOLET}" stop-opacity="0"/></linearGradient>`;
      body += `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="url(#${l.id})" stroke-width="${l.w}" stroke-linecap="round"/>`;
    }
    defs += "</defs>";
    return svg(defs + body + `<circle cx="${head[0]}" cy="${head[1]}" r="32" fill="${VIOLET}"/>`);
  })(),
};

for (const [name, s] of Object.entries(options)) writeFileSync(join(OUT, `tracer-icon-${name}.svg`), s);

// Preview: each option at 16/32/64/256 on dark and light, next to Vixer's.
const sizes = [16, 32, 64, 128, 256];
const row = (name, s, bg) =>
  `<div class="row" style="background:${bg}"><span>${name}</span>${sizes.map((z) => `<img src="data:image/svg+xml;utf8,${encodeURIComponent(s)}" width="${z}" height="${z}">`).join("")}</div>`;
const html = `<!doctype html><meta charset="utf-8"><title>Tracer icon options</title>
<style>body{margin:0;font:12px system-ui;background:#0b0d10;color:#d0d4da}.row{display:flex;align-items:center;gap:22px;padding:14px 18px}.row span{width:90px}</style>
${Object.entries(options).map(([n, s]) => row(n, s, "#181b1f") + row(n, s, "#f6f7f9")).join("")}
<div class="row" style="background:#181b1f"><span>vixer (ref)</span><img src="../../../../vixerapp/public/vixer-logo.png" width="64"></div>`;
writeFileSync(join(OUT, "preview.html"), html);
console.log("wrote", Object.keys(options).map((n) => `tracer-icon-${n}.svg`).join(", "), "+ preview.html");
