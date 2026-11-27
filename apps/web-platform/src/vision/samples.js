// Procedurally drawn sample leaves so the studio works with no files or camera.
const rng = (s) => () => ((s = (s + 0x6d2b79f5) | 0), ((t) => ((t ^ (t >>> 14)) >>> 0) / 4294967296)(Math.imul(s ^ (s >>> 15), 1 | s)));
export const SAMPLES = { "Healthy leaf": { spots: 0, seed: 1 }, "Mild spotting": { spots: 3, seed: 7 }, "Diseased leaf": { spots: 12, seed: 3 } };
export function drawSample(ctx, name, size = 224) {
  const { spots, seed } = SAMPLES[name], r = rng(seed), c = size / 2;
  ctx.fillStyle = "#cfc8b4"; ctx.fillRect(0, 0, size, size);
  const gr = ctx.createLinearGradient(0, 0, size, size); gr.addColorStop(0, "#3f9a45"); gr.addColorStop(1, "#2e7d38");
  ctx.fillStyle = gr; ctx.beginPath(); ctx.ellipse(c, c, size * 0.42, size * 0.27, -0.6, 0, 7); ctx.fill();
  ctx.strokeStyle = "#8fd18f"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(c - 80, c + 55); ctx.lineTo(c + 80, c - 55); ctx.stroke();
  ctx.lineWidth = 1.5; for (let i = -3; i <= 3; i++) { const x = c + i * 22, y = c - i * 14; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 28, y + 36); ctx.moveTo(x, y); ctx.lineTo(x - 28, y - 36); ctx.stroke(); }
  for (let i = 0; i < spots; i++) {
    const t = r() * 6.28, d = r() * 0.55, x = c + Math.cos(t) * 70 * d * 1.4, y = c + Math.sin(t) * 40 * d * 1.4, rad = 6 + r() * 12;
    const g = ctx.createRadialGradient(x, y, 1, x, y, rad); g.addColorStop(0, "#6b3a14"); g.addColorStop(0.7, "#9a5a22"); g.addColorStop(1, "rgba(154,90,34,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, rad, 0, 7); ctx.fill();
  }
}
