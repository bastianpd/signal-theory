// Classical image operations that mirror what a CNN's first layers learn.
export const SIZE = 224;
export const KERNELS = {
  "Sobel X (vertical edges)": [-1, 0, 1, -2, 0, 2, -1, 0, 1],
  "Sobel Y (horizontal edges)": [-1, -2, -1, 0, 0, 0, 1, 2, 1],
  "Laplacian (blobs)": [0, 1, 0, 1, -4, 1, 0, 1, 0],
  "Blur 3×3": [1, 1, 1, 1, 1, 1, 1, 1, 1].map((v) => v / 9),
  "Sharpen": [0, -1, 0, -1, 5, -1, 0, -1, 0],
  "Emboss": [-2, -1, 0, -1, 1, 1, 0, 1, 2],
};
export function channel(d, c, n) { const o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = d[i * 4 + c] / 255; return o; }
export function grayscale(d, n) { const o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255; return o; }
// 3x3 convolution followed by ReLU, as in one CNN filter + activation
export function convolve(src, w, h, k) {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let s = 0; for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) s += src[(y + j - 1) * w + x + i - 1] * k[j * 3 + i];
    out[y * w + x] = Math.max(0, s);
  }
  return out;
}
export function normalize(a) {
  let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const r = hi - lo || 1; return Float32Array.from(a, (v) => (v - lo) / r);
}
