import React, { useEffect, useRef } from "react";
const clamp = (v) => Math.max(0, Math.min(1, v));
const jet = (v) => [255 * clamp(1.5 - Math.abs(4 * v - 3)), 255 * clamp(1.5 - Math.abs(4 * v - 2)), 255 * clamp(1.5 - Math.abs(4 * v - 1))];
// Draws a normalized 2D array as grayscale or jet heatmap, smoothly scaled to `size` px.
export function MapCanvas({ data, w, h, size = 112, mode = "gray", style }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!data) return;
    const off = document.createElement("canvas"); off.width = w; off.height = h;
    const g = off.getContext("2d"), im = g.createImageData(w, h);
    for (let i = 0; i < w * h; i++) { const v = data[i], c = mode === "jet" ? jet(v) : [v * 255, v * 255, v * 255]; im.data.set([c[0], c[1], c[2], 255], i * 4); }
    g.putImageData(im, 0, 0);
    const c = ref.current, x = c.getContext("2d"); x.imageSmoothingEnabled = mode === "jet"; x.clearRect(0, 0, size, size); x.drawImage(off, 0, 0, size, size);
  }, [data, w, h, size, mode]);
  return <canvas ref={ref} width={size} height={size} style={{ width: size, height: size, ...style }} />;
}
export const Thumb = ({ title, children }) => <figure style={{ margin: 0 }}>{children}<figcaption style={{ fontSize: ".75rem", color: "var(--mut)", maxWidth: 112 }}>{title}</figcaption></figure>;
