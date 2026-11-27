import React from "react";
const COLORS = ["#d9480f", "#0b6e69", "#5f3dc4", "#1c7ed6", "#c2255c", "#7c6f00"];
export function Scatter({ points, user, classes }) {
  const xs = [...points, ...(user ? [user] : [])].map((p) => p.x), ys = [...points, ...(user ? [user] : [])].map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), sx = (x) => 20 + ((x - x0) / (x1 - x0 || 1)) * 560, sy = (y) => 300 - ((y - y0) / (y1 - y0 || 1)) * 280;
  return (<svg viewBox="0 0 600 320" style={{ width: "100%", maxWidth: 640, border: "1px solid var(--line)", background: "var(--card)" }} role="img" aria-label="2-D projection of phrase embeddings">
    {points.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r="5" fill={COLORS[classes.indexOf(p.label) % COLORS.length]} opacity=".8"><title>{p.label}: {p.text}</title></circle>)}
    {user && <text x={sx(user.x)} y={sy(user.y)} fontSize="26" textAnchor="middle" dominantBaseline="central" fill="var(--ink)">★</text>}
    {classes.map((c, i) => <g key={c} transform={`translate(10 ${14 + i * 16})`}><circle r="5" cx="5" cy="-4" fill={COLORS[i % COLORS.length]} /><text x="16" fontSize="11" fill="var(--ink)">{c}</text></g>)}
  </svg>);
}
export function Heatmap({ tokens, M, active, onPick }) {
  return (<div style={{ overflowX: "auto" }}><table style={{ width: "auto" }}><thead><tr><th>query ↓ / key →</th>{tokens.map((k, j) => <th key={j} style={{ writingMode: "vertical-rl", fontWeight: 400 }}>{k.t}</th>)}</tr></thead>
    <tbody>{M.map((row, i) => <tr key={i} onClick={() => onPick(i)} style={{ cursor: "pointer", outline: active === i ? "2px solid var(--acc)" : "none" }}>
      <th style={{ fontWeight: active === i ? 700 : 400 }}>{tokens[i].t}</th>
      {row.map((w, j) => <td key={j} title={`${tokens[i].t} → ${tokens[j].t}: ${w.toFixed(3)}`} style={{ width: 26, height: 26, padding: 0, textAlign: "center", fontSize: 9, background: `rgba(11,110,105,${Math.min(1, w * 1.4)})`, color: w > 0.45 ? "#fff" : "var(--ink)" }}>{w >= 0.1 ? w.toFixed(1) : ""}</td>)}</tr>)}</tbody></table></div>);
}
