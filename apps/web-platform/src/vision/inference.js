// In-browser inference. Uses onnxruntime-web when /models/model.onnx + labels.json exist,
// otherwise a deterministic colour-heuristic "mock" so the studio works out of the box.
const ORT_VERSION = "1.18.0"; // UPDATE together with the onnxruntime-web version in package.json
let session = null, ort = null;

export async function loadModel() {
  const base = import.meta.env.BASE_URL;
  try {
    const r = await fetch(base + "models/labels.json");
    if (!r.ok || (r.headers.get("content-type") || "").includes("html")) throw new Error("no labels");
    const meta = await r.json();
    ort = await import("onnxruntime-web");
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    session = await ort.InferenceSession.create(base + "models/model.onnx", { executionProviders: ["wasm"] });
    return { mode: "onnx", classes: meta.classes, mean: meta.mean, std: meta.std, note: "Custom ONNX model loaded." };
  } catch (e) {
    return { mode: "mock", classes: ["healthy", "diseased"], note: "Mock mode: a colour heuristic, not a trained CNN. Place model.onnx and labels.json in public/models/ to use your model." };
  }
}
const softmax = (a) => { const m = Math.max(...a), e = a.map((v) => Math.exp(v - m)), s = e.reduce((x, y) => x + y, 0); return e.map((v) => v / s); };
const norm = (a) => { const r = Float32Array.from(a, (v) => Math.max(0, v)); let hi = 0; for (const v of r) hi = Math.max(hi, v); return hi ? r.map((v) => v / hi) : r; };

export async function classify(imageData, model) {
  const n = imageData.width * imageData.height, d = imageData.data;
  if (model.mode === "onnx") {
    const t = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) t[c * n + i] = (d[i * 4 + c] / 255 - model.mean[c]) / model.std[c];
    const out = await session.run({ input: new ort.Tensor("float32", t, [1, 3, imageData.height, imageData.width]) });
    const probs = softmax(Array.from(out.logits.data)), k = probs.indexOf(Math.max(...probs));
    const [, K, ch, cw] = out.cam.dims, cam = norm(out.cam.data.slice(k * ch * cw, (k + 1) * ch * cw));
    const [, EC, eh, ew] = out.early.dims;
    return { probs, top: k, cam: { data: cam, w: cw, h: ch }, early: { data: out.early.data, c: EC, w: ew, h: eh } };
  }
  // Mock: fraction of brown ("lesion") pixels per cell of a 7x7 grid drives probability and heatmap.
  const G = 7, cell = imageData.width / G, cam = new Float32Array(G * G); let total = 0;
  for (let y = 0; y < imageData.height; y++) for (let x = 0; x < imageData.width; x++) {
    const i = (y * imageData.width + x) * 4;
    if (d[i] > d[i + 1] * 0.85 && d[i] < 200 && d[i + 2] < d[i] * 0.7) { cam[Math.floor(y / cell) * G + Math.floor(x / cell)]++; total++; }
  }
  const frac = total / n, pd = 1 / (1 + Math.exp(-(frac - 0.02) * 120)), probs = [1 - pd, pd];
  return { probs, top: pd >= 0.5 ? 1 : 0, cam: { data: norm(cam), w: G, h: G }, early: null };
}
