// Optional real transformer (MiniLM) running in the browser through transformers.js (ONNX Runtime Web).
// Weights are downloaded once from the Hugging Face Hub (about 23 MB) and cached by the browser.
import { unit } from "./math.js";
export const MODEL_ID = "Xenova/all-MiniLM-L6-v2"; // UPDATE to another transformers.js-compatible feature-extraction model if desired
export const MODEL_VOCAB_SIZE = 30522;
let ext = null;
export async function loadModel(onProgress) {
  const tf = await import("@xenova/transformers");
  tf.env.allowLocalModels = false;
  ext = await tf.pipeline("feature-extraction", MODEL_ID, { progress_callback: onProgress });
}
export async function analyzeModel(text) {
  const out = await ext(text), [, n, d] = out.dims;
  const H = Array.from({ length: n }, (_, i) => Float32Array.from(out.data.slice(i * d, (i + 1) * d)));
  const ids = Array.from(ext.tokenizer(text).input_ids.data, Number);
  const tokens = ids.map((id) => ({ id, t: ext.tokenizer.decode([id]) }));
  const mean = new Float32Array(d); H.forEach((h) => h.forEach((v, i) => (mean[i] += v / n)));
  return { tokens, H, vec: unit(mean) }; // mean pooling + L2 normalization, as in sentence-transformers
}
// Optional trained head exported by train_intent.py (public/models/nlp/intent_head.json).
export async function loadHead() {
  try {
    const r = await fetch(import.meta.env.BASE_URL + "models/nlp/intent_head.json");
    if (!r.ok || (r.headers.get("content-type") || "").includes("html")) return null;
    return await r.json();
  } catch { return null; }
}
