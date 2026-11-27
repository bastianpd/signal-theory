// Self-contained fallback engine: always available, deterministic, no downloads.
// It is NOT a transformer: a WordPiece-style tokenizer plus hashed character n-gram and concept features.
import { CORPUS } from "./trainingData.js";
import { unit } from "./math.js";
const DIM = 256;
const CONCEPTS = {
  heat: ["hot", "warm", "heat", "overheat", "temperature", "degree", "celsius", "cool", "cooling", "thermal"],
  light: ["light", "lighting", "illumination", "dark", "dim", "lux", "bright", "brightness", "lamp", "sunlight", "led", "lights"],
  control: ["turn", "switch", "activate", "start", "stop", "shut", "enable", "open", "close", "disable"],
  device: ["fan", "relay", "pump", "heater", "valve", "motor", "ventilation", "led"],
  query: ["what", "show", "report", "summary", "how", "status", "online", "latest", "current"],
  fault: ["strange", "unexpected", "spike", "noisy", "erratic", "abnormal", "unusual", "failure", "flatlined", "unstable", "vibration"],
  service: ["schedule", "maintenance", "replace", "calibrate", "clean", "update", "inspect", "service", "remind"],
  rise: ["rising", "climbing", "exceeded", "above", "high", "increase", "spike"],
  fall: ["dropped", "fell", "falling", "below", "low", "dim", "dark", "decrease"],
};
const fnv = (s) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; };
export const words = (t) => t.toLowerCase().match(/[a-z]+|\d+|[^\sa-z\d]/g) || [];
export function embedLocal(text) {
  const v = new Float32Array(DIM), add = (f, w) => { const h = fnv(f); v[h % DIM] += ((h >>> 16) & 1 ? 1 : -1) * w; };
  for (const w of words(text)) {
    add("w:" + w, 1); const p = `<${w}>`;
    for (let n = 3; n <= 4; n++) for (let i = 0; i + n <= p.length; i++) add("g:" + p.slice(i, i + n), 0.4);
    for (const [c, l] of Object.entries(CONCEPTS)) if (l.includes(w) || l.some((x) => w.length > 3 && x.length >= 3 && (w.startsWith(x) || x.startsWith(w)))) add("c:" + c, 2.5);
  }
  return unit(v);
}
// Vocabulary: specials, every corpus word, suffix pieces, single characters.
const vocab = new Map(); const addV = (t) => { if (!vocab.has(t)) vocab.set(t, vocab.size); };
["[PAD]", "[UNK]", "[CLS]", "[SEP]"].forEach(addV);
CORPUS.forEach((r) => words(r.text).forEach(addV));
["##s", "##ed", "##ing", "##er", "##ly", "##ion", "##ure", "##ance"].forEach(addV);
"abcdefghijklmnopqrstuvwxyz0123456789".split("").forEach((c) => { addV(c); addV("##" + c); });
export const LOCAL_VOCAB_SIZE = vocab.size;
function pieces(word) {
  if (vocab.has(word)) return [word];
  const out = []; let s = 0;
  while (s < word.length) {
    let e = word.length, hit = null;
    while (e > s) { const p = (s ? "##" : "") + word.slice(s, e); if (vocab.has(p)) { hit = p; break; } e--; }
    if (!hit) return ["[UNK]"]; out.push(hit); s = e;
  }
  return out;
}
export function tokenizeLocal(text) {
  const toks = ["[CLS]", ...words(text).flatMap(pieces), "[SEP]"];
  return toks.map((t) => ({ t, id: vocab.get(t) }));
}
export function analyzeLocal(text) {
  const tokens = tokenizeLocal(text);
  return { tokens, H: tokens.map((k) => embedLocal(k.t.replace("##", ""))), vec: embedLocal(text) };
}
