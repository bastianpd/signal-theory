import React, { useEffect, useMemo, useRef, useState } from "react";
import { asciiTable, entropy, frequencies, sizes } from "../theory/index.js";
import { CORPUS, INTENTS } from "../nlp/trainingData.js";
import { analyzeLocal, embedLocal, LOCAL_VOCAB_SIZE } from "../nlp/localEngine.js";
import { MODEL_ID, MODEL_VOCAB_SIZE, loadModel, analyzeModel, loadHead } from "../nlp/modelEngine.js";
import { attention, dot, pca2, softmax, unit } from "../nlp/math.js";
import { extract, severity, plan } from "../nlp/router.js";
import { Heatmap, Scatter } from "../nlp/components.jsx";

const DEFAULT = "Chamber temperature is rising fast and illumination dropped, trigger cooling and lighting alert";
export default function TransformerNlpLab() {
  const [text, setText] = useState(DEFAULT), [mode, setMode] = useState("local"), [msg, setMsg] = useState(""), [head, setHead] = useState(null);
  const [corpusVecs, setCorpusVecs] = useState(() => CORPUS.map((r) => embedLocal(r.text))), [ana, setAna] = useState(null);
  const [tau, setTau] = useState(0.3), [active, setActive] = useState(0), [url, setUrl] = useState(""), [token, setToken] = useState("change-me"), [log, setLog] = useState([]);
  const run = useRef(0);
  const add = (m) => setLog((l) => [`${new Date().toLocaleTimeString()}  ${m}`, ...l].slice(0, 8));
  const analyze = (t, m = mode) => (m === "model" ? analyzeModel(t) : Promise.resolve(analyzeLocal(t)));

  async function enableModel() {
    setMsg("Downloading model… 0%");
    try {
      await loadModel((p) => p.status === "progress" && setMsg(`Downloading ${p.file || "model"}… ${Math.round(p.progress || 0)}%`));
      setMsg("Embedding the training phrases…");
      const vecs = []; for (const r of CORPUS) vecs.push((await analyzeModel(r.text)).vec);
      setCorpusVecs(vecs); setHead(await loadHead()); setMode("model"); setMsg(`Real transformer active: ${MODEL_ID}.`);
    } catch (e) { setMsg(`Model unavailable (${e.message}). Staying on the local engine.`); setMode("local"); }
  }
  function useLocal() { setCorpusVecs(CORPUS.map((r) => embedLocal(r.text))); setMode("local"); setMsg(""); }
  useEffect(() => { const id = ++run.current; const h = setTimeout(async () => { const a = await analyze(text); if (id === run.current) { setAna(a); setActive(0); } }, 150); return () => clearTimeout(h); }, [text, mode]);

  // Classification: trained head (if exported) in model mode, otherwise cosine to class centroids.
  const cls = useMemo(() => {
    if (!ana) return null;
    let probs;
    if (mode === "model" && head && head.coef[0].length === ana.vec.length) probs = softmax(head.coef.map((w, k) => dot(w, ana.vec) + head.intercept[k]));
    else { const cents = INTENTS.map((c) => unit(corpusVecs.reduce((s, v, i) => (CORPUS[i].label === c ? s.map((x, j) => x + v[j]) : s), new Float32Array(corpusVecs[0].length)))); probs = softmax(cents.map((c) => dot(c, ana.vec) * 12)); }
    const names = mode === "model" && head ? head.classes : INTENTS;
    return names.map((label, i) => ({ label, p: probs[i] })).sort((a, b) => b.p - a.p);
  }, [ana, mode, head, corpusVecs]);
  const proj = useMemo(() => pca2(corpusVecs), [corpusVecs]);
  const pts = useMemo(() => corpusVecs.map((v, i) => { const [x, y] = proj.project(v); return { x, y, label: CORPUS[i].label, text: CORPUS[i].text }; }), [proj, corpusVecs]);
  const user = ana && (([x, y]) => ({ x, y }))(proj.project(ana.vec));
  const near = ana && corpusVecs.map((v, i) => ({ i, s: dot(v, ana.vec) })).sort((a, b) => b.s - a.s).slice(0, 5);
  const M = useMemo(() => ana && attention(ana.H, tau), [ana, tau]);

  const bytes = useMemo(() => [...new TextEncoder().encode(text)], [text]);
  const H = entropy(frequencies(text)), vocabN = mode === "model" ? MODEL_VOCAB_SIZE : LOCAL_VOCAB_SIZE, idBits = Math.ceil(Math.log2(vocabN));
  const params = useMemo(() => extract(text), [text]), top = cls?.[0];
  const sev = top ? severity(top.label, text, params) : "low", actions = top ? plan(top.label, params, sev) : [];
  const payload = top && { event: "nlp", device: "nlp-router", token, ts: new Date().toISOString(), text: text.slice(0, 500), intent: top.label, confidence: +top.p.toFixed(4), severity: sev, params: { sensor: params.sensor, trend: params.trend, location: params.location, measurements: params.measurements }, actions, engine: mode };
  async function send() {
    if (!url) { add("Simulated dispatch (no Web App URL configured)"); return; }
    try { const r = await fetch(url, { method: "POST", headers: { "Content-Type": "text/plain" }, body: JSON.stringify(payload) }); add(`Live POST: HTTP ${r.status} ${(await r.text()).slice(0, 80)}`); }
    catch (e) { add(`Live POST failed: ${e.message}`); }
  }

  return (<section>
    <h2>Transformers &amp; semantic control</h2>
    <p>From bytes to meaning: ASCII fixes <i>how characters are stored</i>; a transformer represents <i>what the text means</i>.</p>
    <label>Instruction or alert note<textarea value={text} onChange={(e) => setText(e.target.value)} /></label>
    <div className="row">
      <span role="status" className={mode === "local" ? "warn" : "hi"}>{mode === "local" ? "Local engine (instant, not a transformer)." : "Transformer engine (MiniLM)."} {msg}</span>
      {mode === "local" ? <button className="p" onClick={enableModel}>Load real transformer (~23 MB)</button> : <button onClick={useLocal}>Switch to local engine</button>}
    </div>

    <h3>1. Tokenization: bytes versus tokens</h3>
    <div className="stats"><div><b>{bytes.length}</b><span>UTF-8 bytes ({sizes(bytes.length * 8).bits} bits)</span></div><div><b>{H.toFixed(3)}</b><span>Shannon entropy (bits/char)</span></div>
      <div><b>{ana?.tokens.length ?? "…"}</b><span>tokens incl. [CLS], [SEP]</span></div><div><b>{ana ? ana.tokens.length * idBits : "…"}</b><span>bits for token IDs ({idBits} bits each, vocab {vocabN.toLocaleString()})</span></div></div>
    <p style={{ fontFamily: "var(--mono, monospace)", fontSize: ".8rem" }}>ASCII hex: {bytes.slice(0, 24).map((b) => b.toString(16).padStart(2, "0")).join(" ")}{bytes.length > 24 ? " …" : ""}</p>
    <div className="row">{ana?.tokens.map((k, i) => <span key={i} style={{ border: "1px solid var(--line)", borderRadius: 4, padding: "2px 6px", background: "var(--card)", fontFamily: "monospace", fontSize: ".85rem" }}>{k.t}<sub style={{ color: "var(--mut)" }}> {k.id}</sub></span>)}</div>
    <p>{mode === "local" ? "IDs come from the local vocabulary, built from the training phrases." : "IDs are real WordPiece IDs from the MiniLM tokenizer."} Rare words split into pieces marked with ##.</p>

    <h3>2. Embeddings and semantic similarity</h3>
    <p>Each phrase becomes a {ana?.vec.length ?? "…"}-dimensional vector; PCA projects it to 2-D. Nearby points mean similar meaning. ★ is your text.</p>
    <Scatter points={pts} user={user} classes={INTENTS} />
    {near && <table><thead><tr><th>Nearest training phrase</th><th>Intent</th><th>Cosine</th></tr></thead><tbody>{near.map(({ i, s }) => <tr key={i}><td>{CORPUS[i].text}</td><td>{CORPUS[i].label}</td><td>{s.toFixed(3)}</td></tr>)}</tbody></table>}

    <h3>3. Self-attention</h3>
    <p>Each row shows how one token distributes attention over all tokens: softmax(cos(qᵢ, kⱼ) / τ) with Q = K = {mode === "model" ? "MiniLM final hidden states" : "local token vectors"} plus position encoding. This is a single-head illustration, not the weights of a specific trained attention head. Click a row to focus it.</p>
    <label>Temperature τ = {tau}<input type="range" min="0.1" max="1" step="0.05" value={tau} onChange={(e) => setTau(+e.target.value)} /></label>
    {M && ana && <Heatmap tokens={ana.tokens} M={M} active={active} onPick={setActive} />}
    {M && ana && <p>“{ana.tokens[active]?.t}” attends most to: {M[active].map((w, j) => ({ w, t: ana.tokens[j].t })).sort((a, b) => b.w - a.w).slice(0, 3).map((x) => `${x.t} (${x.w.toFixed(2)})`).join(", ")}.</p>}

    <h3>4. Semantic automation router</h3>
    {cls && <table><thead><tr><th>Intent</th><th>Probability</th><th></th></tr></thead><tbody>{cls.map((c) => <tr key={c.label}><td>{c.label}</td><td>{(c.p * 100).toFixed(1)}%</td><td><div style={{ background: "var(--acc)", height: 8, width: `${c.p * 200}px` }} /></td></tr>)}</tbody></table>}
    <p>Classifier: {mode === "model" && head ? "trained logistic-regression head on MiniLM embeddings" : "cosine similarity to intent centroids"}. Severity: <b className={sev === "high" ? "warn" : "hi"}>{sev}</b></p>
    <pre>{JSON.stringify({ intent: top?.label, severity: sev, parameters: params, actions }, null, 2)}</pre>
    <div className="row"><label>Apps Script Web App URL (optional)<input style={{ width: 300 }} value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://script.google.com/macros/s/…/exec" /></label>
      <label>Shared secret<input value={token} onChange={(e) => setToken(e.target.value)} /></label><button className="p" disabled={!payload} onClick={send}>Dispatch action payload</button></div>
    <pre>{payload ? JSON.stringify(payload, null, 2) : ""}</pre><pre>{log.join("\n") || "Event log is empty."}</pre>
  </section>);
}
