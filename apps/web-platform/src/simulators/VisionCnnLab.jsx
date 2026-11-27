import React, { useEffect, useMemo, useRef, useState } from "react";
import { toBin, sizes } from "../theory/index.js";
import { SIZE, KERNELS, channel, grayscale, convolve, normalize } from "../vision/imageOps.js";
import { MapCanvas, Thumb } from "../vision/components.jsx";
import { SAMPLES, drawSample } from "../vision/samples.js";
import { loadModel, classify } from "../vision/inference.js";

export default function VisionCnnLab() {
  const src = useRef(null), video = useRef(null), stream = useRef(null), lastSent = useRef(0);
  const [model, setModel] = useState(null), [res, setRes] = useState(null), [pix, setPix] = useState(null), [cam, setCam] = useState(true);
  const [thr, setThr] = useState(0.8), [target, setTarget] = useState("diseased"), [url, setUrl] = useState(""), [token, setToken] = useState("change-me");
  const [auto, setAuto] = useState(true), [log, setLog] = useState([]), [camMsg, setCamMsg] = useState(""), [live, setLive] = useState(false), [busy, setBusy] = useState(false);
  const add = (m) => setLog((l) => [`${new Date().toLocaleTimeString()}  ${m}`, ...l].slice(0, 10));
  useEffect(() => { loadModel().then((m) => { setModel(m); setTarget(m.classes[m.classes.length - 1]); }); return stopCam; }, []);
  useEffect(() => { if (model) sample("Diseased leaf"); }, [model]);

  function stopCam() { stream.current?.getTracks().forEach((t) => t.stop()); stream.current = null; setLive(false); }
  function fit(drawable, w, h) { // centre-crop to a square, then resize to 224x224
    const g = src.current.getContext("2d"), s = Math.min(w, h); g.drawImage(drawable, (w - s) / 2, (h - s) / 2, s, s, 0, 0, SIZE, SIZE);
  }
  async function analyse() {
    if (!model) return; setBusy(true);
    const im = src.current.getContext("2d").getImageData(0, 0, SIZE, SIZE), n = SIZE * SIZE;
    const r = await classify(im, model), gray = grayscale(im.data, n);
    setPix({ R: channel(im.data, 0, n), G: channel(im.data, 1, n), B: channel(im.data, 2, n), maps: Object.entries(KERNELS).map(([k, v]) => [k, normalize(convolve(gray, SIZE, SIZE, v))]), first: [...im.data.slice(0, 4)] });
    setRes(r); setBusy(false); maybeAlert(r);
  }
  function sample(name) { stopCam(); drawSample(src.current.getContext("2d"), name, SIZE); analyse(); }
  function upload(e) {
    const f = e.target.files[0]; if (!f) return; stopCam(); const img = new Image();
    img.onload = () => { fit(img, img.width, img.height); URL.revokeObjectURL(img.src); analyse(); }; img.src = URL.createObjectURL(f);
  }
  async function startCam() {
    if (!navigator.mediaDevices?.getUserMedia) { setCamMsg("Webcam API unavailable here. Use an upload or a sample image."); return; }
    try { stream.current = await navigator.mediaDevices.getUserMedia({ video: true }); video.current.srcObject = stream.current; await video.current.play(); setLive(true); setCamMsg(""); }
    catch (e) { setCamMsg(e.name === "NotAllowedError" ? "Camera permission denied. Allow it in the browser, or use an upload or a sample image." : "No camera found. Use an upload or a sample image."); }
  }
  function capture() { fit(video.current, video.current.videoWidth, video.current.videoHeight); stopCam(); analyse(); }

  const label = res && model.classes[res.top], conf = res ? res.probs[res.top] : 0, hit = res && label === target && conf >= thr;
  const payload = res && { event: "vision", device: "vision-lab", token, ts: new Date().toISOString(), label, confidence: +conf.toFixed(4), threshold: thr, mode: model.mode };
  async function send(p) {
    if (!url) { add("Simulated alert (no Web App URL configured)"); return; }
    try { const r = await fetch(url, { method: "POST", headers: { "Content-Type": "text/plain" }, body: JSON.stringify(p) }); add(`Live POST: HTTP ${r.status} ${(await r.text()).slice(0, 80)}`); }
    catch (e) { add(`Live POST failed: ${e.message}`); }
  }
  function maybeAlert(r) {
    if (!auto || !model) return; const l = model.classes[r.top], c = r.probs[r.top];
    if (l !== target || c < thr) { add(`No alert: ${l} ${(c * 100).toFixed(1)}%`); return; }
    if (Date.now() - lastSent.current < 30000) { add("Alert suppressed by 30 s cooldown"); return; }
    lastSent.current = Date.now(); send({ event: "vision", device: "vision-lab", token, ts: new Date().toISOString(), label: l, confidence: +c.toFixed(4), threshold: thr, mode: model.mode });
  }
  const bits = sizes(SIZE * SIZE * 24);
  const early = useMemo(() => res?.early && Array.from({ length: 8 }, (_, i) => normalize(res.early.data.slice(i * res.early.w * res.early.h, (i + 1) * res.early.w * res.early.h))), [res]);

  return (<section>
    <h2>Vision CNN lab</h2>
    <p role="status" className={model?.mode === "mock" ? "warn" : "hi"}>{model ? `${model.mode === "onnx" ? "ONNX model" : "Mock"} mode. ${model.note}` : "Loading model…"}</p>
    <div className="row">
      <label>Upload image<input type="file" accept="image/*" onChange={upload} /></label>
      <label>Sample<select onChange={(e) => sample(e.target.value)} defaultValue="Diseased leaf">{Object.keys(SAMPLES).map((s) => <option key={s}>{s}</option>)}</select></label>
      {live ? <button className="p" onClick={capture}>Capture frame</button> : <button onClick={startCam}>Start webcam</button>}
    </div>
    {camMsg && <p className="warn" role="status">{camMsg}</p>}
    <video ref={video} muted playsInline style={{ display: live ? "block" : "none", maxWidth: 320, marginBottom: 8 }} />

    <h3>1. Raw pixels and 8-bit RGB channels</h3>
    <div className="row" style={{ alignItems: "start" }}>
      <div style={{ position: "relative", width: 224, height: 224 }}>
        <canvas ref={src} width={SIZE} height={SIZE} />
        {cam && res && <MapCanvas data={res.cam.data} w={res.cam.w} h={res.cam.h} size={SIZE} mode="jet" style={{ position: "absolute", left: 0, top: 0, opacity: 0.55, pointerEvents: "none", border: 0 }} />}
      </div>
      {pix && ["R", "G", "B"].map((c) => <Thumb key={c} title={`${c} channel (0–255)`}><MapCanvas data={pix[c]} w={SIZE} h={SIZE} /></Thumb>)}
    </div>
    <p>Tensor 3 × {SIZE} × {SIZE} = {3 * SIZE * SIZE} values, {bits.bits} bits, {bits.bytes} bytes ({bits.kib.toFixed(0)} KiB) uncompressed. {pix && <>First pixel: R {pix.first[0]} = {toBin(pix.first[0])}, G {pix.first[1]} = {toBin(pix.first[1])}, B {pix.first[2]} = {toBin(pix.first[2])}.</>}</p>

    <h3>2. Convolutional filters and feature maps</h3>
    <p>Each 3×3 kernel slides over the grey image, then ReLU keeps positive responses. Early CNN layers learn filters like these.</p>
    <div className="row" style={{ alignItems: "start" }}>{pix?.maps.map(([k, d]) => <Thumb key={k} title={k}><MapCanvas data={d} w={SIZE} h={SIZE} /></Thumb>)}</div>
    {early ? <><p>First 8 of 16 learned conv1 feature maps exported from your ONNX model:</p>
      <div className="row" style={{ alignItems: "start" }}>{early.map((d, i) => <Thumb key={i} title={`conv1 channel ${i}`}><MapCanvas data={d} w={res.early.w} h={res.early.h} size={96} /></Thumb>)}</div></>
      : <p className="warn">Learned feature maps appear when a custom ONNX model is loaded.</p>}

    <h3>3. Backbone, class probabilities and Grad-CAM</h3>
    <p>Backbone: ResNet18 pretrained on ImageNet, fine-tuned by transfer learning. Grad-CAM highlights the regions that raised the predicted class score.</p>
    <label style={{ flexDirection: "row", gap: 8 }}><input type="checkbox" checked={cam} onChange={(e) => setCam(e.target.checked)} />Show Grad-CAM overlay on the image above</label>
    {res && <table><thead><tr><th>Class</th><th>Probability</th><th></th></tr></thead><tbody>{model.classes.map((c, i) => <tr key={c}><td>{c}{i === res.top ? " ◀" : ""}</td><td>{(res.probs[i] * 100).toFixed(2)}%</td>
      <td><div style={{ background: "var(--acc)", height: 8, width: `${res.probs[i] * 200}px` }} /></td></tr>)}</tbody></table>}
    {busy && <p>Analysing…</p>}

    <h3>4. Alert automation</h3>
    <div className="row">
      <label>Alert when class<select value={target} onChange={(e) => setTarget(e.target.value)}>{model?.classes.map((c) => <option key={c}>{c}</option>)}</select></label>
      <label>Confidence ≥ {Math.round(thr * 100)}%<input type="range" min="0.5" max="0.99" step="0.01" value={thr} onChange={(e) => setThr(+e.target.value)} /></label>
      <label>Web App URL (optional)<input style={{ width: 300 }} value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://script.google.com/macros/s/…/exec" /></label>
      <label>Shared secret<input value={token} onChange={(e) => setToken(e.target.value)} /></label>
      <label style={{ flexDirection: "row", gap: 6 }}><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />Send automatically</label>
      <button className="p" disabled={!res} onClick={() => send(payload)}>Send test alert</button>
    </div>
    {res && <p>Condition: <b className={hit ? "warn" : "hi"}>{hit ? "MET: alert would fire" : "not met"}</b></p>}
    <pre>{payload ? JSON.stringify(payload, null, 2) : "Analyse an image to see the payload."}</pre>
    <pre>{log.join("\n") || "Event log is empty."}</pre>
  </section>);
}
