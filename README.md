# Signal-to-Intelligence Hub

An open platform that follows information from the physical world to machine action:

**Real-world source → transduction and conditioning → sampling and quantization → binary / ASCII / numbers → processing, alerts and automation.**

Phase 1 contains four interactive simulators, the ESP32 hardware design and firmware, and a Google Apps Script email webhook. **Phase 2** adds a fifth studio, **Vision CNN**: in-browser CNN inference, feature maps, Grad-CAM and alerts, with a PyTorch pipeline in `ml-pipelines/vision-cnn-transfer/`. **Phase 3** adds a sixth studio, **Transformers & NLP**: tokenization, embeddings, attention and a semantic automation router, with a Python pipeline in `ml-pipelines/nlp-transformers/`.

## Repository map
| Folder | Contents |
|---|---|
| `apps/web-platform/` | React + Vite app. `src/theory/` = pure math; `src/simulators/` = the four studios |
| `edge-hardware/esp32-firmware/` | `sensor_to_bits.ino` |
| `edge-hardware/schematics-bom/` | Components, pin map, safety rules, BOM |
| `cloud-automation/google-apps-script/` | `Code.gs` and `DEPLOYMENT.md` |
| `ml-pipelines/vision-cnn-transfer/` | Phase 2: PyTorch training, Grad-CAM, ONNX export |
| `ml-pipelines/nlp-transformers/` | Phase 3: intent classifier, evaluation, ONNX notes |
| `docs/` | `ARCHITECTURE.md` |
| `.github/workflows/deploy.yml` | Builds and publishes the site to GitHub Pages |

## End-to-end architecture
```
Signals & bits ──► ESP32 & Apps Script ──► CNN vision & Grad-CAM ──► Transformers & semantic control
 (Text, Sound,       (ADC, hysteresis,       (pixels → features →      (tokens → embeddings →
  Image studios)      HTTPS POST, email)       class + heatmap)          attention → action plan)
        └──────────────────────────────┬──────────────────────────────────┘
                         one webhook: cloud-automation/google-apps-script/Code.gs
              payloads: telemetry (default) · event "vision" · event "nlp"
```
Each stage reuses the previous one: bytes become sensor codes, codes become alerts, pixels become class evidence, and sentences become structured actions. Every studio can send its result to the same Apps Script URL; the `event` field selects the email format. Details: `docs/ARCHITECTURE.md`.

## 1. Run it locally
```bash
npm install
npm run dev        # opens the web platform
npm run build      # production build in apps/web-platform/dist
```
Requires Node.js 18 or newer.

## 2. Publish it on GitHub Pages (fixes the 404)
1. Upload the **contents** of this folder to the repository root, so `package.json` and `apps/` are visible on the repository's front page.
2. **UPDATE (one time):** Settings → Pages → Build and deployment → Source: **GitHub Actions**.
3. Open the **Actions** tab and wait for "Deploy to GitHub Pages" to turn green.
4. The site is at `https://<user>.github.io/<repository>/`.

The repository must be public on a free account.

## 3. Connect the email alerts (optional)
Follow `cloud-automation/google-apps-script/DEPLOYMENT.md`, then paste the `/exec` URL and shared secret into the Automation Studio to test live.

## 4. Build the hardware (optional)
Wire the circuit from `edge-hardware/schematics-bom/README.md`, open `sensor_to_bits.ino` in the Arduino IDE (ESP32 board package installed) and edit the lines marked `UPDATE`.

## Where you must update things
| What | File | Marker |
|---|---|---|
| Wi-Fi name and password, Web App URL, token, device name | `sensor_to_bits.ino` | `UPDATE` |
| Trip/release temperatures, cooldown | `sensor_to_bits.ino` | `UPDATE` |
| Shared secret, alert address, optional Sheet ID | Apps Script **Script properties** | `DEPLOYMENT.md` |
| New code version after editing `Code.gs` | Apps Script **Manage deployments** | `DEPLOYMENT.md` |
| GitHub Pages source | repository Settings | step 2 above |
| Pin map changes | firmware **and** schematics README | keep both in sync |
| Your dataset, class names, trained model | `ml-pipelines/vision-cnn-transfer`, `public/models/` | `UPDATE` in README |
| onnxruntime-web version | `package.json` and `src/vision/inference.js` | keep identical |
| Your intent phrases and rules | `data/intents.csv`, `src/nlp/router.js` | `UPDATE` in the NLP README |
| Transformer model id | `src/nlp/modelEngine.js` | `UPDATE` comment |

## 5. Phase 2: computer vision
The Vision CNN tab works immediately in **mock mode** (a colour heuristic on built-in sample leaves). To use a real model:
1. Train and export following `ml-pipelines/vision-cnn-transfer/README.md`.
2. Commit `apps/web-platform/public/models/model.onnx` and `labels.json`, then push.
3. Reload the site: the tab reports "ONNX model mode". Upload, webcam and samples now use your CNN.
4. For alerts, **UPDATE** `Code.gs` in Apps Script with the version in this repository (it now accepts `event: "vision"`) and create a **new deployment version**. Paste the `/exec` URL into the Vision CNN tab.

## 6. Phase 3: transformers and semantic control
The Transformers & NLP tab works immediately with a built-in local engine. *Load real transformer* downloads MiniLM (about 23 MB) for real tokens, embeddings and attention input. To add a trained classifier head: follow `ml-pipelines/nlp-transformers/README.md`, commit the generated `public/models/nlp/intent_head.json` and `src/nlp/trainingData.js`, and push. UPDATE `Code.gs` once more (it now accepts `event: "nlp"`) and create a new deployment version.

## Deployment notes
- GitHub Pages builds from `.github/workflows/deploy.yml`; no extra step is needed for Phase 2 or 3 assets because files in `public/` are copied automatically.
- Large assets: each file must stay under 100 MB; keep `model.onnx` small (ResNet18 is about 45 MB).
- Webcam and microphone require HTTPS or localhost. The ONNX runtime and the MiniLM weights load from public CDNs and need internet access.
- Never commit your real shared secret or Apps Script URL if the repository is public; type them into the studios at run time.

## Extending the hub
- **New studio:** create `src/simulators/YourLab.jsx` and add one entry to `TABS` in `src/App.jsx`. Put reusable math in `src/theory/` and helpers in a new folder beside `vision/` and `nlp/`.
- **New intent:** add rows to `data/intents.csv`, run `train_intent.py --write-js`, and add an action rule in `src/nlp/router.js` (`plan`).
- **New vision classes:** change the folder names in your dataset and re-run the Phase 2 pipeline; labels travel in `labels.json`.
- **New webhook event:** add a branch in `Code.gs` keyed on `d.event`, redeploy a new version, and send the payload from your studio with the shared secret.
- **New sensor on the ESP32:** add the pin to the firmware and to the pin map in `edge-hardware/schematics-bom/README.md`; keep both in sync.
- Run `npm run build` before every pull request; it is the project's smoke test.

## Known limits of Phase 1, 2 and 3
- The firmware has not been flashed on real hardware by the author; verify readings against a multimeter.
- The firmware uses `setInsecure()` for TLS. Pin Google's root CA before unattended deployment.
- The browser cannot always read Apps Script's reply because of CORS; the email is still sent.
- Image uploads are downscaled to 128 px on the long side; the paint grid is 16×16.
- No automated tests yet.
- Phase 2: the model files are not included; the browser runs mock mode until you train and export one. The Grad-CAM overlay is exact only for the supplied ResNet18 head. The ONNX runtime loads its WebAssembly files from the jsDelivr CDN. Webcam needs HTTPS (GitHub Pages provides it) or localhost.
- Phase 3: the local engine is not a transformer, the attention heatmap is an illustrative single head, intent training data is small (48 phrases), and action routing is rule-based and English-only. Review actions before connecting real actuators.
