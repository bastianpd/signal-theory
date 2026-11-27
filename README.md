# Signal-to-Intelligence Hub

An open platform that follows information from the physical world to machine action:

**Real-world source → transduction and conditioning → sampling and quantization → binary / ASCII / numbers → processing, alerts and automation.**

Phase 1 contains four interactive simulators, the ESP32 hardware design and firmware, and a Google Apps Script email webhook. **Phase 2** adds a fifth studio, **Vision CNN**: in-browser CNN inference, feature maps, Grad-CAM and alerts, with a PyTorch pipeline in `ml-pipelines/vision-cnn-transfer/`. Phase 3 (transformers/NLP) is reserved.

## Repository map
| Folder | Contents |
|---|---|
| `apps/web-platform/` | React + Vite app. `src/theory/` = pure math; `src/simulators/` = the four studios |
| `edge-hardware/esp32-firmware/` | `sensor_to_bits.ino` |
| `edge-hardware/schematics-bom/` | Components, pin map, safety rules, BOM |
| `cloud-automation/google-apps-script/` | `Code.gs` and `DEPLOYMENT.md` |
| `ml-pipelines/vision-cnn-transfer/` | Phase 2: PyTorch training, Grad-CAM, ONNX export |
| `docs/` | `ARCHITECTURE.md` |
| `.github/workflows/deploy.yml` | Builds and publishes the site to GitHub Pages |

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
| Phase 3 work | `ml-pipelines/README.md` | create folder when it starts |

## 5. Phase 2: computer vision
The Vision CNN tab works immediately in **mock mode** (a colour heuristic on built-in sample leaves). To use a real model:
1. Train and export following `ml-pipelines/vision-cnn-transfer/README.md`.
2. Commit `apps/web-platform/public/models/model.onnx` and `labels.json`, then push.
3. Reload the site: the tab reports "ONNX model mode". Upload, webcam and samples now use your CNN.
4. For alerts, **UPDATE** `Code.gs` in Apps Script with the version in this repository (it now accepts `event: "vision"`) and create a **new deployment version**. Paste the `/exec` URL into the Vision CNN tab.

## Known limits of Phase 1 and 2
- The firmware has not been flashed on real hardware by the author; verify readings against a multimeter.
- The firmware uses `setInsecure()` for TLS. Pin Google's root CA before unattended deployment.
- The browser cannot always read Apps Script's reply because of CORS; the email is still sent.
- Image uploads are downscaled to 128 px on the long side; the paint grid is 16×16.
- No automated tests yet.
- Phase 2: the model files are not included; the browser runs mock mode until you train and export one. The Grad-CAM overlay is exact only for the supplied ResNet18 head. The ONNX runtime loads its WebAssembly files from the jsDelivr CDN. Webcam needs HTTPS (GitHub Pages provides it) or localhost.
