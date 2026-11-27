# Vision CNN with transfer learning (Phase 2)

Trains a ResNet18 classifier on your images, explains it with Grad-CAM, exports it to ONNX and runs it in the browser inside the **Vision CNN** tab.

## Why transfer learning
A deep CNN trained from scratch has millions of parameters and needs very large datasets; with a few thousand images it memorizes them (overfits). A backbone pretrained on ImageNet has already learned reusable features: edges and colour blobs in early layers, textures and parts in middle layers. Only the last layers need to learn your task. Result: higher accuracy from fewer images, minutes instead of days of training, and less overfitting. This pipeline trains in two stages: (1) new classifier head only, (2) head plus `layer4` at a ten times lower learning rate.

## From pixels to features
An image is a 3 × 224 × 224 grid of 8-bit values (Phase 1's Image Studio). Convolution slides small learned kernels over it, producing feature maps. Early maps respond to edges; pooling and stride shrink the grid (224 → 112 → 56 → 28 → 14 → 7) while channels grow (3 → 512), so spatial detail is traded for semantic content. Global average pooling collapses the 7 × 7 grid into 512 numbers, and a linear layer turns them into class scores.

## Why Grad-CAM
Accuracy alone does not show whether a model looks at the lesion or at a background artefact. Grad-CAM weights the last feature maps by the gradient of the class score and shows where the evidence is. A defect model whose heat sits on the table, not the leaf, should not be trusted. For this ResNet18 head (global average pool + linear), Grad-CAM equals the class activation map, so the exported ONNX `cam` output is the exact Grad-CAM the browser overlays.

## 1. Prepare data
```
data/
  healthy/*.jpg
  diseased/*.jpg
```
Any number of class folders works. Public options: **PlantVillage** (group leaf images into class folders) or an **MVTec-style** set exported as `good/` and `defect/`. Respect each dataset's licence. UPDATE: use your own class names, and keep the same folders for later phases.

## 2. Train
```bash
cd ml-pipelines/vision-cnn-transfer
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --data /path/to/data
```
Writes to `outputs/`: `best.pt`, `classification_report.txt`, `confusion_matrix.png`, `history.json`. A GPU is optional; CPU works for small sets.

## 3. Grad-CAM figures
```bash
python gradcam.py --images some_leaf.jpg another.jpg
```

## 4. Export to ONNX and deploy
```bash
python export_onnx.py
```
This writes `model.onnx` and `labels.json` into `apps/web-platform/public/models/` and prints the maximum torch-versus-ONNX logit difference (it should be near 1e-5 or smaller). Commit both files, push, and GitHub Pages redeploys. The Vision CNN tab switches from mock mode to ONNX mode automatically. Keep `model.onnx` under GitHub's 100 MB file limit; ResNet18 is about 45 MB.

## ONNX contract the web app expects
Input `input` [1,3,224,224], ImageNet-normalized. Outputs `logits` [1,K], `cam` [1,K,7,7], `early` [1,16,112,112]. `labels.json` holds `classes`, `mean`, `std`. If you change the backbone, keep these outputs.

## Limits
Validation accuracy on a random split overstates real-world performance when images of one plant or part appear in both splits. Split by source if you can. The mock mode in the browser is a colour heuristic for demonstration only.
