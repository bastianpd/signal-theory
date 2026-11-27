"""Export the fine-tuned ResNet18 to ONNX for onnxruntime-web.

Outputs: logits [1,K]; cam [1,K,7,7]; early [1,16,112,112].
For a global-average-pool + linear head, Grad-CAM equals the class activation map
CAM_k = sum_c w_kc * A_c, so the exported `cam` output is the exact Grad-CAM for that class.
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torchvision import models

class Exported(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        m = self.m; x = m.relu(m.bn1(m.conv1(x))); early = x[:, :16]
        f = m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(x)))))
        return m.fc(f.mean((2, 3))), torch.einsum("bchw,kc->bkhw", f, m.fc.weight), early

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--checkpoint", default="outputs/best.pt")
    a.add_argument("--out-dir", default="../../apps/web-platform/public/models")  # UPDATE if your layout differs
    a = a.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu"); classes = ck["classes"]
    m = models.resnet18(); m.fc = nn.Linear(m.fc.in_features, len(classes)); m.load_state_dict(ck["state_dict"]); m.eval()
    net = Exported(m).eval(); dummy = torch.randn(1, 3, 224, 224); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=["input"], output_names=["logits", "cam", "early"], opset_version=17)
    try: torch.onnx.export(net, dummy, out / "model.onnx", dynamo=False, **kw)
    except TypeError: torch.onnx.export(net, dummy, out / "model.onnx", **kw)
    (out / "labels.json").write_text(json.dumps({"classes": classes, "input_size": 224,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, indent=2))
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "model.onnx")); got = sess.run(None, {"input": dummy.numpy()})[0]
    ref = net(dummy)[0].detach().numpy(); print("max |torch - onnx| logits:", float(np.abs(got - ref).max()))
    print(f"Wrote {out/'model.onnx'} and labels.json")

if __name__ == "__main__": main()
