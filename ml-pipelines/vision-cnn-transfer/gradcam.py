"""True Grad-CAM (gradient-weighted) heatmaps for the trained ResNet18, saved as overlay PNGs."""
import argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--checkpoint", default="outputs/best.pt"); a.add_argument("--images", nargs="+", required=True); a.add_argument("--out", default="outputs/gradcam")
    a = a.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.checkpoint, map_location="cpu"); classes = ck["classes"]
    m = models.resnet18(); m.fc = nn.Linear(m.fc.in_features, len(classes)); m.load_state_dict(ck["state_dict"]); m.eval()
    store = {}
    m.layer4.register_forward_hook(lambda _m, _i, o: store.__setitem__("a", o))
    m.layer4.register_full_backward_hook(lambda _m, _gi, go: store.__setitem__("g", go[0]))
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for path in a.images:
        img = Image.open(path).convert("RGB"); x = tf(img).unsqueeze(0)
        logits = m(x); k = int(logits.argmax(1)); m.zero_grad(); logits[0, k].backward()
        w = store["g"].mean((2, 3), keepdim=True)                      # channel weights = pooled gradients
        cam = torch.relu((w * store["a"]).sum(1)).squeeze().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        prob = torch.softmax(logits, 1)[0, k].item()
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img.resize((224, 224))); ax[0].axis("off")
        ax[1].imshow(img.resize((224, 224))); ax[1].imshow(np.array(Image.fromarray((cam * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)), cmap="jet", alpha=0.5); ax[1].axis("off")
        ax[1].set_title(f"{classes[k]} ({prob:.1%})"); fig.tight_layout()
        fig.savefig(Path(a.out) / f"{Path(path).stem}_gradcam.png", dpi=150); plt.close(fig)
        print(f"{path}: {classes[k]} {prob:.3f}")

if __name__ == "__main__": main()
