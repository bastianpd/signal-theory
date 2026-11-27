"""Transfer learning with ResNet18 on an ImageFolder dataset.

Dataset layout (UPDATE: point --data at your own folder):
    data/<class_name>/*.jpg      e.g. data/healthy/..., data/diseased/...
Public sources: PlantVillage (group leaf images by class folder), or any MVTec-style
set exported as good/ and defect/ folders.

Stage 1 trains only the new classifier head. Stage 2 also fine-tunes layer4 at a lower rate.
"""
import argparse, json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def loaders(data, val_split, batch, seed):
    train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(0.2, 0.2, 0.1), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    val_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    full_tr, full_va = datasets.ImageFolder(data, train_tf), datasets.ImageFolder(data, val_tf)
    idx = list(range(len(full_tr))); random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(idx) * val_split))
    tr = DataLoader(Subset(full_tr, idx[n_val:]), batch_size=batch, shuffle=True, num_workers=2)
    va = DataLoader(Subset(full_va, idx[:n_val]), batch_size=batch, num_workers=2)
    return tr, va, full_tr.classes

def run_epoch(model, loader, device, opt=None):
    model.train(opt is not None); loss_sum = correct = n = 0; ys, ps = [], []
    crit = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device); out = model(x); loss = crit(out, y)
            if opt: opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y); pred = out.argmax(1); correct += (pred == y).sum().item(); n += len(y)
            ys += y.cpu().tolist(); ps += pred.cpu().tolist()
    return loss_sum / n, correct / n, ys, ps

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True); a.add_argument("--out", default="outputs")
    a.add_argument("--head-epochs", type=int, default=3); a.add_argument("--finetune-epochs", type=int, default=5)
    a.add_argument("--batch", type=int, default=32); a.add_argument("--val-split", type=float, default=0.2); a.add_argument("--seed", type=int, default=42)
    a = a.parse_args()
    torch.manual_seed(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, classes = loaders(a.data, a.val_split, a.batch, a.seed)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(classes)); model.to(device)
    hist, best = [], 0.0
    def stage(epochs, params, lr, name):
        nonlocal best
        opt = torch.optim.Adam(params, lr=lr)
        for e in range(epochs):
            tl, ta, _, _ = run_epoch(model, tr, device, opt); vl, vacc, ys, ps = run_epoch(model, va, device)
            hist.append(dict(stage=name, epoch=e + 1, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=vacc))
            print(f"[{name}] epoch {e+1}: train {ta:.3f}/{tl:.3f}  val {vacc:.3f}/{vl:.3f}")
            if vacc >= best:
                best = vacc; torch.save({"state_dict": model.state_dict(), "classes": classes}, out / "best.pt")
    stage(a.head_epochs, model.fc.parameters(), 1e-3, "head")
    for p in model.layer4.parameters(): p.requires_grad = True
    stage(a.finetune_epochs, list(model.layer4.parameters()) + list(model.fc.parameters()), 1e-4, "finetune")
    ck = torch.load(out / "best.pt", map_location=device); model.load_state_dict(ck["state_dict"])
    _, acc, ys, ps = run_epoch(model, va, device)
    report = classification_report(ys, ps, target_names=classes, digits=3, zero_division=0)
    (out / "classification_report.txt").write_text(report); print(report)
    cm = confusion_matrix(ys, ps, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 3, 1.2 * len(classes) + 3)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right"); ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)): ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150)
    (out / "history.json").write_text(json.dumps(hist, indent=2)); print(f"Best validation accuracy: {best:.3f}")

if __name__ == "__main__": main()
