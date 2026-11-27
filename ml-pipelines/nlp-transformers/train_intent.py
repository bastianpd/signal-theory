"""Intent classifier: sentence-transformer embeddings + a logistic-regression head.

Input CSV columns: label,text  (UPDATE: add your own phrases to data/intents.csv, at least 5 per label).
Outputs in --out: classification_report.txt, confusion_matrix.png, and intent_head.json,
which the web app reads to classify with real embeddings.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="data/intents.csv"); a.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    a.add_argument("--out", default="outputs"); a.add_argument("--web-models", default="../../apps/web-platform/public/models/nlp")  # UPDATE if your layout differs
    a.add_argument("--write-js", action="store_true", help="Regenerate apps/web-platform/src/nlp/trainingData.js from the CSV")
    a = a.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data); classes = sorted(df.label.unique())
    X = SentenceTransformer(a.model).encode(df.text.tolist(), normalize_embeddings=True); y = df.label.map(classes.index).values
    clf = LogisticRegression(max_iter=3000, C=10)
    folds = StratifiedKFold(n_splits=min(5, int(np.bincount(y).min())), shuffle=True, random_state=42)
    pred = cross_val_predict(clf, X, y, cv=folds)       # every phrase is predicted by a model that never saw it
    report = classification_report(y, pred, target_names=classes, digits=3, zero_division=0)
    (out / "classification_report.txt").write_text(report); print(report); print("Cross-validated accuracy:", round(accuracy_score(y, pred), 3))
    cm = confusion_matrix(y, pred); fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right"); ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)): ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150)
    clf.fit(X, y); head = {"model": a.model, "classes": classes, "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}
    (out / "intent_head.json").write_text(json.dumps(head)); Path(a.web_models).mkdir(parents=True, exist_ok=True)
    (Path(a.web_models) / "intent_head.json").write_text(json.dumps(head)); print("Wrote intent_head.json to", a.web_models)
    if a.write_js:
        js = Path("../../apps/web-platform/src/nlp/trainingData.js")
        js.write_text("// GENERATED from ml-pipelines/nlp-transformers/data/intents.csv by train_intent.py --write-js. Edit the CSV, not this file.\nexport const CORPUS = "
                      + json.dumps(df.to_dict("records"), indent=1) + ";\nexport const INTENTS = [...new Set(CORPUS.map((r) => r.label))];\n"); print("Wrote", js)

if __name__ == "__main__": main()
