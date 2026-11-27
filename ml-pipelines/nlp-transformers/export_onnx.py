"""Export the embedding model to ONNX (for self-hosting, instead of downloading from the Hugging Face Hub).

The shipped web app loads Xenova/all-MiniLM-L6-v2 (an ONNX build) through transformers.js. Run this only if you want
your own model or a self-hosted copy. Afterwards point MODEL_ID in src/nlp/modelEngine.js at the folder (see README).
"""
import argparse
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

a = argparse.ArgumentParser(); a.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2"); a.add_argument("--out", default="outputs/onnx-model")
a = a.parse_args()
ORTModelForFeatureExtraction.from_pretrained(a.model, export=True).save_pretrained(a.out)
AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
print("Exported to", a.out)
