# NLP and transformers pipeline (Phase 3)

Trains an **intent classifier for hardware and monitoring commands** on top of a compact transformer embedding model (`all-MiniLM-L6-v2`), evaluates it, and connects it to the **Transformers & NLP** tab.

## From ASCII to transformers
ASCII and UTF-8 answer *how characters are stored*: one or more bytes per character, with no notion of meaning. Shannon entropy measures how predictable the symbols are, still without meaning. A transformer adds three steps:
1. **Tokenization.** Text is split into subword tokens (WordPiece) and mapped to integer IDs from a fixed vocabulary (30,522 entries for MiniLM). Frequent words stay whole; rare words split into pieces such as `over` + `##heat` + `##ing`.
2. **Embeddings.** Every token becomes a vector, and the network refines it layer by layer using context. Averaging the token vectors gives a sentence vector. Phrases with similar meaning end up close together (high cosine similarity), even without shared words.
3. **Self-attention.** Each token computes weights over all tokens (softmax of scaled query-key dot products) and mixes their values by those weights. This is how "it" can borrow meaning from "chamber".

## How this hub uses it
- **Training (Python):** sentence embeddings + logistic regression head on `data/intents.csv` (6 intents, 8 phrases each). Metrics come from stratified cross-validation, so each phrase is scored by a model that never saw it.
- **Browser, local engine (default, instant):** a WordPiece-style tokenizer, hashed character n-gram and concept features, centroid classifier. It is not a transformer; it keeps the studio responsive offline and while weights download.
- **Browser, transformer engine (optional):** the button *Load real transformer* downloads the ONNX build `Xenova/all-MiniLM-L6-v2` through `@xenova/transformers` (about 23 MB, cached by the browser). Tokens, IDs and embeddings are then real. If `public/models/nlp/intent_head.json` exists, the trained head replaces the centroid classifier.
- **Attention view:** a single-head scaled dot-product illustration over token vectors (the model's final hidden states in transformer mode). It is **not** the weights of a specific trained attention head; the UI says so.
- **Router:** regular expressions extract measurements, trend, targets, state and duration; severity is scored; an action plan is built; the payload goes to Apps Script (`event: "nlp"`).

## Run it
```bash
cd ml-pipelines/nlp-transformers
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_intent.py --write-js
```
This prints precision/recall/F1 and cross-validated accuracy, saves `outputs/confusion_matrix.png`, writes `intent_head.json` into `apps/web-platform/public/models/nlp/`, and (with `--write-js`) regenerates `src/nlp/trainingData.js` from the CSV. Commit both generated files and push; GitHub Pages redeploys.

**UPDATE:** add your own phrases to `data/intents.csv` (at least 5 per label; add new labels freely). Then add rules for them in `src/nlp/router.js` (`plan`) if they need device actions.

## ONNX export and self-hosting
The app loads a ready-made ONNX build from the Hugging Face Hub, so no export is required. To host your own copy or a different model:
```bash
python export_onnx.py --model sentence-transformers/all-MiniLM-L6-v2 --out outputs/onnx-model
```
Copy the folder to `apps/web-platform/public/models/nlp/minilm/`, then in `src/nlp/modelEngine.js` set `tf.env.allowLocalModels = true`, `tf.env.localModelPath = import.meta.env.BASE_URL + "models/nlp/"` and `MODEL_ID = "minilm"`. transformers.js expects `onnx/model.onnx` plus tokenizer files inside that folder; rearrange the exported files to match if needed.

## Limits
- 48 training phrases are enough to demonstrate the method, not to certify a production classifier. Expect weaker results on phrasing unlike the examples.
- Regex extraction is transparent but brittle; it covers English only.
- Severity and actions are rule-based. Review them before connecting real actuators; a misclassified sentence must never switch mains-powered equipment without confirmation.
- The first transformer load needs internet access.
