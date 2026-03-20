"""
diagnose.py — Phân tích class overlap & chẩn đoán tại sao F1 thấp.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# =============================================================================
#  Constants
# =============================================================================

EMOTION_COLS  = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
CLASS_NAMES   = EMOTION_COLS + ["neutral"]
NUM_CLASSES   = 7

CLASS_COLORS = {
    "anger":    "#e74c3c",
    "disgust":  "#8e44ad",
    "fear":     "#2c3e50",
    "joy":      "#f1c40f",
    "sadness":  "#3498db",
    "surprise": "#e67e22",
    "neutral":  "#95a5a6",
}

# =============================================================================
#  Data loading
# =============================================================================

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    texts    = df["text"].astype(str).tolist()
    labels_6 = df[EMOTION_COLS].astype(np.float32).values
    neutral  = (labels_6.sum(axis=1) == 0).astype(np.float32).reshape(-1, 1)
    labels_7 = np.concatenate([labels_6, neutral], axis=1)
    return texts, labels_7


def get_single_label_mask(labels_7):
    return labels_7.sum(axis=1) == 1


def get_class_label(labels_7):
    n = labels_7.sum(axis=1)
    out = np.full(len(labels_7), -1, dtype=int)
    mask = (n == 1)
    out[mask] = np.argmax(labels_7[mask], axis=1)
    return out


# =============================================================================
#  Feature extraction
# =============================================================================

def extract_tfidf(texts, max_features=5000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3)
    return vec.fit_transform(texts).toarray()


def extract_model_embeddings(texts, model_path, batch_size=64, device_str="auto"):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("Thiếu torch/transformers")
        return None

    if not os.path.exists(model_path):
        print(f"Không tìm thấy model: {model_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)
    pretrained_name = ckpt.get("pretrained_name", "bert-base-uncased")

    model = AutoModel.from_pretrained(pretrained_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    state = {k.replace("backbone.", ""): v
             for k, v in ckpt["model_state"].items()
             if k.startswith("backbone.")}

    model.load_state_dict(state)
    model.eval()

    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")

        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )

        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls)

    return np.vstack(embeddings)


# =============================================================================
#  Plot utils
# =============================================================================

def reduce_to_2d(X):
    if X.shape[1] > 50:
        X = PCA(n_components=50).fit_transform(X)
    return TSNE(n_components=2, perplexity=40).fit_transform(X)


def plot_tsne(emb, class_ids, out_path):
    mask = class_ids >= 0
    emb = emb[mask]
    cids = class_ids[mask]

    plt.figure(figsize=(8,6))
    for cid, name in enumerate(CLASS_NAMES):
        idx = cids == cid
        plt.scatter(emb[idx,0], emb[idx,1], s=5, label=name)

    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_class_overlap(X, class_ids, out_path):
    mask = class_ids >= 0
    X = X[mask]
    cids = class_ids[mask]

    centroids = []
    for i in range(NUM_CLASSES):
        centroids.append(X[cids == i].mean(axis=0))
    centroids = np.vstack(centroids)

    sim = cosine_similarity(centroids)

    plt.imshow(sim)
    plt.colorbar()
    plt.savefig(out_path)
    plt.close()

    return sim


# =============================================================================
#  Main
# =============================================================================

def diagnose(
    data_path,
    model_path=None,
    out_dir=None,
    max_samples=8000,
    use_model=True,
):
    if out_dir is None:
        out_dir = "diagnose"
    os.makedirs(out_dir, exist_ok=True)

    texts, labels_7 = load_data(data_path)
    class_ids = get_class_label(labels_7)

    if len(texts) > max_samples:
        idx = np.random.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in idx]
        class_ids = class_ids[idx]

    print("Extract TF-IDF...")
    X_tfidf = extract_tfidf(texts)

    print("t-SNE TF-IDF...")
    emb = reduce_to_2d(X_tfidf)
    plot_tsne(emb, class_ids, os.path.join(out_dir, "tsne_tfidf.png"))

    print("Overlap...")
    plot_class_overlap(X_tfidf, class_ids,
                       os.path.join(out_dir, "class_overlap.png"))

    if use_model and model_path is not None:
        print("Extract model embeddings...")
        X_model = extract_model_embeddings(texts, model_path)

        if X_model is not None:
            emb = reduce_to_2d(X_model)
            plot_tsne(emb, class_ids,
                      os.path.join(out_dir, "tsne_model.png"))

    print(f"Done. Output: {out_dir}")


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = r"D:\USTH\nlp\NLP_Emotion_Group_14\end-to-end\data\data1_train.csv"
MODEL_PATH = r"D:\USTH\nlp\electra\best.pth"
OUT_DIR = None
MAX_SAMPLES = 8000

if __name__ == "__main__":
    diagnose(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        out_dir=OUT_DIR,
        max_samples=MAX_SAMPLES,
        use_model=(MODEL_PATH is not None),
    )