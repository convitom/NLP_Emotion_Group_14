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
hieu = "train"
DATA_PATH = r"D:\USTH\nlp\NLP_Emotion_Group_14\end-to-end\data\data1_{hieu}.csv".format(hieu=hieu)
MODEL_PATH = r"end-to-end\results\bert\checkpoints\bert_e2e_emotion.pth"
OUT_DIR = r"end-to-end\diagnose"
MAX_SAMPLES = 80000

plot_overlap = 1 
plot_tsne_tfidf = 1

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

def get_model_suffix(model_path: Optional[str]):
    if model_path is None:
        return ""

    name = os.path.basename(model_path)          # electra_e2e_emotion.pth
    name = os.path.splitext(name)[0]             # electra_e2e_emotion
    name = name.replace("_emotion", "")          # electra_e2e
    return name

def get_dataset_suffix(data_path: Optional[str]):
    if data_path is None:
        return ""

    name = os.path.basename(data_path)          # data1_train.csv
    name = os.path.splitext(name)[0]             # data1_train
    return name

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
        cls_vecs = X[cids == i]
        if len(cls_vecs) == 0:
            centroids.append(np.zeros(X.shape[1]))
        else:
            centroids.append(cls_vecs.mean(axis=0))

    centroids = np.vstack(centroids)

    sim = cosine_similarity(centroids)

    plt.figure(figsize=(8, 6))

    # 🔥 dùng colormap đẹp hơn (đậm khi cao)
    im = plt.imshow(sim, cmap="YlOrRd", vmin=0, vmax=1)

    plt.colorbar(im)

    # 🔥 set label = tên class
    plt.xticks(ticks=np.arange(NUM_CLASSES), labels=CLASS_NAMES, rotation=45)
    plt.yticks(ticks=np.arange(NUM_CLASSES), labels=CLASS_NAMES)

    # 🔥 hiển thị số trong từng ô
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            value = sim[i, j]
            plt.text(
                j, i,
                f"{value:.2f}",
                ha="center", va="center",
                color="black" if value < 0.7 else "white",
                fontsize=8
            )

    plt.title("Class Overlap (Cosine Similarity)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
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
    plot_overlap=True,
    plot_tsne_tfidf=True,
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

    suffix_model = get_model_suffix(model_path)
    suffix_model = f"_{suffix_model}" if suffix_model else ""
    suffix_dataset = get_dataset_suffix(data_path)
    suffix_dataset = f"_{suffix_dataset}" if suffix_dataset else ""
    
    if plot_tsne_tfidf:
        print("Extract TF-IDF...")
        X_tfidf = extract_tfidf(texts)

        print("t-SNE TF-IDF...")
        emb = reduce_to_2d(X_tfidf)
        
        
        
        plot_tsne(emb, class_ids, os.path.join(out_dir, f"tsne_tfidf{suffix_dataset}.png"))

    if plot_overlap:
        print("Overlap...")
        plot_class_overlap(X_tfidf, class_ids,
                        os.path.join(out_dir, f"class_overlap{suffix_dataset}.png"))

    if use_model and model_path is not None:
        print("Extract model embeddings...")
        X_model = extract_model_embeddings(texts, model_path)

        if X_model is not None:
            emb = reduce_to_2d(X_model)
            plot_tsne(emb, class_ids,
                      os.path.join(out_dir, f"tsne_model{suffix_model}{suffix_dataset}.png"))

    print(f"Done. Output: {out_dir}")


# =============================================================================
# CONFIG
# =============================================================================



if __name__ == "__main__":
    diagnose(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        out_dir=OUT_DIR,
        max_samples=MAX_SAMPLES,
        use_model=(MODEL_PATH is not None),
    )