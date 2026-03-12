import pandas as pd
import ast

# ======================================================
# Ekman emotions
# ======================================================
OUTPUT = r"NLP_SemEval_36\data\data1_test.csv"
semeval = pd.read_csv(
    r"NLP_SemEval_36\data\2018-E-c-En-test-gold.txt",
    sep="\t"
)
go = pd.read_csv(r"NLP_SemEval_36\data\test.csv")

EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# ======================================================
# SemEval -> Ekman mapping
# (anticipation, trust removed)
# ======================================================
semeval_map = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "surprise",
    "love": "joy",
    "optimism": "joy",
    "pessimism": "sadness"
}

# ======================================================
# GoEmotions -> Ekman mapping
# ======================================================
goemotion_map = {

    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "joy",
    "confusion": "surprise",
    "curiosity": "surprise",
    "desire": "joy",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise"
}

# ======================================================
# Load SemEval
# ======================================================
print("Loading SemEval...")



rows = []

for _, row in semeval.iterrows():

    text = row["Tweet"]

    emotions = []

    for col in semeval_map:
        if row[col] == 1:
            emotions.append(semeval_map[col])

    if not emotions:
        label = ["neutral"]
    else:
        label = list(set(emotions))

    onehot = {e: 0 for e in EKMAN}

    for e in label:
        if e in onehot:
            onehot[e] = 1

    rows.append({
        "text": text,
        "label": label,
        **onehot
    })

semeval_df = pd.DataFrame(rows)

print("SemEval samples:", len(semeval_df))

# ======================================================
# Load GoEmotions
# ======================================================
print("Loading GoEmotions...")


rows = []

for _, row in go.iterrows():

    text = row["text"]

    label_names = ast.literal_eval(row["label_name"])

    if "neutral" in label_names:
        label = ["neutral"]

    else:

        mapped = []

        for l in label_names:
            if l in goemotion_map:
                mapped.append(goemotion_map[l])

        label = list(set(mapped)) if mapped else ["neutral"]

    onehot = {e: 0 for e in EKMAN}

    for e in label:
        if e in onehot:
            onehot[e] = 1

    rows.append({
        "text": text,
        "label": label,
        **onehot
    })

go_df = pd.DataFrame(rows)

print("GoEmotions samples:", len(go_df))

# ======================================================
# Merge datasets
# ======================================================
print("Merging datasets...")

merged = pd.concat([semeval_df, go_df], ignore_index=True)

print("Total samples:", len(merged))

# ======================================================
# Save dataset
# ======================================================
merged.to_csv(
    OUTPUT,
    index=False
)

print(f"Saved to {OUTPUT}")
print(merged.head())