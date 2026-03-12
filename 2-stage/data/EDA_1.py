import pandas as pd

# đọc file txt (tab separated)
df = pd.read_csv(
    r"D:\USTH\nlp\NLP_SemEval_36\data\data1_train.csv",
    sep=","
)

print(df.head())

# cột text
text_col = "text"

# các cột emotion (bỏ ID + Tweet)
emotion_cols = df.columns.drop(["label", "text"])

# =========================
# Count samples per emotion
# =========================

emotion_counts = df[emotion_cols].sum().sort_values(ascending=False)

print("Emotion counts:")
print(emotion_counts)

# =========================
# Number of emotions per sample
# =========================

df["num_emotions"] = df[emotion_cols].sum(axis=1)

print("\nNumber of emotions per sample:")
print(df["num_emotions"].value_counts().sort_index())

# =========================
# Correlation between emotions
# =========================

emotion_matrix = df[emotion_cols]

corr = emotion_matrix.corr()
co_matrix = emotion_matrix.T.dot(emotion_matrix)

pairs = []
emotions = corr.columns

for i in range(len(emotions)):
    for j in range(i+1, len(emotions)):
        e1 = emotions[i]
        e2 = emotions[j]

        correlation = corr.loc[e1, e2]
        co_occurrence = co_matrix.loc[e1, e2]

        pairs.append((e1, e2, correlation, co_occurrence))

pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)

print("\nTop 20 emotion pairs by correlation:")
for e1, e2, corr_val, count in pairs_sorted[:20]:
    print(f"{e1:15} {e2:15} corr={corr_val:.3f}  co-occur={count}")

# =========================
# Sentence length analysis
# =========================

df["char_len"] = df[text_col].str.len()
df["word_len"] = df[text_col].str.split().apply(len)

print("\nCharacter length statistics:")
print(df["char_len"].describe())

print("\nWord length statistics:")
print(df["word_len"].describe())

longest = df.sort_values("word_len", ascending=False).head(10)
print("\nLongest sentences:")
print(longest[["word_len", text_col]])

shortest = df.sort_values("word_len").head(10)
print("\nShortest sentences:")
print(shortest[["word_len", text_col]])

# =========================
# Emotion vs sentence length
# =========================

emotion_matrix["length"] = df["word_len"]
emotion_length = emotion_matrix.groupby("length").mean()

print("\nAverage emotion presence by sentence length:")
print(emotion_length.head(10))

# =========================
# Visualization
# =========================

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm")

plt.figure()
plt.hist(df["word_len"], bins=50)
plt.xlabel("Sentence length (words)")
plt.ylabel("Frequency")
plt.title("Sentence Length Distribution")

plt.show()