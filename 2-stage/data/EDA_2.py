import pandas as pd
import re
from collections import Counter
from wordfreq import zipf_frequency

# đọc file SemEval txt
df = pd.read_csv(
    r"D:\USTH\nlp\NLP_SemEval_36\data\data1_train.csv",
    sep=","
)

text_col = "text"

# =========================
# Tokenize
# =========================

def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())

df["tokens"] = df[text_col].apply(tokenize)

# =========================
# Word frequency
# =========================

all_tokens = []

for tokens in df["tokens"]:
    all_tokens.extend(tokens)

word_freq = Counter(all_tokens)

print(f"Number of unique words: {len(word_freq)}")
print(f"Most common words: {word_freq.most_common(10)}")

# =========================
# Elongated words
# =========================

print("\n")
print("==================== ELONGATED WORDS ===================")
print("\n")

elongated_words = [
    w for w in word_freq
    if re.search(r"(.)\1{2,}", w)
]

print(elongated_words[:50])

# =========================
# Unusual words
# =========================

print("\n")
print("==================== UNUSUAL WORDS ===================")
print("\n")

unusual_words = [
    w for w in word_freq
    if zipf_frequency(w, "en") < 2
]

print(unusual_words[:50])