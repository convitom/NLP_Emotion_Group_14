import pandas as pd
import ast

# Load data
df = pd.read_csv(r"end-to-end\data\data1_train.csv")

def parse_label(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df["label_list"] = df["label"].apply(parse_label)
df_single = df[df["label_list"].apply(len) == 1].copy()
df_single["emotion"] = df_single["label_list"].apply(lambda x: x[0])
class_counts = df_single["emotion"].value_counts()
print("Class distribution:\n", class_counts)


min_count = class_counts.min()
print("Min samples per class:", min_count)

df_balanced = (
    df_single.groupby("emotion")
    .apply(lambda x: x.sample(n=min_count, random_state=42))
    .reset_index(drop=True)
)

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
cols = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

for col in cols:
    df_balanced[col] = (df_balanced["emotion"] == col).astype(int)
df_balanced.to_csv(r"end-to-end\data\data2_train.csv", index=False)

print("Saved balanced dataset!")
print(df_balanced["emotion"].value_counts())
