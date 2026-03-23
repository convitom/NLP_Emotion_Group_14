import pandas as pd
import ast

df = pd.read_csv(r"end-to-end\data\data1_test.csv")

def parse_label(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df["label_list"] = df["label"].apply(parse_label)
df_single = df[df["label_list"].apply(len) == 1].copy()
df_single["emotion"] = df_single["label_list"].apply(lambda x: x[0])
cols = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

for col in cols:
    df_single[col] = (df_single["emotion"] == col).astype(int)
df_single = df_single.sample(frac=1, random_state=42).reset_index(drop=True)
df_single.to_csv(r"end-to-end\data\data3_test.csv", index=False)

print("Saved single-label dataset!")
print(df_single["emotion"].value_counts())
