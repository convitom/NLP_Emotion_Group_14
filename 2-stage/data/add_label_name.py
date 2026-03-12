import pandas as pd

df_train = pd.read_csv("D:\\USTH\\nlp\\final_prj\\data\\train.csv")
df_test = pd.read_csv("D:\\USTH\\nlp\\final_prj\\data\\test.csv")
df_val = pd.read_csv("D:\\USTH\\nlp\\final_prj\\data\\val.csv")

emotion_map = {
    0:"admiration",
    1:"amusement",
    2:"anger",
    3:"annoyance",
    4:"approval",
    5:"caring",
    6:"confusion",
    7:"curiosity",
    8:"desire",
    9:"disappointment",
    10:"disapproval",
    11:"disgust",
    12:"embarrassment",
    13:"excitement",
    14:"fear",
    15:"gratitude",
    16:"grief",
    17:"joy",
    18:"love",
    19:"nervousness",
    20:"optimism",
    21:"pride",
    22:"realization",
    23:"relief",
    24:"remorse",
    25:"sadness",
    26:"surprise",
    27:"neutral"
    }

def parse_labels(x):
    nums = x.strip("[]").split()
    nums = [int(i) for i in nums]
    return [emotion_map[i] for i in nums]

def add_label_name(df):
    df['label_name'] = df['labels'].apply(parse_labels)
    return df

df_train = add_label_name(df_train)
df_test = add_label_name(df_test)
df_val = add_label_name(df_val)

df_train.to_csv("D:\\USTH\\nlp\\final_prj\\data\\train.csv", index=False)
df_test.to_csv("D:\\USTH\\nlp\\final_prj\\data\\test.csv", index=False)
df_val.to_csv("D:\\USTH\\nlp\\final_prj\\data\\val.csv", index=False)