import json
x = 1
file_path = r"D:\USTH\nlp\NLP_SemEval_36\data\train.jsonl"
K = 2922

if x == 1:
    with open(file_path, "r", encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == K:
                sample = json.loads(line)
                print(json.dumps(sample, indent=2, ensure_ascii=False))
                break
else:    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == K:
                print("LINE INDEX:", i)
                print("RAW LINE:")
                print(line[:50000])  # chỉ in raw text
                sample = json.loads(line)
                print("\nKEYS:", sample.keys())
                break