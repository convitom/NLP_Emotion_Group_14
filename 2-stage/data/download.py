from datasets import load_dataset


dataset = load_dataset("alex-shvets/EmoPillars", trust_remote_code=True)

print(dataset['train'].head())

