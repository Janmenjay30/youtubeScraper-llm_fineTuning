from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = open("dataset/cleaned_dataset.txt").read()

tokens = tokenizer(text)

print(tokens["input_ids"][:50])