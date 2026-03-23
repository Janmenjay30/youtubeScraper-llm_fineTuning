from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from pathlib import Path
import torch
import os

max_seq_len = 128
base_dir = Path(__file__).resolve().parent
train_file = base_dir / "dataset" / "cleaned_dataset.txt"
output_dir = base_dir / "mkbhd_model"

use_cuda = torch.cuda.is_available()
gpu_vram_gb = (
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if use_cuda else 0
)
default_model = "distilgpt2" if use_cuda and gpu_vram_gb < 6 else "Qwen/Qwen3-0.6B"
model_name = os.getenv("MODEL_NAME", default_model)
print(f"Using device: {'cuda' if use_cuda else 'cpu'}")
print(f"Model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
)
model.config.pad_token_id = tokenizer.pad_token_id
if use_cuda:
    model.gradient_checkpointing_enable()

dataset=load_dataset(
    "text",
    data_files={"train": str(train_file)}

)

def tokenize(examples):
    tokens = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset=dataset.map(tokenize, batched=True)

training_args=TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_strategy="epoch",
    logging_steps=10,
    fp16=use_cuda,
    use_cpu=not use_cuda,
    dataloader_pin_memory=use_cuda,
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()
trainer.save_model(str(output_dir))
tokenizer.save_pretrained(str(output_dir))