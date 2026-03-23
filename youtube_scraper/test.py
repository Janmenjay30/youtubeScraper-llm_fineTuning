from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = Path(__file__).resolve().parent / "mkbhd_model"

if not model_dir.is_dir():
    raise FileNotFoundError(
        f"Local model folder not found: {model_dir}. "
        "Train/save the model first or update this path."
    )

# Training outputs may only contain checkpoint-* folders. Pick the latest valid checkpoint.
model_path = model_dir
if not (model_path / "config.json").is_file():
    checkpoints = sorted(
        [p for p in model_dir.glob("checkpoint-*") if (p / "config.json").is_file()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No usable model found in {model_dir}. Expected config.json or checkpoint-* with config.json."
        )
    model_path = checkpoints[-1]

try:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    probe = tokenizer("probe", return_tensors="pt")
    if getattr(tokenizer, "vocab_size", 0) == 0 or probe["input_ids"].shape[-1] == 0:
        raise ValueError("Invalid tokenizer in checkpoint path")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

prompts = [
    "User: What do you think about this phone?\nMKBHD:",
    "User: Is this laptop worth buying in 2026?\nMKBHD:",
    "User: Give me three quick pros and cons of this camera.\nMKBHD:",
    "User: Should I upgrade from last year's model?\nMKBHD:",
]

for idx, prompt in enumerate(prompts, start=1):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    continuation_ids = output_ids[0][prompt_len:]
    generated = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()

    print(f"\n--- Sample {idx} ---")
    print("Prompt:")
    print(prompt)
    print("Model Output:")
    print(generated if generated else "[No continuation generated. Retrain longer or use a richer prompt.]")