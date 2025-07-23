import os
from dotenv import load_dotenv
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
from project_config import DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR


# * --- CONFIG ---
prompt = "What is the capital of France?"
n_tokens_to_generate = 5
saved_dir = Path(
    "~/dev/deepseek_local_runner/documents/model_comparison/generated_logits"
).expanduser()
saved_dir.mkdir(parents=True, exist_ok=True)

# --- Full precision model ---
load_dotenv()

MODEL_NAME_HF = os.getenv("MODEL_NAME_HF")
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

model_fp = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_HF, token=HF_TOKEN, local_files_only=True
)
tokenizer_fp = AutoTokenizer.from_pretrained(
    MODEL_NAME_HF, token=HF_TOKEN, local_files_only=True, trust_remote_code=True
)

input_ids = tokenizer_fp(prompt, return_tensors="pt")["input_ids"].to(model_fp.device)

logits_fp = []
with torch.no_grad():
    for _ in range(n_tokens_to_generate):
        out = model_fp(input_ids)
        logits = out.logits[:, -1, :]  # logits for next token
        logits_fp.append(logits.cpu())
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

torch.save(
    {
        "prompt": prompt,
        "generated_ids": input_ids.cpu(),
        "logits_fp": logits_fp,
    },
    saved_dir / "generated_logits_fp.pt",
)

logger.info

del model_fp
torch.cuda.empty_cache()

# * --- Quantized model ---
model_q = AutoAWQForCausalLM.from_quantized(
    DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR,
    device_map="auto",
    fuse_layers=False,
    trust_remote_code=True,
)
tokenizer_q = AutoTokenizer.from_pretrained(DEEPSEEK_R1_DISTILL_QUANT_MODEL_DIR)

device = next(model_q.parameters()).device
inputs = tokenizer_q(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

logits_q = []
with torch.no_grad():
    for _ in range(n_tokens_to_generate):
        out = model_q(**inputs)
        logits = out.logits[:, -1, :]  # logits for next token
        logits_q.append(logits.cpu())

        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)

torch.save(
    {
        "prompt": prompt,
        "generated_ids": input_ids.cpu(),
        "logits_q": logits_q,
    },
    saved_dir / "generated_logits_scrooge.pt",
)

del model_q
torch.cuda.empty_cache()
