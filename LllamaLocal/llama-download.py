import transformers
import torch
import os

os.environ["HF_TOKEN"] = "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"

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",  # This will automatically use CUDA if available
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Ensure inputs are on the correct device
messages = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in m.items()} for m in messages]

with torch.cuda.amp.autocast(enabled=True):
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

print(outputs[0]["generated_text"][-1])