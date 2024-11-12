import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import Dataset
from tqdm import tqdm

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load Model and Tokenizer with 4-bit Quantization
model_name = "meta-llama/Llama-3.1-8B"
print("Loading model and tokenizer with 4-bit quantization...")

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto'
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Load and Prepare Text Data
def load_text_data(file_path):
    print(f"Loading text data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

txt_file_path = """LllamaLocal\output_text.txt"""
text_data = load_text_data(txt_file_path)

# Step 3: Prepare Dataset
print("Tokenizing text data...")
data = {"text": [text_data]}
dataset = Dataset.from_dict(data)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 4: Fine-Tuning with QLoRA
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    dataloader_num_workers=4,
)

print("Starting fine-tuning with QLoRA...")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Step 5: Save Model
print("Saving fine-tuned model...")
model.save_pretrained("fine_tuned_llama")
tokenizer.save_pretrained("fine_tuned_llama")

# Step 6: Downstream Task - Text Generation
def generate_text(prompt, max_length=100):
    model = AutoModelForCausalLM.from_pretrained("fine_tuned_llama").to(device)
    tokenizer = AutoTokenizer.from_pretrained("fine_tuned_llama")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage of the generate_text function
input_text = "Your input text here."
print("Generated text:", generate_text(input_text))
