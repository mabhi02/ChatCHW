import transformers
import torch
import os
from typing import List, Dict, Union

# Set environment variables
os.environ["HF_TOKEN"] = "YOUR_TOKEN_HERE"  # Replace with your token
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

# Model configuration
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_gpu_pipeline():
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")
    
    # Initialize pipeline with GPU optimizations
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        },
        device_map="auto",
    )
    
    return pipeline

def generate_response(
    pipeline: transformers.Pipeline,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9
) -> str:
    # Set up terminators
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Enable automatic mixed precision for faster inference
    with torch.amp.autocast(device):
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = pipeline(
                messages,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pipeline.tokenizer.pad_token_id,
                num_return_sequences=1,
            )
    
    return outputs[0]["generated_text"]

def main():
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize pipeline
    pipeline = setup_gpu_pipeline()
    
    # Example messages
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]
    
    # Generate response
    try:
        response = generate_response(pipeline, messages)
        print("Generated response:", response)
    except Exception as e:
        print(f"Error during generation: {str(e)}")
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()