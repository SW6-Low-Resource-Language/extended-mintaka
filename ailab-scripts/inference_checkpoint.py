import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

# 1. Load model and tokenizer
pre_trained_model = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
model = AutoModelForCausalLM.from_pretrained(pre_trained_model)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


# 2. Load checkpoint (your original code)
checkpoint_dir = "checkpoints"
if os.listdir(checkpoint_dir):
    latest_checkpoint = max([int(file.split('.')[0]) for file in os.listdir(checkpoint_dir) if file.endswith('.pt')])
    checkpoint = torch.load(os.path.join(checkpoint_dir, f'{latest_checkpoint}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

# 3. Set model to evaluation mode
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. Define inference function
def generate_text(prompt, max_length=50, temperature=0.7, top_k=50):
    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True  # For randomness; set False for greedy decoding
        )
    
    # Decode and return
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 5. Run inference
prompt = "we are great"
generated_text = generate_text(prompt)
print("Generated Text:")
print(generated_text)