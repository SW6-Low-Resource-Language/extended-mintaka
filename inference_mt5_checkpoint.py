import json
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

lang = "da"
pre_trained_model = "google/mt5-base"  # Base model for tokenizer and config
checkpoint_path = "checkpoints_" + lang + "/checkpoint_49.pt"  # Path to Accelerate checkpoint

# Load the tokenizer from the base model
tokenizer = MT5Tokenizer.from_pretrained(pre_trained_model, legacy=False)



# Load the model configuration from the base model
model = MT5ForConditionalGeneration.from_pretrained(pre_trained_model)

# Load the Accelerate checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

# Check if the checkpoint contains a 'module' key (common with Accelerate)
if "module" in checkpoint:
    state_dict = checkpoint["module"]
elif "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint  # Assume the checkpoint is the state_dict itself

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Remove 'module.' prefix from keys if it exists
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("module.", "")  # Remove 'module.' prefix
    new_state_dict[new_key] = value

# Load the state dictionary into the model
model.load_state_dict(new_state_dict)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load the input data
with open('./data/mintaka_test_extended.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

local_question_answer = []
for entry in data:
    if lang == 'en':
        question = entry['question']
    else:
        question = entry['translations'][lang]

    qa_entity = {
            "id": entry["id"],
            "question": question, 
        }
    answer = entry['answer']
    if answer['answerType'] in ["numerical", "boolean", "date", "string"]:
        qa_entity['answer'] = answer['answer'][0]
    elif answer['answer'] == None:
        continue
    elif answer['answer'][0]['label'][lang] != None:
        qa_entity['answer'] = answer['answer'][0]['label'][lang]
    else:
        continue
    local_question_answer.append(qa_entity)


prompts = [entry['question'] for entry in local_question_answer]

print("Tokenizer vocab size:", len(tokenizer))
print("First 5 prompts:", prompts[:5])

results = []
for idx, prompt in enumerate(prompts):
    print(idx)
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        num_return_sequences=5,
        early_stopping=True
    )
    
    answers = []
    print(outputs)
    # for i in range(len(outputs)):
    #     answer = tokenizer.decode(outputs[i], skip_special_tokens=True)
    #     answers.append(f"{i}: {answer}")
    for output in outputs:
        answers.append(tokenizer.decode(output, skip_special_tokens=True))


    results.append({
        "id": local_question_answer[idx]["id"],
        "question": local_question_answer[idx]["question"],
        "true_answer": local_question_answer[idx]["answer"],
        "predicted_answers": answers
    })
    break

print(results)

with open('./results/inference_results_'+lang+'.json', 'w', encoding='utf-8') as result_file:
    json.dump(results, result_file, ensure_ascii=False, indent=4)