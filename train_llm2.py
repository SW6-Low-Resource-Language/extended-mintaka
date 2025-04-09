import os
import torch
import torch.optim as optim
# import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset

from dotenv import load_dotenv

# import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return the number of GPUs



print("Finished importing modules")

torch.cuda.empty_cache()

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
# load_dotenv()
# os.environ = os.getenv("hugging_face")


lang='da'


pre_trained_model = "openai-community/gpt2"
#pre_trained_model = "meta-llama/Llama-3.2-1B-Instruct"
# pre_trained_model = "meta-llama/Llama-3.3-70B-Instruct"

for i in range(torch.cuda.device_count()):
    torch.cuda.set_per_process_memory_fraction(0.95, device=i)
    allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 2)  # Convert to MB
    reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 2)  # Convert to MB
    print(f"GPU {i}: Allocated Memory: {allocated_memory:.2f} MB, Reserved Memory: {reserved_memory:.2f} MB")


#for i in range(torch.cuda.device_count()):
 #   torch.cuda.set_per_process_memory_fraction(0.95, device=i)
  #  gpu_mem = torch.cuda.get_device_properties(i).total_memory
   # print(f"GPU {i} memory: {gpu_mem}")


tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
model = AutoModelForCausalLM.from_pretrained(pre_trained_model)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})


optimizer = optim.Adam(model.parameters(), lr=3e-5)
# num_epochs = 100
num_epochs = 5

validation_file = "./data/mintaka_test_extended.json"
training_file = "./data/mintaka_dev_extended.json"



def load_data(filename):
    # training_file = "data/mintaka_train_extended.json"

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # question_answer_pairs = {}
    question_answer_pairs = []
    for entry in data:
        id = entry['id']
        if lang != 'en':
            question = entry['translations'][lang]
        else:
            question = entry['question']

        if entry['answer']['answer'] == None:
             continue

        if entry['answer']['answerType'] in ["numerical", "boolean", "date", "string"]:
            answer = entry['answer']['answer'][0]
        else:
            answer = entry['answer']['answer'][0]['label'][lang]

        if 'supportingEnt' in entry['answer']:
            supporting_entities = []
            for entity in entry['answer']['supportingEnt']:
                ent = entity['label'][lang]
                if ent != None:
                    supporting_entities.append(ent)

            supporting_entities_string = ", ".join(supporting_entities)
            answer = str(answer) + ", " + supporting_entities_string
        if answer == None:
             continue
        # question_answer_pairs[id] = {
        #     'question': question,
        #     'answer': answer
        # }
        pair_obj = {
            "id": str(id),
            "question": str(question),
            "answer": str(answer)
        }
        question_answer_pairs.append(pair_obj)

    print(f"{len(question_answer_pairs)} question-answer pairs loaded from {filename}")
    with open(filename.replace('.json', '_qa_pairs_' + lang + '.json'), 'w', encoding='utf-8') as f:
         json.dump({"data": question_answer_pairs}, f, ensure_ascii=False, indent=4)
        # f.write("\n".join(question_answer_pairs))


load_data(training_file)
load_data(validation_file)

training_data_pairs = training_file.replace('.json', '_qa_pairs_' + lang + '.json')
validation_data_pairs = validation_file.replace('.json', '_qa_pairs_' + lang + '.json')

dataset = load_dataset("json", data_files={"train": training_data_pairs, "validation": validation_data_pairs}, field="data")



def preprocess_data(data_file):
    # https://huggingface.co/transformers/v3.0.2/preprocessing.html
    # Look at 'Preprocessing pars of sentences'
    # encoded_input = tokenizer(["How old are you?", "what's your name?"], ["I'm 6 years old", "Magnus"])
    # print(encoded_input)

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    batch_questions = [
        #LIST OF ALL QUESTIONS IN ORDER
    ]

    batch_answers = [
        #LIST OF ALL ANSWERS IN SAME ORDER AS QUESTIONS
    ]

    for datapoint in data:
        batch_questions.append(datapoint['question'])
        batch_answers.append(datapoint['answer'])

    batch_questions = batch_questions[:100]
    batch_answers = batch_questions[:100]

    encoded_inputs = tokenizer(batch_questions, batch_answers, padding=True, truncation=True, return_tensors="pt")

    return encoded_inputs

encoded_training_inputs = preprocess_data(training_data_pairs)
encoded_test_inputs = preprocess_data(validation_data_pairs)

training_dataset = TensorDataset(encoded_training_inputs["input_ids"], encoded_training_inputs["attention_mask"])
validation_dataset = TensorDataset(encoded_test_inputs["input_ids"], encoded_test_inputs["attention_mask"])


training_loader = DataLoader(training_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

num_training_steps = num_epochs * len(training_loader)

# train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
#     train_dataloader, eval_dataloader, model, optimizer
# )

if os.listdir(checkpoint_dir):
    latest_checkpoint = max([int(file.split('.')[0]) for file in os.listdir(checkpoint_dir)])
    checkpoint = torch.load(os.path.join(checkpoint_dir, f'{latest_checkpoint}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = latest_checkpoint + 1
else:
    start_epoch = 0

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

accelerator = Accelerator(
    mixed_precision="fp16",
    cpu=False,
    gradient_accumulation_steps=1
)

model, optimizer, training_loader, test_loader = accelerator.prepare(
    model, optimizer, training_loader, test_loader
)


print(f"Beginning training from epoch {start_epoch}")
for epoch in range(start_epoch, num_epochs):
        # Check dedicated VRAM usage (actual GPU memory)
    allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
    print(f"Dedicated VRAM: Allocated = {allocated:.2f} GB, Reserved = {reserved:.2f} GB")
    print(f"start of training loop {epoch}")
    model.train()
    for batch_idx, (input_ids, attention_mask) in enumerate(training_loader):
        
        print("for batch")
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        accelerator.backward(loss)

        print("take a step")
        optimizer.step()
        optimizer.zero_grad()

    print(f'Epoch {epoch}: Loss {loss.item()}')

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for batch_idx, (input_ids, attention_mask) in enumerate(test_loader):
            output = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            val_loss += loss.item()

    val_loss /= len(test_loader)  # Average validation loss
    print(f'Epoch {epoch}: Validation Loss {val_loss}')

    # Save checkpoint every epoch
    accelerator.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, os.path.join(checkpoint_dir, f'{epoch}.pt'))



# model.train()
# for epoch in range(num_epochs):
#     #do stuff here with our loss function and backwards propagation
#     for batch in train_dataloader:
#         outputs=model(**batch)
#         loss = outputs.loss
#         accelerator.backward(loss)

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
