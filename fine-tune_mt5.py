import os
import json
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist


from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, MT5Tokenizer, MT5ForConditionalGeneration
from deepspeed import initialize
from deepspeed.ops.adam import DeepSpeedCPUAdam

print("Finished importing modules")

if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

rank = dist.get_rank()

os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

lang='da'

isMT5 = True
pre_trained_model = "google/mt5-xl"

# training_file = "./data/mintaka_dev_extended.json"
training_file = "./data/mintaka_train_extended.json"
validation_file = "./data/mintaka_test_extended.json"

train_micro_batch_size_per_gpu = 8
gradient_accumulation_steps = 8

num_epochs = 100
learning_rate = 3e-5
max_length = 64 


torch.cuda.empty_cache()
for i in range(torch.cuda.device_count()):
    torch.cuda.set_per_process_memory_fraction(0.95, device=i)

if isMT5:
    tokenizer = MT5Tokenizer.from_pretrained(pre_trained_model)
    model = MT5ForConditionalGeneration.from_pretrained(pre_trained_model)
else:
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
    model = AutoModelForCausalLM.from_pretrained(pre_trained_model)
model.gradient_checkpointing_enable()

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

optimizer = DeepSpeedCPUAdam(model.parameters(), lr=learning_rate)


def load_data(filename):
    with open('./configurations/comparative_dict.json', 'r', encoding='utf-8') as f:    
        comparative_dict = json.load(f)
        comparative_dict_lang = comparative_dict[lang]

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_answer_pairs = []

    for entry in data:
        id = entry['id']
        if lang != 'en':
            question = entry['translations'][lang]
        else:
            question = entry['question']

        if entry['answer']['answer'] == None:
             continue

        if entry['answer']['answerType'] == 'string':
            answer = entry['answer']['answer'][0]
            if answer.lower() in comparative_dict_lang:
                answer = comparative_dict_lang[answer.lower()]
        elif entry['answer']['answerType'] == 'boolean':
            answer = entry['answer']['answer'][0]
            if answer is True:
                answer = comparative_dict_lang['true_list'][0]
            elif answer is False:
                answer = comparative_dict_lang['false_list'][0] 
        elif entry['answer']['answerType'] in ["numerical", "boolean", "date", "string"]:
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
       
        pair_obj = {
            "id": str(id),
            "question": str(question),
            "answer": str(answer)
        }
        question_answer_pairs.append(pair_obj)

    print(f"{len(question_answer_pairs)} question-answer pairs loaded from {filename}")
    with open(filename.replace('.json', '_qa_pairs_' + lang + '.json'), 'w', encoding='utf-8') as f:
         json.dump({"data": question_answer_pairs}, f, ensure_ascii=False, indent=4)


#load_data(training_file)
#load_data(validation_file)

training_data_pairs = training_file.replace('.json', '_qa_pairs_' + lang + '.json')
validation_data_pairs = validation_file.replace('.json', '_qa_pairs_' + lang + '.json')


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


    encoded_inputs = tokenizer(batch_questions, 
                                batch_answers, 
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                                max_length=max_length
                            )

    return encoded_inputs

encoded_training_inputs = preprocess_data(training_data_pairs)
encoded_test_inputs = preprocess_data(validation_data_pairs)

training_dataset = TensorDataset(encoded_training_inputs["input_ids"], encoded_training_inputs["attention_mask"])
validation_dataset = TensorDataset(encoded_test_inputs["input_ids"], encoded_test_inputs["attention_mask"])

training_loader = DataLoader(training_dataset, batch_size=train_micro_batch_size_per_gpu, shuffle=True, num_workers=6)
test_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=6)


num_training_steps = num_epochs * len(training_loader)

model_engine, optimizer, _, lr_scheduler = initialize(
    model=model,
    optimizer=optimizer,
    config_params={
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            }
        },
        "offload_param" : {
            "device": "cpu",
            "pin_memory": True
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 16
        },
         "scheduler": {
            "type": "WarmupLR",  # Use DeepSpeed's WarmupLR scheduler
            "params": {
                "warmup_min_lr": 0,  # Minimum learning rate during warmup
                "warmup_max_lr": learning_rate,  # Target learning rate after warmup
                "warmup_num_steps": int(0.05 * num_training_steps)  # 10% warmup steps
            }
        }
    }
)

checkpoint_dir = 'checkpoints_' + lang
os.makedirs(checkpoint_dir, exist_ok=True)
# if os.listdir(checkpoint_dir):
#     latest_checkpoint = max([int(file.split('.')[0]) for file in os.listdir(checkpoint_dir)])
#     model_engine.load_checkpoint(checkpoint_dir, tag=f'{latest_checkpoint}')
#     start_epoch = latest_checkpoint + 1
# else:
#     start_epoch = 0
start_epoch = 0

start = time.time()

print(f"[Rank {rank}] Beginning training from epoch {start_epoch}")
for epoch in range(start_epoch, num_epochs):
    print(f"[Rank {rank}] Start of training loop {epoch}")
    if rank == 0:
        print(f"Time elapsed: {time.time() - start} seconds")
    model_engine.train()
    print(f"[Rank {rank}] Model_engine set to training mode")
    for batch_idx, (input_ids, attention_mask) in enumerate(training_loader):
        print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}")

        input_ids = input_ids.to(model_engine.device)
        attention_mask = attention_mask.to(model_engine.device)

        outputs = model_engine(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        model_engine.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model_engine.step()
        optimizer.zero_grad()

    if rank == 0:
        print(f'Epoch {epoch}: Loss {loss.item()}')


    model_engine.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in enumerate(test_loader):
            input_ids = input_ids.to(model_engine.device)
            attention_mask = attention_mask.to(model_engine.device)

            output = model_engine(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = output.loss
            val_loss += loss.item()
        val_loss /=len(test_loader)

    if rank == 0:
        print(f'Epoch {epoch}: Validation Loss {val_loss}')

if rank == 0:
    print(f"Training completed in {time.time() - start} seconds")
    model_engine.save_checkpoint(checkpoint_dir, tag=f'{pre_trained_model}_{lang}')

if dist.is_initialized():
    dist.destroy_process_group()