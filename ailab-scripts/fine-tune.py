import os
import json
import time
import torch.distributed as dist

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, MT5Tokenizer, MT5ForConditionalGeneration

from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="no")

print("Finished importing modules")

if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

rank = int(os.getenv("RANK", 0))	
world_size = int(os.getenv("WORLD_SIZE", 1))
 

os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

lang='da'
last_epoch = 81

isMT5 = True
pre_trained_model = "google/mt5-base"

#training_file = "./data/mintaka_dev_extended.json"
training_file = "./data/mintaka_train_extended.json"
validation_file = "./data/mintaka_test_extended.json"

train_micro_batch_size_per_gpu = 4
gradient_accumulation_steps = 4
val_batch_size = 16

num_epochs = 100
learning_rate = 1e-7
max_length = 128


torch.cuda.empty_cache()
for i in range(torch.cuda.device_count()):
    torch.cuda.set_per_process_memory_fraction(0.95, device=i)


if isMT5:
    tokenizer = MT5Tokenizer.from_pretrained(pre_trained_model, legacy=False)
    model = MT5ForConditionalGeneration.from_pretrained(pre_trained_model)
else:
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
    model = AutoModelForCausalLM.from_pretrained(pre_trained_model)

model.gradient_checkpointing_enable()

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

optimizer = AdamW(model.parameters(), lr=learning_rate)

def preprocess_data(filename):
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
            if answer.lower() in comparative_dict_lang and lang != 'en':
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


def pair_and_tokenize_data(data_file):
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
                                max_length=128
                            )

    return encoded_inputs

if rank == 0:
    preprocess_data(training_file)
    preprocess_data(validation_file)

accelerator.wait_for_everyone()

training_data_pairs = training_file.replace('.json', '_qa_pairs_' + lang + '.json')
validation_data_pairs = validation_file.replace('.json', '_qa_pairs_' + lang + '.json')

encoded_training_inputs = pair_and_tokenize_data(training_data_pairs)
encoded_test_inputs = pair_and_tokenize_data(validation_data_pairs)

training_dataset = TensorDataset(
    encoded_training_inputs["input_ids"],
    encoded_training_inputs["attention_mask"]
)

validation_dataset = TensorDataset(
    encoded_test_inputs["input_ids"],
    encoded_test_inputs["attention_mask"]
)


training_sampler = DistributedSampler(
    training_dataset,
    num_replicas=dist.get_world_size(),
    rank=rank,
    shuffle=True
)

validation_sampler = DistributedSampler(
    validation_dataset,
    num_replicas=dist.get_world_size(),
    rank=rank,
    shuffle=False
)


training_loader = DataLoader(
    training_dataset,
    batch_size=train_micro_batch_size_per_gpu,
    sampler=training_sampler,
    num_workers=6,
    pin_memory=True,
    drop_last=True
)

test_loader = DataLoader(
    validation_dataset,
    batch_size=val_batch_size,
    num_workers=6,
    sampler=validation_sampler, 
    pin_memory=True,
    drop_last=False
)


if rank == 0:
    print(f"Training dataset size: {len(training_dataset)}")
    print(f"DataLoader batch size: {training_loader.batch_size}")
    print(f"Effective number of batches per epoch: {len(training_loader) // dist.get_world_size()}")
    print(f"DataLoader batch size: {training_loader.batch_size}")

print(f"[Rank {rank}] Total samples after padding: {len(training_sampler)}")

model, optimizer, training_loader, test_loader = accelerator.prepare(
    model, optimizer, training_loader, test_loader
)

num_training_steps = num_epochs * len(training_loader)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)

checkpoint_dir = 'checkpoints_' + lang
checkpoint_path = checkpoint_dir + '/checkpoint_'+str((last_epoch-1))+'.pt'

os.makedirs(checkpoint_dir, exist_ok=True)

start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)

    # Load the model state dictionary
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load the optimizer state dictionary
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Set the start epoch to resume training
    start_epoch = last_epoch
    print(f"Resuming training from epoch {start_epoch}")
else:
    print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")



start = time.time()

print(f"[Rank {rank}] Beginning training from epoch {start_epoch}")
for epoch in range(start_epoch, num_epochs):
    print(f"[Rank {rank}] Start of training loop {epoch}")

    training_sampler.set_epoch(epoch)

    if rank == 0:
        print(f"Time elapsed: {time.time() - start:.2f} seconds")

    model.train()
    print(f"[Rank {rank}] model set to training mode")
    for batch_idx, (input_ids, attention_mask) in enumerate(training_loader):
        with accelerator.accumulate(model):
            if batch_idx % 10 == 0:
                print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}")

            input_ids = input_ids.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()

    if rank == 0:
        print(f'Epoch {epoch}: Loss {loss.item()}')
        print(f"Time elapsed: {time.time() - start:.2f} seconds")

    model.eval()
    print(f"[Rank {rank}] model set to evaluation mode")

    val_loss = 0
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"[Rank {rank}] Epoch {epoch} - validation Batch {batch_idx}")

            input_ids = input_ids.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)

            output = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = output.loss.detach()
            val_loss += loss.item()
        val_loss /=len(test_loader)

    if rank == 0:
        print(f'Epoch {epoch}: Validation Loss {val_loss}')

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        accelerator.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"{checkpoint_dir}/checkpoint_{epoch}.pt")

    if rank == 0:
        print(f"Training completed in {time.time() - start:.2f} seconds")
        print(f"Model saved to {checkpoint_dir}")

    accelerator.wait_for_everyone()

if dist.is_initialized():
    dist.destroy_process_group()

accelerator.free_memory()
