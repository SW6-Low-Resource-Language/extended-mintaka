import os
import time
import torch.distributed as dist

from preprocess_data import pair_and_tokenize_data, preprocess_data

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, MT5Tokenizer, MT5ForConditionalGeneration
# from deepspeed import initialize

from accelerate import Accelerator

accelerator = Accelerator()

print("Finished importing modules")

if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

#rank = dist.get_rank()
rank = int(os.getenv("RANK", 0))	
world_size = int(os.getenv("WORLD_SIZE", 1))
# rank = int(os.getenv("LOCAL_RANK", 0))
 

os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

lang='da'

isMT5 = True
pre_trained_model = "google/mt5-xl"

#training_file = "./data/mintaka_dev_extended.json"
training_file = "./data/mintaka_train_extended.json"
validation_file = "./data/mintaka_test_extended.json"

train_micro_batch_size_per_gpu = 4
gradient_accumulation_steps = 8
val_batch_size = 16

num_epochs = 2
learning_rate = 1e-7
max_length = 128


# torch.cuda.empty_cache()
# for i in range(torch.cuda.device_count()):
#     torch.cuda.set_per_process_memory_fraction(0.95, device=i)


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

if rank == 0:
    preprocess_data(training_file, lang)
    preprocess_data(validation_file, lang)

accelerator.wait_for_everyone()

training_data_pairs = training_file.replace('.json', '_qa_pairs_' + lang + '.json')
validation_data_pairs = validation_file.replace('.json', '_qa_pairs_' + lang + '.json')

encoded_training_inputs = pair_and_tokenize_data(training_data_pairs, tokenizer)
encoded_test_inputs = pair_and_tokenize_data(validation_data_pairs, tokenizer)

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


model, optimizer, training_loader, test_loader = accelerator.prepare(
    model, optimizer, training_loader, test_loader
)



if rank == 0:
    print(f"Training dataset size: {len(training_dataset)}")
    print(f"DataLoader batch size: {training_loader.batch_size}")
    print(f"Effective number of batches per epoch: {len(training_loader) // dist.get_world_size()}")
    print(f"DataLoader batch size: {training_loader.batch_size}")

print(f"[Rank {rank}] Total samples after padding: {len(training_sampler)}")

num_training_steps = num_epochs * len(training_loader)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)

checkpoint_dir = 'checkpoints_' + lang
os.makedirs(checkpoint_dir, exist_ok=True)
# if os.path.exists(f"{checkpoint_dir}/checkpoint_{epoch}.pt"):
#     checkpoint = torch.load(f"{checkpoint_dir}/checkpoint_{epoch}.pt", map_location=accelerator.device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# else:
#     start_epoch = 0
start_epoch = 0

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
            print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}")

            input_ids = input_ids.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            model.backward(loss)

            # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
            # print(f"[Rank {rank}] Gradient Norm: {total_norm}")
        
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            print(f"[Rank {rank}] Epoch {epoch} - validation Batch {batch_idx}")

            input_ids = input_ids.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)

            output = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = output.loss.detach()
            val_loss += loss.item()
        val_loss /=len(test_loader)

    if rank == 0:
        print(f'Epoch {epoch}: Validation Loss {val_loss}')


    accelerator.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, f"{checkpoint_dir}/checkpoint_{epoch}.pt")

    # model.save_checkpoint(
    #     checkpoint_dir,
    #     tag=f'{pre_trained_model}_{lang}_{epoch}',
    #     client_state={"save_optimizer_states": False})
    if rank == 0:
        print(f"Training completed in {time.time() - start:.2f} seconds")
        print(f"Model saved to {checkpoint_dir}")

    accelerator.wait_for_everyone()

if dist.is_initialized():
    dist.destroy_process_group()

accelerator.free_memory()