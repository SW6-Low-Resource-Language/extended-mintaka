import os
import time
import torch.distributed as dist

from preprocess_data import pair_and_tokenize_data, preprocess_data

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, MT5Tokenizer, MT5ForConditionalGeneration
from deepspeed import initialize

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

training_file = "./data/mintaka_dev_extended.json"
# training_file = "./data/mintaka_train_extended.json"
validation_file = "./data/mintaka_test_extended.json"

train_micro_batch_size_per_gpu = 16
gradient_accumulation_steps = 4
val_batch_size = 16

num_epochs = 2
learning_rate = 1e-6
max_length = 128


# torch.cuda.empty_cache()
# for i in range(torch.cuda.device_count()):
#     torch.cuda.set_per_process_memory_fraction(0.95, device=i)

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

optimizer = AdamW(model.parameters(), lr=learning_rate)

if rank == 0:
    preprocess_data(training_file, lang)
    preprocess_data(validation_file, lang)

dist.barrier()


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


if rank == 0:
    print(f"Training dataset size: {len(training_dataset)}")
    print(f"DataLoader batch size: {training_loader.batch_size}")
    print(f"Effective number of batches per epoch: {len(training_loader) // dist.get_world_size()}")
    print(f"DataLoader batch size: {training_loader.batch_size}")

print(f"[Rank {rank}] Total samples after padding: {len(training_sampler)}")

num_training_steps = num_epochs * len(training_loader)

model_engine, optimizer, _, lr_scheduler = initialize(
    model=model,
    optimizer=optimizer,
    config_params={
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "zero_optimization": {
            "stage": 3
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "initial_scale_power": 16,
             "min_loss_scale": 128 
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

    training_sampler.set_epoch(epoch)

    if rank == 0:
        print(f"Time elapsed: {time.time() - start:.2f} seconds")

    model_engine.train()
    print(f"[Rank {rank}] Model_engine set to training mode")
    for batch_idx, (input_ids, attention_mask) in enumerate(training_loader):
        print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}")

        input_ids = input_ids.to(model_engine.device)
        attention_mask = attention_mask.to(model_engine.device)

        outputs = model_engine(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        model_engine.backward(loss)

        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model_engine.parameters() if p.grad is not None]), 2)
        # print(f"[Rank {rank}] Gradient Norm: {total_norm}")
    
        torch.nn.utils.clip_grad_norm_(model_engine.parameters(), max_norm=1.0)
        model_engine.step()

    if rank == 0:
        print(f'Epoch {epoch}: Loss {loss.item()}')
        print(f"Time elapsed: {time.time() - start:.2f} seconds")

    model_engine.eval()
    print(f"[Rank {rank}] Model_engine set to evaluation mode")

    val_loss = 0
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in enumerate(test_loader):
            print(f"[Rank {rank}] Epoch {epoch} - validation Batch {batch_idx}")

            input_ids = input_ids.to(model_engine.device)
            attention_mask = attention_mask.to(model_engine.device)

            output = model_engine(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = output.loss.detach()
            val_loss += loss.item()
        val_loss /=len(test_loader)

    if rank == 0:
        print(f'Epoch {epoch}: Validation Loss {val_loss}')


    model_engine.save_checkpoint(
        checkpoint_dir,
        tag=f'{pre_trained_model}_{lang}_{epoch}',
        client_state={"save_optimizer_states": False})
    if rank == 0:
        print(f"Training completed in {time.time() - start:.2f} seconds")
        print(f"Model saved to {checkpoint_dir}")

    dist.barrier()

if dist.is_initialized():
    dist.destroy_process_group()
