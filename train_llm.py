import os
# import torch
# import torch.nn as nn
# import torch.optim as optim

import json
# from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

lang='da'

pre_trained_model = "meta-llama/Llama-3.3-70B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)

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
        question_answer_pairs.append(str(pair_obj))

    print(f"{len(question_answer_pairs)} question-answer pairs loaded from {filename}")
    with open(filename.replace('.json', '_qa_pairs_' + lang + '.json'), 'w', encoding='utf-8') as f:
         json.dump({"data": question_answer_pairs}, f, ensure_ascii=False, indent=4)
        # f.write("\n".join(question_answer_pairs))


load_data(training_file)
load_data(validation_file)

training_data_pairs = training_file.replace('.json', '_qa_pairs_' + lang + '.json')
validation_data_pairs = validation_file.replace('.json', '_qa_pairs_' + lang + '.json')

dataset = load_dataset("json", data_files={"train": training_data_pairs, "validation": validation_data_pairs}, field="data")



def preprocess_data(data):
	# https://huggingface.co/transformers/v3.0.2/preprocessing.html
	# Look at 'Preprocessing pars of sentences'
	# encoded_input = tokenizer(["How old are you?", "what's your name?"], ["I'm 6 years old", "Magnus"])
	# print(encoded_input)

	batch_questions = [
		#LIST OF ALL QUESTIONS IN ORDER
	]
	
	batch_answers = [
		#LIST OF ALL ANSWERS IN SAME ORDER AS QUESTIONS
	]

	# encoded_inputs = tokenizer(batch_questions, batch_answers)

	return {}


# from transformers import get_scheduler, AutoModelForCausalLM



# training_loader = DataLoader(training_data, batch_size=32, shuffle=True)

# model = AutoModelForCausalLM.from_pretrained(pre_trained_model)
# optimizer = optim.Adam(model.parameters())
# loss_fn = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=3e-5)

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

num_epochs = 100
# num_training_steps = num_epochs * len(train_dataloader)

# train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
#     train_dataloader, eval_dataloader, model, optimizer
# )

# if os.listdir(checkpoint_dir):
#     latest_checkpoint = max([int(file.split('.')[0]) for file in os.listdir(checkpoint_dir)])
#     checkpoint = torch.load(os.path.join(checkpoint_dir, f'{latest_checkpoint}.pt'))
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = latest_checkpoint + 1
# else:
#     start_epoch = 0

# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps
# )

# for epoch in range(start_epoch, num_epochs):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         data = data.view(data.size(0), -1)
#         output = model(data)
#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch}: Loss {loss.item()}')

#     # Save checkpoint every epoch
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss
#     }, os.path.join(checkpoint_dir, f'{epoch}.pt'))



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