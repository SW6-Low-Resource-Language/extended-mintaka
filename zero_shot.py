from huggingface_hub import login
from dotenv import load_dotenv
from transformers import pipeline
import os
import json

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Hugging Face token
hf_token = os.getenv("hf_token2")

# Log in to Hugging Face
login(token=hf_token)


# Load the JSON data
with open('../dataset-generation/data/mintaka_test_extended.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

lang = 'da'

local_question_answer = []
for question in data:
    qa_entity = {
            "id": question['id'],
            "question": question['translations'][lang],
        }
    answer = question['answer']
    if answer['answerType'] in ["numerical", "boolean", "date", "string"]:
        qa_entity['answer'] = answer['answer'][0]
    elif answer['answer'] == None:
        continue
    elif answer['answer'][0]['label'][lang] != None:
        qa_entity['answer'] = answer['answer'][0]['label'][lang]
    else:
        continue
    local_question_answer.append(qa_entity)


print(local_question_answer[0])



# Define the pre-trained model
# pretrained_model = "meta-llama/Llama-3.3-70B-Instruct"
pretrained_model = "meta-llama/Llama-3.2-1B-Instruct"

# Initialize the text generation pipeline
generator = pipeline('text-generation', model=pretrained_model)

# Placeholder for results
results = []

# Example usage of the generator
# Uncomment and modify the following lines as needed
# prompt = "Svar kort: \n" + local_question_answer[2]['question']
# generated_text = generator(prompt, max_length=50, num_return_sequences=5)
# print(generated_text)

# qa_result = local_question_answer[2]
# qa_result['model_answer'] = generated_text
# results.append(qa_result)

# [{'generated_text': 'Svar kort: \nHvem er ældst, The Weeknd eller Drake? \nSvar: \nThe Weeknd (2015) \nDrake (2016) \n\nHvis du vil vide mere om denne spørg'},
#  {'generated_text': 'Svar kort: \nHvem er ældst, The Weeknd eller Drake? \n\nSvar: \nHvis man ser på følgende liste, så er The Weeknd ældst af de to. Han følger'},
#  {'generated_text': 'Svar kort: \nHvem er ældst, The Weeknd eller Drake? \nHvis du spørger om The Weeknd, er han ældst af de to. \nHvis du spørger om Drake,'},
#  {'generated_text': 'Svar kort: \nHvem er ældst, The Weeknd eller Drake? \nHvis du spørger om ældst, er Drake ældst, men The Weeknd er også ældst.\n\nJeg er'},
#  {'generated_text': 'Svar kort: \nHvem er ældst, The Weeknd eller Drake? \n\nSvar: \nHvis man ligger i orden, så er The Weeknd ældst. Han født i 1990, mens'}]

# Iterate over the local_question_answer dataset
for entry in local_question_answer:
    prompt = "Svar kort: \n" + entry['question']
    generated_text = generator(prompt, max_length=35, num_return_sequences=5)
    qa_result = entry
    qa_result['model_answer'] = generated_text

    results.append(qa_result)

# Save the results to a JSON file
output_file = "results_" + pretrained_model.replace("/", "_") + '_' + lang + '.json'

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)