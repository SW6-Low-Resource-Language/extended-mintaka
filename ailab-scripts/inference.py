import json
from transformers import AutoTokenizer, AutoModelForCausalLM, MT5Tokenizer, MT5ForConditionalGeneration
#from transformers import ByT5Tokenizer, T5ForConditionalGeneration
import torch

os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
load_dotenv()
os.environ = os.getenv("hf_token2")


# Load the JSON data
with open('./data/mintaka_test_extended.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

lang = 'en'
# lang = 'bn'

local_question_answer = []
for question in data:
    qa_entity = {
            "id": question['id'],
            "question": question['question'],
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
    

# pre_prompt = "নিম্নলিখিত প্রশ্নের উত্তর দিন এবং প্রতিটি উত্তরের জন্য একটি কনফিডেন্স স্কোর প্রদান করুন\n প্রশ্ন:"
# post_prompt = "\nউত্তর:"

pre_prompt = "Answer this question and provide a confidence score for the answer. Question: "
post_prompt = "Answer: "

prompts = [pre_prompt + entry['question'] + post_prompt
        for entry in local_question_answer]

# pretrained_model = "meta-llama/Llama-3.2-1B-Instruct"
pretrained_model = "meta-llama/Llama-3.3-70B-Instruct"
# pretrained_model = "google/mt5-xl"

#pre_trained_model = "google/mt5-large" 

#pre_trained_model = "google/byt5-large" 
isMT5 = False

if isMT5: # fra fine-tune_mt5.py 
    #tokenizer = MT5Tokenizer.from_pretrained(pre_trained_model)
    #model = MT5ForConditionalGeneration.from_pretrained(pre_trained_model)
    tokenizer = ByT5Tokenizer.from_pretrained(pre_trained_model)
    model = T5ForConditionalGeneration.from_pretrained(pre_trained_model)   
else:
   tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
   model = AutoModelForCausalLM.from_pretrained(pre_trained_model) 
   
   #tokenizer = ByT5Tokenizer.from_pretrained(pre_trained_model)
   #model = T5ForConditionalGeneration.from_pretrained(pre_trained_model)
   


results = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #to speed up local processing for now
model = model.to(device)

for idx, prompt in enumerate(prompts): 
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512) #most mt5's use max token length 512 as default
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    results.append({
        "id": local_question_answer[idx]["id"],
        "question": local_question_answer[idx]["question"],
        "true_answer": local_question_answer[idx]["answer"],
        "predicted_answer": answer
    })

print(results)
