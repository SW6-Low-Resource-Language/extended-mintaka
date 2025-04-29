import json 
from transformers import AutoModelForCausalLM, AutoTokenizer


# Load the JSON data
with open('./data/mintaka_test_extended.json', 'r', encoding='utf-8') as file:
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


pre_prompt = " " #qwen echoe'er preprompt/postprompt i predicted_answer så derfor er de en empty string
post_prompt = " "

prompts = [pre_prompt + entry['question'] + post_prompt
        for entry in local_question_answer]


#model_name = "Qwen/Qwen2.5-14B-Instruct" - ændre til 14b modellen eller højere - 32? 
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

device = next(model.parameters()).device 

results = []


for idx, prompt in enumerate(prompts[:10]):
    messages = [
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    _, _, answer = decoded_output.partition(post_prompt) #et forsøg på at formindske occurences af interne tokens som \nAssistant i predicted answer. Virker ikke optimalt
    answer = answer.replace("\nAssistant:", "").replace("Assistant:", "").strip() #kan ikke helt gennemskue hvor meget af de skrald svar der er problemer med setuppet, 
                                                                                    #eller fordi jeg tester med 0,5b modellen og den er dårlig på multilingual 
    results.append({
        "id": local_question_answer[idx]["id"],
        "question": local_question_answer[idx]["question"],
        "true_answer": local_question_answer[idx]["answer"],
        "predicted_answer": answer
    })

# gør det bare lidt lækrere at læse resultaterne
print(json.dumps(results, indent=2, ensure_ascii=False))









