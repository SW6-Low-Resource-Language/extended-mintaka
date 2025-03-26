import re
import json

with open('vllm_textgen_output_da.txt', 'r', encoding='utf-8') as f:
    data = f.read()

with open('./data/mintaka_test_extended.json', 'r', encoding='utf-8') as file:
    data_for_id = json.load(file)

lang = "da"  
# [{id: "blabla", question: "yaya", label: "xdd"}]


pattern = re.compile(
    r"(?P<index>\d+):\s*Prompt:\s*'(?P<prompt>.*?)',\s*Generated text:\s*'(?P<gen_text>.*?)'",
    re.DOTALL
)

matches = list(pattern.finditer(data))

local_question_with_id = []
for question in data_for_id:
    qa_entity = {
            "id": question['id'],
            "question": question['translations'][lang],
        }
    local_question_with_id.append(qa_entity)


data = []
if matches:
    def extract_question(prompt):
        q_match = re.search(r"Spørgsmål:\s*(.*?)(\\n|$)", prompt)
        return q_match.group(1).strip() if q_match else prompt.strip()

    for i in range(0, len(matches), 5):
        group = matches[i:i+5]
        if len(group) < 5:
            continue
        question_text = extract_question(group[0].group('prompt'))
        answers = [m.group('gen_text').strip() for m in group]
        data.append({
            'question': question_text,
            'answers': answers
        })


lookup = {qa["question"].strip(): qa["id"] for qa in local_question_with_id}


for entry in data:
    q_text = entry["question"].strip()
    if q_text in lookup:
        entry["id"] = lookup[q_text]
    else:
        entry["id"] = None  
    

print(data[:1])  # printer x antal første spørgsmål til tjekning.








