from vllm import LLM, SamplingParams
import os
import json
from dotenv import load_dotenv

os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
# load_dotenv()
# os.environ = os.getenv("hf_token2")


# Load the JSON data
with open('./data/mintaka_test_extended.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

lang = 'da'
# [{id: "blabla", question: "yaya", label: "xdd"}]

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
    

pre_prompt = "Svar på det følgende spørgsmål:\n Spørgsmål: "
post_prompt = "\nSvar: "

prompts = [pre_prompt + entry['question'] + post_prompt
        for entry in local_question_answer]

#pretrained_model = "meta-llama/Llama-3.2-1B-Instruct"
pretrained_model = "meta-llama/Llama-3.3-70B-Instruct"

llm = LLM(model=pretrained_model, 
        tensor_parallel_size=8,
        dtype="float16",
        gpu_memory_utilization=1,
        cpu_offload_gb=32,
        enforce_eager=True)
sampling_params = SamplingParams(
    max_tokens=50,
    temperature=0.7, 
    n=5 
)

# Generate responses for each prompt
results = []


outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    for i in range(len(output.outputs)):
        print(f"{i}: Prompt: {output.prompt!r}, Generated text: {output.outputs[i].text!r}")

#for prompt, output in zip(prompts, outputs):
 #   results.append({
  #      "prompt": prompt,
   #     "responses": outputs
   # })

#output_file = "results_" + pretrained_model.replace("/", "_") + '_' + lang + '.json'
#with open(output_file, 'w', encoding='utf-8') as file:
 #   json.dump(results, file, ensure_ascii=False, indent=4)
