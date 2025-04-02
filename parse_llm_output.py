import re
import json

lang = "da"  
questions_label = "Spørgsmål" if lang == "da" else "প্রশ্ন"
model_input_txt_path = "vllm_textgen_output_" + lang + ".txt"
# model_input_txt_path = "vllm_textgen_input_da.txt"
dataset_input_json_path = "./data/mintaka_test_extended.json"

output_json_path = "output_" + lang + "_sprgsml_w_id_and_true_answer.json"
# output_json_path = "output_dk_sprgsml_w_id_and_true_answer.json"

def parse_llm_output(
model_input_txt_path = model_input_txt_path, 
dataset_input_json_path = dataset_input_json_path, 
output_json_path = output_json_path,
questions_label = questions_label,
lang = lang):
    """
    Parses the output of a language model (LLM) and maps questions to their corresponding IDs and true answers.

    This function processes a text file containing prompts and generated answers from an LLM, matches the questions
    with their IDs and true answers from a dataset, and outputs the results in a structured JSON format.

    Args:
        model_input_txt_path (str): Path to the text file containing LLM prompts and generated answers.
        dataset_input_json_path (str): Path to the JSON file containing the dataset with question IDs and true answers.
        output_json_path (str): Path to save the output JSON file with parsed questions, answers, IDs, and true answers.
        questions_label (str): Label used to identify questions in the LLM prompts (e.g., "Spørgsmål").
        lang (str): Language code used to extract the correct translation of questions and answers (e.g., "da" for Danish).

    Returns:
        None: The function writes the parsed data to the specified output JSON file.

    Notes:
        - The function assumes that the LLM output uses a specific format for prompts and generated text.
        - It handles cases where the LLM output uses inconsistent quotation marks for "Generated text".
        - Questions and answers are matched using their text content, and unmatched entries are assigned `None` for IDs and true answers.
    """
    
    with open(model_input_txt_path, 'r', encoding='utf-8') as f:
        data = f.read()

    with open(dataset_input_json_path, 'r', encoding='utf-8') as file:
        data_for_id = json.load(file)


    # [{id: "blabla", question: "yaya", label: "xdd"}]


    # Sometimes in the file there is written Generated text: " instead of Generated text: ' which messes up with the parsing
    data = data.replace("Generated text: \"", "Generated text: '")


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
        answer = question['answer']
        if answer['answerType'] in ["numerical", "boolean", "date", "string"]:
            qa_entity['true_answer'] = answer['answer'][0]
        elif answer['answer'] == None:
            continue
        elif answer['answer'][0]['label'][lang] != None:
            qa_entity['true_answer'] = answer['answer'][0]['label'][lang]
        else:
            continue
        local_question_with_id.append(qa_entity)


    data = []
    if matches:
        def extract_question(prompt):
            q_match = re.search(rf"{questions_label}:\s*(.*?)(\\n|$)", prompt)
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


    lookup = {qa["question"].strip().replace("\\", ""): qa["id"] for qa in local_question_with_id}
    lookup2 = {qa["question"].strip().replace("\\", ""): qa["true_answer"] for qa in local_question_with_id}


    for entry in data:
        q_text = entry["question"].strip().replace("\\", "")
        if q_text in lookup:
            entry["id"] = lookup[q_text]
            entry["true_answer"] = lookup2[q_text]
        else:
            entry["id"] = None  
            entry["true_answer"] = None
        

    print(data[:1])  # printer x antal første spørgsmål til tjekning.

    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parse_llm_output()









