import re
import json
from utils.get_valid_entities import get_valid_entities



def parse_llm_output(
model_input_txt_path, 
dataset_input_json_path, 
output_json_path,
questions_label,
answers_label,
lang):
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
    
    with open(model_input_txt_path, 'r', encoding='utf-8') as file:
        lines_txt = file.read()

    lines = lines_txt.splitlines()
    print(len(lines))

    startindex = -1
    found = False
    while(not found):
        startindex += 1
        line = lines[startindex]
        if "0: Prompt:" in line:
            found = True

    print(f"Start index: {startindex}")
        
    data = []


    for index in range(startindex, len(lines), 5):
        group = lines[index:index+5]
        question_match = re.search(rf'{questions_label}:\s*(.*?)\s*\\n{answers_label}:', group[0], re.DOTALL)
        question = question_match.group(1).strip() if question_match else "NO_QUESTION_FOUND"
        if question == "NO_QUESTION_FOUND":
            continue
            

        answers = []
        for entry in group:
            entry = entry.replace("Generated text: \"", "Generated text: '")
            answer_match = re.search(r"Generated text:\s*'(.*?)'", entry, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else "NO_ANSWER_FOUND"
            if "\\n" in answer:
                print(f"Answer contains \\n: {answer}")
                answer = answer.replace("\\n", " ")

            answers.append(answer)
        # Extract answer (after "Generated text: '")
        data.append({
            'question': question,
            'answers': answers
        })

    print(len(data))
    with open(dataset_input_json_path, 'r', encoding='utf-8') as file:
        data_for_id = json.load(file)

    valid_entities = get_valid_entities(data_for_id, lang)
    if len(valid_entities) == len(data):
        for i in range(len(data)):
            data[i]["true_answer"] = valid_entities[i]["true_answer"]
            data[i]["id"] = valid_entities[i]["id"]
            data[i]["answerType"] = valid_entities[i]["answerType"]
    else:
        print("NO")
        raise Exception("The number of valid entities does not match the number of parsed answers.")


    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
    return data










