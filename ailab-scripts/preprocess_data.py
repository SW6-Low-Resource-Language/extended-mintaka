import json

def preprocess_data(filename, lang):
    with open('./configurations/comparative_dict.json', 'r', encoding='utf-8') as f:    
        comparative_dict = json.load(f)
        comparative_dict_lang = comparative_dict[lang]

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_answer_pairs = []

    for entry in data:
        id = entry['id']
        if lang != 'en':
            question = entry['translations'][lang]
        else:
            question = entry['question']

        if entry['answer']['answer'] == None:
             continue

        if entry['answer']['answerType'] == 'string':
            answer = entry['answer']['answer'][0]
            if answer.lower() in comparative_dict_lang and lang != 'en':
                answer = comparative_dict_lang[answer.lower()]
        elif entry['answer']['answerType'] == 'boolean':
            answer = entry['answer']['answer'][0]
            if answer is True:
                answer = comparative_dict_lang['true_list'][0]
            elif answer is False:
                answer = comparative_dict_lang['false_list'][0] 
        elif entry['answer']['answerType'] in ["numerical", "boolean", "date", "string"]:
            answer = entry['answer']['answer'][0]
        else:
            answer = entry['answer']['answer'][0]['label'][lang]

        if 'supportingEnt' in entry['answer']:
            supporting_entities = []
            for entity in entry['answer']['supportingEnt']:
                ent = entity['label'][lang]
                if ent != None:
                    supporting_entities.append(ent)

            supporting_entities_string = ", ".join(supporting_entities)
            answer = str(answer) + ", " + supporting_entities_string
        if answer == None:
             continue
       
        pair_obj = {
            "id": str(id),
            "question": str(question),
            "answer": str(answer)
        }
        question_answer_pairs.append(pair_obj)

    print(f"{len(question_answer_pairs)} question-answer pairs loaded from {filename}")
    with open(filename.replace('.json', '_qa_pairs_' + lang + '.json'), 'w', encoding='utf-8') as f:
         json.dump({"data": question_answer_pairs}, f, ensure_ascii=False, indent=4)


def pair_and_tokenize_data(data_file, tokenizer):
    # https://huggingface.co/transformers/v3.0.2/preprocessing.html
    # Look at 'Preprocessing pars of sentences'
    # encoded_input = tokenizer(["How old are you?", "what's your name?"], ["I'm 6 years old", "Magnus"])
    # print(encoded_input)

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    batch_questions = [
        #LIST OF ALL QUESTIONS IN ORDER
    ]

    batch_answers = [
        #LIST OF ALL ANSWERS IN SAME ORDER AS QUESTIONS
    ]

    for datapoint in data:
        batch_questions.append(datapoint['question'])
        batch_answers.append(datapoint['answer'])


    encoded_inputs = tokenizer(batch_questions[:50], 
                                batch_answers[:50], 
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                                max_length=64
                            )

    return encoded_inputs
