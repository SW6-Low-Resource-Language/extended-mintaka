def get_valid_entities(mintaka_data, lang):
    entities = []
    for question in mintaka_data:
        qa_entity = {
            "id": question['id'],
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
        entities.append(qa_entity)
    return entities