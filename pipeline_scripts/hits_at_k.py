import json
import re

#https://pykeen.readthedocs.io/en/stable/api/pykeen.metrics.ranking.HitsAtK.html#pykeen.metrics.ranking.HitsAtK


def isHit(recommendation, truth):
    if truth in recommendation:
        return 1
    return 0

def hits_at_k(k, recommendations, truth):
    hits = 0
    for i in range(len(recommendations)):
        if (recommendations[i]['idx'] < k):
            hits += isHit(recommendations[i]['answer'], truth)
    return (hits, k)

def pre_process_boolean(answer, bool_comparative_dict, lang):
    if answer is True:
         return bool_comparative_dict[lang]['true_list']
    elif answer is False:    
        return bool_comparative_dict[lang]['false_list']
    elif answer == "After":
        return bool_comparative_dict[lang]['after']
    elif answer == "Before":
        return  bool_comparative_dict[lang]['before']
    elif answer == "Same":
        return bool_comparative_dict[lang]['same']
    elif answer == "Less":
        return bool_comparative_dict[lang]['less']
    elif answer == "Both":
        return bool_comparative_dict[lang]['both']
    else:
        return answer

def match_pattern(answer, true_answer):
    """Check if the true answer matches the answer string using regex."""
    pattern = rf'(?<!\w){re.escape(str(true_answer).lower())}(?!\w)'
    return re.search(pattern, answer.lower())

def check_annotations(answer, true_answer):
    """Check if the true answer exists in the annotations."""
    annotations = answer.get("annotations", [])
    return any(annot == true_answer for annot in annotations)

def process_answer(index, answer, true_answer, bool_hits, bool_string, hits_at):
    """Process a single answer and check for hits."""
    hit = False
    if isinstance(answer, dict) and "annotations" in answer:
        # Check annotations and answer string
        if check_annotations(answer, true_answer) or match_pattern(answer["answer"], true_answer):
            hit = True
            if hits_at is None:
                hits_at = index + 1
    else:
        # Check plain string answers
        if match_pattern(answer, true_answer):
            hit = True
            if bool_string is not None:
                bool_hits[bool_string][true_answer] += 1
            if hits_at is None:
                hits_at = index + 1
    return hit, hits_at


def hits_at_k_string_match(h_answers, bool_comparative_dict, lang, output_path):
    hits_obj = {}
    true_list = bool_comparative_dict[lang]['true_list']
    false_list = bool_comparative_dict[lang]['false_list']
    bool_hits = {"True": {i: 0 for i in true_list}, "False": {i: 0 for i in false_list}}

    for answer in h_answers:
        question, answers, id, true_answer = answer['question'], answer['answers'], answer['id'], answer['true_answer']
        bool_answer = isinstance(true_answer, bool)
        bool_string = "True" if true_answer is True else "False" if true_answer is False else None

        # Preprocess true_answer
        true_answer_list = pre_process_boolean(true_answer, bool_comparative_dict, lang)
        if not isinstance(true_answer_list, list):
            true_answer_list = [true_answer_list]

        hits = []
        hits_at = None

        # Process each answer
        for index, answer in enumerate(answers):
            for t_answer in true_answer_list:
                hit, hits_at = process_answer(index, answer, t_answer, bool_hits, bool_string, hits_at)
                if hit:
                    break  # Stop checking other true answers if a hit is found
            hits.append({"idx": index + 1, "hit": hit})

            
        hits_obj[id] = {}
        hits_obj[id]['question'] = question
        hits_obj[id]['true_answer'] = true_answer
        hits_obj[id]['hits'] = hits
        hits_obj[id]['hits_at'] = hits_at
        hits_obj[id]['answers'] = answers


    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(hits_obj, json_file, ensure_ascii=False, indent=4)
    return hits_obj, bool_hits


def count_hits(hits_object):
    none_count = 0
    hits_1 = 0
    hits_2 = 0
    hits_3 = 0
    hits_4 = 0
    hits_5 = 0

    for key, value in hits_object.items():
        hit = value['hits_at']
        if hit == None:
            none_count += 1
        elif hit == 1:
            hits_1 += 1
        elif hit == 2:
            hits_2 += 1
        elif hit == 3:
            hits_3 += 1
        elif hit == 4:
            hits_4 += 1
        elif hit == 5:
            hits_5 += 1

    return none_count, hits_1, hits_2, hits_3, hits_4, hits_5

def sum_hits(hits_1, hits_2, hits_3, hits_4, hits_5):
    hits_2 += hits_1
    hits_3 += hits_2
    hits_4 += hits_3
    hits_5 += hits_4

    return hits_2, hits_3, hits_4, hits_5
    

def pp_hits(none_count, hits_1, hits_2, hits_3, hits_4, hits_5):
    print(f"Hits@1: {hits_1}")
    print(f"Hits@2: {hits_2}")
    print(f"Hits@3: {hits_3}")
    print(f"Hits@4: {hits_4}")
    print(f"Hits@5: {hits_5}")
    print(f"None: {none_count}")




if __name__ == "__main__":
    lang = 'bn'
    # lang = 'bn'

    true_list = bool_comparative_dict[lang]['true_list']
    false_list = bool_comparative_dict[lang]['false_list']
    bool_hits = {i : 0 for i in true_list + false_list}
    
    input_file = "./output_" + lang + "_sprgsml_w_id_and_true_answer.json"

    with open(input_file, 'r', encoding='utf-8') as file:
        answers = json.load(file)


    hits_object = hits_at_k_string_match(answers, )
    print(hits_object)

    none_count, hits_1, hits_2, hits_3, hits_4, hits_5 = count_hits(hits_object)

    hits_2, hits_3, hits_4, hits_5 = sum_hits(hits_1, hits_2, hits_3, hits_4, hits_5)

    print(f"Bool hits: {bool_hits}")

    