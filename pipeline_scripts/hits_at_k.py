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


def hits_at_k_string_match(h_answers, bool_comparative_dict, lang, output_path):
    hits_obj = {}

    true_list = bool_comparative_dict[lang]['true_list']
    false_list = bool_comparative_dict[lang]['false_list']
    bool_hits_true = {i : 0 for i in true_list}
    bool_hits_false = {i : 0 for i in false_list}
    bool_hits = {"True" : bool_hits_true, "False" : bool_hits_false}

    for answer in h_answers:
        question, answers, id, true_answer = answer['question'], answer['answers'], answer['id'], answer['true_answer']
        bool_answer = true_answer is True or true_answer is False
        bool_string = None
        if true_answer is True:
            bool_string = "True"
        elif true_answer is False:
            bool_string = "False"

        answer_strings = 1 
        true_answer = pre_process_boolean(true_answer, bool_comparative_dict, lang)
        hits = []
        # pre_process_boolean return lists of potential strings if boolean answer
        if (bool_answer):
            answer_strings = len(true_answer)
        else:
            true_answer = [true_answer]
        hits_at = None
        for index, answer in enumerate(answers):
            hit = False
            # if bool_answer:
            for i in range(answer_strings):
                t_answer = true_answer[i]
                # Use a regular expression to match the word with boundaries
                pattern = rf'(?<!\w){re.escape(str(t_answer).lower())}(?!\w)'
                if re.search(pattern, answer.lower()): 
                    if(bool_string is not None):
                        bool_hits[bool_string][t_answer] += 1
                    hit = True
                    # print("I'm hit!")
                    # print(f"Hits@{index+1}: {question} - {answer} - {true_answer}")
                    if hits_at == None:
                        hits_at = index + 1
                    break
            hits.append({"idx": index+1, "hit": hit})
            
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

    