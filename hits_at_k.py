import json

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


with open("./output_dk_sprgsml_w_id_and_true_answer.json", 'r', encoding='utf-8') as file:
    answers = json.load(file)

def pre_process_boolean(answer):
    if true_answer == True:
         return ["Sandt", "Rigtig", "Rigtigt", "Korrekt", "Ja"]
    elif answer == False:    
        return ["Falsk", "Forkert", "Nej"]
    elif answer == "After":
        return "Efter"
    elif answer == "Before":
        return "Før"
    elif answer == "Same":
        return "samme"
    elif answer == "Less":
        return "Mindre"
    elif answer == "Both":
        return "Begge"
    else:
        return answer


hits_obj = {}
for answer in answers:
    question, answers, id, true_answer = answer['question'], answer['answers'], answer['id'], answer['true_answer']
    bool_answer = true_answer in [True, False]
    answer_strings = 1 
    true_answer = pre_process_boolean(true_answer)
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
            if str(t_answer).lower() in answer.lower(): 
                hit = True
                print("I'm hit!")
                print(f"Hits@{index+1}: {question} - {answer} - {true_answer}")
                if hits_at == None:
                    # hits_at = index + 1
                    hits_at = index + 1
                break
            else:
                print(f"Missed me: {true_answer} - {answer}")
        hits.append({"idx": index+1, "hit": hit})
        
    hits_obj[id] = {}
    hits_obj[id]['question'] = question
    hits_obj[id]['true_answer'] = true_answer
    hits_obj[id]['hits'] = hits
    hits_obj[id]['hits_at'] = hits_at
    hits_obj[id]['answers'] = answers

none_count = 0
hits_1 = 0
hits_2 = 0
hits_3 = 0
hits_4 = 0
hits_5 = 0

for key, value in hits_obj.items():
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

hits_2 += hits_1
hits_3 += hits_2
hits_4 += hits_3
hits_5 += hits_4


print(f"Hits@1: {hits_1}")
print(f"Hits@2: {hits_2}")
print(f"Hits@3: {hits_3}")
print(f"Hits@4: {hits_4}")
print(f"Hits@5: {hits_5}")
print(f"None: {none_count}")

with open("hits_da.json", "w", encoding="utf-8") as json_file:
    json.dump(hits_obj, json_file, ensure_ascii=False, indent=4)



    