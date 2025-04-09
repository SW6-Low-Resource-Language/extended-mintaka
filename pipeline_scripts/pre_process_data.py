from utils.number_proccesor import replace_entire_word_if_digit
from utils.google_translation import google_translate_line_by_line
from text_to_num import text2num
import json
def pre_process_data(parsed_llm): 
    changes = 0
    entries_modified = 0
    numerical_text_answers_indexes = []
    answers_for_translation = []

    
    for e_index, entry in enumerate(parsed_llm):
        if entry['answerType'] == "numerical":
            true_answer = entry['true_answer']
            answers = entry['answers']
            any_changes = False
            for a_index, answer in enumerate(answers):
                answer = answer.replace("\\n", " ")
                mod_answer = replace_entire_word_if_digit(answer).replace("\\200", "\\u200b")

                ans_split = answer.split(" ")
                mod_split = mod_answer.split(" ")
                for i in range(len(ans_split)):
                    if ans_split[i] != mod_split[i] and "%" not in ans_split[i] and "\\u" not in ans_split[i]:
                        changes += 1
                        any_changes = True
                        print(f"Changed {answer} to {mod_answer}")
                        break
                else:
                    filtered_answer = " ".join([word for word in answer.split(" ") if "%" not in word and "\\u" not in word])
                    if any(char.isdigit() for char in filtered_answer):
                        print(f"already contains arabic numerals: {answer}") 
                    else:
                        numerical_text_answers_indexes.append([e_index, a_index])
                        answers_for_translation.append(mod_answer)
            if any_changes:
                entries_modified += 1

    print(f"Total changes made: {changes} / {655 * 5}")
    print(f"Total entries affected: {entries_modified}")
        
                
    with open('translation_indexes.json', 'w', encoding='utf-8') as f:
        json.dump(numerical_text_answers_indexes, f, ensure_ascii=False, indent=4)
    
    with open('answers_for_translation.txt', 'w', encoding='utf-8') as f:
        for answer in answers_for_translation:
            f.write(answer + "\n")

    print(f"Will translate {len(answers_for_translation)} to check for numerical text answers in english")
    # translated_answer = google_translate_line_by_line(answers_for_translation, "translated_answers.txt", target_language="en-US", source_language="bn")
    with open('translated_answers2.txt', 'r', encoding='utf-8') as f:
        translated_answer = f.readlines()

    number_annotations = []
    for i in range(len(translated_answer)):
        t_ans = translated_answer[i].strip().split(" ")
        print(f"Looking for numeral representation in: {t_ans}")
        num = None
        for w in t_ans:
            try: 
                num = text2num(w.lower(), "en")

                if num != None:
                    print(f"word: {w} to number: {num}")
                    break
                
            except Exception as e:
                print(f"Error converting word '{w}' to number: {e}")
                continue
        number_annotations.append(num)

    with open('number_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(number_annotations, f, ensure_ascii=False, indent=4)
        
    