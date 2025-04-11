from utils.number_proccesor import replace_entire_word_if_digit, is_same_answer
from utils.google_translation import google_translate_line_by_line
from text_to_num import text2num
from dateparser.search import search_dates
import json
def pre_process_data(parsed_llm, lang): 

    def process_numerical():
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
                    if not is_same_answer(answer, mod_answer):
                        entry['answers'][a_index] = mod_answer
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
            
                    
        """ with open('translation_indexes.json', 'w', encoding='utf-8') as f:
            json.dump(numerical_text_answers_indexes, f, ensure_ascii=False, indent=4)
        
        with open('answers_for_translation.txt', 'w', encoding='utf-8') as f:
            for answer in answers_for_translation:
                f.write(answer + "\n") """

        # Uncomment this to actually use the translation service
        """ for answer in answers_for_translation:
            translated_answers = google_translate_line_by_line(answer, "translated_answers.txt", target_language="en-US", source_language=lang) """


        # Read the translated answers from a file
        with open('translated_answers2.txt', 'r', encoding='utf-8') as f:
            translated_answers = f.readlines()

        number_annotations = []
        for i in range(len(translated_answers)):
            t_ans = translated_answers[i].strip().split(" ")
            num = None
            for w in t_ans:
                try: 
                    num = text2num(w.lower(), "en")

                    if num != None:
                        break
                    
                except Exception as e:
                    print(f"Error converting word '{w}' to number: {e}")
                    continue
            number_annotations.append(num)

        """ with open('number_annotations.json', 'w', encoding='utf-8') as f:
            json.dump(number_annotations, f, ensure_ascii=False, indent=4) """

        if len(number_annotations) != len(numerical_text_answers_indexes):
            raise ValueError(f"Number of annotations does not match number of indexes {len(number_annotations)} != {len(numerical_text_answers_indexes)}")
            
        for i in range (len(numerical_text_answers_indexes)):
            e_index, a_index = numerical_text_answers_indexes[i]
            entry = parsed_llm[e_index]
            answer = entry['answers'][a_index]
            number_annotation = number_annotations[i]

            if number_annotation is not None:
                # Update the answer directly in the parsed_llm structure
                entry['answers'][a_index] = {
                    "answer": entry['answers'][a_index],
                    "annotations": [number_annotation]
                }
    def process_temporal():
        print("processing_temporal")
        for index, e in enumerate(parsed_llm):
            if e["answerType"] == "date":
                for i, ans in enumerate(e["answers"]):
                    mod_ans = replace_entire_word_if_digit(ans)
                    dates = search_dates(mod_ans, languages=[lang])
                    if dates != None:
                        formatted_dates = []
                        for _, date_obj in dates:
                            formatted_date = date_obj.strftime('%Y-%m-%d')
                            formatted_dates.append(formatted_date)
                        # Update the answer directly in the parsed_llm structure
                        parsed_llm[index]['answers'][i] = {
                            "answer": mod_ans,
                            "annotations": formatted_dates
                        }
                    else:
                        parsed_llm[index]['answers'][i] = mod_ans
                   
                    
                

    process_numerical()
    process_temporal()


    return parsed_llm
    