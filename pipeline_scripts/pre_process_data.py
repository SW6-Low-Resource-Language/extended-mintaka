from utils.number_proccesor import replace_entire_word_if_digit, is_same_answer
from utils.google_translation import google_translate_line_by_line
from text_to_num import text2num
from dateparser.search import search_dates
from utils.get_generation_path import get_generation_path
import json
import copy
def pre_process_data(parsed_llm, mode, lang): 
    """
    Pre-processes parsed LLM data by handling numerical and temporal (date) answers.

    Args:
        parsed_llm (list): A list of dictionaries representing parsed LLM data. Each dictionary contains:
            - 'answerType' (str): The type of the answer (e.g., "numerical", "date").
            - 'true_answer' (str): The correct answer for the entry.
            - 'answers' (list): A list of answers to process.
        lang (str): The language code (e.g., "en", "da") used for processing temporal data.

    Returns:
        list: The modified `parsed_llm` structure with processed numerical and temporal answers on form 
        {answer: text, annotation: value} instead of just text.
    """
    parsed_llm_copy = copy.deepcopy(parsed_llm)

    def process_numerical():
        """
        Processes numerical answers in the parsed LLM data.

        - 1. Replacing any numerical unicode character with their corresponding Arabic numeral.
        - 2. If no modifications where done, check if the answer already contains any Arabic numerals.
        - 3. If not, the answer will be translated to english and checked for textual representation of numbers.
        - 4. Finally the correct numerical representation will be added to the answer as an annotation.
        Raises:
            ValueError: If the number of annotations does not match the number of numerical text answers.
        """
        changes = 0
        entries_modified = 0
        numerical_text_answers_indexes = []
        answers_for_translation = []

        
        for e_index, entry in enumerate(parsed_llm_copy):
            if entry['answerType'] == "numerical":
                true_answer = entry['true_answer']
                answers = entry['answers']
                any_changes = False
                for a_index, answer in enumerate(answers):
                    answer = answer.replace("\\n", " ")
                    mod_answer = replace_entire_word_if_digit(answer).replace("\\200", "\\u200b")

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



        translation_output_path = get_generation_path("processesing_numerical_translations", mode, lang)
        """ # Uncomment this to actually use the translation service
        translated_answers = google_translate_line_by_line(answers_for_translation, "translated_num_answers_da.txt", target_language="en-US", source_language=lang) """


        # Read the translated answers from a file
        with open(translation_output_path, 'r', encoding='utf-8') as f:
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
            entry = parsed_llm_copy[e_index]
            answer = entry['answers'][a_index]
            number_annotation = number_annotations[i]

            if number_annotation is not None:
                # Update the answer directly in the parsed_llm_copy structure
                entry['answers'][a_index] = {
                    "answer": entry['answers'][a_index],
                    "annotations": [number_annotation]
                }
    def process_temporal():
        """
        Processes temporal (date) answers in the parsed LLM data.

        - Identifies answers of type "date" and extracts date information using `dateparser`.
        - Formats extracted dates into the 'YYYY-MM-DD' format.
        - Updates the `parsed_llm_copy` structure with the processed temporal answers and their annotations.
        """
        print("processing_temporal")
        for index, e in enumerate(parsed_llm_copy):
            if e["answerType"] == "date":
                for i, ans in enumerate(e["answers"]):
                    mod_ans = replace_entire_word_if_digit(ans)
                    dates = search_dates(mod_ans, languages=[lang])
                    if dates != None:
                        formatted_dates = []
                        for _, date_obj in dates:
                            formatted_date = date_obj.strftime('%Y-%m-%d')
                            formatted_dates.append(formatted_date)
                        # Update the answer directly in the parsed_llm_copy structure
                        parsed_llm_copy[index]['answers'][i] = {
                            "answer": mod_ans,
                            "annotations": formatted_dates
                        }
                    else:
                        parsed_llm_copy[index]['answers'][i] = mod_ans
                   
                    
                

    process_numerical()
    process_temporal()


    return parsed_llm_copy
    