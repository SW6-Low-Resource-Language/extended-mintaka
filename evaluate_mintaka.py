import json
from pipeline_scripts.parse_llm_output import parse_llm_output
from pipeline_scripts.pre_process_data import pre_process_data
from pipeline_scripts.hits_at_k import hits_at_k_string_match  
from pipeline_scripts.hits_stats import calc_hits_at_ks
from pipeline_scripts.semantic_similarity import perform_semantic_similarity
from pipeline_scripts.sem_stat import run_semantic_similarity_analysis 
from utils.get_generation_path import get_generation_path 
from utils.get_intersecting_entries import get_intersecting_entries

def run_mintaka_analysis(lang, mode, comparative_dict, questions_label_dict):
    """
    Run the Mintaka analysis pipeline for a given language.

    Args:
        lang (str): The language code (e.g., 'bn' for Bengali).
        dataset_input_json_path (str): Path to the input JSON file containing the dataset.
        model_input_txt_path (str): Path to the input text file containing LLM prompts and generated answers.
        output_json_path (str): Path to save the output JSON file with parsed questions, answers, IDs, and true answers.

    Returns:
        None
    """
    # Step 1: Parse LLM output
    dataset_input_json_path = get_generation_path("test_data_extended")
    with open(dataset_input_json_path, 'r', encoding='utf-8') as file:
        dataset_input_json = json.load(file)
    questions_label = questions_label_dict['question'][lang]
    answers_label = questions_label_dict['answer'][lang]
    model_input_txt_path = get_generation_path("model_answers", mode, lang)
    parsed_answer_path = get_generation_path("parsed_answers_json", mode, lang)
    parsed_answers = parse_llm_output(
        model_input_txt_path,
        dataset_input_json_path,
        parsed_answer_path,
        questions_label,
        answers_label,
        lang
    )
    processed_answers = pre_process_data(parsed_answers,mode, lang)  # This function is assumed to be defined in pre_process_data.py
    processed_data_path = get_generation_path("processed_test_data", mode, lang)
    with open(processed_data_path, 'w', encoding='utf-8') as file:
        json.dump(processed_answers, file, ensure_ascii=False, indent=4)
    with open(processed_data_path, 'r', encoding='utf-8') as file:
        processed_answers = json.load(file) 


    hits_output_path = get_generation_path("hit_annotation_json", mode, lang)
    hits_obj, bool_hits = hits_at_k_string_match(processed_answers, comparative_dict, lang, hits_output_path)
    hits_excel_path = get_generation_path("hits_k_excel", mode, lang)
    # Subsets hits_obj
    sub_entries = get_intersecting_entries(dataset_input_json, ["da", "bn", "fi"])
    # dont pass sub_entries to calc_hits_at_ks if you want to calculate for all entries in the language
    calc_hits_at_ks(hits_obj,5, hits_excel_path, lang, dataset_input_json)
    calc_hits_at_ks(hits_obj,5, hits_excel_path, lang, dataset_input_json, sub_entries)

    true_hits = bool_hits["True"]  
    false_hits = bool_hits["False"]

    max_hit_true_label = max(true_hits, key=true_hits.get)
    max_hit_false_label = max(false_hits, key=false_hits.get) 

    """ #Step 3 : Calculate semantic similarity scores
    sem_score_output_path = get_generation_path("sem_scores_json", mode, lang)
    perform_semantic_similarity(
        lang=lang, 
        dataset_input_json_path=dataset_input_json_path, 
        model_answer_json_path=parsed_answer_path, 
        true_label=max_hit_true_label, 
        false_label=max_hit_false_label,
        output_json_path=sem_score_output_path
    )  
    #dont pass sub_entries to run_semantic_similarity_analysis if you want to calculate for all entries in the language
    run_semantic_similarity_analysis(lang, mode) 
    run_semantic_similarity_analysis(lang, mode, sub_entries) """
    
    



if __name__ == "__main__":
    # load comparative dict
    with open('./configurations/comparative_dict.json', 'r', encoding='utf-8') as file:
        comparative_dict = json.load(file)  

    with open('./configurations/questions_label_lang_dict.json', 'r', encoding='utf-8') as file:
        questions_label_dict = json.load(file)

    lang = "bn"  
    mode = "zeroshot"

    run_mintaka_analysis(lang, mode, comparative_dict, questions_label_dict)