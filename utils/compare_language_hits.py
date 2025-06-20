from utils.get_generation_path import get_generation_path
from utils.get_intersecting_entries import get_intersecting_entries
import json
import os

def compare_language_hits(lang1, lang2):
    hits_path1 = get_generation_path("hit_annotation_json", "zeroshot", lang1)
    hits_path2 = get_generation_path("hit_annotation_json", "zeroshot", lang2)
    
    with open("data/mintaka_test_extended2.json", 'r', encoding='utf-8') as file:
        mintaka_data = json.load(file)
    intersecting_entries = get_intersecting_entries(mintaka_data)


    with open(hits_path1, 'r', encoding='utf-8') as file1, open(hits_path2, 'r', encoding='utf-8') as file2:
        if os.stat(hits_path1).st_size == 0:
            print(f"File {hits_path1} is empty!")
            return
        print(hits_path1, hits_path2)
        hits_data1 = json.load(file1)
        hits_data2 = json.load(file2)
        hits1_greater = {}
        hits1_less = {}
        hits1_equal = {}
        for id, hits_obj in hits_data1.items():
            hits1_counter = 0
            hits2_counter = 0
            if id in hits_data2 and intersecting_entries[id] is True:
                hits_obj2 = hits_data2[id]
                hits1 = hits_obj['hits']
                hits2 = hits_obj2['hits']
                for i in range(len(hits1)):
                    if hits1[i]["hit"]:
                        hits1_counter += 1
                    if hits2[i]["hit"]:
                        hits2_counter += 1
                entry = {
                    f"{lang1}_hits": hits1_counter,
                    f"{lang2}_hits": hits2_counter,
                    f"{lang1}_true_answer": hits_obj['true_answer'],
                    f"{lang2}_true_answer": hits_obj2['true_answer'],
                    f"{lang1}_answers": hits_obj['answers'],
                    f"{lang2}_answers": hits_obj2['answers'],
                    "question": hits_obj['question'],
                    "answerType": hits_obj['answerType']
                }   
                if hits1_counter > hits2_counter:
                    hits1_greater[id] = entry
                elif hits1_counter < hits2_counter:
                    hits1_less[id] = entry
                else:
                    hits1_equal[id] = entry
        print(f"Entries where {lang1} has more hits than {lang2}: {len(hits1_greater)}")
        print(f"Entries where {lang1} has less hits than {lang2}: {len(hits1_less)}")
        print(f"Entries where {lang1} has equal hits to {lang2}: {len(hits1_equal)}")
        print(f"Total entries compared: {len(hits1_greater) + len(hits1_less) + len(hits1_equal)}")
        # Save the results to JSON files
        with open(f'outputs/hits_lang_comparisons/{lang1}_{lang2}/hits_{lang1}_greater_than_{lang2}.json', 'w', encoding='utf-8') as file:
            json.dump(hits1_greater, file, ensure_ascii=False, indent=4)
        with open(f'outputs/hits_lang_comparisons/{lang1}_{lang2}/hits_{lang1}_less_than_{lang2}.json', 'w', encoding='utf-8') as file:
            json.dump(hits1_less, file, ensure_ascii=False, indent=4)
        with open(f'outputs/hits_lang_comparisons/{lang1}_{lang2}/hits_{lang1}_equal_to_{lang2}.json', 'w', encoding='utf-8') as file:
            json.dump(hits1_equal, file, ensure_ascii=False, indent=4)

                    
            
        
        
    
