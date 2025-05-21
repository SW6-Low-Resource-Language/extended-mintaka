
import json
for lang in ["da", "bn", "fi", "en"]:
    with open(f"outputs\parsed_LLM_output_jsons\output_zeroshot_questions_w_id_and_true_answer_{lang}.json", encoding="utf-8") as f:
        data = json.load(f)
        lengths = 0
        for entry in data:
            for ans in entry["answers"]:
                lengths += len(ans)
        print(f"Average length of answers in {lang}: {lengths / (len(data) * 5)}")
