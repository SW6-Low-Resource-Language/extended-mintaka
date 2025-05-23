import json

# Prepare a list to store stats for each language
stats = []

for lang in ["da", "bn", "fi", "en"]:
    with open(f"outputs\\parsed_LLM_output_jsons\\output_zeroshot_questions_w_id_and_true_answer_{lang}.json", encoding="utf-8") as f:
        data = json.load(f)
        q_lengths = 0
        q_min = 100
        q_max = 0
        q_min_q = ""
        q_max_q = ""
        a_lengths = 0
        a_min = 100
        a_max = 0
        a_min_ans = ""
        a_max_ans = ""
        for entry in data:
            question = entry["question"]
            q_word_count = len(question.split())
            q_lengths += q_word_count
            if q_word_count < q_min:
                q_min = q_word_count
                q_min_q = question
            if q_word_count > q_max:
                q_max = q_word_count
                q_max_q = question

            for ans in entry["answers"]:
                word_count = len(ans.split())
                a_lengths += word_count
                if word_count < a_min:
                    a_min = word_count
                    a_min_ans = ans
                if word_count > a_max:
                    a_max = word_count
                    a_max_ans = ans
        stats.append({
            "lang": lang,
            "q_avg": q_lengths / len(data),
            "q_min": q_min,
            "q_max": q_max,
            "a_avg": a_lengths / (len(data) * 5),
            "a_min": a_min,
            "a_max": a_max,
        })

# Print table header
header = [
    "Lang", "Q_Avg", "Q_Min", "Q_Max", "A_Avg", "A_Min", "A_Max"
]
print("{:<6} {:<8} {:<7} {:<7} {:<8} {:<7} {:<7}".format(*header))

# Print table rows
for s in stats:
    print("{:<6} {:<8.2f} {:<7} {:<7} {:<8.2f} {:<7} {:<7}".format(
        s["lang"], s["q_avg"], s["q_min"], s["q_max"], s["a_avg"], s["a_min"], s["a_max"]
    ))

