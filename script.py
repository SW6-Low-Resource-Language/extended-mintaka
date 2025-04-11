from utils.number_proccesor import replace_entire_word_if_digit, is_same_answer
from utils.get_generation_path import get_generation_path
import json
from datetime import datetime
from dateparser import parse
from dateparser.search import search_dates
p = get_generation_path("parsed_answers_json", "zeroshot", "bn")
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)

def extract_date_from_lang(str, lang):
    # Parse the string
    parsed_date = search_dates(str, languages=[lang])
    
    if parsed_date:
        # Format the date in yyyy-mm-dd
        return parsed_date
    else:
        return None  # Return None if parsing fails

text_with_date = "এই ইভেন্টটি ১১ এপ্রিল ২০২৫ তারিখে অনুষ্ঠিত হবে।"  # "This event will take place on 11 April 2025."
date = extract_date_from_lang(text_with_date, "bn")
print(date)  # Output: 2025-04-11
for e in data:
    if e["answerType"] == "date":
        for i, ans in enumerate(e["answers"]):
            dates = extract_date_from_lang(ans, "bn")
            if dates != None:
                for date_text, date_obj in dates:
                    formatted_date = date_obj.strftime('%Y-%m-%d')

