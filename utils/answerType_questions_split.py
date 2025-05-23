from collections import Counter
import json
datasets = ["test"]
Langs = ["en", "da", "bn", "fi"]

cnt = {}

for d_set in datasets:
    with open (f"data/mintaka_{d_set}_extended2.json", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            ansType = entry["answer"]["answerType"]
            if ansType == "entity":
               if "entity" not in cnt:
                     cnt["entity"] = {}
               answer = entry["answer"]["answer"]
               if answer != None: 
                     labels = answer[0]["label"]
                     for lang in Langs:
                         if labels[lang] != None:
                            if lang not in cnt["entity"]:
                                cnt["entity"][lang] = 0
                            cnt["entity"][lang] += 1
            else:
                if ansType not in cnt:
                    cnt[ansType] = 0
                cnt[ansType] += 1
print(cnt)