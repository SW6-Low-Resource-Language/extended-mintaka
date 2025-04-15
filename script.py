from pipeline_scripts.hits_stats import calc_hits_at_ks
import json

with open(r"outputs\hits_jsons\outputs_on_unprocessed_data\hits_zeroshot_da.json", "r", encoding="utf-8") as f:
    hits = json.load(f)

calc_hits_at_ks(hits, 5)