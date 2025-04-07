import json

def hits_at_k(hits_data, k):
    hits_k = 0
    for _, question in hits_data.items():
        question_hits = question['hits']
        for elem in question_hits:
            if elem['hit'] == True and elem['idx'] <= k:
                hits_k += 1
    hits_tested = len(hits_data)*k
    hits_percent = hits_k / hits_tested
    print(f"Hits at {k}: {hits_k} out of {hits_tested} - {hits_percent:.2%}")

def calc_hits_at_ks(hits_data, k):
    for i in range(1, k+1):
        hits_at_k(hits_data, i)

if __name__ == "__main__":
    with open('hits_bn.json', 'r', encoding="utf-8") as f:
        hits_data = json.load(f)
    
    calc_hits_at_ks(hits_data, 5)