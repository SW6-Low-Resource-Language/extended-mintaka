# import json

#https://pykeen.readthedocs.io/en/stable/api/pykeen.metrics.ranking.HitsAtK.html#pykeen.metrics.ranking.HitsAtK

def isHit(recommendation, truth):
    if truth in recommendation:
        return 1
    return 0

def hits_at_k(k, recommendations, truth):
    hits = 0
    for i in range(len(recommendations)):
        if (recommendations[i]['idx'] < k):
            hits += isHit(recommendations[i]['answer'], truth)
    return (hits, k)
