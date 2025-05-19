from sentence_transformers import SentenceTransformer, CrossEncoder
import json

# Load https://huggingface.co/sentence-transformers/all-mpnet-base-v2

def semantic_similarity(model_answer, dataset_answer):
    # can determine the similarity between two sentences semantically, struggles with negation
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings_model = model.encode([
        model_answer
    ])
    embeddings_dataset = model.encode([
        dataset_answer
    ])
    similarities = model.similarity(embeddings_model, embeddings_dataset)
    return similarities.item()

def natural_language_inference(model_answer, dataset_answer):
    # can determine "not" negation and contradictions among two sentences (in english :C )
    model = CrossEncoder('cross-encoder/nli-deberta-base')
    scores = model.predict([(model_answer), (dataset_answer)])

    print("Scores:", scores)
    #Convert scores to labels
    label_mapping = ['contradiction', 'entailment', 'neutral']
    
    if len(scores.shape) == 1:  # Check if scores is 1D
        labels = [label_mapping[scores.argmax()]]
    else:  # If scores is 2D, use axis=1
        labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

    return labels

# Example usage
if __name__ == "__main__":
    lang = 'da'
    with open("./data/mintaka_test_extended.json" , 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    supporting_ent_map = {}
    for entry in dataset:
        id = entry["id"]
        answer = entry["answer"]
        supporting_ent_map[id] = None
        if "supportingEnt" in answer:
            supporting_ent_map[id] = answer["supportingEnt"]
    
    with open("./output_dk_sprgsml_w_id_and_true_answer.json", 'r', encoding='utf-8') as file:
        answers = json.load(file)
    
    for answer in answers:
        question, answers, id, true_answer = answer['question'], answer['answers'], answer['id'], answer['true_answer']
        supporting_ent = supporting_ent_map[id]
        compare_string = str(true_answer)
        if supporting_ent != None:
            for entity in supporting_ent:
                compare_string += ", " + str(entity['label'][lang])
        best_ans = [0, None]
        for index, ans in enumerate(answers):
            sem_score = semantic_similarity(ans, compare_string)
            if sem_score > best_ans[0]:
                best_ans[0] = sem_score
                best_ans[1] = ans
            print(f"Semantic similarity: {sem_score} - {question} - {ans} - {compare_string}")
        answer["best_answer"] = best_ans
    with open("./sem_test.json", 'w', encoding='utf-8') as file:
        json.dump(answers, file, ensure_ascii=False, indent=4)
            



    """    model_answer = "Brad Pitt er højere end george clooney med 8 centimeter"
        dataset_answer = "Brad pitt"
        similarity = semantic_similarity(model_answer, dataset_answer)
        print(similarity) """
