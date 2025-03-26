from sentence_transformers import SentenceTransformer, CrossEncoder

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
    model_answer = "Ja det er rigtigt"
    dataset_answer = "Nej det er ikke rigtigt"
    similarity = semantic_similarity(model_answer, dataset_answer)
    print(similarity)
