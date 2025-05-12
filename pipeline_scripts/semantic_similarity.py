from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
import json

# Load https://huggingface.co/sentence-transformers/all-mpnet-base-v2

# can determine the similarity between two sentences semantically, struggles with negation


""" def semantic_similarity_handler(dataset_answer, model_answers): 
    embeddings_dataset = model.encode([
        dataset_answer])
    sem_scores = []
    for model_answer in model_answers:
        embeddings_model = model.encode([
            model_answer])
        similarities = model.similarity(embeddings_model, embeddings_dataset)
        sem_scores.append(similarities.item())
    return sem_scores 
     """
def semantic_similarity_handler(model, dataset_answer, model_answers): 
    # Encode the dataset answer
    embeddings_dataset = model.encode([dataset_answer], convert_to_tensor=True)
    # Encode all model answers as a batch
    embeddings_model = model.encode(model_answers, convert_to_tensor=True)
    # Compute cosine similarity between the dataset embedding and all model embeddings
    similarities = cos_sim(embeddings_dataset, embeddings_model)
    # Convert the similarities to a list of scores
    sem_scores = similarities.squeeze(0).tolist()
    return sem_scores


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

lang = 'bn'
dataset_input_json_path = "./data/mintaka_test_extended.json"
model_answer_json_path = f"./output_{lang}_sprgsml_w_id_and_true_answer.json"
true_label = "হ্যাঁ"
false_label = "না"
output_json_path = f"sem_scores_{lang}.json"

def perform_semantic_similarity(
lang = lang, 
dataset_input_json_path = dataset_input_json_path, 
model_answer_json_path = model_answer_json_path, 
true_label = true_label, 
false_label = false_label,
output_json_path = output_json_path):
    """
    Performs semantic similarity analysis between model answers and dataset answers.
    This function loads a dataset and model answers, computes semantic similarity scores,
    and saves the results to a JSON file.
    Args:
        lang (str): Language code used for processing.
        dataset_input_json_path (str): Path to the JSON file containing the dataset.
        model_answer_json_path (str): Path to the JSON file containing model answers.
        true_label (str): Label representing a positive answer.
        false_label (str): Label representing a negative answer.
        output_json_path (str): Path to save the output JSON file with semantic scores.
    Returns:
    None: The function writes the semantic scores to the specified output JSON file.
    Notes:
        - The function assumes that the dataset contains a field "supportingEnt" for supporting entities.
        - The function uses the SentenceTransformer model to compute semantic similarity scores.
        - The results are saved in a JSON file with the specified output path.
        - The function prints the progress of processing each answer.
    """
    model_name = "all-mpnet-base-v2"
    print(f"Loading sentence transformer model: {model_name} into memory for semantic similarity...")
    model = SentenceTransformer("all-mpnet-base-v2")
    with open(dataset_input_json_path , 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    supporting_ent_map = {}
    for entry in dataset:
        id = entry["id"]
        answer = entry["answer"]
        supporting_ent_map[id] = None
        if "supportingEnt" in answer:
            supporting_ent_map[id] = answer["supportingEnt"]

    with open(model_answer_json_path, 'r', encoding='utf-8') as file:
        answers = json.load(file)
    a_len = len(answers)
    for index, answer in enumerate(answers):
        question, model_answers, id, true_answer = answer['question'], answer['answers'], answer['id'], answer['true_answer']
        supporting_ent = None
        if true_answer is True:
            true_answer = true_label
        elif true_answer is False:
            true_answer = false_label
        if id != None:
            supporting_ent = supporting_ent_map[id]
        compare_string = str(true_answer)
        if supporting_ent != None:
            for entity in supporting_ent:
                compare_string += ", " + str(entity['label'][lang])
        sem_scores = semantic_similarity_handler(model, compare_string, model_answers)

        answer["compare_string"] = compare_string
        print(f"Semantic similarity: {question} - {compare_string} - {sem_scores}")
        answer["sem_scores"] = sem_scores
    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(answers, file, ensure_ascii=False, indent=4)
            


if __name__ == "__main__":
    lang = 'bn'
    dataset_input_json_path = "./data/mintaka_test_extended.json"
    model_answer_json_path = f"./output_{lang}_sprgsml_w_id_and_true_answer.json"
    true_label = "হ্যাঁ"
    false_label = "না"
    output_json_path = f"sem_scores_{lang}.json"
    perform_semantic_similarity(
        lang=lang, 
        dataset_input_json_path=dataset_input_json_path, 
        model_answer_json_path=model_answer_json_path, 
        true_label=true_label, 
        false_label=false_label,
        output_json_path=output_json_path
    )
    # with open(model_answer_json_path, 'r', encoding='utf-8') as file:
    #     answers = json.load(file)
    # for answer in answers:
    #     print(answer["sem_scores"])

