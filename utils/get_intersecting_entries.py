import json
"""[
    {
        "id": "9ace9041",
        "question": "What is the fourth book in the Twilight series?",
        "translations": {
            "ar": "ما عنوان الكتاب الرابع في سلسلة \"الشفق\"؟",
            "de": "Welches ist das vierte Buch in der Twilight-Saga-Serie?",
            "ja": "トワイライトシリーズの四番目の本はなんですか？",
            "hi": "Twilight सीरीज की चौथी किताब कौन सी है?",
            "pt": "Qual é o quarto livro da série Crespúsculo?",
            "es": "¿Cuál es el cuarto libro de la saga Crepúsculo?",
            "it": "Qual è il quarto libro della serie Twilight?",
            "fr": "Quel est le quatrième livre de la série Twilight ?",
            "da": "Hvad er den fjerde bog i Twilight-serien?",
            "bn": "টোয়াইলাইট সিরিজের চতুর্থ বই কোনটি?",
            "fi": "Mikä on Twilight-sarjan neljäs kirja?"
        },
        "questionEntity": [
            {
                "name": "Q44523",
                "entityType": "entity",
                "label": "Twilight",
                "mention": "Twilight",
                "span": [
                    31,
                    39
                ]
            },
            {
                "name": 4,
                "entityType": "ordinal",
                "mention": "fourth",
                "span": [
                    12,
                    18
                ]
            }
        ],"""



def get_intersecting_entries(mintaka_data, languages = ["da", "bn"]): 
    """
    Get the intersecting entries for the specified languages from the mintaka data.
    Args:
        mintaka_data (list): The mintaka data loaded from the JSON file.
        languages (list): The list of languages to check for intersecting entries.
    Returns:
        dict: A dictionary containing the intersecting entries for the specified languages.
    """
    overlapping_dict = {}
    for entry in mintaka_data:
        id = entry['id']
        answerType = entry['answer']['answerType']
        if answerType != 'entity':
            overlapping_dict[id] = True
        elif entry['answer']['answer'] == None:
            overlapping_dict[id] = False
        else: 
            print(f"Processing {id}")
            # Check if the answer is a list of entities
            label_dict = entry['answer']['answer'][0]['label']
            for lang in languages:
                if lang not in label_dict or label_dict[lang] == None:
                    overlapping_dict[id] = False
                    break
            else:
                overlapping_dict[id] = True
    return overlapping_dict


if __name__ == "__main__":
    # Load the mintaka data
    with open("./data/mintaka_test_extended.json", 'r', encoding='utf-8') as file:
        mintaka_data = json.load(file)

    # Get the intersecting entries for the specified languages
    intersecting_entries = get_intersecting_enries(mintaka_data, languages=["da", "bn"])
    with open("intersecting_entries.json", 'w', encoding='utf-8') as file:
        json.dump(intersecting_entries, file, ensure_ascii=False, indent=4)
    print(intersecting_entries)


