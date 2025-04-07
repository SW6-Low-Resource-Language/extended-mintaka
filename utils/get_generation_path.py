import json

def get_generation_path(key, mode = None, lang = None):
    """
    Get the path to the generation file based on the provided key, mode, and language.

    Args:
        key (str): The key to identify the generation file.
        mode (str): The mode of the generation (e.g., "zeroshot", "finetune").
        lang (str): The language code (e.g., "en", "da").

    Returns:
        str: The path to the generation file.
    """
    with open('configurations\generation_paths.json', 'r', encoding='utf-8') as file:
        generations_paths = json.load(file)
    template = generations_paths[key]["path"]
    if mode is not None:
        template = template.replace("MODE", mode)
    if lang is not None:
        template = template.replace("LANG", lang)

    return template