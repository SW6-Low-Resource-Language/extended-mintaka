import os
from dotenv import load_dotenv
from google.cloud import translate_v3

def google_translate_line_by_line(lines, output_path, target_language="bn", source_language="en-US"):
    load_dotenv()
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    """
    Translates the content of a large text file line by line using Google Cloud Translation API and saves the translated text to an output file.
    Args:
        input_file (str): The path to the input text file to be translated.
        output_file (str): The path to the output text file where the translated text will be saved.
        target_language (str, optional): The target language code for translation (default is "da" for Danish).
    """
    client = translate_v3.TranslationServiceClient()
    parent = f"projects/{PROJECT_ID}/locations/global"
 
    translated_lines = []
    count = 0
    lines_len = len(lines)
    for line in lines:
        print(f"Translating line {count + 1} of {lines_len}")
        count += 1
        response = client.translate_text(
            parent=parent,
            contents=[line],
            mime_type="text/plain",
            target_language_code=target_language,
            source_language_code=source_language,
        )
        translated_text = response.translations[0].translated_text.replace("\n", "")

        translated_lines.append(translated_text)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(translated_lines))
    return translated_lines




