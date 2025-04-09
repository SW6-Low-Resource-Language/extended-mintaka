import os
from dotenv import load_dotenv
from google.cloud import translate_v3

load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
print(f"Project ID: {PROJECT_ID}")

def chunk_text(text, max_size=20000):
    """
    Splits the input text into chunks, each under the specified maximum size in bytes.

    Args:
        text (str): The input text to be split into chunks.
        max_size (int, optional): The maximum size of each chunk in bytes. Defaults to 30000.

    Returns:
        list: A list of text chunks, each under the specified maximum size.
    """
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) + 1 > max_size:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line if current_chunk else line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def google_translate_line_by_line(lines, output_path, target_language="bn", source_language="en-US"):
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



def google_translate_chunks(input_file, output_file, target_language="bn", source_language="en-US"):
    """
    Translates the content of a large text file using Google Cloud Translation API and saves the translated text to an output file.
    Args:
        input_file (str): The path to the input text file to be translated.
        output_file (str): The path to the output text file where the translated text will be saved.
        target_language (str, optional): The target language code for translation (default is "da" for Danish).
    """
    client = translate_v3.TranslationServiceClient()
    parent = f"projects/{PROJECT_ID}/locations/global"

    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()
    chunks = chunk_text(text)
    translated_chunks = []
    for chunk in chunks:
        response = client.translate_text(
            parent=parent,
            contents=[chunk],
            mime_type="text/plain",
            target_language_code=target_language,
            source_language_code=source_language,
        )
        
        # "Google sometimes returns double newlines in chunks, so we replace them with single newlines"
        translated_text = response.translations[0].translated_text.replace("\n\n", "\n")
        translated_chunks.append(translated_text)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(translated_chunks))

def google_translate_text(text, target_language):
    """
    Translates a text string using the Google Cloud Translation API.
    Args:
        text (str): The text to be translated.
        target_language (str): The target language code for translation.
    Returns:
        str: The translated text.
    """
    client = translate_v3.TranslationServiceClient()
    parent = f"projects/{PROJECT_ID}/locations/global"
    response = client.translate_text(
        parent=parent,
        contents=[text],
        mime_type="text/plain",
        target_language_code=target_language,
        source_language_code="en-US",
    )
    translated_text = response.translations[0].translated_text
    return translated_text
