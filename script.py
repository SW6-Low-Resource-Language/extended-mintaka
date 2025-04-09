from utils.google_translation import google_translate_line_by_line, google_translate_chunks

def get_file_lines_as_array(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]  # Strip newline characters
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Specify the path to the file
file_path = "answers_for_translation.txt"
lines_array = get_file_lines_as_array(file_path)
#print(lines_array)
#print(len(lines_array))

google_translate_line_by_line(lines_array, "translated_answers2.txt", target_language="en-US", source_language="bn")