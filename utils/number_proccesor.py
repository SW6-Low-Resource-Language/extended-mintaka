import regex as re

text = "৪টি দেশ ছিল। \\nকনফ "


def replace_entire_word_if_digit(text):
    """
    Replaces any word in the input text that contains Unicode digits with its corresponding Arabic numeral representation.

    This function scans the input text for words containing Unicode digits (e.g., Bengali, Arabic, or other numeral systems) and replaces the entire word with the Arabic numeral equivalent. Words without digits remain unchanged.

    Args:
        text (str): The input text containing words with Unicode digits.

    Returns:
        str: The modified text where words with Unicode digits are replaced by their Arabic numeral equivalents.
    """
    def extract_and_convert(match):
        word = match.group(0)
        # Extract all Unicode digits from the word and convert them
        digits = re.findall(r'\p{Nd}', word)
        if digits:
            arabic_digits = ''.join([str(int(d)) for d in digits])
            return arabic_digits  # Replace the entire word with just the digits
        return word  # Fallback (shouldn't happen since we match words with digits)
    
    return re.sub(r'\w*\p{Nd}\w*', extract_and_convert, text)

def is_same_answer(answer, mod_answer):
    """
    Check if the answer and mod_answer are "same" based on specific conditions.

    Args:
        answer (str): The original answer string.
        mod_answer (str): The modified answer string.

    Returns:
        bool: True if the answer and mod_answer are "same", False otherwise.
    """
    ans_split = answer.split(" ")
    mod_split = mod_answer.split(" ")

    for i in range(len(ans_split)):
        if ans_split[i] != mod_split[i] and "%" not in ans_split[i] and "\\u" not in ans_split[i]:
            return False
    return True