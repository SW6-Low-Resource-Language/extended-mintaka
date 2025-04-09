import regex as re

text = "৪টি দেশ ছিল। \\nকনফ "


def replace_entire_word_if_digit(text):
    def extract_and_convert(match):
        word = match.group(0)
        # Extract all Unicode digits from the word and convert them
        digits = re.findall(r'\p{Nd}', word)
        if digits:
            arabic_digits = ''.join([str(int(d)) for d in digits])
            return arabic_digits  # Replace the entire word with just the digits
        return word  # Fallback (shouldn't happen since we match words with digits)
    
    return re.sub(r'\w*\p{Nd}\w*', extract_and_convert, text)

converted = replace_entire_word_if_digit(text)
print(converted)