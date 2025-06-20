from utils.compare_language_hits import compare_language_hits

languages = ["bn", "da", "fi"]
for lang in languages:
    compare_language_hits("en", lang)