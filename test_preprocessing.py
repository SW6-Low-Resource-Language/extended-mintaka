from preprocess_data import preprocess_data

filename = './data/mintaka_train_extended2.json'

lang = 'en'
preprocess_data(filename, lang)

lang = 'da'
preprocess_data(filename, lang)

lang = 'fi'
preprocess_data(filename, lang)

lang = 'bn'
preprocess_data(filename, lang)


filename = './data/mintaka_test_extended2.json'
lang = 'en'
preprocess_data(filename, lang)

lang = 'da'
preprocess_data(filename, lang)

lang = 'fi'
preprocess_data(filename, lang)

lang = 'bn'
preprocess_data(filename, lang)