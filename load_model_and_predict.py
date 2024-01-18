import numpy as np
from tensorflow.keras.models import load_model
from functions import n_gram_seqs, fit_tokenizer_on_corpus
from tensorflow.keras.preprocessing.sequence import pad_sequences

FILE = './bible.txt'
SPLIT = '.'
model = load_model('./model.h5')
seed_text = input("Start the NLP word prediction with seed text: ")

while True:
    try:
        next_words = int(input("How many next words would you like the model to predict: "))
        break
    except ValueError:
        print("ERROR: You must enter an integer value for the number of next words to predict. Try again.\n")
print('\n')

corpus, tokenizer = fit_tokenizer_on_corpus(FILE, SPLIT)
input_sequences = n_gram_seqs(corpus, tokenizer)
max_sequence_len = max([len(x) for x in input_sequences])

for i in range(next_words):
    # Convert the text into sequences
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    # Pad the sequences
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    # Get the probabilities of predicting a word
    predicted = model.predict(token_list, verbose=0)
    # Choose the next word based on the maximum probability
    predicted = np.argmax(predicted, axis=-1).item()
    # Get the actual word from the word index
    output_word = tokenizer.index_word[predicted]
    # Append to the current text

    if (i+1) % 15 == 0:
        seed_text += '\n' + output_word
    else:
        seed_text += " " + output_word

print(seed_text)
