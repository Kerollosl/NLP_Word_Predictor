import random
from functions import *

FILE = './bible.txt'
SPLIT = '.'
RAND_SAMPLES = 500

corpus, tokenizer = fit_tokenizer_on_corpus(FILE, SPLIT)
total_words = len(tokenizer.word_index) + 1

print(f"There are {len(corpus)} sentences in this text\n")
print(f"First sentence in corpus: {corpus[0]}")
print(f"First text tokenized sequence: {tokenizer.texts_to_sequences([corpus[0]])[0]}")

sample_sequences = n_gram_seqs(corpus[0:4], tokenizer)

# For each tokenized sequence, create all possible sequences from 2 words to n words
print("n_gram sequences for next 5 examples look like this:\n")
print(sample_sequences)

# Apply the n_gram_seqs transformation to the whole corpus
input_sequences = n_gram_seqs(corpus, tokenizer)

max_sequence_len = max([len(x) for x in input_sequences])
print(f"\nTotal number of n gram sequences: {len(input_sequences)}")
print(f"Max sequence length: {max_sequence_len}\n")

# Show first 5 n gram sequences
sample_padded_sequences = pad_seqs(sample_sequences, max([len(s) for s in sample_sequences]))
print(f"Sample Padded Sequences: {sample_padded_sequences}")

# Pad the whole corpus
input_sequences = pad_seqs(input_sequences, max_sequence_len)
print(f"padded corpus has shape: {input_sequences.shape}")

# Take a random n number of sequences to train with
random.shuffle(input_sequences)
random_elements = input_sequences[:RAND_SAMPLES]

# Get input sequences and output labels
features, labels = features_and_labels(random_elements, total_words)
print(f"features have shape: {features.shape}")
print(f"labels have shape: {labels.shape}")

# Create and train model
model = create_model(total_words, max_sequence_len)
history = model.fit(features, labels, epochs=15, verbose=1)
model.save('./model.h5')

plot_metrics(history)
