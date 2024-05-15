from transformers import AutoTokenizer, AutoModel
import torch
import re
import spacy
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import stanza
import spacy_stanza 
from stanza.pipeline.core import DownloadMethod
from sklearn.metrics.pairwise import cosine_similarity

nlp_pos = stanza.Pipeline('ro', download_method=DownloadMethod.REUSE_RESOURCES)

poem = ''' 
Plumb
Dormeau adânc sicriele de plumb,
Și flori de plumb și funerar veștmânt -
Stam singur în cavou… și era vânt…
Și scârțâiau coroanele de plumb.

Dormea întors amorul meu de plumb
Pe flori de plumb, și-am început să-l strig -
Stam singur lângă mort… și era frig…
Și-i atârnau aripile de plumb.
'''
poem = poem.replace("...", " ")
pattern = r'\w+|\.\.\.|[.,!?;]'
poem = re.findall(pattern, poem)
poem_words = ["..." if token == "..." else token for token in poem]
poem = ' '.join(poem_words)
#print(poem)
doc = nlp_pos(poem)

filtered_tokens = []

for sentence in doc.sentences:
    for word in sentence.words:
        #print(f"word: {word.text}, pos: {word.pos}") 
        if word.pos not in ['PUNCT', 'CCONJ', 'ADP']:
            filtered_tokens.append(word.text)
print("len of tokens: ", len(filtered_tokens))
print(filtered_tokens)        

#stemmer = SnowballStemmer("romanian")
#nlp = spacy.load("ro_core_news_lg")


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")


# Tokenize the poem
#tokenized_poem = tokenizer(plumb_poem, return_tensors="pt")
inputs = tokenizer(filtered_tokens, return_tensors="pt", padding=True,truncation=True, max_length=128)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
outputs = model(input_ids, attention_mask=attention_mask)

# get encoding
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

print(last_hidden_states)
print("len of last_hidden_states: ", len(last_hidden_states))

# Define a function to calculate cosine similarity between two vectors
def calculate_similarity(embedding1, embedding2):
    embedding1_np = embedding1.squeeze(0).detach().numpy()
    embedding2_np = embedding2.squeeze(0).detach().numpy()
    #return cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))[0][0]
    return cosine_similarity(embedding1_np, embedding2_np)[0][0]
# Calculate the number of related words for each word in the poem
related_words_counts = []
threshold = 0.9  # Define a threshold for similarity
for i, word_embedding in enumerate(last_hidden_states):
    related_word_count = 0
    reference_embedding = word_embedding.unsqueeze(0)  # Reshape the word embedding for cosine similarity calculation
    for j, other_embedding in enumerate(last_hidden_states):
        if i != j:  # Exclude comparing a word with itself
            similarity = calculate_similarity(reference_embedding, other_embedding)
            if similarity > threshold:
                related_word_count += 1
    related_words_counts.append(related_word_count)

print("Number of related words for each word:", related_words_counts)
