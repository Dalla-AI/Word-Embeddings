import gensim
from gensim.models import Word2Vec

# Load preprocessed text file
with open(r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 2\wiki_preprocessed.txt", "r", encoding="utf-8") as f:
    sentences = [line.split() for line in f.readlines()]  # Tokenize each line into a list of words

print(f"Loaded {len(sentences)} sentences for training.")

# Skip Gram training
skipgram_model = Word2Vec(sentences, vector_size=100, window=5, sg=1, min_count=5, workers=4)
skipgram_model.save(r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 2\skipgram.model")
print("Skip-gram model saved!")

# CBOW training
cbow_model = Word2Vec(sentences, vector_size=100, window=5, sg=0, min_count=5, workers=4)
cbow_model.save(r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 2\cbow.model")
print("CBOW model saved!")
