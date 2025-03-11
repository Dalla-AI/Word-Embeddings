from gensim.models import Word2Vec
import gensim.downloader as api

# Load trained models
cbow_model = Word2Vec.load(r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 2\cbow.model")
skipgram_model = Word2Vec.load(r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 2\skipgram.model")

print("Trained embeddings loaded successfully!")

# Load pre-trained GloVe embeddings (100-dimensional)
glove_model = api.load("glove-wiki-gigaword-100")
print("GloVe embeddings loaded!")

# Load pre-trained FastText embeddings (300-dimensional) without a limit
fasttext_model = api.load("fasttext-wiki-news-subwords-300")
print("FastText embeddings loaded!")

# Trim FastText model's vocabulary to the first 200,000 words
limit = 200000
fasttext_model.wv.vectors = fasttext_model.wv.vectors[:limit]
fasttext_model.wv.index_to_key = fasttext_model.wv.index_to_key[:limit]
fasttext_model.wv.key_to_index = {word: idx for idx, word in enumerate(fasttext_model.wv.index_to_key)}
print(f"FastText model trimmed to {limit} words.")

# Function to compare embeddings
def compare_embeddings(word, models):
    results = {}
    for name, model in models.items():
        try:
            results[name] = model.most_similar(word, topn=5)
        except KeyError:
            results[name] = "Word not in vocabulary"
    return results

# Define test words
test_words = ["king", "computer", "Paris", "dog", "doctor"]

# Dictionary of models
models = {
    "CBOW": cbow_model.wv,
    "Skip-gram": skipgram_model.wv,
    "GloVe": glove_model,
    "FastText": fasttext_model.wv
}

# Run comparisons
for word in test_words:
    print(f"\n### Most similar words to '{word}' ###")
    results = compare_embeddings(word, models)
    for model_name, similar_words in results.items():
        print(f"{model_name}: {similar_words}")
