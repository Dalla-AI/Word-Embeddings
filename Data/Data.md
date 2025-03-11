Data was taken from Simple English Wikipedi "train-00000-of-00001.parquet" 

Pre-trained Embeddings:

GloVe (glove-wiki-gigaword-100)
FastText (fasttext-wiki-news-subwords-300)

Downloaded using:

python

import gensim.downloader as api
glove_model = api.load("glove-wiki-gigaword-100")
fasttext_model = api.load("fasttext-wiki-news-subwords-300")
