import os
import re
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Download required NLTK resources (if not already present)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

# Create a set of standard English words (all lowercased)
english_vocab = set(w.lower() for w in words.words())

# Define input and output paths
input_path = r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 2\train-00000-of-00001.parquet"
output_path = r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 2\wiki_preprocessed.txt"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, max_token_length=25):
    """
    Preprocess the input text by:
    - Lowercasing and removing non-ASCII characters
    - Tokenizing
    - Removing stopwords and non-English words
    - Lemmatizing
    - Removing uncommon tokens
    """
    # Lowercase the text
    text = text.lower()
    
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Replace newlines/tabs with a space and remove extra spaces
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Regex pattern: only allow tokens with alphabetical characters
    pattern = re.compile("^[a-z]+$")
    # Filter tokens: must be alphabetic, longer than 2 characters, and not exceed max_token_length
    tokens = [token for token in tokens if pattern.match(token) and 2 < len(token) <= max_token_length]
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Remove tokens that are not in the standard English vocabulary
    tokens = [token for token in tokens if token in english_vocab]
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

# --------------------------------------------------
# Step 1: Read Parquet File
# --------------------------------------------------
df = pd.read_parquet(input_path)

# Ensure the dataset has the expected column
if "text" not in df.columns:
    raise ValueError("Expected 'text' column not found in Parquet file!")

# --------------------------------------------------
# Step 2: Preprocess Text Data
# --------------------------------------------------
all_tokens = []
processed_texts = []

for text in tqdm(df["text"], desc="Processing Wikipedia articles"):
    tokens = preprocess_text(text, max_token_length=15)
    all_tokens.extend(tokens)
    processed_texts.append(" ".join(tokens))  # Store as a single cleaned document

# --------------------------------------------------
# Step 3: Remove Rare Tokens
# --------------------------------------------------
token_freq = Counter(all_tokens)
tokens_to_keep = {token for token, count in token_freq.items() if count > 1}

processed_texts_filtered = [
    " ".join([token for token in text.split() if token in tokens_to_keep])
    for text in processed_texts
]

# --------------------------------------------------
# Step 4: Save Preprocessed Data
# --------------------------------------------------
with open(output_path, "w", encoding="utf-8") as f:
    for text in processed_texts_filtered:
        f.write(text + "\n")

print(f"Preprocessed text saved to: {output_path}")
