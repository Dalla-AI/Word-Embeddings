# NLP Word Embeddings Project

## Project Overview
This project explores **word embeddings**, specifically:
- Training embeddings (CBOW & Skip-gram) on Simple English Wikipedia.
- Comparing trained embeddings to pre-trained embeddings (GloVe, FastText).
- Quantifying biases using vector arithmetic and RNSB.
- Evaluating embeddings as features for text classification.

---

## 📁 **Project Structure**
- `data/`: Original and preprocessed datasets.
- `models/`: Trained embeddings (CBOW, Skip-gram).
- `notebooks/`: Jupyter notebooks for each project step.
- `requirements.txt`: Python dependencies.
- `report.pdf`: Final detailed project report.

---

## 📖 **Dataset**
- Source: [Simple English Wikipedia](https://huggingface.co/datasets/wikipedia)
- File used: `train-00000-of-00001.parquet`
- Preprocessed file: `wiki_preprocessed.txt`

---

## 🛠 **Installation**
Run the following command in your Python environment (use a virtual environment or Google Colab):
```bash
pip install -r requirements.txt
