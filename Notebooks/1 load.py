
import pandas as pd

# Load the dataset
df = pd.read_parquet(r"C:\Users\dalla\Downloads\train-00000-of-00001.parquet")

# Check the first few rows
print(df.head())

# Check available columns
print(df.columns)

# Check for missing values
print(df.isnull().sum())
