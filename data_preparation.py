import pandas as pd
from datasets import load_dataset
import os
import re

DATA_DIR = "data"
PREPROCESSED_FILE = os.path.join(DATA_DIR, "english_quotes_preprocessed.json")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data():
    print("--- Data Preparation ---")
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(PREPROCESSED_FILE):
        print(f"Preprocessed data found. Loading from {PREPROCESSED_FILE}")
        return pd.read_json(PREPROCESSED_FILE, orient='records', lines=True)

    print("Downloading and preprocessing data...")
    dataset = load_dataset("Abirate/english_quotes")
    df = dataset['train'].to_pandas()

    df.dropna(subset=['quote', 'author'], inplace=True)
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])

    df['quote'] = df['quote'].apply(clean_text)
    df['author'] = df['author'].apply(clean_text)
    df['tags'] = df['tags'].apply(lambda tag_list: [clean_text(tag) for tag in tag_list if isinstance(tag, str) and clean_text(tag)])
    
    df.drop_duplicates(subset=['quote', 'author'], keep='first', inplace=True)
    df = df[df['quote'].apply(lambda x: len(x.split()) > 3)] 

    print(f"Number of rows after cleaning: {len(df)}")

    df['combined_text_for_embedding'] = df.apply(
        lambda row: f"Quote: {row['quote']} Author: {row['author']} Tags: {', '.join(row['tags'])}", 
        axis=1
    )

    df.to_json(PREPROCESSED_FILE, orient='records', lines=True)
    print(f"Preprocessed data saved to {PREPROCESSED_FILE}")
    return df

if __name__ == "__main__":
    quotes_df = prepare_data()
    print("\nSample of preprocessed data:")
    if quotes_df is not None: # Add check for None if file was just loaded
        print(quotes_df.head())
    else:
        print("Data preparation did not return a DataFrame.")