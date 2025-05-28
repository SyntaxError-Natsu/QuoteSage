import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

from data_preparation import prepare_data, DATA_DIR

MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "quotes_faiss.index")
QUOTES_DATA_FILE = os.path.join(DATA_DIR, "quotes_data.pkl")

def create_index_and_save_data(df):
    print("--- Model Finetuning ---")
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(QUOTES_DATA_FILE):
        print("FAISS index and quotes data already exist. Skipping creation.")
        return

    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Creating embeddings for quotes...")
    corpus_embeddings = model.encode(df['combined_text_for_embedding'].tolist(), show_progress_bar=True)
    corpus_embeddings = np.array(corpus_embeddings).astype('float32')

    embedding_dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(corpus_embeddings)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"FAISS index saved to {FAISS_INDEX_FILE}")

    quotes_list = df[['quote', 'author', 'tags']].to_dict(orient='records')
    # Optional: Remove the sample print from here if it's too verbose for final code
    # print("\nSample of quotes_list being saved (first 2 entries):") 
    # for i, item in enumerate(quotes_list[:2]):
    #     print(f"Quote {i}: {item['quote'][:50]}... Author: {item['author']}, Tags: {item['tags']}")
    with open(QUOTES_DATA_FILE, 'wb') as f:
        pickle.dump(quotes_list, f)
    print(f"Quotes data saved to {QUOTES_DATA_FILE}")

def load_indexed_data(): # This function isn't directly used by other scripts, but good utility
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(QUOTES_DATA_FILE):
        print("Error: Index or quotes data not found. Run model_finetuning.py first.")
        return None, None
        
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(QUOTES_DATA_FILE, 'rb') as f:
        quotes_list = pickle.load(f)
    return index, quotes_list

if __name__ == "__main__":
    quotes_df = prepare_data()
    if quotes_df is not None and not quotes_df.empty:
        create_index_and_save_data(quotes_df)
        print("\nEmbedding and Indexing(model tuning) process complete.") # Changed message
    else:
        print("DataFrame is empty. Cannot proceed.")