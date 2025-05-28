# rag_system.py
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from transformers import pipeline, logging as hf_logging
import json
import os
import re

hf_logging.set_verbosity_error()

# Assuming the file is named model_finetuning.py
from model_finetuning import MODEL_NAME, FAISS_INDEX_FILE, QUOTES_DATA_FILE 

_embedding_model = None
_faiss_index = None
_quotes_data = None
_llm_generator = None

def _initialize_resources():
    global _embedding_model, _faiss_index, _quotes_data, _llm_generator
    
    resources_loaded_this_call = False

    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer(MODEL_NAME)
        resources_loaded_this_call = True
    
    if _faiss_index is None or _quotes_data is None:
        print("Loading FAISS index and quotes data...")
        if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(QUOTES_DATA_FILE):
            raise FileNotFoundError(
                f"FAISS index ({FAISS_INDEX_FILE}) or quotes data ({QUOTES_DATA_FILE}) not found. "
                "Run model_finetuning.py first."
            )
        _faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        with open(QUOTES_DATA_FILE, 'rb') as f:
            _quotes_data = pickle.load(f)
        resources_loaded_this_call = True
            
    if _llm_generator is None:
        print("Loading LLM for generation (GPT-2)...")
        try:
            _llm_generator = pipeline("text-generation", model="gpt2", max_new_tokens=200, device=-1)
        except Exception as e:
            print(f"Warning: Could not load LLM: {e}. Summary generation will be basic.")
            _llm_generator = None
        resources_loaded_this_call = True
    
    if resources_loaded_this_call:
        print("RAG resources initialized/verified.")

def retrieve_relevant_quotes(query_text, k=5):
    _initialize_resources()
    
    query_embedding = _embedding_model.encode([query_text])
    query_embedding = np.array(query_embedding).astype('float32')
    
    distances, indices = _faiss_index.search(query_embedding, k)
    
    retrieved_docs = []
    retrieved_scores = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if 0 <= idx < len(_quotes_data):
            retrieved_docs.append(_quotes_data[idx])
            similarity_score = 1 / (1 + distances[0][i]) if distances[0][i] >= 0 else 0 
            retrieved_scores.append(similarity_score)
    return retrieved_docs, retrieved_scores

def generate_response_with_llm(query, retrieved_docs):
    _initialize_resources()

    if not retrieved_docs:
        return {"summary": "No relevant quotes found to generate a summary.", "processed_quotes": []}

    context_str = ""
    for i, doc in enumerate(retrieved_docs):
        context_str += f"Doc {i+1}: {doc['quote']} (Author: {doc['author']})\n"

    prompt = f"""Based on the following documents:
{context_str}
User Query: "{query}"

Respond in JSON format with two keys: "summary_text" (a brief summary related to the query based ONLY on the provided documents) and "relevant_quotes_from_context" (a list of the most relevant documents you used from the context, each with 'quote', 'author', 'tags').
Example for 'relevant_quotes_from_context' item: {{"quote": "Text.", "author": "Name", "tags": ["tag1"]}}

JSON Output:
"""
    
    if _llm_generator:
        try:
            llm_output_list = _llm_generator(prompt)
            raw_llm_text = llm_output_list[0]['generated_text']
            
            json_match = re.search(r'\{.*\}', raw_llm_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    if "summary_text" in parsed_json and "relevant_quotes_from_context" in parsed_json:
                        return {
                            "summary": parsed_json["summary_text"],
                            "processed_quotes": parsed_json["relevant_quotes_from_context"]
                        }
                    else:
                        print("LLM JSON missing required keys. Falling back.")
                except json.JSONDecodeError:
                    print(f"LLM output was not valid JSON. Falling back. Output snippet: {json_str[:100]}")
            else:
                print("No JSON object found in LLM output. Falling back.")
        except Exception as e:
            print(f"Error during LLM generation: {e}. Falling back.")
    
    return {
        "summary": "Summary generation failed or LLM not available. Displaying raw retrieved quotes.",
        "processed_quotes": retrieved_docs
    }

def query_system(user_query, k_retrieval=5):
    print(f"\nProcessing query: '{user_query}'")
    retrieved_docs, _ = retrieve_relevant_quotes(user_query, k=k_retrieval)
    
    llm_response_data = generate_response_with_llm(user_query, retrieved_docs)
    
    final_response = {
        "query": user_query,
        "llm_summary": llm_response_data.get("summary"),
        "llm_processed_quotes": llm_response_data.get("processed_quotes"),
        "raw_retrieved_for_context": retrieved_docs
    }
    return final_response

if __name__ == "__main__":
    _initialize_resources()
    test_queries = [
        "quotes about hope by Oscar Wilde",
        "Einstein on insanity",
        "motivational accomplishment quotes"
    ]
    for q in test_queries:
        response = query_system(q, k_retrieval=3)
        print("\n--- RAG System Response ---")
        print(json.dumps(response, indent=2))
        print("---------------------------\n")