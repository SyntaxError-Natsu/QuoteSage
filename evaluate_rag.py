from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision

from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings as LangchainHuggingFaceEmbeddings

from transformers import pipeline as hf_transformers_pipeline 

import os
import json
import logging
import pandas as pd
import rag_system 

from rag_system import query_system, _initialize_resources as initialize_rag_resources 
from data_preparation import DATA_DIR

# Configure basic logging - INFO for general, WARNING for RAGAS to reduce noise
logging.basicConfig(level=logging.INFO)
ragas_logger = logging.getLogger("ragas")
ragas_logger.setLevel(logging.WARNING) 

EVAL_RESULTS_FILE = os.path.join(DATA_DIR, "ragas_evaluation_final_attempt_results.json")

def run_evaluation():
    print("--- RAG Evaluation with RAGAS (Minimal Configuration) ---")
    initialize_rag_resources() 

    RAGAS_CRITIC_INFO = "Main RAG GPT-2" 
    print(f"Using {RAGAS_CRITIC_INFO} as RAGAS critic LLM.")
    try:
        if rag_system._llm_generator is None:
            raise ValueError("Main RAG LLM (gpt2) from rag_system._llm_generator is not loaded.")
        ragas_langchain_llm = HuggingFacePipeline(pipeline=rag_system._llm_generator)
        print(f"RAGAS critic LLM configured using {RAGAS_CRITIC_INFO}.")
    except Exception as e:
        print(f"ERROR: Could not configure RAGAS critic LLM: {e}")
        return

    print("Configuring RAGAS embeddings (all-MiniLM-L6-v2)...")
    try:
        ragas_embeddings = LangchainHuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        print("RAGAS embeddings configured.")
    except Exception as e:
        print(f"ERROR: Could not load RAGAS embeddings: {e}")
        return

    eval_queries = [
        "Quotes about insanity attributed to Einstein",
        "Motivational quotes tagged 'accomplishment'",
    ]
    print(f"Using {len(eval_queries)} queries for RAGAS evaluation.")

    questions_for_ragas = []
    answers_for_ragas = []
    contexts_for_ragas = []
    ground_truth_answers_for_ragas = [] 

    print("Generating RAG outputs for evaluation queries...")
    for query in eval_queries: 
        print(f"Processing query for RAGAS: {query}")
        rag_output = query_system(query, k_retrieval=3) 
        
        generated_answer = rag_output.get("llm_summary", "No summary available.")
        questions_for_ragas.append(query)
        answers_for_ragas.append(generated_answer)
        ground_truth_answers_for_ragas.append(
            generated_answer if generated_answer != "No summary available." else "Placeholder answer due to summary failure."
        ) 

        retrieved_docs_quotes = [doc['quote'] for doc in rag_output.get("raw_retrieved_for_context", [])]
        if not retrieved_docs_quotes:
            retrieved_docs_quotes = ["No context documents retrieved."]
        contexts_for_ragas.append(retrieved_docs_quotes)

    eval_dataset_dict = {
        "question": questions_for_ragas,
        "answer": answers_for_ragas,
        "contexts": contexts_for_ragas,
        "ground_truth": ground_truth_answers_for_ragas 
    }
    eval_dataset = Dataset.from_dict(eval_dataset_dict)

    metrics_to_run = [context_precision]
    
    print(f"Running RAGAS evaluation with {RAGAS_CRITIC_INFO} as critic (1 metric)...")
    
    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics_to_run,
            llm=ragas_langchain_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False 
        )
        print("RAGAS evaluation call completed.")
    except Exception as e:
        print(f"ERROR during RAGAS evaluate call: {e}")
        import traceback
        traceback.print_exc()
        return

    if results is None:
        print("RAGAS evaluation returned None. Cannot proceed with results processing.")
        return
        
    results_df = results.to_pandas()
    
    columns_to_display = ['question', 'answer'] 
    if 'ground_truth' in results_df.columns: columns_to_display.append('ground_truth')
    for metric in metrics_to_run:
      if metric.name in results_df.columns:
        columns_to_display.append(metric.name)
    
    if 'contexts' in results_df.columns:
        results_df['contexts_preview'] = results_df['contexts'].apply(
            lambda x: [str(c)[:70] + "..." for c in x] if isinstance(x, list) else str(x)[:70]+"..."
        )
        if 'contexts_preview' not in columns_to_display : columns_to_display.append('contexts_preview')

    print(f"\n--- RAGAS Evaluation Results (Using {RAGAS_CRITIC_INFO} as Critic - Unreliable Scores) ---")
    existing_columns_to_display = [col for col in columns_to_display if col in results_df.columns]
    
    if not results_df.empty and existing_columns_to_display:
        print(results_df[existing_columns_to_display])
    elif not results_df.empty:
        print("Results DataFrame is not empty, but standard columns for display might be missing. Printing all available columns:")
        print(results_df)
    else:
        print("Results DataFrame is empty.")
        print("Raw RAGAS results object:", results)

    if not results_df.empty:
        results_df.to_json(EVAL_RESULTS_FILE, orient='records', indent=2)
        print(f"\nEvaluation results saved to {EVAL_RESULTS_FILE}")

        print("\nAverage Scores (interpret with caution):")
        for metric in metrics_to_run:
            if metric.name in results_df.columns:
                if pd.api.types.is_numeric_dtype(results_df[metric.name]):
                    avg_score = results_df[metric.name].mean()
                    if pd.notna(avg_score):
                        print(f"  Average {metric.name}: {avg_score:.4f}")
                    else:
                        print(f"  Average {metric.name}: NaN (Likely due to metric calculation failures)")
                else:
                    print(f"  Metric {metric.name} is not numeric. Sample values: {results_df[metric.name].unique()[:3]}")
    else:
        print("No results to save or calculate averages from.")

    print(f"\nNote: Using {RAGAS_CRITIC_INFO} as critic. RAGAS scores primarily demonstrate pipeline execution, not reliable quality assessment.")

if __name__ == "__main__":
    run_evaluation()
    print("\nEvaluation script finished.")