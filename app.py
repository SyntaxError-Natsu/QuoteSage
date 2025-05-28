import streamlit as st
import json
import os
import signal 

from rag_system import query_system, _initialize_resources as initialize_rag_resources, retrieve_relevant_quotes
from model_finetuning import FAISS_INDEX_FILE, QUOTES_DATA_FILE # Ensure this matches your filename
from data_preparation import PREPROCESSED_FILE, DATA_DIR 

st.set_page_config(page_title="Quote Finder", layout="wide")

@st.cache_resource 
def load_app_resources():
    print("Streamlit: Initializing RAG resources for the app...")
    try:
        initialize_rag_resources()
        return True
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure data preparation and indexing scripts have been run successfully.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        return False

def run_streamlit_app():
    st.title("📚 Semantic Quote Finder")
    st.caption("Find quotes using natural language with RAG.")

    if not load_app_resources():
        st.warning("Application resources could not be loaded. Please check the console for errors.")
        return

    user_query = st.text_input("What kind of quotes are you looking for?", 
                               "quotes about courage by women authors")
    
    st.sidebar.header("Options")
    k_slider = st.sidebar.slider("Max quotes for context:", 1, 10, 5, key="k_slider_main")
    show_raw_retrieval = st.sidebar.checkbox("Show raw retrieved quotes & scores", True, key="show_raw_cb")
    
    st.sidebar.markdown("---") 

    if st.sidebar.button("🔴 Exit Application", key="exit_button"):
        st.warning("Shutting down Streamlit application...")
        print("Streamlit application is shutting down via Exit button.")
        os.kill(os.getpid(), signal.SIGTERM)

    if st.button("Search Quotes", key="search_button"):
        if not user_query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching..."):
                rag_result = query_system(user_query, k_retrieval=k_slider)

                st.subheader("💡 Generated Summary (from LLM)")
                st.info(rag_result.get("llm_summary", "No summary available."))

                st.subheader("🗣️ Processed Quotes (selected/formatted by LLM, if successful)")
                processed_quotes = rag_result.get("llm_processed_quotes", [])
                if processed_quotes:
                    for i, item in enumerate(processed_quotes):
                        expander_title = f"LLM Processed Quote {i+1}: {item.get('quote', 'N/A')[:60]}..."
                        with st.expander(expander_title):
                            st.markdown(f"**Quote:** {item.get('quote', 'N/A')}")
                            st.markdown(f"**Author:** {item.get('author', 'N/A')}")
                            tags_list = item.get('tags', [])
                            if tags_list:
                                st.markdown(f"**Tags:** {', '.join(tags_list)}")
                else:
                    st.write("No quotes were specifically processed by the LLM, or LLM fallback occurred.")
                
                if show_raw_retrieval:
                    st.subheader(f"🔍 Raw Retrieved Quotes (Top {k_slider})")
                    # Re-retrieving to get scores; can be optimized by passing scores from query_system
                    raw_docs_for_display, raw_scores_for_display = retrieve_relevant_quotes(user_query, k=k_slider)
                    
                    if raw_docs_for_display:
                        for i, (doc, score) in enumerate(zip(raw_docs_for_display, raw_scores_for_display)):
                            expander_title_raw = f"Retrieved Doc {i+1} (Similarity: {score:.3f}): {doc['quote'][:60]}..."
                            with st.expander(expander_title_raw):
                                st.markdown(f"**Quote:** {doc['quote']}")
                                st.markdown(f"**Author:** {doc['author']}")
                                tags_list_raw = doc.get('tags', [])
                                if tags_list_raw:
                                    st.markdown(f"**Tags:** {', '.join(tags_list_raw)}")
                                st.markdown(f"**Similarity Score:** {score:.3f} (higher is better)")
                    else:
                        st.write("No raw documents retrieved for this query.")

                with st.expander("View Full RAG System JSON Response"):
                    st.json(rag_result)
    else:
        st.info("Enter a query and click 'Search Quotes' to begin.")

if __name__ == "__main__":
    required_data_files = [
        os.path.join(DATA_DIR, "english_quotes_preprocessed.json"),
        FAISS_INDEX_FILE,
        QUOTES_DATA_FILE
    ]
    if not os.path.exists(DATA_DIR) or not all(os.path.exists(f) for f in required_data_files):
        st.error("Core data files are missing! Please run these scripts first in order:")
        st.code("1. python data_preparation.py\n2. python model_finetuning.py")
    else:
        run_streamlit_app()