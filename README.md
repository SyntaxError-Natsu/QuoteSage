# RAG-Based Semantic Quote Retrieval & Structured QA

This project implements a Retrieval Augmented Generation (RAG) system for semantic quote retrieval and answering questions based on a dataset of English quotes. The system adapts a sentence embedding model, indexes quotes for efficient retrieval, uses a local Large Language Model (LLM) for generation, evaluates the pipeline, and provides an interactive Streamlit application.

## 📋 **Core Features:**
*   **Semantic Quote Retrieval:** Finds quotes based on natural language queries.
*   **RAG Pipeline:** Combines a retriever (adapted sentence model + FAISS) with a generator (local LLM).
*   **Model Adaptation:** Uses a pre-trained sentence transformer adapted for the quote dataset.
*   **Structured Output:** Aims to provide JSON-like output with quotes, authors, tags, and a summary.
*   **Interactive UI:** A Streamlit application for easy querying and result visualization.
*   **RAG Evaluation:** Performance assessed using the RAGAS framework.

---

## 🎞️ Short Video Walkthrough

*   **[Watch on Youtube](https://youtu.be/Xtk4wuafZk4) - Full feature walkthrough**
*   **Summary:** This video provides a comprehensive demonstration of the RAG-Based Semantic Quote Retrieval (QuoteSage) project. It covers the project setup, data preparation for the quote dataset, the "fine-tuning" process (embedding generation and FAISS indexing), an overview of the RAG evaluation with RAGAS, and an interactive session with the Streamlit application showcasing how to query for quotes and interpret the results. Key features, system architecture, design choices, and project challenges are also highlighted.
 
---

## 🚀 **Quick Start & Setup**

**1. Prerequisites:**
*   Python 3.8+
*   Git (for cloning, if applicable)

**2. Environment Setup:**
```
# Clone the repository (if you haven't already)
# git clone https://github.com/SyntaxError-Natsu/QuoteSage.git
# cd QuoteSage

# Create and activate a Python virtual environment
python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
*Note: `faiss-cpu` is used for CPU-based indexing. For GPU support, consider `faiss-gpu` (may require CUDA setup).*

**3. Data Preparation & Model Indexing (Run in order):**

   **a. Prepare Data:** Downloads, cleans, and preprocesses the `Abirate/english_quotes` dataset.
   ```
   python data_preparation.py
   ```
   *Output: Creates `data/english_quotes_preprocessed.json`.*

   **b. "Fine-Tune" Model (Embed & Index Quotes):** Generates embeddings for quotes using `all-MiniLM-L6-v2` and builds a FAISS index.
   ```
   python model_finetuning.py
   ```
   *Outputs: `data/quotes_faiss.index` and `data/quotes_data.pkl`. This may take a few minutes.*

**4. Run RAG Evaluation (Optional, but Recommended):**
   Evaluates the RAG system using RAGAS. This can be time-consuming (10-30+ minutes).
   ```
   python evaluate_rag.py
   ```
   *Output: Prints metrics to console and saves results to `data/ragas_evaluation_results.json`.*

**5. Launch the Streamlit Application:**
   ```
   streamlit run app.py
   ```
   Open your browser and navigate to the URL provided (usually `http://localhost:8501`).

---

## 🏗️ **Project Structure**
```
.
├── data/                     # Stores datasets, index, evaluation results
├── screenshots/              # Contains screenshots of the application
├── venv/                     # Virtual environment (if named 'venv')
├── app.py                    # Streamlit application
├── data_preparation.py       # Script for data downloading and preprocessing
├── model_finetuning.py       # Script for embedding generation and FAISS indexing
├── rag_pipeline.py           # Core RAG logic (retrieval and generation)
├── evaluate_rag.py           # Script for RAG evaluation using RAGAS
├── requirements.txt          # Python dependencies
└── README.md                 # This documentation file
```
*(The `data/` directory and its contents will be created when you run the scripts. Create the `screenshots/` directory manually if it doesn't exist.)*

---

<h2>🌐 Streamlit Application Interface</h2>

<p>The Streamlit app provides a user-friendly interface for interacting with the RAG system.</p>

<ul>
    <li>
        <p><strong>Home Page / Main Interface:</strong> Users can input natural language queries. Options to control retrieval (max quotes) and display raw results are available in the sidebar.</p>
        <p align="center">
            <img src="./screenshots/Home Page.png" alt="Streamlit App - Home Page and Query Input" width="700"/>
        </p>
    </li>
    <li>
        <p><strong>Raw Retrieved Quotes:</strong> If enabled (via sidebar checkbox), this section shows the top quotes directly retrieved by the FAISS index along with their similarity scores, before any LLM processing.</p>
        <p align="center">
            <img src="./screenshots/Raw Retrieved Quotes.png" alt="Streamlit App - Raw Retrieved Quotes and Scores" width="700"/>
        </p>
    </li>
    <li>
        <p><strong>LLM Processed Quotes & Summary:</strong> Displays the summary generated by the LLM and any quotes it specifically selected or reformatted from the retrieved context. These are presented in expandable sections.</p>
        <p align="center">
            <img src="./screenshots/Processed Quotes.png" alt="Streamlit App - LLM Processed Quotes and Summary" width="700"/>
        </p>
    </li>
    <li>
        <p><strong>Full RAG JSON Response:</strong> An expandable section that shows the complete JSON object returned by the RAG system. This is useful for debugging or detailed inspection of the entire pipeline's output for a given query.</p>
        <p align="center">
            <img src="./screenshots/RAG JSON.png" alt="Streamlit App - Full RAG JSON Response" width="700"/>
        </p>
    </li>
</ul>

---

## 🔧 **System Architecture & Design Decisions**

*   **Dataset:** `Abirate/english_quotes` (HuggingFace).
*   **Data Preprocessing (`data_preparation.py`):**
    *   Standard cleaning: Lowercasing, handling missing values, removing duplicates, filtering short quotes.
    *   **Key Feature:** Creation of a `combined_text_for_embedding` field (e.g., "Quote: [text] Author: [name] Tags: [tags]") for rich contextual embeddings.
*   **Embedding Model & Adaptation (`model_finetuning.py`):**
    *   Utilizes `sentence-transformers/all-MiniLM-L6-v2`.
    *   "Fine-tuning" refers to generating specialized embeddings for our structured quote data with this pre-trained model.
*   **Vector Store (`model_finetuning.py`):**
    *   `FAISS (IndexFlatL2)` for efficient local similarity search.
*   **Retrieval-Augmented Generation (`rag_pipeline.py`):**
    *   **Retriever:** Encodes user query with `all-MiniLM-L6-v2`; uses FAISS for top-k retrieval.
    *   **Generator (LLM):** Uses `gpt2` via Hugging Face `transformers` pipeline. Chosen for local, free availability.
    *   **Prompt Engineering:** Guides `gpt2` to summarize and extract information. Includes fallback for JSON parsing failures.
*   **Evaluation (`evaluate_rag.py`):**
    *   `RAGAS` framework; Metrics: `faithfulness`, `answer_relevancy`, `context_precision`.
*   **User Interface (`app.py`):**
    *   Streamlit for interactive querying and results display. Uses `st.cache_resource`.

---

## 🎯 **Key Design Choices**

*   **Embedding Model:** `all-MiniLM-L6-v2` for a balance of performance and local resource needs.
*   **LLM Choice (gpt2):** Prioritized a free, locally runnable model, accepting its generative limitations.
*   **Vector Store (FAISS):** Fast and efficient for local deployment.
*   **"Practical" Fine-Tuning:** Focused on adapting a pre-trained model via input engineering.
*   **RAGAS Metrics:** Selected metrics suitable for open-ended QA without strict ground-truth answers.

---

## 📈 **Evaluation Insights & Performance**

*(This section **MUST** be populated with a summary of your RAGAS evaluation results from `data/ragas_evaluation_results.json` or the console output of `evaluate_rag.py`.)*

*   **Example Placeholder (Replace with your actual findings):**
    *   The RAGAS evaluation indicated average scores of:
        *   `faithfulness`: [Your Score] - *e.g., The LLM's generations were generally grounded in the retrieved context.*
        *   `answer_relevancy`: [Your Score] - *e.g., The answers provided were relevant to the input queries.*
        *   `context_precision`: [Your Score] - *e.g., The retriever effectively fetched relevant documents for the LLM.*
*   **Key Observations:**
    *   Retrieval of semantically similar quotes using FAISS and MiniLM was generally effective.
    *   The primary challenge was the generative quality and instruction-following capability of `gpt2`, particularly for consistent structured output.
    *   *(Add any other specific insights or observations from your RAGAS results and testing.)*

---

## ⚠️ **Challenges & Limitations**

1.  **LLM (GPT-2) Capabilities:**
    *   **Structured Output:** Struggles with consistent valid JSON generation.
    *   **Summarization Quality:** Summaries can be basic.
    *   **Resource Constraint:** Better LLMs require more resources.
2.  **Nature of "Fine-Tuning":** The project adapts a pre-trained model rather than performing full weight retraining (which requires extensive data and compute).
3.  **RAGAS Evaluation Time:** The evaluation script can be time-consuming due to RAGAS's internal model usage.
4.  **Open-Ended QA Evaluation:** Evaluating without manually curated ground-truth is inherently challenging; RAGAS provides useful proxy metrics.

---

## 🔮 **Future Improvements**
*   **Integrate a More Capable LLM:** Replace `gpt2` with a larger open-source model (e.g., Mistral, Llama variants) or an API-based LLM for better generation.
*   **Advanced Fine-Tuning:** Explore contrastive learning for the sentence transformer if suitable data can be created.
*   **Hybrid Search:** Combine dense (FAISS) and sparse (e.g., BM25) retrieval.
*   **Refine Prompt Engineering:** Further iterate on LLM prompts.
*   **Expanded Evaluation:** Include human evaluation or more detailed metrics.

---

## 📝 **Dependencies**
Key dependencies are listed in `requirements.txt`. Main libraries include:
`streamlit`, `sentence-transformers`, `faiss-cpu`, `transformers`, `datasets`, `ragas`, `pandas`.

---

## 📄 License
This project was created for educational purposes as part of an assignment. Feel free to use, modify, and adapt the code for learning and development.

---

## 👨‍💻 Developer

Developed with ❤️ by [Your Name](https://github.com/SyntaxError-Natsu)

---

⭐ Star this repository if you found it helpful!

---

## 🙏 **Acknowledgments**
Special thanks to the creators of the `Abirate/english_quotes` dataset, the Hugging Face team for `transformers` and `datasets`, the developers of `sentence-transformers`, `FAISS`, `RAGAS`, Streamlit, and the assignment provider for this learning opportunity.
