# RAG Graph-Based Retrieval System

## Overview
This repository implements a **Retrieval-Augmented Generation (RAG) pipeline** using a graph-based workflow. The system enhances question-answering by retrieving relevant documents, assessing their relevance, rewriting queries, and performing web searches when necessary.

The architecture is designed using **LangChain, OpenAI API, and LangGraph**, ensuring efficient retrieval and response generation.

## Features
- **Graph-Based RAG Workflow:** Uses LangGraph to structure the pipeline.
- **Retrieval-Augmented Generation:** Combines document retrieval with LLM-based text generation.
- **Query Rewriting & Web Search:** Enhances question formulation and retrieves additional relevant documents from the web.
- **Modular Components:** Designed for flexibility and easy extension.

## Repository Structure
```
├── config/
│   ├── key.env              # API Keys (ignored in Git)
│   ├── config.env           # Configuration settings
├── data/
│   ├── urls/retrieval_urls.json # URLs used for document retrieval
├── src/
│   ├── __init__.py
│   ├── llm_setup.py         # LLM setup (retriever, RAG pipeline, rewriter)
│   ├── web_search_tool.py   # Web search integration (Google, Tavily)
├── log/                     # Logs for pipeline executions (ignored in GIT)
├── scripts/
│   ├──rag_graph_pipeline.py     # Core pipeline implementation
│   ├──run_rag_graph.py         # Execution script
├── README.md                # Documentation
```

## Setup
### 1. Clone the Repository
```bash
git clone https://github.com/leem18/GraphRAG-LLM.git
cd GraphRAG-LLM
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Set Up API Keys
Create a `.env` file inside `config/` and add the required API keys:
```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_google_key
GOOGLE_CSE_ID=your_google_cse_id
LANGCHAIN_API_KEY=your_langchain_key
```
### 4. Run the Pipeline
```bash
python3 run_rag_graph.py
```

## Explanation of Key Files
### **1. config.env**
Defines system parameters, including:
- **Model settings:** LLMs used for retrieval grading, question rewriting, and generation.
- **Search parameters:** Max search depth, API selection (Google, Tavily, SerpAPI).
- **Logging configurations.**

### **2. llm_setup.py**
Handles the initialization of:
- **Document retrievers** (ChromaDB VectorStore, WebBaseLoader).
- **LLM-powered retrieval grading** (binary relevance scoring).
- **RAG chain for generating responses.**
- **Question rewriting and hypothetical document generation.**

### **3. web_search_tool.py**
- Integrates Google and Tavily search engines.
- Standardizes and formats search results for retrieval.

### **4. rag_graph_pipeline.py**
This script defines the **graph-based pipeline** using LangGraph:
- Retrieves documents related to the query.
- Generates hypothetical documents for better context.
- Grades document relevance to the query.
- Rewrites the query if necessary.
- Conducts web searches for additional information.
- Passes relevant data to an LLM for final answer generation.

### **5. run_rag_graph.py**
- Loads environment variables.
- Sets API keys.
- Executes `rag_graph_pipeline.py`.

## Example Usage
### Input:
```bash
python3 run_rag_graph.py
```
Prompt: *"Who are the competitors of OpenAI in 2024?"*

### Output:
- **Retrieves documents** from stored sources and web search.
- **Grades document relevance.**
- **Rewrites question** for better clarity if needed.
- **Generates a final response** using RAG-based text generation.

## Customization
- Modify `config.env` to change model parameters and search tools.
- Add new sources in `data/urls/retrieval_urls.json`.
- Extend `rag_graph_pipeline.py` to integrate new modules.

## Future Improvements
- Support for additional search engines.
- More advanced ranking and filtering of retrieved documents.
- Expansion of dataset integration for better responses.

---
This project is actively maintained and open for contributions!

