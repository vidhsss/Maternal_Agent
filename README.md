# Maternal Health RAG Project (`nivi`)

This repository provides a comprehensive suite for **Retrieval-Augmented Generation (RAG)**, document processing, and evaluation, with a focus on maternal health. It includes modular RAG pipelines, document chunking, vector store management, evaluation scripts, and a rich set of Jupyter notebooks for experimentation and analysis.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Folder and File Descriptions](#folder-and-file-descriptions)
- [How to Use](#how-to-use)
- [Notebooks Overview](#notebooks-overview)
- [RAG Pipeline Overview](#rag-pipeline-overview)
- [Data and Results](#data-and-results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure

```
nivi/
│
├── .gitignore
├── replacement.txt
├── maternal_health_pipeline.log
├── eSaathi Resources_ANC_PNC_CMU.xlsx
├── llm_judge_evaluation.log
├── stacked_by_question.csv
├── sorted_response_weekly_tilljan2025.csv
├── sorted_response_weekly_tilljan2025
├── Nivi_ANCSessions_edited.csv
├── Nivi_ANCUsers_2023.01.01-2025.01.20.csv
├── anc-guideline-presentationb8005106-0947-4329-adb1-61aea46db1db (1).pptx
├── analysis.ipynb
├── RAG/
│   ├── (core RAG pipeline code, chunking, vector store, scripts, etc.)
│   └── README.md
├── Notebooks/
│   ├── (Jupyter notebooks for experiments, evaluation, and testing)
│   └── test_docs/
├── Results/
│   ├── (Model outputs, evaluation results, plots, and summaries)
├── maternal_healthcare_evaluation/
│   ├── (Evaluation plots, reports, and summaries)
├── maternal_healthcare_evaluation_new/
│   ├── (Updated evaluation results and plots)
├── maternal-documents/
│   ├── (PDFs and source documents for maternal health)
├── data/
│   ├── user_data/
├── question_details/
│   ├── (CSV files with question/answer details)
├── model_responses/
│   ├── (JSON files with model outputs)
└── .venv/
    └── (Python virtual environment)
```

---

## Setup Instructions

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/nivi.git
cd nivi
```

### 2. **Set Up Python Environment**
It is recommended to use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. **Install Dependencies**
Install the main requirements for the RAG pipeline:
```bash
pip install -r RAG/requirements.txt
```
This will install:
- langchain
- faiss-cpu
- sentence-transformers
- openai
- scikit-learn
- numpy
- pandas

If you use notebooks, also install Jupyter:
```bash
pip install jupyter
```

### 4. **(Optional) Set Up OpenAI API Key**
If you use OpenAI models, set your API key as an environment variable:
```bash
export OPENAI_API_KEY=sk-...
```
**Never commit your API key to the repository!**

---

## Folder and File Descriptions

### **Top-Level Files**
- `.gitignore` — Lists files to be ignored by git (e.g., large CSVs).
- `replacement.txt` — Used for secret removal (BFG).
- `maternal_health_pipeline.log`, `llm_judge_evaluation.log` — Log files from pipeline runs.
- `eSaathi Resources_ANC_PNC_CMU.xlsx` — Reference Excel file.
- `stacked_by_question.csv`, `sorted_response_weekly_tilljan2025.csv` — Processed data files.
- `Nivi_ANCSessions_edited.csv`, `Nivi_ANCUsers_2023.01.01-2025.01.20.csv` — Large data files (ignored by git).
- `analysis.ipynb` — Main analysis notebook.

### **RAG/**
- **README.md** — Detailed documentation for the RAG package.
- **requirements.txt** — Python dependencies for RAG.
- **core.py, EnhancedMedicalRAG.py, Medical_rag.py, Reasoning_based_rag.py, etc.** — Main RAG pipeline implementations for various use cases (medical, multilingual, reasoning-based, etc.).
- **chunking.py, chunking_vector_store.py, semantic_chunker.py, etc.** — Document chunking and vector store utilities.
- **scripts/** — Helper scripts for building databases, running RAG, and evaluation.
- **utils.py, config.py, prompts.py** — Utility and configuration modules.

### **Notebooks/**
- **testing_different_llms.ipynb** — Main notebook for comparing LLMs and RAG pipelines.
- **NEW-building_datbase_and_RAG.ipynb** — Building vector stores and running RAG.
- **evaluation_initial.ipynb, Initial_results_with_different_llms.ipynb, etc.** — Evaluation and results analysis.
- **test_docs/** — Test documents and model outputs for notebook experiments.

### **Results/**
- **model_comparisons.xlsx, model_comparison_structured.xlsx, etc.** — Model comparison results.
- **results_*.json, *.png** — Model outputs, evaluation results, and plots.

### **maternal_healthcare_evaluation/** and **maternal_healthcare_evaluation_new/**
- Evaluation plots, reports, and summary files for model and RAG performance.

### **maternal-documents/**
- Source PDFs and reference documents for maternal health.

### **data/user_data/**
- User-specific or experimental data.

### **question_details/**
- CSV files with detailed Q&A for various maternal health topics.

### **model_responses/**
- JSON files with model-generated answers for benchmarking.

### **.venv/**
- Python virtual environment (not tracked by git).

---

## How to Use

### **Run a RAG Pipeline from the Command Line**
Example (from `RAG/README.md`):
```bash
cd RAG
python core.py --question "What are the early signs of pregnancy complications?" --vector_store_path <path_to_vector_store> --model medical
```
Or, for enhanced multilingual RAG:
```bash
python EnhancedMedicalRAG.py --question "..." --vector_store_path <path> --model_name <hf_model>
```

### **Build a Vector Store**
```bash
python scripts/build_database.py --input_dir <docs_dir> --output_db <db_dir>
```

### **Run a Notebook**
```bash
jupyter notebook
```
Open any notebook in the `Notebooks/` directory for interactive experiments.

---

## Notebooks Overview

- **testing_different_llms.ipynb** — Compare different LLMs and RAG pipelines on maternal health queries.
- **NEW-building_datbase_and_RAG.ipynb** — End-to-end pipeline for building vector stores and running RAG.
- **evaluation_initial.ipynb** — Initial evaluation of model and RAG performance.
- **Document_extracting_building.ipynb** — Extract and process documents for vector store creation.
- **test_docs/** — Contains test outputs and sample model answers for reproducibility.

---

## RAG Pipeline Overview

The RAG system is modular and supports:
- Multiple chunking strategies (semantic, hierarchical, etc.)
- Vector store backends (FAISS)
- Multiple LLMs (OpenAI, HuggingFace, etc.)
- Multilingual and reasoning-based retrieval
- Evaluation and benchmarking scripts

See `RAG/README.md` for detailed usage of each module.

---

## Data and Results

- **maternal-documents/** — Source PDFs for building knowledge bases.
- **Results/** — Model outputs, evaluation results, and plots.
- **question_details/** — CSVs with real-world maternal health questions and answers.
- **model_responses/** — Model-generated answers for benchmarking.

---

## Contributing

1. Fork the repo and create your branch.
2. Commit your changes.
3. Push to your fork and submit a pull request.

---

## License

Specify your license here (e.g., MIT, Apache 2.0, etc.).

---

**For more details, see the docstrings in each file or contact the maintainers.** 