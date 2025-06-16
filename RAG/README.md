# RAG (Retrieval-Augmented Generation) Python Package for Maternal Health

This package provides modular implementations of Retrieval-Augmented Generation (RAG) systems for medical and general document QA, chunking, and evaluation, with a focus on maternal health.

## Main Model Files & Usage

| File                                      | Description                                              | Example Usage                                                                 |
|--------------------------------------------|----------------------------------------------------------|-------------------------------------------------------------------------------|
| `core.py`                                 | Main RAG pipeline (MedicalRAG, EnhancedMedicalRAG)        | `python core.py --question "..." --vector_store_path <path> --model medical|enhanced` |
| `rag_core.py`                             | Minimal RAG pipeline for medical QA                      | `python rag_core.py --question "..." --vector_store_path <path>`            |
| `EnhancedMedicalRAG.py`                   | Advanced RAG with multilingual, stage-aware, reranking   | `python EnhancedMedicalRAG.py --question "..." --vector_store_path <path> --model_name <hf_model>` |
| `Medical_rag.py`                          | Medical PDF pipeline: build vector store, run query      | `python Medical_rag.py --pdf_path <input.pdf> --output_db <db_dir> --query "..."` |
| `Basic_rag_multilingual_with_no_rag.py`   | Basic multilingual RAG                                   | `python Basic_rag_multilingual_with_no_rag.py --question "..." --vector_store_path <path> --model_name <hf_model>` |
| `basic_rag_huggingface.py`                | Basic RAG using HuggingFace models                       | `python basic_rag_huggingface.py --question "..." --vector_store_path <path> --model_name <hf_model>` |
| `basic_rag_gpt.py`                        | Basic RAG using OpenAI GPT models                        | `python basic_rag_gpt.py --question "..." --vector_store_path <path>`        |
| `Reasoning_based_rag.py`                  | RAG with multi-document reasoning                        | `python Reasoning_based_rag.py --question "..." --vector_store_path <path> --model_name <hf_model>` |
| `new_enhancedrag.py`                      | Enhanced RAG, similar to EnhancedMedicalRAG              | `python new_enhancedrag.py --question "..." --vector_store_path <path> --model_name <hf_model>` |
| `MaternalHealthDocumentProcessor.py`      | Specialized processor for maternal health docs           | `python MaternalHealthDocumentProcessor.py --input_file <input.txt> --output_db <db_dir>` |

## Scripts
- `scripts/build_database.py`: Build a vector database from a directory of documents.
- `scripts/run_rag.py`: Run a RAG query using the standard MedicalRAG pipeline or EnhancedMedicalRAG.
- `scripts/evaluate.py`: Run the evaluation pipeline for RAG models.

## Utility Modules
- `chunking.py`, `vector_store.py`, `config.py`, `utils.py`, `prompts.py`, etc.: Importable modules for chunking, vector store management, configuration, and utilities.

## Requirements
See `requirements.txt` for dependencies. Recommended Python 3.8+.

## Notes
- All main model files are directly executable with a CLI.
- Each file prints answers and sources in a standardized format.
- The code is modular and easy to extend for new chunking, retrieval, or evaluation strategies.

---
For further details, see docstrings in each file or contact the maintainers. 