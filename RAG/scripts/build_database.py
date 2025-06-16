"""
Script to build the vector database from documents.
"""

import argparse
from rag.core import MedicalRAG
from rag.vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Build the vector database from documents.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with raw documents')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed vector DB')
    args = parser.parse_args()

    # Load and process documents
    vector_store = VectorStore()
    vector_store.build_from_directory(args.input_dir, args.output_dir)
    print(f"Vector database built and saved to {args.output_dir}")

if __name__ == "__main__":
    main() 