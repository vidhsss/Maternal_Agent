"""
Script to run the evaluation pipeline for RAG models.
"""

import argparse
from rag.evaluation import evaluate_rag_system

def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save evaluation results')
    args = parser.parse_args()

    # Run evaluation (assume function exists in evaluation.py)
    # NOTE: This is a stub. Real evaluation logic should load responses and ground truth.
    evaluate_rag_system(results_dir=args.results_dir)
    print(f"Evaluation complete. Results saved to {args.results_dir}")

if __name__ == "__main__":
    main() 