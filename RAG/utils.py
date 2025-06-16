"""
utils.py
--------
Provides utility functions for the RAG system, such as token counting.

Functions:
    - count_tokens: Count the number of tokens (words) in a string.
"""

def count_tokens(text: str) -> int:
    """Count the number of tokens (words) in a string."""
    return len(text.split()) 