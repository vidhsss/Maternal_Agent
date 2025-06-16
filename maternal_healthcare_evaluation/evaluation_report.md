# Maternal Healthcare Response Evaluation Report

## Overview

This report presents an evaluation of 6 models across 3 languages, with a total of 0 responses evaluated.

Evaluation was performed by 1 LLM judges: gemini-pro

## Overall Results

### RAG vs Non-RAG Performance

#### Judge: gemini-pro

## Language-Specific Analysis

### English

**RAG vs Non-RAG performance:**

### Hindi

**RAG vs Non-RAG performance:**

### Hinglish

**RAG vs Non-RAG performance:**

## Model-Specific Analysis

### gpt-4-turbo

![Model Performance](visuals/gemini-pro_gpt-4-turbo_language_performance.png)

**Performance across languages:**

**Judge: gemini-pro**

### gpt-4-turbo_rag

![Model Performance](visuals/gemini-pro_gpt-4-turbo_rag_language_performance.png)

**Performance across languages:**

**Judge: gemini-pro**

### llama

![Model Performance](visuals/gemini-pro_llama_language_performance.png)

**Performance across languages:**

**Judge: gemini-pro**

### llama_rag

![Model Performance](visuals/gemini-pro_llama_rag_language_performance.png)

**Performance across languages:**

**Judge: gemini-pro**

### mixtral

![Model Performance](visuals/gemini-pro_mixtral_language_performance.png)

**Performance across languages:**

**Judge: gemini-pro**

### mixtral_rag

![Model Performance](visuals/gemini-pro_mixtral_rag_language_performance.png)

**Performance across languages:**

**Judge: gemini-pro**

## Recommendations

- **Mixed approach recommended**: RAG and Non-RAG models show similar performance overall. Consider using RAG for specific languages or criteria where it shows advantage.

### Language-Specific Recommendations

- **English**: Either approach works similarly well

- **Hindi**: Either approach works similarly well

- **Hinglish**: Either approach works similarly well

## Methodology

This evaluation uses LLM judges to assess maternal healthcare responses across four criteria:

1. **Medical Correctness**: Are the medical claims in the response accurate? (1 = all correct, 3 = not correct)

2. **Completeness**: Does the answer cover all necessary information? (1 = covers everything, 3 = omits significant information)

3. **Language Clarity**: Is the response clear for users with average literacy? (1 = completely understandable, 3 = unacceptable)

4. **Cultural Appropriateness**: Is the response appropriate for the cultural context? (1 = completely appropriate, 3 = inappropriate)

For all scores, **lower is better** (1 is the best possible score, 3 is the worst).