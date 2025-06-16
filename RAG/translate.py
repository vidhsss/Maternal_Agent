"""
translate.py
------------
Provides translation utilities for cross-lingual RAG, including functions to translate between Hindi and English and a cross-lingual QA pipeline.

Functions:
    - translate_to_english: Translate Hindi text to English.
    - translate_to_hindi: Translate English text to Hindi.
    - crosslingual_qa: Full cross-lingual QA pipeline using translation and RAG.
"""

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from googletrans import Translator  # Or use your custom translator

# --- STEP 1: Translation Setup ---
translator = Translator()

def translate_to_english(hindi_text):
    """Translate Hindi text to English."""
    return translator.translate(hindi_text, src="hi", dest="en").text

def translate_to_hindi(english_text):
    """Translate English text to Hindi."""
    return translator.translate(english_text, src="en", dest="hi").text

# --- STEP 2: Load Vector Store and Embeddings ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model)

# --- STEP 3: LLM Setup ---
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# --- STEP 4: QA Chain using LangChain ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def crosslingual_qa(hindi_query):
    """Full cross-lingual QA pipeline: Hindi -> English -> RAG -> Hindi."""
    # Step 1: Translate query
    query_en = translate_to_english(hindi_query)
    # Step 2: RAG pipeline
    result = qa_chain(query_en)
    # Step 3: Translate answer back to Hindi
    answer_hi = translate_to_hindi(result['result'])
    # Optional: Also return the context documents or original answer
    return {
        "answer_hindi": answer_hi,
        "answer_english": result['result'],
        "sources": result['source_documents']
    }

# --- Example ---
query_hi = "मैं 2.5 महीने की गर्भवती हूँ और उल्टी नहीं हो रही है, क्या यह सामान्य है?"
response = crosslingual_qa(query_hi)

print("🤖 उत्तर (Hindi):", response["answer_hindi"])
print("\n💬 Original Answer (English):", response["answer_english"])
