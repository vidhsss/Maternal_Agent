from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document
import os
from typing import List

class VectorStore:
    """
    Wrapper for FAISS vector store with add, search, load, and save methods.
    """
    def __init__(self, faiss_store: FAISS = None):
        self.faiss_store = faiss_store

    def add_documents(self, docs: List[Document]):
        if self.faiss_store is None:
            raise ValueError("FAISS store must be initialized before adding documents.")
        self.faiss_store.add_documents(docs)

    def similarity_search(self, query: str, k: int = 3):
        return self.faiss_store.similarity_search(query, k=k)

    @staticmethod
    def load(folder_path: str, embeddings=None):
        if embeddings is None:
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        faiss_store = FAISS.load_local(folder_path=folder_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        return VectorStore(faiss_store)

    @staticmethod
    def load_default():
        from .config import FAISS_INDEX_PATH
        return VectorStore.load(FAISS_INDEX_PATH)

    def save(self, folder_path: str):
        if self.faiss_store is not None:
            self.faiss_store.save_local(folder_path)

    def build_from_directory(self, input_dir: str, output_dir: str):
        """Build the vector store from documents in a directory and save it."""
        from langchain.schema import Document
        from .chunking import SemanticChunker, HierarchicalChunker, get_default_embeddings
        import glob, os

        # Collect all text/pdf files (for demo, only .txt)
        file_paths = glob.glob(os.path.join(input_dir, '*.txt'))
        documents = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Use semantic chunker for demo
                chunker = SemanticChunker()
                chunks = chunker.chunk(text)
                for chunk in chunks:
                    doc = Document(page_content=chunk, metadata={"source": file_path})
                    documents.append(doc)
        embeddings = get_default_embeddings()
        self.faiss_store = FAISS.from_documents(documents, embeddings)
        self.save(output_dir)

def load_faiss_vector_store(folder_path: str, embeddings=None) -> FAISS:
    """Load a FAISS vector store from disk."""
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.load_local(folder_path=folder_path, embeddings=embeddings, allow_dangerous_deserialization=True) 



# # Load embeddings model (use OpenAI or HuggingFace embeddings)
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# # Path where FAISS index is stored
# faiss_index_path = "/data/user_data/vidhij2/medical_db/heirchical_chunking_rag/heirchical_chunking_rag_new"

# # Load FAISS vector store
# vector_store = FAISS.load_local(folder_path=faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)

# print(f"FAISS index loaded successfully with {vector_store.index.ntotal} vvectors.")

# embeddings = HuggingFaceEmbeddings(
#     model_name="google/muril-base-cased",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

# vector_store_2 = FAISS.load_local(
#     folder_path="/data/user_data/vidhij2/medical_db/rag",
#     embeddings=embeddings,  # This must be the properly initialized embeddings
#     allow_dangerous_deserialization=True
# )