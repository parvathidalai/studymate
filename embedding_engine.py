from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def create_embeddings(self, chunks: List[dict]) -> np.ndarray:
        """Create embeddings for text chunks"""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts)
        return embeddings
    
    def build_faiss_index(self, chunks: List[dict]):
        """Build FAISS index from chunks"""
        self.chunks = chunks
        embeddings = self.create_embeddings(chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def search_similar_chunks(self, query: str, k: int = 3) -> List[dict]:
        """Search for similar chunks based on query"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return relevant chunks
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results