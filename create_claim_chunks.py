from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import json
import os
from tqdm import tqdm
import sqlite3
import numpy as np
import faiss

class ClaimChunks:
    def __init__(self, embedding_model_name: str = "Alibaba-NLP/gte-base-en-v1.5"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50, 
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embedding_model =  SentenceTransformer(embedding_model_name, trust_remote_code=True)
        self.dimension = 768
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = {}
        self.embeddings_temp = []
     
    
    def read_document(self, document_path: str) -> str:
        """Read document content from file"""
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                text = content.get('text', '') or content.get('raw_text', '')
                return text
        except Exception as e:
            print(f"Error reading document {document_path}: {e}")
            return ""
    
    def get_index(self, key):
        try:
            conn = sqlite3.connect("outputs/index.db", check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM index_table WHERE key = ?", (key,))
            result = cursor.fetchone() 
            return result
        
        except Exception as e:
            print("An error occurred:", e)
            return None
    
    
    def process_document(self, text: str, claim_id: str, document_key: str):
        """Process a single document into chunks with embeddings"""
        chunks = self.text_splitter.split_text(text)
        # chunk_metadata = []
        print(len(chunks))
        
        # Generate embeddings for all chunks at once for efficiency
        embeddings = self.embedding_model.encode(chunks)
        
        # for i, chunk in enumerate(chunks):
        #     chunk_id = f"{claim_id}-{document_key}-{i}"
        #     chunk_metadata.append({
        #         'chunk_id': chunk_id,
        #         'claim_id': claim_id,
        #         'document_key': document_key,
        #         'chunk_text': chunk,
        #         'chunk_index': i  # Store index for mapping back to embeddings
        #     })
        
        return chunks, embeddings
    
    
    def get_claim_documents(self):
        with open("outputs/search_results.json", "r") as fp:
            search_results = json.load(fp)
            document_folder = "outputs/documents"
            os.makedirs(document_folder, exist_ok=True)
        for claim_index, (claim, queries) in enumerate(tqdm(search_results.items(), desc="Processing Claims")):
            claim_id = f"claim_{claim_index}"
            # claim_chunks = []
            # claim_metadata = []
            # claim_embeddings = []
            if claim_index == 2:
                break
            for query_index, (query, page_results) in enumerate(queries.items()):
                for page_num, results in page_results.items():
                    for webpage_index, result_object in enumerate(results):
                        doc_key = f"{claim_index}-{query_index}-{page_num}-{webpage_index}"
                        index_result = self.get_index(doc_key)
                        if not index_result:
                            continue
                        
                        document_path = index_result[1]
                        document_text = self.read_document(document_path)
                        
                        if not document_text:
                            continue
                        # Process document into chunks
                        chunks, embeddings = self.process_document(
                            document_text, 
                            claim_id, 
                            doc_key,
                        )
                        # claim_chunks.extend(chunks)
                        # claim_metadata.extend(metadata)
                        self.embeddings_temp.append(np.array(embeddings).astype('float32'))
            if self.embedding_model:
                all_embeddings = np.vstack(self.embeddings_temp)
                self.index.add(all_embeddings)
                self.embeddings_temp = []
            faiss.write_index(self.index, f"outputs/faiss/faiss_index_{claim_index}.index")
                        
                        
    
    
                        
    # def load_index(self, claim_index):
    #     """Load FAISS index and chunks for a specific claim"""
    #     # Load FAISS index
       
                           
                        
    def search(self, query: str, k: int = 5):
        """Search for k most similar sentences to the query"""
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        # Search the index
        self.index = faiss.read_index(f"outputs/faiss/faiss_index_0.index")
        distances, indices = self.index.search(query_embedding, k)
        
        # Get the corresponding chunks
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            print(idx)
            if idx < 216:  # Ensure index is valid
                # chunk_id, text = self.chunks[idx]
                results.append({
                    'rank': i + 1,
                    # 'chunk_id': chunk_id,
                    # 'text': text,
                    'similarity_score': 1 - distance  # Convert distance to similarity
                })
        
        return results
                        
if __name__ == "__main__":
    claim_chunks = ClaimChunks()
    # claim_chunks.get_claim_documents()
    # claim_chunks.load_index(0)
    res = claim_chunks.search("Diaspora remittances contributed upwards of KSh290.")
    print(res)
    
## boolean based searches
## google based dorking
## advanced queries - re


