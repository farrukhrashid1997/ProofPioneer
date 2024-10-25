import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np



def create_embeddings(chunks, model_name='Alibaba-NLP/gte-base-en-v1.5'):
    # Initialize the embedding model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Extract text from chunks
    texts = [chunk.page_content for chunk in chunks]
    
    # Create embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Combine chunks with their embeddings
    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({
            'text': chunk.page_content,
            'embedding': embedding,
            'metadata': chunk.metadata
        })
    
    return embedded_chunks



def calculate_similarity(claim_embedding: np.ndarray, evidence_embedding: np.ndarray) -> float:
    return np.dot(claim_embedding, evidence_embedding) / (
        np.linalg.norm(claim_embedding) * np.linalg.norm(evidence_embedding)
    )


def evaluate_evidence_relevance(claims: List[str], embedded_chunks: List[Dict], 
                              model_name='Alibaba-NLP/gte-base-en-v1.5', 
                              similarity_threshold=0.63):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    claim_embeddings = model.encode(claims, show_progress_bar=True)
    
    relevant_chunks = 0
    relevance_scores = []
    relevant_evidence = []  # To store relevant chunks

    
    for chunk in embedded_chunks:
        # Calculate max similarity with any claim
        chunk_similarities = [
            calculate_similarity(claim_emb, chunk['embedding'])
            for claim_emb in claim_embeddings
        ]
        max_similarity = max(chunk_similarities)
        relevance_scores.append(max_similarity)
        
        
        if max_similarity >= similarity_threshold:
            relevant_chunks += 1
            relevant_evidence.append({
                'text': chunk['text'],
            })
    
    stats = {
        'total_chunks': len(embedded_chunks),
        'relevant_chunks': relevant_chunks,
        'relevance_ratio': relevant_chunks / len(embedded_chunks),
        'avg_similarity': np.mean(relevance_scores),
        'max_similarity': max(relevance_scores),
        'min_similarity': min(relevance_scores),
        'relevant_evidence': relevant_evidence
    }
    
    return stats



def save_relevant_chunks_to_file(relevant_chunks, output_file):
    # Save relevant chunks to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(relevant_chunks, f, ensure_ascii=False, indent=4)
    print(f"Relevant chunks saved to {output_file}")


def process_documents(folder_path):
    docs = []
    
    # Ensure folder path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder path {folder_path} does not exist")
    # Process each JSON file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                # Open and read the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    
                    # Check if 'text' key exists in the JSON data
                    if "text" in json_data:
                        # Create a Document object instead of a dictionary
                        docs.append(
                            Document(
                                page_content=json_data["text"],
                                metadata={"source": filename}
                            )
                        )
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file {filename}: {e}")
            except Exception as e:
                print(f"Unexpected error processing file {filename}: {e}")
    
    return docs


if __name__ == "__main__":
    folder_path = "outputs/documents/"
    
    docs = process_documents(folder_path)
        
    if not docs:
        print("No documents were processed. Check if the folder contains valid JSON files.")
        exit(1)


   # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(docs)
    
    # Check the number of splits
    print(f"Total chunks created: {len(chunks)}")
    
      # Create embeddings for chunks
    embedded_chunks = create_embeddings(chunks)
    print(f"Created embeddings for {len(embedded_chunks)} chunks")
    
    # Evaluate evidence relevance
    relevance_stats = evaluate_evidence_relevance(["Aqui São Paulo nós estamos com 2,7 milhões com os tais R 600 que viraram R 300 [ do auxílio emergencial ]"], embedded_chunks)
    print("\nEvidence Relevance Statistics:")
    print(relevance_stats['total_chunks'])
    print(relevance_stats['relevant_chunks'])
    print(relevance_stats['relevance_ratio'])
    print(relevance_stats['avg_similarity'])

 
    output_file = "relevant_chunks.json"
    save_relevant_chunks_to_file(relevance_stats['relevant_evidence'], output_file)