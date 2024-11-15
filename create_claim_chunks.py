import sqlite3
import json
import os
import re
from tqdm import tqdm
import logging
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated Import
from langchain_community.vectorstores import Qdrant  # Updated Import
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
global text_splitter
global embedding_model
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def initialize_qdrant(
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "fact_checking",
    vector_size: int = 384,
    distance_metric: str = "Cosine", 
) -> Qdrant:
    """
    Initializes the Qdrant client and ensures the specified collection exists.

    Args:
        host (str): Hostname where Qdrant is running.
        port (int): Port number for Qdrant.
        collection_name (str): Name of the collection to use/create.
        vector_size (int): Dimensionality of the vectors.
        distance_metric (str): Distance metric to use (e.g., "Cosine", "Euclidean").

    Returns:
        Qdrant: An instance of LangChain's Qdrant vector store.
    """
    try:
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at {host}:{port}...")
        qdrant_client = QdrantClient(host=host, port=port)
        logger.info("Successfully connected to Qdrant.")

        # Check if the collection exists
        logger.info(f"Checking if collection '{collection_name}' exists...")
        existing_collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in existing_collections.collections]

        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' already exists.")
        else:
            # Create the collection with the specified parameters
            logger.info(f"Creating collection '{collection_name}' with vector size {vector_size} and distance metric '{distance_metric}'...")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_metric,  # Options: "Cosine", "Euclidean", "Dot"
                )
            )
            logger.info(f"Collection '{collection_name}' created successfully.")

        # Initialize LangChain's Qdrant vector store
        qdrant_vectorstore = Qdrant(
            client=qdrant_client,  # Correct parameter name
            collection_name=collection_name,
            embeddings=embedding_model
        )
        logger.info(f"LangChain Qdrant vector store for '{collection_name}' initialized.")

        return qdrant_vectorstore

    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
        raise e

def get_file_path(key: str, db_path: str = "outputs/index.db") -> str:
    """
    Retrieves the file path from the SQLite database based on the provided key.

    Args:
        key (str): The unique key to search for in the database.
        db_path (str): Path to the SQLite database file.

    Returns:
        str: The file path associated with the key, or None if not found.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM index_table WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            logger.warning(f"No path found for key: {key}")
            return None
    except Exception as e:
        logger.error(f"Unexpected Error for key '{key}': {e}")
        return None

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing unwanted characters and formatting.

    Args:
        text (str): The raw text to clean.

    Returns:
        str: The cleaned text.
    """
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)     # Replace multiple whitespace with single space
    text = text.strip()
    return text

def process_and_store_claim_chunks(claim_id: str, document_json: dict, qdrant_vectorstore: Qdrant):
    """
    Processes the document text, splits it into chunks, generates embeddings,
    and stores them in Qdrant.

    Args:
        claim_id (str): The identifier for the claim.
        document_json (dict): The document containing 'hostname' and 'text'.
        qdrant_vectorstore (Qdrant): The initialized Qdrant vector store.
    """
    hostname = document_json["hostname"]
    text = document_json["text"]
    cleaned_text = clean_text(text)
    chunks = text_splitter.split_text(cleaned_text)

    # Prepare texts, metadatas, and ids
    texts = chunks
    metadatas = [
        {
            "claim_id": claim_id,
            "source": hostname,
            "chunk_number": chunk_number
        } for chunk_number in range(len(chunks))
    ]
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    
    try:
        qdrant_vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully inserted {len(texts)} vectors for claim '{claim_id}'.")
    except Exception as e:
        logger.error(f"Error inserting vectors for claim '{claim_id}': {e}")


def main():

    # Initialize Qdrant vector store
    qdrant_vectorstore = initialize_qdrant(
        host="localhost",
        port=6333,
        collection_name="fact_checking",
        vector_size=384,  # Dimensionality of 'sentence-transformers/all-MiniLM-L6-v2'
        distance_metric="Cosine"  # Choose based on your similarity requirements
    )
    # Load search results
    search_results_path = "outputs/search_results.json"
    if not os.path.exists(search_results_path):
        logger.error(f"Search results file not found at path: {search_results_path}")
        return

    with open(search_results_path, "r") as fp:
        try:
            search_results = json.load(fp)
            logger.info(f"Loaded search results from {search_results_path}.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {search_results_path}: {e}")
            return

    document_folder = "outputs/documents"
    os.makedirs(document_folder, exist_ok=True)

    # Process each claim
    for claim_index, (claim, queries) in enumerate(tqdm(search_results.items())):
        for query_index, (query, page_results) in enumerate(queries.items()):
            for page_num, results in page_results.items():
                for webpage_index, result_object in enumerate(results):
                    key = f"{claim_index}-{query_index}-{page_num}-{webpage_index}"
                    file_path = get_file_path(key)
                    if file_path and os.path.exists(file_path):
                        try:
                            with open(file_path, "r") as fp:
                                document_json = json.load(fp)
                                if document_json:
                                    process_and_store_claim_chunks(claim_index, document_json, qdrant_vectorstore)
                                else:
                                    logger.warning(f"Empty JSON found for key: {key}, file: {file_path}")
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding JSON from file {file_path}: {e}")
                        except Exception as e:
                            logger.error(f"Error processing file {file_path} for key {key}: {e}")
                    else:
                        logger.warning(f"File path not found or does not exist for key: {key}, path: {file_path}")

if __name__ == "__main__":
    main()
