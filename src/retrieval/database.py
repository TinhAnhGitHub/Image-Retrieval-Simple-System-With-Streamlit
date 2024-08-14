"""
This module defines an `ImageDatabase` class for managing image embeddings and performing similarity searches using ChromaDB.

The `ImageDatabase` class provides functionalities to:
- Add image embeddings along with their corresponding image paths to a database collection.
- Search for similar images based on a query embedding, returning the most similar images and their associated similarity scores.

Key Features:
- Utilizes ChromaDB for efficient storage and retrieval of image embeddings.
- Supports adding new embeddings to an existing collection or creating a new one.
- Allows for similarity searches to find top-K similar images in the database.

Dependencies:
- chromadb
- numpy

Example usage:
    db = ImageDatabase("my_image_collection")
    db.add_embeddings(embeddings, image_paths)
    similar_images = db.search_similar_images(query_embedding, top_k=10)
"""

import os
import sys
def get_project_root() -> str:
    current_abspath = os.path.abspath(os.getcwd())
    while True:
        if os.path.split(current_abspath)[1] == 'Image-Retrieval-Simple-System-With-Streamlit':
            project_root = current_abspath
            break
        else:
            current_abspath = os.path.dirname(current_abspath)
    
    return project_root

PROJECR_ROOT = get_project_root()
os.chdir(PROJECR_ROOT)
sys.path.append(PROJECR_ROOT)



import chromadb
import numpy as np
from typing import List, Tuple

class ImageDatabase:
    """
    A class to handle image database operations from Chroma Database
    """
    def __init__(self ,collection_name = "image_collection"):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_embeddings(self, embeddings: np.ndarray, image_paths: List[str]) -> None:
        """Add image embedding and its corresponding image path, for each image

        Args:
            embeddings (np.ndarray): An matrix embeddings vector. Shape: (n_images, n_dimension = 512)
            image_paths (List[str]): A list of image path
        """
        self.collection.add(
            embeddings=embeddings.tolist(),
            ids = [f"id_{i}" for i in range(len(image_paths))],
            metadatas= [{'path': path} for path in image_paths]
        )
    
    def search_image_similarity(self, query_embedding: np.ndarray, top_k_retrieval: int = 20) -> List[Tuple[str, float]]:
        """Search for similar images in the database

        Args:
            query_embedding (np.ndarray): Query embedding vector. Shape: (n_dimension = 512,)
            top_k_retrieval (int, optional): the maximum length of retrieved images. Defaults to 20.

        Returns:
            List[Tuple[str, float]]: a list of tuple, with this format (image_path, confidence score). This function use cosine similarity

        """

        results = self.collection.query(
            query_embeddings= query_embedding.reshape(1, -1),
            n_results= top_k_retrieval,
            
        )
        return [(metadata['path'], cosine_score) for metadata, cosine_score in zip(results['metadatas'][0], results['distances'][0])]
    

# if __name__ == "__main__":
#     db = ImageDatabase("test_collection")
    
#     # Create dummy embeddings and paths
#     dummy_embeddings = np.random.rand(5, 512)  # 5 images, 512-dimensional embeddings
#     dummy_paths = [f"image_{i}.jpg" for i in range(5)]
    
#     # Add embeddings to the database
#     db.add_embeddings(dummy_embeddings, dummy_paths)
    
#     # Query with a new random embedding
#     query_embedding = np.random.rand(512)
#     similar_images = db.search_image_similarity(query_embedding, top_k_retrieval=3)
#     print(similar_images)
    






