"""
This module defines a `CLIPEncoder` class that provides utilities for working with CLIP (Contrastive Language-Image Pretraining) embeddings using the OpenCLIP framework. 

The `CLIPEncoder` class allows for the extraction of image embeddings from input images, which can be used in various downstream tasks such as image retrieval, similarity matching, and more. 

The main functionalities include:
- Reading and resizing images from a file path.
- Converting images into embeddings using the CLIP model.

Dependencies:
- chromadb.utils.embedding_functions.OpenCLIPEmbeddingFunction
- numpy
- cv2 

Example usage:
    encoder = CLIPEncoder()
    image = encoder.read_image_from_path("path/to/image.jpg")
    embedding = encoder.get_single_image_embedding(image)
"""


import os
import sys
import cv2
import numpy as np
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from tqdm import tqdm
from typing import List, Tuple


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


class CLIPEncoder:
    """
    A class to handle CLIP encoding operations
    """
    def __init__(self):
        print("Loading model, this could take a while...")
        self.embedding_function = OpenCLIPEmbeddingFunction()
        print("CLIP model is successfully loaded!")
    
    def get_single_image_embedding(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Get the feature extraction embedding vector for the image

        Args:
            image (np.ndarray): Image array. Example shape: (height, width, color_channel = 3)

        Returns:
            np.ndarray: Embedding vector (n,). In this case, n = 512
        """
        embedding = self.embedding_function._encode_image(image)
        return np.array(embedding)
    
    @staticmethod
    def read_image_from_path(
        path: str,
        size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """Read an image from a specific path

        Args:
            path (str): The path to the image
            size (Tuple[int, int], optional): The desired size of the image, before fetching it into the modelCLIP. Defaults to (224, 224).

        Returns:
            np.ndarray: return numpy array representation of the image (height, width, channel)
        """
        image_cv2 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image_cv2, size)
        return image
    


