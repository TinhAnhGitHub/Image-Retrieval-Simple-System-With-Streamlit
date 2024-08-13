"""
This is the testing script for CLIP model APIs
Directory: "./src/retrieval/model.py"
"""

import unittest
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

from src.retrieval import CLIPEncoder
import numpy as np

class TestCLIPEncoder(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case by initializing the CLIPEncoder and loading a test image.
        """
        self.encoder = CLIPEncoder()
        self.test_image_path = './data/processed/train/basketball/n02802426_3881.JPEG'
        self.expected_image_shape = (224, 224, 3)
        self.expected_embedding_shape = (512,)

    def test_read_image_from_path(self):
        """
        Test the image reading and resizing functionality.
        """
        image = self.encoder.read_image_from_path(self.test_image_path)
        self.assertIsInstance(image, np.ndarray, "Image should be a numpy array")
        self.assertEqual(image.shape, self.expected_image_shape, f"Image shape should be {self.expected_image_shape}")

    def test_get_single_image_embedding(self):
        """
        Test the embedding extraction functionality.
        """
        image = self.encoder.read_image_from_path(self.test_image_path)
        embedding = self.encoder.get_single_image_embedding(image)
        self.assertIsInstance(embedding, np.ndarray, "Embedding should be a numpy array")
        self.assertEqual(embedding.shape, self.expected_embedding_shape, f"Embedding shape should be {self.expected_embedding_shape}")

if __name__ == "__main__":
    unittest.main()
