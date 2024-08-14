import os
import sys
def get_project_root() -> str:
    """
    Find and return the project root directory.
    """
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


import streamlit as st
from src.retrieval import CLIPEncoder, ImageDatabase
from src.data import load_embeddings
import cv2
import random
import math
from typing import List, Tuple
import tempfile







def save_upload_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')as temp_file:
        temp_file.write(uploaded_file.get_value())
        return temp_file.name

def get_random_image_from_test() -> str:
    test_path = './data/processed/test'
    folders = os.listdir(test_path)
    random_folder = random.choice(folders)
    random_image = random.choice(os.listdir(os.path.join(test_path, random_folder)))
    return os.path.join(test_path, random_folder, random_image)


def find_similarities(encoder, database, query_path, top_k: str):
    query_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
    query_embedding = encoder.get_single_image_embedding(query_img)

    similary_image_score = database.search_image_similarity(
        query_embedding=query_embedding,
        top_k_retrieval=top_k
    )
    return similary_image_score


    