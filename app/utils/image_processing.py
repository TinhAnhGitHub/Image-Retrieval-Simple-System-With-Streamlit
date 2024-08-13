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



from src.retrieval import CLIPEncoder, ImageDatabase
from src.data import load_embeddings
import cv2
import random
import tempfile


EMBEDDING_DIR = './data/embeddings'

def find_similarities(query_path, top_k: str):
    encoder = CLIPEncoder()
    database = ImageDatabase()

    matrix_embeddings, global_index2img_path = load_embeddings(
        EMBEDDING_DIR
    )

    image_paths = []
    for _, path in global_index2img_path.items():
        image_paths.append(path)

    database.add_embeddings(
        matrix_embeddings,
        image_paths
    )

    query_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
    query_embedding = encoder.get_single_image_embedding(query_img)

    similary_image_score = database.search_image_similarity(
        query_embedding=query_embedding,
        top_k_retrieval=top_k
    )
    return similary_image_score


    