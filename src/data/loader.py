"""
This module provides functions for saving and loading image embeddings and their corresponding file paths.

Functions:
- save_embeddings: Saves an array of image embeddings and a list of image paths to specified files.
- load_embeddings: Loads image embeddings and image paths from saved files.

Dependencies:
- numpy: For handling and saving/loading embeddings.
- json: For saving/loading image paths.
- os: For creating directories and handling file paths.
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


import numpy as np
import json
from typing import  Tuple, Dict
from src.retrieval import CLIPEncoder
from .preprocessing import precomputing_embeddings


def save_embeddings(
    output_dir: str,
    all_embeddings: np.ndarray,
    global_index2img_path: Dict[int, str]
):
    """
    Save embeddings and image paths to files.

    Args:
        output_dir (str): Directory to save the files
        embeddings (np.ndarray): Array of embeddings of shape (n_images, embedding_dim)
        global_index2img_path (Dict[int, str]): Dictionary of index:img_path 
    """
    
    os.makedirs(output_dir, exist_ok= True)
    np.save(os.path.join(output_dir, 'all_image_embeddings.npy'), all_embeddings)
    with open(os.path.join(output_dir, 'global_index2img_path.json'), 'w') as f:
        json.dump(global_index2img_path, f, indent = 4)


def load_embeddings(input_dir:str)-> Tuple[np.ndarray, Dict[int, str]]:
    """loading the embeddings and the json 

    Args:
        input_dir (str): Path to the embedding information

    Returns:
        Tuple[np.ndarray, Dict[int, str]]: return a tuple, including the matrix embeddings vector and the dictionary of globalindex2imgpath
    """
    all_embeddings = np.load(
        os.path.join(input_dir, 'all_image_embeddings.npy')
    )
    with open(os.path.join(input_dir, 'global_index2img_path.json'), 'r') as f:
        global_index2img_path = json.load(f)
    
    return all_embeddings, global_index2img_path


# if __name__ == "__main__":

#     ROOT_PATH = './data/processed/train'
#     CLASS_NAME = sorted(list(os.listdir(ROOT_PATH)))

#     encoder = CLIPEncoder()

#     matrix_embeddings, global_index2img_path = precomputing_embeddings(
#         root_path= ROOT_PATH,
#         folder_list_name= CLASS_NAME,
#         encoder= encoder
#     )
#     embedding_dir = './data/embeddings'

#     save_embeddings(
#         output_dir= embedding_dir,
#         all_embeddings= matrix_embeddings,
#         global_index2img_path=global_index2img_path
#     )

#     hehe1, hehe2 = load_embeddings(embedding_dir)
#     print(hehe1.shape)
#     print(isinstance(hehe2, dict))
#     print(hehe2['0'])