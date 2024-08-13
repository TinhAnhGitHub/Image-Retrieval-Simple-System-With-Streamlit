"""
This script precomputes image embeddings for a given dataset using a CLIP model encoder.

The main functionality provided by this script is encapsulated in the `precompute_embeddings` function, which:
- Iterates through a list of class names to locate images within each class directory.
- Uses the provided `CLIPEncoder` instance to generate embeddings for each image.
- Aggregates these embeddings and the corresponding image paths into a list.

Key Features:
- Efficiently processes large datasets by leveraging the `tqdm` library to display a progress bar during computation.
- Ensures that the embeddings and their associated image paths are returned as a tuple, making them easy to store or further process.

Dependencies:
- numpy
- tqdm
- CLIPEncoder class from the `src.retrieval.model` module

Example usage:
    encoder = CLIPEncoder()
    root_path = "/path/to/dataset"
    class_names = ["class1", "class2", "class3"]
    embeddings, image_paths = precompute_embeddings(root_path, class_names, encoder)
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


from typing import List, Tuple, Dict
from src import CLIPEncoder
from tqdm import tqdm
import numpy as np

def precomputing_embeddings(
    root_path: str,
    folder_list_name: List[str],
    encoder: CLIPEncoder
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Precompute embeddings for all images in the given class

    Args:
        root_path (str): Path to the images, used for retrieval. This images could be places in data/processed/train
        folder_list_name (List[str]): List of folder names
        encoder (CLIPEncoder): CLIPEncoder class for image embedding

    Returns:
        Tuple[np.ndarray, ]: return a Tuple(matrix, Dict of imagePath with global index), where the matrix's shape is (len(image_paths), n_dimension=512)
    """

    all_embeddings = []
    global_index2img_path = {}
    i = 0 

    for folder_name in tqdm(
        folder_list_name,
        desc="Computing Embeddings...",
        unit= "images"  
    ):
        folder_path = os.path.join(root_path, folder_name)
        for img_name in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            image = encoder.read_image_from_path(img_path)
            image_np = encoder.get_single_image_embedding(image = image)
            all_embeddings.append(image_np)
            global_index2img_path[i] = img_path
            i += 1
    
    return np.array(all_embeddings), global_index2img_path

# if __name__ == "__main__":
#     ROOT_PATH = './data/processed/train'
#     CLASS_NAME = sorted(list(os.listdir(ROOT_PATH)))

#     encoder = CLIPEncoder()

#     matrix_embeddings, global_index2img_path = precomputing_embeddings(
#         root_path= ROOT_PATH,
#         folder_list_name= CLASS_NAME,
#         encoder= encoder
#     )
#     print(matrix_embeddings.shape)
#     print(global_index2img_path)