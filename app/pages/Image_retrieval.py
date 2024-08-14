import streamlit as st
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import math
import tempfile
import random
from utils.image_processing import find_similarities, get_random_image_from_test
import sys

st.session_state.current_page = "Image Retrieval"

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








st.markdown("""
<style>
    
    .stButton {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state() -> None:
    """
    Initialize session state variables if they don't exist
    """

    
    if 'manual_image' not in st.session_state: 
        st.session_state.manual_image = None
    if 'random_image' not in st.session_state:
        st.session_state.random_image = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None


    if 'query_path_manual' not in st.session_state:
        st.session_state.query_path_manual = None
    if 'query_path_random' not in st.session_state:
        st.session_state.query_path_random = None
    if 'query_path_upload' not in st.session_state:
        st.session_state.query_path_upload = None
    
    if 'process_started' not in st.session_state:
        st.session_state.process_started = False
    if 'top_k_return' not in st.session_state:
        st.session_state.top_k_return = 15

   

def display_tags(tags: list) -> None:
    """
    Display a list of tags in a visually appealing manner.

    Args:
    tags (list): A list of strings representing tags.
    """
    palette = ['#ffadad', '#ffd6a5', '#fdffb6', '#caffbf', '#9bfbc0', '#a0c4ff', '#b9b9f5', '#f5a1a1']
    tags_html = ""
    for tag in tags:
        bg_color = random.choice(palette)
        font_color =    '#000000'
        tags_html += f'<div class="tag" style="background-color: {bg_color}; color: {font_color};">{tag}</div>'
    
    st.markdown("""
    <style>
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .tag {
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 14px;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="tag-container">{tags_html}</div>', unsafe_allow_html=True)

def display_image(image: np.ndarray, size: Tuple[int, int] = (300, 300)) -> None:
    """
    display an image with the desire size
    The image must be in BGR 
    """
    image_resize = cv2.resize(image, size)[:, :, ::-1]
    st.image(
        image= image_resize,
    )
    
def select_image_from_test_folder(folder_path: str) -> None:
    """Display images from a folder and allow user to select one. Use for test set selection

    Args:
        folder_path (str): the path to the image folder

    Returns:
        _type_: None
    """
    image_name = os.listdir(folder_path)[0] # test folder only has 1 image
    img_path = os.path.join(folder_path, image_name)
    img = cv2.imread(img_path)
    tile = st.container(height=350)
    with tile:
        display_image(img)
    if st.button(
        "Select Image", key = 'select_btn'
    ):
        st.session_state.manual_image = img
        st.session_state.query_path_manual = img_path
        st.session_state.process_started = False

def display_similar_images(
    similar_image_score: List[Tuple[str, float]]
):
    """Display summary images with their scores

    Args:
        similar_image_score (List[Tuple[str, float]]): A list of tuples, where the tuple contains the image_path, and the confidence score

    Returns:
        _type_: None
    """
    similar_image_score = sorted(similar_image_score, key=lambda x: x[1], reverse=True)
    col = st.columns(2)
    number_of_row  = math.ceil(st.session_state.top_k_return / 2)
    rows = [st.columns(2) for _  in range(number_of_row)]
    for row in rows:
        col += row


    for i, ax in enumerate(col):
        if i < len(similar_image_score):
            tile = ax.container(height= 350)
            with tile:
                path, score = similar_image_score[i]
                image = cv2.imread(path)[:, :, ::-1]
                image_resize = cv2.resize(image, (700, 700))
                st.image(image_resize, caption=score)



def handle_uploaded_image(uploaded_file) -> Optional[str]:
    """
    Handle uploaded file, an object from Streamlit object, and return the path to that file, for image reading
    """    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            return temp_file.name
    return None


def main():

    
    st.title("Image Retrieval System")
    st.write("""
    Welcome to the Image Retrieval System! This application demonstrates the power of CLIP embeddings
    and vector databases for efficient image similarity search. You can either choose a test image or
    upload your own to find similar images in our database.
    """)
    
    initialize_session_state()

    choosing_options = st.selectbox(
        "Options: Either upload your own images, or you can select an image in the databse(manually or randomly) from our test set.",
        ("Choose an image from the testset", "Upload images" )
    )

    if choosing_options == 'Choose an image from the testset':
        st.subheader("Option 1: Choose a test image")
        choice = st.radio("Select method", ["Choose manually", "Random selection"])

        if choice == "Choose manually":
            test_path = "./data/processed/test"
            folder = os.listdir(test_path)
            selected_folder = st.selectbox('Select a folder (Please do not choose too fast. When changing the folders, the image displaying might get delayed!)', folder)
            selected_folder_path = os.path.join(test_path, selected_folder)
            select_image_from_test_folder(selected_folder_path)

            if st.session_state.manual_image is not None:
                folder_name_selected = os.path.split(os.path.dirname(st.session_state.query_path_manual))[1]
                st.success(f"Image selected from {folder_name_selected} folder")
                top_k = st.selectbox(
                    "Top k Image retrieval",
                    ("5 images", "10 images", "20 images", "30 images")
                )
                st.session_state.top_k_return = int(top_k.split(" ")[0])

                if st.button(
                    "Start Image Retrieval", key = 'retrieval_manual'
                ):
                    st.session_state.process_started = True
                   
                
                if st.session_state.process_started:
                    with st.status(
                        "Finding the best images to display (loading the model to memory), please wait!"
                    ) as Status:
                        similar_image_score_tuple = find_similarities(
                            encoder= st.session_state.encoder,
                            database= st.session_state.database,
                            query_path= st.session_state.query_path_manual,
                            top_k= st.session_state.top_k_return
                        )
                        Status.update(
                            label = "List of images found! Search completed! The results might not be semantically identical to the query images, due to the usage of old models.", state='complete', expanded= False
                        )
                    display_similar_images(similar_image_score=similar_image_score_tuple)
                    if st.button("Make a new selection", key='new_selection_manual'):
                        st.session_state.manual_image = None
                        st.session_state.query_path_manual = None
                        st.session_state.process_started = False
                        st.rerun()

        else:  # Random selection
            if st.button(
                label="Select Random Image",
                key= 'btn_random'
            ):
                image_path = get_random_image_from_test()
                image_read = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_RGB2BGR)
                st.session_state.random_image = image_read
                st.session_state.query_path_random = image_path
                st.session_state.process_started = False

            if st.session_state.random_image is not None:
                st.image(st.session_state.random_image)
                folder_name_selected = os.path.split(os.path.dirname(st.session_state.query_path_random))[1]
                st.success(f"Image selected from {folder_name_selected} folder")
                top_k = st.selectbox(
                    "Top k Image retrieval",
                    ("5 images", "10 images", "20 images", "30 images")
                )
                st.session_state.top_k_return = int(top_k.split(" ")[0])

                if st.button(
                    "Start Image Retrieval", key = 'retrieval_random', disabled= st.session_state.process_started
                ):
                    st.session_state.process_started = True
                
                if st.session_state.process_started:
                    with st.status(
                        "Finding the best images to display (loading the model to memory), please wait!"
                    ) as Status:
                        similar_image_score_tuple = find_similarities(
                            encoder= st.session_state.encoder,
                            database= st.session_state.database,
                            query_path= st.session_state.query_path_random,
                            top_k= st.session_state.top_k_return
                        )
                        Status.update(
                            label = "List of images found! Search completed! The results might not be semantically identical to the query images, due to the usage of old models.", state='complete', expanded= False
                        )
                    display_similar_images(similar_image_score=similar_image_score_tuple)
                    if st.button("Make a new selection", key='new_selection_random'):
                        st.session_state.random_image = None
                        st.session_state.query_path_random = None
                        st.session_state.process_started = False
                        st.rerun()

    elif choosing_options == "Upload images":
        st.subheader("Option 2: Upload your image. Considering")
        st.write("For better retrieval, considering using these tags as a choice for image retrieval")

        ROOT = './data/processed'
        CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))
        st.markdown("""
        #### Recommendation of tags
        """)
        tile = st.container(height=350)
        with tile:
            display_tags(CLASS_NAME)

        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "png", "jpeg"]
        )
        if uploaded_file is not None:
            temp_file_path = handle_uploaded_image(uploaded_file)
            
            if temp_file_path is not None:
                image_read = cv2.cvtColor(cv2.imread(temp_file_path), cv2.COLOR_RGB2BGR)
                st.session_state.uploaded_image = image_read
                st.session_state.query_path_upload = temp_file_path
                st.session_state.process_started = False

            if st.session_state.uploaded_image is not None:
                st.image(st.session_state.uploaded_image)
                folder_name_selected = os.path.split(os.path.dirname(st.session_state.query_path_upload))[1]
                st.success(f"Image selected from {folder_name_selected} folder")
                top_k = st.selectbox(
                    "Top k Image retrieval",
                    ("5 images", "10 images", "20 images", "30 images")
                )
                st.session_state.top_k_return = int(top_k.split(" ")[0])

                if st.button(
                    "Start Image Retrieval", key = 'retrieval_random', disabled= st.session_state.process_started
                ):
                    st.session_state.process_started = True
                
                if st.session_state.process_started:
                    with st.status(
                        "Finding the best images to display (loading the model to memory), please wait!"
                    ) as Status:
                        similar_image_score_tuple = find_similarities(
                            encoder= st.session_state.encode,
                            database= st.session_state.database,
                            query_path= st.session_state.query_path_upload,
                            top_k= st.session_state.top_k_return
                        )
                        Status.update(
                            label = "List of images found! Search completed! The results might not be semantically identical to the query images, due to the usage of old models.", state='complete', expanded= False
                        )
                    display_similar_images(similar_image_score=similar_image_score_tuple)
                    if st.button("Make a new selection", key='new_selection_random'):
                        st.session_state.uploaded_image = None
                        st.session_state.query_path_upload = None
                        st.session_state.process_started = False
                        st.rerun()

if __name__ == "__main__":
    main()