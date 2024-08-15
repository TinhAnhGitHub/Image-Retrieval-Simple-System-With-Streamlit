import streamlit as st
import os
import sys
import chromadb


st.set_page_config(
    page_title="Basic Image Retrieval System",
    page_icon= "🐧",
    layout='wide',
)


if 'database' not in st.session_state:
    st.session_state.database = None

if 'encoder' not  in st.session_state:
    st.session_state.encoder = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

st.session_state.current_page = 'Home'

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

from src.data import load_embeddings
from src.retrieval import CLIPEncoder, ImageDatabase
EMBEDDING_DIR = './data/embeddings'

if st.session_state.database is None:
   
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
    st.session_state.database = database

if st.session_state.encoder is None:
    with st.spinner("Loading the model to memory, please wait"):
        encoder = CLIPEncoder()
    st.session_state.encoder = encoder



if 'first_run' not in st.session_state:
    st.session_state.first_run = True



st.title("Main Features")

st.sidebar.title("About the Developer")
st.sidebar.markdown(
    """
    This simple application is developed by TA.
    - [GitHub](https://github.com/TinhAnhGitHub)
    - [Facebook](https://www.facebook.com/nguyennhutinhanh/)
    """
)
if st.session_state.first_run or st.session_state.current_page == "Home":
    
    st.write("Welcome to the Basic Image Retrieval System!")
    st.write("Go ahead and navigate to Image_retrival to try out the system!")

    st.markdown("### Feature Development Checklist")
    st.markdown("""
    - [x] **Home Page**: Display the main features of the system.
    - [x] **Image Retrieval System**: Implement a system for searching and retrieving similar images (Test image picked only)
    - [ ] **User Upload Functionality**: Allow users to upload images for searching.
    - [ ] **Image Browser**: Develop a browser for users to explore the image database.
    - [ ] **Performance Optimization**: Optimize the system for faster image retrieval and processing.
    - [ ] **UI/UX Enhancements**: Improve the user interface and experience.
    """)
    
    st.session_state.first_run = False
else:
    from pages import Image_browser, Image_retrieval
    
    
