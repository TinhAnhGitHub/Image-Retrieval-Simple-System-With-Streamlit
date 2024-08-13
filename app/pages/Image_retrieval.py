import streamlit as st
import os
import cv2
import math
from utils.image_processing import find_similarities
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

st.markdown("""
<style>
    
    .stButton {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
if 'query_path' not in st.session_state:
    st.session_state.query_path = None
if 'process_started' not in st.session_state:
    st.session_state.process_started = False
if 'top_k_return' not in st.session_state:
    st.session_state.top_k_return = 5



st.title("Image Retrieval System")
st.write("""
Welcome to the Image Retrieval System! This application demonstrates the power of CLIP embeddings
and vector databases for efficient image similarity search. You can either choose a test image or
upload your own to find similar images in our database.
""")

#option1: choose from test images

choosing_options = st.selectbox(
    "Options: Either upload your own images, or you can select an image/random image from out test set.",
    ("Upload images", "Choose an image from the testset")
)

if choosing_options == 'Choose an image from the testset':
    st.subheader("Option 1: Choose a test image")
    choice = st.radio("Select method", ["Choose manually", "Random selection"])

    if choice == "Choose manually":
        test_path = "./data/processed/test"
        folder = os.listdir(test_path)
        selected_folder = st.selectbox('Select a folder(Please do not choose to fast. When changing the folders, the image displaying might get delayed!)', folder)
        selected_folder_path = os.path.join(test_path, selected_folder)
 
           
        selected_image = None
        for i, image in enumerate(os.listdir(selected_folder_path)):
            img_path = os.path.join(selected_folder_path, image)
            image = cv2.imread(img_path)[: , :, ::-1]

            height, width = image.shape[:2]
            max_dim = 200
            scale = max_dim / max(height, width)
            new_width = int(scale * width)
            new_height = int(scale * height)

            img_resize = cv2.resize(image, (new_height, new_width))
            
            st.image(img_resize, use_column_width=False, width=new_width)
            
            if st.button("Select image", key=f"btn_{i}"):
                st.session_state.selected_image = image
                st.session_state.query_path = img_path
                st.session_state.process_started = False
            st.markdown('</div>', unsafe_allow_html=True)


        if st.session_state.selected_image is not None:
            st.success(f"Image selected from {selected_folder} folder")
            query_path = img_path

            top_k = st.selectbox(
                "Top k Image retrieval",
                ("5 images", "10 images", "20 images", "30 images")
            )
            st.session_state.top_k_return = int(top_k.split(" ")[0])

            if st.button(
                "Verify top k image display selection",
                key= 'top_k',
                disabled= st.session_state.process_started
            ):
                st.session_state.process_started = True

            # Move this block outside of the button click condition
            if st.session_state.process_started:
                with st.spinner("Finding the best images to display, please wait!"):
                    similary_image_score = find_similarities(
                        query_path=query_path,
                        top_k= st.session_state.top_k_return
                    )

                st.success("Searching Successfully!")

                col = st.columns(2)
                row = st.columns(math.ceil(st.session_state.top_k_return     / 2))

                for i, ax in enumerate(col + row):
                    if i < len(similary_image_score):
                        tile = ax.container(height=350)
                        with tile:
                            metadata_select = similary_image_score[i]
                            temper_path = metadata_select[0]
                            temper_score =  metadata_select[1]
                            image = cv2.imread(temper_path)[:, :, ::-1]
                            image_resize = cv2.resize(image, (700, 700))
                            print(image_resize.shape)

                            st.image(image_resize, caption=temper_score)

                if st.button("Make new selection"):
                    st.session_state.process_started = False
                    st.session_state.selected_image = None
                    st.session_state.query_path = None
                    st.rerun()


    else:
        
        st.write("⚠️ Under developement")


elif choosing_options == "Upload images":
    st.write("Under development : D")
           

