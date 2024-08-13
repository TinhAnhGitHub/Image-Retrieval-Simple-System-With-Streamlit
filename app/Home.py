import streamlit as st




st.set_page_config(
    page_title="Basic Image Retrieval System",
    page_icon= "🐧",
    layout='wide',
)

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
if st.session_state.first_run:
    
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
