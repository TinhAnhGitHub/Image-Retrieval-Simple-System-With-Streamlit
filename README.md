# Image-Retrieval-Simple-System-With-Streamlit

## The application is under developement
- [x] CLIP Testing
- [x] Basic utilities (image similarity)
- [x] ChromaDB for retrieval
- [x] Provide a detailed testing for each utilities, functions
- [x] Building Basic Streamlit application with Similarity search
- [ ] Selenium to scrape images to build database with ChromaDB (flickr.com)
- [ ] Adding AutoWebscraping for building database, based on different image search engine
## Folder Structure Template Usage
```txt
Version 1 Folder Structure
image_retrieval_project/
│
├── data/
│   ├── raw/                 # Raw scraped images
│   ├── processed/           # Processed images
│   │   ├── train/
│   │   └── test/
│   └── embeddings/          # Stored image embeddings from CLIP
│
├── src/
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── similarity.py    # Similarity measurement functions
│   │   ├── model.py         # CLIP model wrapper
│   │   └── database.py      # Vector database operations: Chroma
│   │
│   ├── scraping/
│   │   ├── __init__.py
│   │   ├── scraper.py       # Web scraping functionality
│   │   └── url_extractor.py # URL extraction methods
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py # Data cleaning and preprocessing
│   │   └── loader.py        # Data loading utilities
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py       # Miscellaneous helper functions
│
├── notebooks/
│   ├── data_exploration.ipynb #
│   └── model_evaluation.ipynb
|   └── Clip_embedding.ipynb
│
├── scripts/
│   ├── scrape_images.py
│   └── process_data.py
│
├── tests/
│   ├── test_retrieval.py
│   ├── test_scraping.py
│   └── test_data.py
│
├── app/
│   ├── main.py              # Streamlit app main file
│   ├── pages/               # Streamlit multipage app structure
│   │   ├── home.py
│   │   ├── scrape.py
│   │   └── retrieve.py
│   └── components/          # Reusable Streamlit components
│       └── image_display.py
│
├── config/
│   └── config.yaml          # Configuration files
│
├── requirements.txt
├── README.md
└── .gitignore
```


## 2. Installation
### 2.1. Clone this repo and install the requirements
```bash
git clone https://github.com/TinhAnhGitHub/Image-Retrieval-Simple-System-With-Streamlit.git
cd Image-Retrieval-Simple-System-With-Streamlit
pip install -r requirements.txt
```

### 2.2. Run the application
```bash
streamlit run ./app/Home.py 
```