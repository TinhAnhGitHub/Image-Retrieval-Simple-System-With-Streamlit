import os
import json
import time
from typing import List
import urllib.request
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
import cv2
from PIL import Image
import io
import sys


def get_project_root():
    current_abspath = os.path.abspath('./')
    while True:
        if os.path.split(current_abspath)[1] == 'Image-Retrieval-Simple-System-With-Streamlit':
            project_root = current_abspath
            break
        else:
            current_abspath = os.path.dirname(current_abspath)
    return project_root

PROJECT_ROOT = get_project_root()
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

CHROME_DRIVER = './chromedriver.exe'

class URLScraper:
    """A class to scrape images URLs from Flickr."""
    def __init__(
        self,
        url_template: str,
        max_images: int = 50, 
        max_workers: int = 4,

    ):
        """Initialize the URLScraper

        Args:
            url_template (str): The URL template for Flickr search
            max_images (int, optional): Maximum number of images to scrape per term. Defaults to 50.
            max_workers (int, optional): Maximum number of concurrent threads. Defaults to 4.
        """
        self.url_template = url_template
        self.max_images = max_images
        self.max_workers = max_workers
    
    def setup_environment(self):
        """Set up the environment for Selenium."""
        service = Service(CHROME_DRIVER)
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--disable-infobars")
        self.options.add_argument("--start-maximized")
        self.options.add_argument("--disable-extensions")
        self.options.add_argument('--window-size=1920,1080')
        self.options.add_argument('--no-sandbox') # Docker environment
        self.options.add_argument('--disable-notifications')
        self.options.add_argument('--disable-infobars')
        self.driver = webdriver.Chrome(
            service= service,
            options= self.options
        )

    def get_url_images(self, term: str) -> List[str]:
        """Scrape image URLs for a given search term

        Args:
            term (str): The search term

        Returns:
            List[str]: A list of image URLs.
        """
        url = self.url_template.format(search_term = term)
        self.driver.get(url=url)
        urls = []
        more_content_available = True

        with tqdm(total = self.max_images, desc=f"Fetching Images for {term}", unit= "image") as pbar:
            while len(urls) <= self.max_images:
                ...





