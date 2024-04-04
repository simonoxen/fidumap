import os
import requests
from io import BytesIO
from zipfile import ZipFile

def download_data(dest_folder):
    if not os.path.exists(dest_folder):
        print('Downloading models data...')
        response = requests.get('https://github.com/simonoxen/fidumap/releases/download/v0.1/data.zip')
        with ZipFile(BytesIO(response.content)) as zipf:
            zipf.extractall(dest_folder)


models_dir = os.path.join(os.path.dirname(__file__), 'data')

download_data(models_dir)