import os
import tempfile
import zipfile

import requests


def download_model_and_unzip(url):

    # Create a temporary directory
    model_dir = tempfile.mkdtemp()

    # Download the zip file
    r = requests.get(url, allow_redirects=True)
    zip_path = os.path.join(model_dir, 'model.zip')
    open(zip_path, 'wb').write(r.content)

    if not os.path.exists(zip_path):
        raise ValueError(f"Failed to download model from {url}")

    # Unzip the zip file into the temporary directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

    # Remove the zip file
    os.remove(zip_path)

    # Verify model downloaded by checking if dir is empty
    if len(os.listdir(model_dir)) == 0:
        raise ValueError(f"Failed to unzip model from {url}")

    # Return the temporary directory
    return model_dir