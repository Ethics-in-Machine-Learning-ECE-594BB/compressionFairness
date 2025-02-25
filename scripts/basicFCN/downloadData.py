import os
import requests

# Define the URL and file paths
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
DATA_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
DATA_DIR = "/Users/ethan3048/Documents/school/winter25/ece594bbEthics/compressionFairness/data/raw"
TRAIN_FILE = os.path.join(DATA_DIR, "adult.csv")
TEST_FILE = os.path.join(DATA_DIR, "adult_test.csv")

# Ensure the directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, save_path):
    """Downloads a file from a given URL and saves it to the specified path."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download {url}")

# Download the training and test data
download_file(DATA_URL, TRAIN_FILE)
download_file(DATA_TEST_URL, TEST_FILE)

print("Adult Income dataset download complete.")
