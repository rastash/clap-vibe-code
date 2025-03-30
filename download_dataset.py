import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file from a URL to a destination with progress bar
    """
    if os.path.exists(destination):
        print(f"{destination} already exists. Skipping download.")
        return
    
    print(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a destination
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        
    print(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting"):
            zip_ref.extract(member, extract_to)

def main():
    # Create data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Download ESC-50 dataset
    dataset_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = os.path.join(data_dir, "ESC-50.zip")
    
    download_file(dataset_url, zip_path)
    extract_zip(zip_path, data_dir)
    
    print("Dataset downloaded and extracted successfully!")
    print(f"Dataset location: {os.path.join(data_dir, 'ESC-50-master')}")

if __name__ == "__main__":
    main()
