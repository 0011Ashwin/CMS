import os
import requests
import subprocess
from pathlib import Path

# Configuration
DATASET_DIR = Path("D:/datasets")
ENV_FILE = Path("env.sh")

# URLs (Using typical public links for demonstration, replace with official Zenodo/CERN links if specific ones required)
# JetClass is hosted on Zenodo (Record 6619768). Part 0 is usually the first file.
# QuarkGluon is another standard dataset.
DATASETS = {
    "JetClass": {
        "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part0.tar?download=1", 
        "filename": "JetClass_Pythia_train_100M_part0.tar"
    },
    "QuarkGluon": {
        "url": "https://zenodo.org/record/3164691/files/QG_jets.npz",
        "filename": "QG_jets.npz"
    }
}

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"File {dest_path} already exists. Skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")
    try:
        from tqdm import tqdm
        # standard optimized download
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        print(f"Downloaded {dest_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def update_env_file():
    # Update or create env.sh
    if ENV_FILE.exists():
        content = ENV_FILE.read_text()
    else:
        content = ""
    
    export_cmd = f'export DATADIR="{DATASET_DIR.as_posix()}"'
    
    if export_cmd not in content:
        with open(ENV_FILE, "a") as f:
            f.write(f"\n{export_cmd}\n")
        print(f"Updated {ENV_FILE} with DATADIR")
    else:
        print(f"{ENV_FILE} already contains DATADIR")

def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, info in DATASETS.items():
        dest = DATASET_DIR / info["filename"]
        download_file(info["url"], dest)
        
    update_env_file()
    print("Data acquisition complete.")

if __name__ == "__main__":
    main()
