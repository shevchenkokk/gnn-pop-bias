import os
import argparse
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import kagglehub

DATASETS_INFO = {
    "ml-1m": {"type": "url", "value": "https://files.grouplens.org/datasets/movielens/ml-1m.zip"},
    "kion": {"type": "url", "value": "https://github.com/irsafilo/KION_DATASET/blob/main/data_en.zip?raw=true"},
    "lastfm": {"type": "url", "value": "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"},
    "yelp2021": {
        "type": "kagglehub",
        "ref": "yelp-dataset/yelp-dataset/versions/3"
    }
}


def download_with_progress(url, file_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(file_path, "wb") as f, tqdm(
        desc=f"Downloading {file_path.name}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def download_kaggle_with_hub(root: Path, name: str, hub_info: dict):
    """
    Download and extract Kaggle dataset using Kaggle API.
    """
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / f".{name}.done"

    if marker.exists():
        print(f"--- {name} dataset has already been downloaded and processed ---")
        return

    try:
        current_cwd = os.getcwd()
        os.chdir(out_dir)

        downloaded_path_hub = kagglehub.dataset_download(hub_info['ref'])
        downloaded_path_hub = Path(downloaded_path_hub)

        if downloaded_path_hub.is_dir():
            print(f"--- Copying files from {downloaded_path_hub} to {out_dir} ---")
            for item in os.listdir(downloaded_path_hub):
                s = downloaded_path_hub / item
                d = out_dir / item
                if s.is_dir():
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            print(f"--- Copying completed  ---")
        else:
            # Если скачался один файл, а не директория (редко для датасетов)
            shutil.copy2(downloaded_path_hub, out_dir / downloaded_path_hub.name)
            print(f"--- File {downloaded_path_hub.name} was copied to {out_dir} ---")
        
    except Exception as e:
        print(f"Error when downloading Kaggle dataset '{hub_info['ref']}' within kagglehub: {e}")
        print("Please make sure that your Kaggle API key is configured correctly and has access to the dataset..")
        return
        
    marker.write_text('')
    print(f"--- {name} dataset successfully processed ---")

def download_and_extract(root: Path, name: str, url: str):
    """
    Download archive by url in root/name
    """
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = url.split('?')[0]

    if base_url.endswith('.zip'):
        ext = '.zip'
    elif base_url.endswith('.gz'):
        ext = '.gz'
    else:
        ext = ''

    archive_path = out_dir / f"{name}{ext}"
    marker = out_dir / f".{name}.done"

    if marker.exists():
        print(f"--- {name} dataset has already been downloaded ---")
        return

    print(f"--- Downloading the {name} dataset ---")
    download_with_progress(url, archive_path)

    if ext == '.zip':
        print(f"--- Extracting {archive_path} in {out_dir} ---")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            namelist = zip_ref.namelist()
            all_top_level_components = {
                member.split(os.sep)[0] + os.sep
                for member in namelist
                if os.sep in member and member.split(os.sep)[0] != ''
            }

            common_root_to_strip = ""
            if len(all_top_level_components) == 1:
                potential_root = list(all_top_level_components)[0]
                if potential_root in namelist or any(m.startswith(potential_root) for m in namelist if not m.endswith('/')):
                    common_root_to_strip = potential_root

            for member in namelist:
                if member.endswith('/') and member != common_root_to_strip:
                        continue

                if common_root_to_strip and member.startswith(common_root_to_strip):
                    target_path = out_dir / member[len(common_root_to_strip):]
                else:
                    target_path = out_dir / member

                # Create parent directories if they don't exist
                if target_path.parent:
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract only files, not directories directly
                if not member.endswith('/'):
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
        print(f"--- Extraction for {name} complete ---")
    elif ext == '.gz':
        print(f"--- Extracting {archive_path} in {out_dir} ---")
        output_file = out_dir / f"{name}.txt"
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        # For direct files, just keep as is
        print(f"--- No extraction needed for {name} ---")
        output_file = out_dir / f"{name}.inter"
        shutil.copy(archive_path, output_file)
    marker.write_text('')
    if os.path.exists(archive_path):
        os.remove(archive_path)
        print(f"--- Removed archive file {archive_path} ---")
    print(f"--- {name} dataset successfully processed ---")

def main():
    parser = argparse.ArgumentParser(
        description="Download and extract datasets to datasets/ directory"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ml-1m", "kion", "yelp2021", "lastfm"],
        help="List of datasets: ml-1m, kion, yelp2021, lastfm"
    )

    parser.add_argument(
        "--root",
        type=str,
        default="datasets/",
        help="Root dir to download"
    )
    
    args = parser.parse_args()
    root = Path(args.root)

    for name in args.datasets:
        if name not in DATASETS_INFO:
            print(f"Unknown dataset: {name}")
            continue
        info = DATASETS_INFO[name]
        if info["type"] == "url":
            download_and_extract(root, name, info["value"])
        elif info["type"] == "kagglehub":
            download_kaggle_with_hub(root, name, info)

if __name__ == '__main__':
    main()