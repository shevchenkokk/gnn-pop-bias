import os
import argparse
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

DATASET_URLS = {
    "ml-32m":   "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    "gowalla": "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
    "yelp2018": ("https://raw.githubusercontent.com/hexiangnan/"
                "neural_graph_collaborative_filtering/master/Data/"
                "yelp2018/yelp2018.inter"),
    "lastfm": "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
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

def download_and_extract(root: Path, name: str, url: str):
    """
    Download archive by url in root/name
    """
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if url.endswith('.zip'):
        ext = '.zip'
    elif url.endswith('.gz'):
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
            zip_contents = [f for f in zip_ref.namelist() if not f.endswith('/')]
            common_prefix = os.path.commonpath(zip_contents) if zip_contents else ''
            if common_prefix and common_prefix.endswith('/'):
                temp_extract_path = out_dir / f"temp_extract_{name}"
                temp_extract_path.mkdir(exist_ok=True)
                zip_ref.extractall(temp_extract_path)

                source_path = temp_extract_path / common_prefix
                for item in source_path.iterdir():
                    shutil.move(str(item), str(out_dir))
                shutil.rmtree(temp_extract_path)
                print(f"--- Flattened extraction for {name} ---")
            else:
                zip_ref.extractall(out_dir)
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
        default=["ml-32m", "gowalla", "yelp", "lastfm"],
        help="List of datasets: ml-32m, gowalla, yelp, lastfm"
    )

    parser.add_argument(
        "--root",
        type=str,
        default=os.path.dirname(__file__),
        help="Root dir to download"
    )
    
    args = parser.parse_args()
    root = Path(args.root)

    for name in args.datasets:
        if name not in DATASET_URLS:
            print(f"Unknown dataset: {name}")
            continue
        download_and_extract(root, name, DATASET_URLS[name])

if __name__ == '__main__':
    main()