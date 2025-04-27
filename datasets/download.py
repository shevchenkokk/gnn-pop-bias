import os
import argparse
import requests
import zipfile
from pathlib import Path

DATASET_URLS = {
    "ml-32m":   "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    "gowalla": "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
    "yelp2018":("https://raw.githubusercontent.com/hexiangnan/"
                "neural_graph_collaborative_filtering/master/Data/"
                "yelp2018/yelp2018.inter"),
}


def download_and_extract(root: Path, name: str, url: str, ext='.zip'):
    """
    Download archive by url in root/name
    """
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_path = out_dir / f"{name}{ext}"
    marker = out_dir / f".{name}.done"
    print(marker)
    if marker.exists():
        print(f"--- {name} dataset has already been downloaded ---")
        return

    print(f"--- Downloading the {name} dataset ---")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"--- Extracting {archive_path} in {out_dir}  ---")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    marker.write_text('')


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract datasets to datasets/ directory"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ml-32m", "yelp", "lastfm"],
        help="List of datasets: ml-32m, yelp, lastfm"
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
