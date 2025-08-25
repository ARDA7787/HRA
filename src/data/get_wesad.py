#!/usr/bin/env python3
"""
Download WESAD (~2.5GB) without login and extract.
Link: https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/index.html.en?lang=en
Direct zip: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx (served via web UI; we will instruct user to manually download if curl fails)
"""
import argparse
import os
import sys
import shutil
import zipfile
from pathlib import Path

try:
    import requests
except Exception:
    requests = None


def download_file(url: str, dest_path: Path) -> bool:
    if requests is None:
        print("requests not installed; please install it or download manually:")
        print(url)
        return False
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("If this link requires a browser redirect, please download manually from:")
        print("https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/index.html.en?lang=en")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data_raw", help="Output directory for raw data")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The sciebo link may require interactive browser; provide instruction.
    zip_path = out_dir / "WESAD.zip"

    print("Attempting direct download of WESAD. If it fails, follow manual instructions below.")
    url = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"  # typical sciebo pattern
    ok = download_file(url, zip_path)

    if not ok:
        print("\nManual download steps:")
        print(
            "1) Open: https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/index.html.en?lang=en"
        )
        print("2) Click 'ICMI'18 dataset (2.5 GB zipped)'")
        print("3) Download the zip and place it at:", zip_path)
        print("4) Re-run this script to extract.")
        if not zip_path.exists():
            sys.exit(1)

    # Extract
    extract_dir = out_dir / "WESAD"
    if extract_dir.exists():
        print(f"Folder already exists: {extract_dir}")
        sys.exit(0)

    print("Extracting zip (this may take a while)...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("Done. Data at:", extract_dir)
    except zipfile.BadZipFile:
        print("Zip extraction failed. The downloaded file may be HTML (redirect) or corrupted.")
        print("Please download manually using a browser, then run again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
