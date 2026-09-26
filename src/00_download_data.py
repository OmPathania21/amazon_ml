"""Step 0: download the challenge zip from Google Drive and arrange the data.

Result:
  data/train/train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
  data/test/test_source1.tsv,  test_source2.tsv,  test_source3.tsv

Usage:
  python src/00_download_data.py                  # download + unzip + arrange
  python src/00_download_data.py --zip file.zip   # already downloaded by hand
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DRIVE_FILE_ID = "10kOaB9eVp0096S069W8a8mT9-IEm3m8f"

EXPECTED = {
    "train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"],
    "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"],
}


def all_present():
    return all((DATA_DIR / split / name).is_file() for split, names in EXPECTED.items() for name in names)


def download(dest):
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is missing: run  pip install -r requirements.txt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading Google Drive file {DRIVE_FILE_ID} -> {dest}")
    out = gdown.download(id=DRIVE_FILE_ID, output=str(dest), quiet=False)
    if not out or not dest.is_file():
        sys.exit("Download failed (Drive quota or permissions). Download the zip in a browser from\n"
                 f"  https://drive.google.com/file/d/{DRIVE_FILE_ID}/view\n"
                 "and run:  python src/00_download_data.py --zip <path to zip>")


def arrange(zip_path, keep_extract=False):
    if not zipfile.is_zipfile(zip_path):
        sys.exit(f"{zip_path} is not a zip file (the download may be an HTML error page).")
    extract_dir = DATA_DIR / "_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)

    missing = []
    for split, names in EXPECTED.items():
        (DATA_DIR / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            hits = [p for p in extract_dir.rglob(name) if "__MACOSX" not in p.parts]
            if not hits:
                missing.append(name)
                continue
            shutil.move(str(hits[0]), DATA_DIR / split / name)

    # keep the organisers' helper files (validator, template) next to the data
    for extra in ["validate_submission.py", "Documentation_template.md", "README.md"]:
        hits = [p for p in extract_dir.rglob(extra) if "__MACOSX" not in p.parts]
        if hits:
            shutil.copy2(hits[0], DATA_DIR / extra)

    if not keep_extract:
        shutil.rmtree(extract_dir, ignore_errors=True)
    if missing:
        sys.exit(f"these files were not found in the zip: {missing}")


def main():
    p = argparse.ArgumentParser(description="Download and arrange the challenge data")
    p.add_argument("--zip", type=Path, help="use an already downloaded zip instead of downloading")
    p.add_argument("--force", action="store_true", help="redo even if data/ is already complete")
    p.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip afterwards")
    args = p.parse_args()

    if all_present() and not args.force:
        print("data/ already complete, nothing to do (use --force to redo)")
    else:
        zip_path = args.zip
        downloaded = False
        if zip_path is None:
            zip_path = DATA_DIR / "_download" / "student_resource.zip"
            if not zip_path.is_file():
                download(zip_path)
                downloaded = True
        arrange(zip_path)
        if downloaded and not args.keep_zip:
            shutil.rmtree(zip_path.parent, ignore_errors=True)

    print("\ndata files:")
    for split, names in EXPECTED.items():
        for name in names:
            f = DATA_DIR / split / name
            print(f"   {f.relative_to(ROOT)}  {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
