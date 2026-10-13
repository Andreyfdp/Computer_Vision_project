
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

BASE_URL = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/"

FILES = {
    "FullIJCNN2013.zip": BASE_URL + "FullIJCNN2013.zip",
}


def download_file(url: str, destination: Path):
    if destination.exists():
        print(f"Already downloaded: {destination.name}")
        return

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved: {destination}")


def extract_zip(archive: Path, destination: Path):
    print(f"Extracting {archive.name}")

    with zipfile.ZipFile(archive) as z:
        z.extractall(destination)

    print("Extraction complete")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():
        archive = RAW_DIR / filename

        download_file(url, archive)
        extract_zip(archive, RAW_DIR)

    images = list(RAW_DIR.rglob("*.ppm"))
    annotations = list(RAW_DIR.rglob("gt.txt"))

    print(f"Images found: {len(images)}")
    print(f"Annotation files found: {len(annotations)}")


if __name__ == "__main__":
    main()
