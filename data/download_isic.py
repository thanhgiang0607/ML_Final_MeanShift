import argparse
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018/"
DATA = Path(__file__).resolve().parent
ROOT = DATA / "isic2018"
SPLITS = DATA / "splits"

PARTS = [
    ("ISIC2018_Task1-2_Validation_Input", ".jpg", 100, False),
    ("ISIC2018_Task1_Validation_GroundTruth", ".png", 100, False),
    ("ISIC2018_Task1_Test_GroundTruth", ".png", 1000, False),
    ("ISIC2018_Task1-2_Test_Input", ".jpg", 1000, True),
]

TEST_SAMPLE_SIZE = 200
SPLIT_SEED = 42


def count_files(folder, suffix):
    return len(list(folder.glob(f"*{suffix}"))) if folder.exists() else 0


def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url, dest, retries=5, timeout=60):
    tmp = dest.with_suffix(".part")
    for attempt in range(1, retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(req, context=ssl_context(), timeout=timeout) as resp:
                if done and resp.status != 206:
                    done = 0
                total = done + int(resp.headers.get("Content-Length", 0))
                with open(tmp, "ab" if done else "wb") as f:
                    while chunk := resp.read1(1 << 18):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f}%)",
                                  end="", flush=True)
            print()
            tmp.rename(dest)
            return
        except (OSError, urllib.error.URLError) as e:
            print(f"\n  Lỗi mạng ({e}); thử lại lần {attempt}/{retries}...")
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError(f"Không tải được {url}. Kiểm tra mạng rồi chạy lại lệnh (sẽ tải tiếp).")


def safe_extract(zip_path, target):
    target = target.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            dest = (target / member).resolve()
            if not dest.is_relative_to(target):
                raise RuntimeError(f"Đường dẫn không an toàn trong zip: {member}")
        zf.extractall(target)


def fetch_part(name, suffix, expected, keep_zips):
    folder = ROOT / name
    have = count_files(folder, suffix)
    if have == expected:
        print(f"[bỏ qua] {name}: đã có đủ {have} file")
        return
    zips = ROOT / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    zip_path = zips / f"{name}.zip"
    if not zip_path.exists():
        print(f"[tải]    {name}.zip")
        download(BASE_URL + f"{name}.zip", zip_path)
    print(f"[giải nén] {name}")
    safe_extract(zip_path, ROOT)
    have = count_files(folder, suffix)
    if have != expected:
        raise RuntimeError(f"{name}: có {have} file, mong đợi {expected}. Xoá {zip_path} rồi chạy lại.")
    print(f"[xong]   {name}: {have} file")
    if not keep_zips:
        zip_path.unlink()


def ids_from_masks(folder):
    return sorted(p.name.replace("_segmentation.png", "") for p in folder.glob("*_segmentation.png"))


def write_splits():
    SPLITS.mkdir(exist_ok=True)
    tune = ids_from_masks(ROOT / "ISIC2018_Task1_Validation_GroundTruth")
    pool = ids_from_masks(ROOT / "ISIC2018_Task1_Test_GroundTruth")
    test = sorted(np.random.RandomState(SPLIT_SEED).choice(pool, TEST_SAMPLE_SIZE, replace=False))
    (SPLITS / "isic_tune.txt").write_text("\n".join(tune) + "\n")
    (SPLITS / "isic_test.txt").write_text("\n".join(test) + "\n")
    print(f"[chia tập] tune: {len(tune)} ảnh, test: {len(test)} ảnh -> {SPLITS}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-test-images", action="store_true", help="bỏ qua ảnh Test (2,2 GB)")
    parser.add_argument("--keep-zips", action="store_true", help="giữ file zip sau khi giải nén")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    for name, suffix, expected, is_big in PARTS:
        if is_big and args.no_test_images:
            print(f"[bỏ qua] {name} (--no-test-images)")
            continue
        fetch_part(name, suffix, expected, args.keep_zips)

    zips = ROOT / "zips"
    if zips.exists() and not args.keep_zips and not any(zips.iterdir()):
        shutil.rmtree(zips)
    write_splits()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nĐã dừng. Chạy lại lệnh để tiếp tục (phần đã xong sẽ được bỏ qua).")
