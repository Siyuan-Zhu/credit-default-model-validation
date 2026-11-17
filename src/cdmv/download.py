"""Download the UCI dataset once and verify it against committed checksums.

The archive is treated as untrusted data: only the single expected .xls member
is extracted, and nothing in it is executed. Parsing happens later in
``warehouse.load_raw`` with xlrd.
"""
import hashlib
import sys
import urllib.request
import zipfile

from .config import CHECKSUMS, DATA_URL, RAW_DIR, XLS_MEMBER, XLS_NAME, ZIP_NAME


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_checksums():
    out = {}
    for line in CHECKSUMS.read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            out[name] = digest
    return out


def verify(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}. "
            "Delete data/raw/ and rerun `make data`; if it persists the source has changed."
        )


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sums = expected_checksums()
    zip_path = RAW_DIR / ZIP_NAME
    xls_path = RAW_DIR / XLS_NAME

    if not zip_path.exists():
        print(f"Downloading {DATA_URL}")
        tmp = zip_path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(DATA_URL, tmp)
        except Exception as exc:  # network failures should stop the run
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"Download failed: {exc}") from exc
        tmp.rename(zip_path)
    verify(zip_path, sums[ZIP_NAME])

    if not xls_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            if names != [XLS_MEMBER]:
                raise SystemExit(f"Unexpected archive contents: {names}")
            xls_path.write_bytes(zf.read(XLS_MEMBER))
    verify(xls_path, sums[XLS_NAME])
    print(f"Data verified: {xls_path.relative_to(RAW_DIR.parents[1])}")


if __name__ == "__main__":
    sys.exit(main())
