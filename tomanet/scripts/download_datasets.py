#!/usr/bin/env python3
"""Download the tomato datasets listed in configs/datasets.yaml.

    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --tier core
    python scripts/download_datasets.py --only taiwan plantseg
    python scripts/download_datasets.py --tier core --dry-run

Zenodo, Mendeley and GitHub download without credentials. Kaggle needs
~/.kaggle/kaggle.json. Anything marked `source: manual` prints its URL and why.

Each dataset lands in data/raw/<name>/ with a .provenance.json recording the URL,
sha256, licence and download date, so the paper can state exactly what was used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "configs" / "datasets.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw"

CHUNK = 1 << 20  # 1 MiB
TIMEOUT = 60


# ----------------------------------------------------------------- helpers ----
def human(n: int | None) -> str:
    if not n:
        return "?"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def download(url: str, dest: Path, desc: str) -> str:
    """Stream `url` to `dest`, return the sha256. Resumes nothing - keep it simple."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()

    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=desc,
            leave=False,
            disable=not sys.stderr.isatty(),
        ) as bar:
            for chunk in response.iter_content(CHUNK):
                handle.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))

    tmp.rename(dest)
    return digest.hexdigest()


def _extract_7z(archive: Path, target: Path) -> bool:
    """Unpack a .7z with the system binary if present, else py7zr. False if neither."""
    binary = next((b for b in ("7z", "7za", "7zr") if shutil.which(b)), None)
    if binary:
        subprocess.run([binary, "x", "-y", f"-o{target}", str(archive)], check=True,
                       stdout=subprocess.DEVNULL)
        return True

    try:
        import py7zr
    except ImportError:
        print(f"  ! {archive.name}: install p7zip-full or py7zr to extract - left in place")
        return False

    with py7zr.SevenZipFile(archive) as sz:
        sz.extractall(target)
    return True


def extract(archive: Path, target: Path) -> None:
    """Unpack zip / tar / 7z into `target`, then delete the archive."""
    target.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()

    if suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    elif suffix in {".gz", ".tgz", ".tar", ".bz2", ".xz"}:
        with tarfile.open(archive) as tf:
            tf.extractall(target)
    elif suffix == ".7z":
        if not _extract_7z(archive, target):
            return
    else:
        print(f"  ! unknown archive type {suffix} - left in place")
        return

    archive.unlink()


def write_provenance(target: Path, spec: dict, files: list[dict]) -> None:
    payload = {
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": spec.get("source"),
        "licence": spec.get("licence"),
        "caveats": spec.get("caveats", []),
        "files": files,
    }
    (target / ".provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- handlers ----
def fetch_zenodo(name: str, spec: dict, target: Path) -> list[dict]:
    record = spec["record"]
    meta = requests.get(f"https://zenodo.org/api/records/{record}", timeout=TIMEOUT)
    meta.raise_for_status()
    files = meta.json().get("files", [])
    if not files:
        raise RuntimeError(f"Zenodo record {record} lists no files")

    written = []
    for entry in files:
        url = entry["links"]["self"]
        archive = target / entry["key"]
        sha = download(url, archive, f"{name}/{entry['key']}")
        written.append({"name": entry["key"], "url": url, "sha256": sha, "size": entry.get("size")})
        extract(archive, target)
    return written


def fetch_mendeley(name: str, spec: dict, target: Path) -> list[dict]:
    dataset_id, version = spec["mendeley_id"], spec.get("version", 1)
    listing = requests.get(
        f"https://data.mendeley.com/public-api/datasets/{dataset_id}/files",
        params={"folder_id": "root", "version": version},
        timeout=TIMEOUT,
    )
    listing.raise_for_status()

    written = []
    for entry in listing.json():
        details = entry.get("content_details", {})
        url = details.get("download_url")
        if not url:
            continue
        archive = target / entry["filename"]
        sha = download(url, archive, f"{name}/{entry['filename']}")
        written.append(
            {"name": entry["filename"], "url": url, "sha256": sha, "size": details.get("size")}
        )
        extract(archive, target)
    return written


def fetch_github(name: str, spec: dict, target: Path) -> list[dict]:
    repo, branch = spec["repo"], spec.get("branch", "main")
    url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
    archive = target / f"{repo.split('/')[-1]}-{branch}.zip"
    sha = download(url, archive, f"{name}/{archive.name}")
    extract(archive, target)
    return [{"name": archive.name, "url": url, "sha256": sha, "size": None}]


def fetch_kaggle(name: str, spec: dict, target: Path) -> list[dict]:
    slug = spec.get("kaggle") or spec.get("alt_kaggle")
    if not slug:
        raise RuntimeError("no kaggle slug in the registry entry")
    if shutil.which("kaggle") is None:
        raise RuntimeError("kaggle CLI not found - `pip install kaggle` and add ~/.kaggle/kaggle.json")

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"],
        check=True,
    )
    return [{"name": slug, "url": f"https://www.kaggle.com/datasets/{slug}", "sha256": None, "size": None}]


HANDLERS = {
    "zenodo": fetch_zenodo,
    "mendeley": fetch_mendeley,
    "github": fetch_github,
    "kaggle": fetch_kaggle,
}


# -------------------------------------------------------------------- main ----
def load_registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["datasets"]


def show_list(registry: dict) -> None:
    header = f"{'name':<28} {'tier':<6} {'source':<9} {'auto':<5} licence"
    print(header)
    print("-" * len(header))
    for name, spec in registry.items():
        auto = "yes" if spec["source"] in HANDLERS else "NO"
        print(f"{name:<28} {spec['tier']:<6} {spec['source']:<9} {auto:<5} {spec.get('licence','?')}")
    print(f"\n{len(registry)} datasets. Sizes and caveats: configs/datasets.yaml")


def is_populated(path: Path) -> bool:
    return path.exists() and any(p for p in path.iterdir() if p.name != ".provenance.json")


def fetch_one(name: str, spec: dict, force: bool, dry_run: bool) -> str:
    target = RAW_DIR / name
    source = spec["source"]

    if source not in HANDLERS:
        print(f"[manual]  {name}")
        print(f"          {spec.get('url', '?')}")
        print(f"          reason: {spec.get('manual_reason', 'not automatable')}")
        print(f"          place the files at: {target}")
        return "manual"

    if is_populated(target) and not force:
        print(f"[skip]    {name} - already at {target} (use --force to redownload)")
        return "skip"

    if dry_run:
        print(f"[dry-run] {name} via {source} -> {target}  ({spec.get('size_hint', 'size unknown')})")
        return "dry-run"

    print(f"[fetch]   {name} via {source}  ({spec.get('size_hint', 'size unknown')})")
    target.mkdir(parents=True, exist_ok=True)
    try:
        files = HANDLERS[source](name, spec, target)
    except Exception as exc:  # noqa: BLE001 - one bad dataset must not kill the run
        print(f"[FAIL]    {name}: {exc}")
        return "fail"

    write_provenance(target, spec, files)
    print(f"[done]    {name} -> {target}")
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="show the registry and exit")
    parser.add_argument("--tier", choices=["core", "extra", "fruit", "all"], help="download a whole tier")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="download named datasets")
    parser.add_argument("--force", action="store_true", help="redownload even if present")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen")
    args = parser.parse_args()

    registry = load_registry()

    if args.list or not (args.tier or args.only):
        show_list(registry)
        return

    if args.only:
        unknown = [n for n in args.only if n not in registry]
        if unknown:
            sys.exit(f"unknown dataset(s): {', '.join(unknown)}\nrun --list to see valid names")
        selected = {n: registry[n] for n in args.only}
    else:
        selected = {
            n: s for n, s in registry.items() if args.tier == "all" or s["tier"] == args.tier
        }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {name: fetch_one(name, spec, args.force, args.dry_run) for name, spec in selected.items()}

    print("\nsummary:")
    for status in ("ok", "skip", "manual", "dry-run", "fail"):
        names = [n for n, s in results.items() if s == status]
        if names:
            print(f"  {status:<8} {len(names):>2}  {', '.join(names)}")


if __name__ == "__main__":
    main()
