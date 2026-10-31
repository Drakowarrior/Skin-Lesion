#!/usr/bin/env python3
"""
Production Weights Provisioning Script for ConvNeXt-Small Model.

Downloads the active production checkpoint from a trusted URL, streams to a temporary
file, verifies its SHA-256 against the committed manifest, and atomically places it
at the expected destination.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
import urllib.request
import urllib.error

# Expected canonical production values
DEFAULT_TARGET_PATH = Path("models/production/convnext_small_skin_lesion_v1.pt")
DEFAULT_MANIFEST_PATH = Path("models/production/model_sha256.txt")
FALLBACK_SHA256 = "67f4d8d04edef9389aeac77688dcc90e193b717de67f8c4c0c1ebf15483a3fe3"


def get_expected_sha256(manifest_path: Path, filename: str) -> str:
    """Read expected SHA-256 from manifest file or fall back to known hash."""
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1].endswith(filename):
                    return parts[0].lower()
    return FALLBACK_SHA256.lower()


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum in chunks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower()


def download_and_verify(
    url: str,
    target_path: Path,
    expected_sha256: str,
    force: bool = False,
    timeout: int = 120,
) -> bool:
    """
    Stream download from url to temporary file, verify SHA-256, and move to target_path.
    """
    target_path = Path(target_path)
    
    # 1. Existing file check
    if target_path.exists():
        if not force:
            print(f"[INFO] Existing file found at: {target_path}")
            print("[INFO] Verifying checksum of existing file...")
            current_sha = compute_sha256(target_path)
            if current_sha == expected_sha256:
                print(f"[SUCCESS] Existing checkpoint is already valid (SHA-256: {current_sha}).")
                print("         Skipping download. Use --force to re-download.")
                return True
            else:
                print(f"[WARNING] Existing checkpoint checksum mismatch (found: {current_sha}, expected: {expected_sha256}).")
                print("          Proceeding with clean download.")
        else:
            print(f"[INFO] Force flag active: re-downloading to {target_path}...")

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Download to temporary file in target parent dir for atomic replace
    temp_dir = target_path.parent
    temp_file = None

    try:
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, prefix="download_", suffix=".tmp") as tmp:
            temp_file = Path(tmp.name)
            print(f"[INFO] Connecting to: {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "DermaVision-Production-Model-Loader/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB

                print(f"[INFO] Content length: {total_size / (1024 * 1024):.2f} MB")
                print("[INFO] Streaming payload to disk...")

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\r[INFO] Progress: {downloaded / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB ({percent:.1f}%)")
                        sys.stdout.flush()

        print("\n[INFO] Download finished. Computing cryptographic checksum...")
        computed_sha = compute_sha256(temp_file)
        print(f"[INFO] Computed SHA-256: {computed_sha}")
        print(f"[INFO] Expected SHA-256: {expected_sha256}")

        # 3. Checksum verification
        if computed_sha != expected_sha256:
            raise ValueError(
                f"Cryptographic verification failed!\n"
                f"Expected: {expected_sha256}\n"
                f"Computed: {computed_sha}"
            )

        # 4. Atomic move to target
        shutil.move(str(temp_file), str(target_path))
        print(f"[SUCCESS] Production model verified and installed: {target_path}")
        return True

    except Exception as e:
        print(f"\n[ERROR] Download or verification failed: {e}", file=sys.stderr)
        if temp_file and temp_file.exists():
            print(f"[INFO] Cleaning up incomplete temporary file: {temp_file}")
            try:
                temp_file.unlink()
            except Exception:
                pass
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify production ConvNeXt-Small checkpoint."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("PRODUCTION_MODEL_URL", ""),
        help="Download URL for model checkpoint (or set PRODUCTION_MODEL_URL env var)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=str(DEFAULT_TARGET_PATH),
        help=f"Target file path (default: {DEFAULT_TARGET_PATH})",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST_PATH),
        help=f"Manifest file path (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default="",
        help="Optional explicit expected SHA-256 override",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download and replace even if file exists",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Network timeout in seconds (default: 120)",
    )

    args = parser.parse_args()

    if not args.url:
        print("[ERROR] No download URL provided. Pass --url <URL> or set PRODUCTION_MODEL_URL.", file=sys.stderr)
        sys.exit(1)

    target_path = Path(args.target)
    manifest_path = Path(args.manifest)

    expected_sha = (
        args.expected_sha256.strip().lower()
        if args.expected_sha256
        else get_expected_sha256(manifest_path, target_path.name)
    )

    success = download_and_verify(
        url=args.url,
        target_path=target_path,
        expected_sha256=expected_sha,
        force=args.force,
        timeout=args.timeout,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

