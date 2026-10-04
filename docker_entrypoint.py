"""Run a container command and optionally upload its artifacts to S3."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOTS = ("data", "models", "reports")


def upload_artifacts() -> None:
    bucket = os.environ.get("ARTIFACTS_BUCKET")
    if not bucket:
        return

    prefix = os.environ.get("ARTIFACTS_PREFIX", "runs").strip("/")
    run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    s3 = boto3.client("s3")
    uploaded = 0

    for root_name in ARTIFACT_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for artifact in root.rglob("*"):
            if not artifact.is_file() or artifact.name == ".gitignore":
                continue
            relative_path = artifact.relative_to(PROJECT_ROOT).as_posix()
            key = f"{prefix}/{run_id}/{relative_path}"
            s3.put_object(Bucket=bucket, Key=key, Body=artifact.read_bytes())
            uploaded += 1

    lock_file = PROJECT_ROOT / "dvc.lock"
    if lock_file.is_file():
        key = f"{prefix}/{run_id}/dvc.lock"
        s3.put_object(Bucket=bucket, Key=key, Body=lock_file.read_bytes())
        uploaded += 1

    print(f"Uploaded {uploaded} artifacts to s3://{bucket}/{prefix}/{run_id}/")


def main() -> None:
    command = sys.argv[1:] or ["dvc", "repro"]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    upload_artifacts()


if __name__ == "__main__":
    main()