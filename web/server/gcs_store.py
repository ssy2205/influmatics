from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


try:
    from google.cloud import storage
except ImportError:  # pragma: no cover - local dev can run without GCS support
    storage = None


class GCSRunStore:
    """Persist completed web analysis runs to Google Cloud Storage."""

    def __init__(self, bucket_name: str, prefix: str = "runs") -> None:
        if storage is None:
            raise RuntimeError("google-cloud-storage is not installed.")
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    @classmethod
    def from_env(cls) -> Optional["GCSRunStore"]:
        bucket_name = (
            os.getenv("INFLUMATICS_RUNS_BUCKET")
            or os.getenv("GCS_BUCKET_NAME")
            or ""
        ).strip()
        if not bucket_name:
            return None
        prefix = os.getenv("INFLUMATICS_RUNS_PREFIX", "runs")
        return cls(bucket_name=bucket_name, prefix=prefix)

    def upload_file(self, local_path: Path, run_id: str, relative_path: str) -> None:
        blob = self.bucket.blob(self._object_name(run_id, relative_path))
        blob.upload_from_filename(str(local_path))

    def upload_json(self, payload: dict[str, Any], run_id: str, relative_path: str) -> None:
        blob = self.bucket.blob(self._object_name(run_id, relative_path))
        blob.upload_from_string(
            json.dumps(payload, indent=2),
            content_type="application/json",
        )

    def upload_run_dir(self, run_dir: Path, run_id: str) -> None:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(run_dir).as_posix()
            self.upload_file(path, run_id, relative_path)

    def download_json(self, run_id: str, relative_path: str) -> Optional[dict[str, Any]]:
        text = self.download_text(run_id, relative_path)
        if text is None:
            return None
        return json.loads(text)

    def download_text(self, run_id: str, relative_path: str) -> Optional[str]:
        blob = self.bucket.blob(self._object_name(run_id, relative_path))
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def download_file(self, run_id: str, relative_path: str, destination: Path) -> bool:
        blob = self.bucket.blob(self._object_name(run_id, relative_path))
        if not blob.exists():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
        return True

    def list_result_files(self, run_id: str) -> list[dict[str, Any]]:
        result_prefix = self._object_name(run_id, "results") + "/"
        files = []
        for blob in self.client.list_blobs(self.bucket_name, prefix=result_prefix):
            if blob.name.endswith("/"):
                continue
            relative_name = blob.name.removeprefix(result_prefix)
            if not relative_name:
                continue
            files.append(
                {
                    "name": relative_name,
                    "size": blob.size or 0,
                    "url": f"/analyses/{run_id}/files/{relative_name}",
                }
            )
        return sorted(files, key=lambda item: item["name"])

    def _object_name(self, run_id: str, relative_path: str) -> str:
        parts = [part for part in (self.prefix, run_id, relative_path.strip("/")) if part]
        return "/".join(parts)
