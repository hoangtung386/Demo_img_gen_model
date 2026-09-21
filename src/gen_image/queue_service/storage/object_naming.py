"""
Build the GCS/S3 object key for an upload.

Format:
    {folder}/{app_id}/{feature}/{YYYY}/{MM}/{DD}/{filename}.{ext}

Where filename is:
    os-{android|ios}_package_name-{app_id}_device_id-{device_id}_{YYYYMMDDHHmmss}_unique{uuid}

All inputs are sanitized to lowercase + safe characters. UTC is used for the
timestamp + date partition. The `feature` and `folder` come from StorageConfig
— never hardcoded in calling code.
"""

from __future__ import annotations

import os as _os
import re
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

# Python 3.10 compat — datetime.UTC was added in 3.11.
UTC = timezone.utc

# Allowed in path/filename segments after sanitization. Letters, digits, dot
# (for package_name like com.example.app), dash, underscore.
_SAFE_RE = re.compile(r"[^a-z0-9._-]")


def _sanitize(value: str | None, fallback: str = "unknown") -> str:
    """Lowercase + replace unsafe chars with `_`. Empty/None → fallback."""
    if not value:
        return fallback
    return _SAFE_RE.sub("_", value.strip().lower()) or fallback


def _normalize_os(os_name: str | None) -> str:
    """Per spec, allowed values are 'android' or 'ios'. Other → 'unknown'."""
    v = (os_name or "").strip().lower()
    if v in ("android", "ios"):
        return v
    return "unknown"


def _ext_for(source_path: str) -> str:
    """File extension WITHOUT the leading dot, lowercased."""
    ext = Path(source_path).suffix.lstrip(".").lower()
    return ext or "bin"


def build_object_key(
    *,
    folder: str,
    app_id: str,
    feature: str,
    os_name: str,
    device_id: str,
    source_path: str,
    now: datetime | None = None,
    request_unique: str | None = None,
) -> str:
    """
    Build the full GCS object key for an upload.

    Args:
        folder       : env/tenant root (e.g. "prd/perm"). From StorageConfig.folder.
        app_id       : app package name (e.g. "com.example.app"). From the request.
        feature      : kebab-case feature label (e.g. "gen-image").
                       From StorageConfig.feature.
        os_name      : "android" or "ios". From the request.
        device_id    : device identifier from the request.
        source_path  : local file path; only the extension is used.
        now          : optional timestamp (defaults to UTC now). Useful in tests.
        request_unique : optional uuid hex (defaults to uuid4). Useful in tests.

    Returns:
        The full object key, NOT including a leading slash.
    """
    now = now or datetime.now(UTC)
    ts = now.strftime("%Y%m%d%H%M%S")
    y, m, d = now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")

    safe_app = _sanitize(app_id, fallback="unknown-app")
    safe_feature = _sanitize(feature, fallback="default")
    safe_os = _normalize_os(os_name)
    safe_device = _sanitize(device_id, fallback="unknown-device")
    ext = _ext_for(source_path)
    unique = (request_unique or _uuid.uuid4().hex).replace("-", "").lower()

    filename = (
        f"os-{safe_os}_package_name-{safe_app}_device_id-{safe_device}"
        f"_{ts}_unique{unique}.{ext}"
    )

    # `folder` may legitimately contain '/' (e.g. "prd/perm"); strip surrounding
    # slashes.
    folder_clean = (folder or "").strip("/")
    return _os.path.join(folder_clean, safe_app, safe_feature, y, m, d, filename)
