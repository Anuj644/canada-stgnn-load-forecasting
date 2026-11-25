"""Atomic checkpoint writes resilient to temporary Windows/OneDrive file locks."""
from io import BytesIO
import json
import os
from pathlib import Path
import time
import uuid

import torch


def atomic_bytes(path, payload):
    path = Path(path)
    # Keep the sibling name short for Windows' legacy path-length limit.
    temporary = path.with_name(f".ckpt-{uuid.uuid4().hex[:16]}.tmp")
    # Serialize before opening the destination; a failed save cannot truncate it.
    for attempt in range(12):
        try:
            temporary.write_bytes(payload)
            break
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.1 * (attempt + 1), 1))
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.1 * (attempt + 1), 1))


def save_checkpoint(payload, path):
    buffer = BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue())


def save_json(payload, path):
    atomic_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))
