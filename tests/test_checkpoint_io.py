"""A failed save must not truncate a usable training checkpoint."""
import torch
import pytest

from src.models.checkpoint_io import save_checkpoint, atomic_bytes
import src.models.checkpoint_io as checkpoint_io


def test_serialization_failure_preserves_previous_checkpoint(tmp_path):
    path = tmp_path / "latest.pt"
    save_checkpoint({"weight": torch.ones(3)}, path)
    original = path.read_bytes()

    class Unserializable:
        def __reduce__(self):
            raise ValueError("test serialization failure")

    with pytest.raises(ValueError, match="serialization failure"):
        save_checkpoint({"bad": Unserializable()}, path)
    assert path.read_bytes() == original
    torch.testing.assert_close(torch.load(path, weights_only=True)["weight"], torch.ones(3))


def test_temporary_file_lock_retries_without_truncation(tmp_path, monkeypatch):
    path = tmp_path / "latest.bin"
    path.write_bytes(b"old")
    real_replace, calls = checkpoint_io.os.replace, []

    def briefly_locked(source, target):
        calls.append(1)
        assert path.read_bytes() == b"old"
        if len(calls) == 1:
            raise PermissionError("temporary lock")
        return real_replace(source, target)

    monkeypatch.setattr(checkpoint_io.os, "replace", briefly_locked)
    monkeypatch.setattr(checkpoint_io.time, "sleep", lambda duration: None)
    atomic_bytes(path, b"new")
    assert path.read_bytes() == b"new"
    assert len(calls) == 2
