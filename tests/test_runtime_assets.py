from io import BytesIO
from urllib.error import URLError

import pytest

from src import runtime_assets


class Response(BytesIO):
    headers = {"Content-Length": "6"}

    def read(self, size=-1):
        assert 0 < size <= 1024 * 1024
        return super().read(size)


def test_stream_and_reuse_cache(tmp_path, monkeypatch):
    calls = []

    def fetch(request, timeout):
        calls.append(request.full_url)
        return Response(b"abcdef")

    monkeypatch.setattr(runtime_assets, "urlopen", fetch)
    path = tmp_path / "cache/model.joblib"
    runtime_assets.download_asset("https://huggingface.co/file", path)
    monkeypatch.setattr(runtime_assets, "urlopen", lambda *a, **k: pytest.fail("Cache redownloaded"))
    runtime_assets.download_asset("https://huggingface.co/file", path)
    assert path.read_bytes() == b"abcdef"
    assert len(calls) == 1


def test_failure_does_not_leave_partial_file(tmp_path, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise URLError("interrupted")

    monkeypatch.setattr(runtime_assets, "urlopen", fail)
    monkeypatch.setattr(runtime_assets.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Could not download"):
        runtime_assets.download_asset("https://huggingface.co/file", tmp_path / "model")
    assert len(calls) == 3
    assert not list(tmp_path.iterdir())


def test_local_fallback_and_revision_cache(tmp_path, monkeypatch):
    for key in ["FRAUD_HISTORY_URL", "FRAUD_MODEL_URL", "HF_DATASET_PREFIX", "HF_MODEL_FILE"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FRAUD_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_DATASET_REPO", "S1H6647/fraudlens")
    monkeypatch.setenv("HF_DATASET_REVISION", "version1")
    monkeypatch.setattr(runtime_assets, "urlopen", lambda *a, **k: Response(b"abcdef"))
    data, model = tmp_path / "train.csv", tmp_path / "model.joblib"
    first = runtime_assets.ensure_runtime_assets(data, model)
    monkeypatch.setenv("HF_DATASET_REVISION", "version2")
    second = runtime_assets.ensure_runtime_assets(data, model)
    assert first != second
    data.write_bytes(b"local")
    model.write_bytes(b"local")
    assert runtime_assets.ensure_runtime_assets(data, model) == (data, model)
