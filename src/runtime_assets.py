"""Stream Hugging Face assets to an atomic, reusable disk cache."""

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def download_asset(url: str, destination: Path, token: str | None = None) -> Path:
    if destination.is_file() and destination.stat().st_size:
        return destination
    if urlparse(url).scheme != "https":
        raise RuntimeError("Asset URLs must use HTTPS.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url)
    if token and urlparse(url).hostname == "huggingface.co":
        request.add_header("Authorization", f"Bearer {token}")
    for attempt in range(3):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                logger.info("Downloading runtime asset %s", destination.name)
                with urlopen(request, timeout=120) as response:
                    expected = response.headers.get("Content-Length")
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        size += len(chunk)
                    if not size or (expected is not None and size != int(expected)):
                        raise OSError("Empty or incomplete download")
            temporary.replace(destination)
            return destination
        except (URLError, OSError) as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(
                    f"Could not download {destination.name}; check Hugging Face access and asset settings."
                ) from exc
            time.sleep(attempt + 1)
    raise RuntimeError("Download retries exhausted")


def ensure_runtime_assets(data_path: Path, model_path: Path) -> tuple[Path, Path]:
    repo = os.getenv("HF_DATASET_REPO", "S1H6647/fraudlens").strip()
    revision = os.getenv("HF_DATASET_REVISION", "main").strip()
    model_file = os.getenv("HF_MODEL_FILE", "xgb_recall_055.joblib").strip("/")
    prefix = os.getenv("HF_DATASET_PREFIX", "").strip("/")
    cache = Path(os.getenv("FRAUD_CACHE_DIR", str(data_path.parent.parent / ".cache/assets")))
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    paths = []
    for local, override, filename in [
        (data_path, "FRAUD_HISTORY_URL", "train.csv"),
        (model_path, "FRAUD_MODEL_URL", model_file),
    ]:
        explicit_url = os.getenv(override)
        if not explicit_url and local.is_file():
            paths.append(local)
            continue
        remote = f"{prefix}/{filename}" if prefix else filename
        url = explicit_url or (
            f"https://huggingface.co/datasets/{quote(repo, safe='/')}/resolve/"
            f"{quote(revision, safe='')}/{quote(remote, safe='/')}?download=true"
        )
        # URL/revision changes select a new cache entry rather than stale content.
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        paths.append(download_asset(url, cache / key / local.name, token))
    return paths[0], paths[1]
