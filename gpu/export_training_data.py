"""Fetch one consistent dashboard export; never fall back to stale local data."""
import argparse
from http.client import IncompleteRead
from http.cookiejar import CookieJar
import json
from pathlib import Path
import sys
import tempfile
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import build_opener, HTTPCookieProcessor, Request

from train_laya import load_dataset


def fetch_export(url, key_file):
    origin = url.rstrip("/")
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("Dashboard URL must contain only http(s)://host:port.")
    key = key_file.read_text(encoding="utf-8-sig").strip()
    if not key:
        raise ValueError("Dashboard access key is empty.")
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    login = Request(origin + "/login", data=urlencode({"code": key}).encode(), headers={"Origin": origin})
    with opener.open(login, timeout=10):
        pass
    with opener.open(origin + "/api/training-export", timeout=30) as response:
        if response.headers.get_content_type() != "application/x-ndjson":
            raise ValueError("Dashboard did not return a training export. Check the access key.")
        return response.read()


def save_export(payload, output_dir):
    # Refuse an existing destination so a failed download cannot reuse an older run.
    if output_dir.exists():
        raise ValueError("Export destination already exists; use a new run directory.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output_dir.parent) as folder:
        staging = Path(folder)
        snapshot = staging / "snapshot.jsonl"
        snapshot.write_bytes(payload)
        train, val = load_dataset(snapshot)
        if not val:
            raise ValueError("No validation examples available. Review more emails before training.")
        for split, samples in (("train", train), ("val", val)):
            text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in samples)
            (staging / f"laya-mail-{split}.jsonl").write_text(text, encoding="utf-8")
        snapshot.unlink()
        # Publish both validated files together, on the same filesystem.
        staging.rename(output_dir)
    return len(train), len(val)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        train_count, val_count = save_export(fetch_export(args.url, args.key_file), args.output_dir)
    except IncompleteRead:
        print("Export failed: download incomplete. No partial or old data will be used. Please retry.", file=sys.stderr)
        return 1
    except URLError:
        print("Export failed: dashboard unreachable or access denied. No old data will be used.", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1
    print(f"Downloaded and validated {train_count} training and {val_count} validation examples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
