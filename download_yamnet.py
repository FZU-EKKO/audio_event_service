from __future__ import annotations

import shutil
from pathlib import Path

import tensorflow_hub as hub


MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
SERVICE_DIR = Path(__file__).resolve().parent
TARGET_DIR = SERVICE_DIR / "models" / "yamnet"


def main() -> None:
    resolved = Path(hub.resolve(MODEL_HANDLE))
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    shutil.copytree(resolved, TARGET_DIR)
    print(f"YAMNet downloaded to: {TARGET_DIR}")
    print(f"Resolved cache path: {resolved}")


if __name__ == "__main__":
    main()
