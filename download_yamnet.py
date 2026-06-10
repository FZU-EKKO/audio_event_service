"""
下载 YAMNet 模型并保存为 TensorFlow SavedModel 格式。

用法（一行搞定）:
    python download_yamnet.py

首次运行会下载 ~15MB 的 yamnet.h5，然后转为 models/yamnet/ 目录。
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import requests
import tensorflow as tf

MODEL_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)

SERVICE_DIR = Path(__file__).resolve().parent
TARGET_DIR = SERVICE_DIR / "models" / "yamnet"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 下载 yamnet.h5 ──
    h5_path = TARGET_DIR / "yamnet.h5"
    if not h5_path.exists():
        print(f"⬇  Downloading yamnet.h5 (~15 MB)...")
        print(f"   {MODEL_URL}")
        resp = requests.get(MODEL_URL, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(h5_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r   {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="")
        print()
        print(f"   ✅ Saved: {h5_path}")
    else:
        print(f"   ⏭  Already exists: {h5_path}")

    # ── 2. 下载 class_map.csv ──
    csv_path = TARGET_DIR / "yamnet_class_map.csv"
    if not csv_path.exists():
        print(f"⬇  Downloading class map...")
        resp = requests.get(CLASS_MAP_URL, timeout=30)
        resp.raise_for_status()
        csv_path.write_text(resp.text, encoding="utf-8")
        print(f"   ✅ Saved: {csv_path}")
    else:
        print(f"   ⏭  Already exists: {csv_path}")

    # ── 3. 加载 Keras 模型 → 存为 SavedModel ──
    saved_model_marker = TARGET_DIR / "saved_model.pb"
    if not saved_model_marker.exists():
        print("🔄 Converting to SavedModel...")
        model = tf.keras.models.load_model(str(h5_path))
        tf.saved_model.save(model, str(TARGET_DIR))
        print(f"   ✅ SavedModel → {TARGET_DIR}")
    else:
        print(f"   ⏭  SavedModel already exists: {saved_model_marker}")

    # ── 4. 验证 ──
    print("🔍 Verifying...")
    loaded = tf.saved_model.load(str(TARGET_DIR))
    infer = loaded.signatures["serving_default"]
    input_key = list(infer.structured_input_signature[1].keys())[0]
    print(f"   Input key : {input_key}")

    # 验证一次前向传播
    import numpy as np
    dummy = tf.constant(np.zeros(16000, dtype=np.float32))
    outputs = infer(**{input_key: dummy})
    output_keys = list(outputs.keys())
    print(f"   Output keys: {output_keys}")

    # 验证 class map
    with open(csv_path, "r", encoding="utf-8") as f:
        labels = [row["display_name"] for row in csv.DictReader(f)]
    print(f"   Labels    : {len(labels)}")

    print(f"\n✅ YAMNet 模型就绪！路径: {TARGET_DIR}")
    print(f"   在 .env 中确保: EKKO_AUDIO_EVENT_MODEL_PATH={TARGET_DIR}")


if __name__ == "__main__":
    main()
