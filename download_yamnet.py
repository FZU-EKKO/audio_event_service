"""
下载 YAMNet 完整模型（SavedModel + class_map）到本地。

用法:
    pip install tensorflow_hub   # 仅下载时需要，之后可卸载
    python download_yamnet.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
TARGET_DIR = SERVICE_DIR / "models" / "yamnet"
MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 用 tensorflow_hub 解析并下载（谷歌云存储，国内可通）
    print(f"⬇  Resolving {MODEL_HANDLE} ...")
    import tensorflow_hub as hub
    import tensorflow as tf

    resolved = Path(hub.resolve(MODEL_HANDLE))
    print(f"   Cache: {resolved}")

    # 复制 SavedModel
    saved_model = TARGET_DIR / "saved_model.pb"
    if saved_model.exists():
        print(f"   ⏭  SavedModel already exists, removing...")
        shutil.rmtree(TARGET_DIR)
        TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📦 Copying to {TARGET_DIR} ...")
    shutil.copytree(resolved, TARGET_DIR, dirs_exist_ok=True)

    # 提取 class_map.csv
    model = hub.load(str(TARGET_DIR))
    class_map_src = Path(model.class_map_path().numpy().decode("utf-8"))
    class_map_dst = TARGET_DIR / "yamnet_class_map.csv"
    shutil.copy(class_map_src, class_map_dst)

    # 验证
    print("🔍 Verifying...")
    loaded = tf.saved_model.load(str(TARGET_DIR))
    infer = loaded.signatures["serving_default"]
    input_key = list(infer.structured_input_signature[1].keys())[0]
    print(f"   Input key : {input_key}")

    import numpy as np
    import csv
    dummy = tf.constant(np.zeros(16000, dtype=np.float32))
    outputs = infer(**{input_key: dummy})
    print(f"   Outputs   : {list(outputs.keys())}")

    with open(class_map_dst, "r", encoding="utf-8") as f:
        labels = [row["display_name"] for row in csv.DictReader(f)]
    print(f"   Labels    : {len(labels)}")

    print(f"\n✅ YAMNet 模型就绪！路径: {TARGET_DIR}")
    print(f"   可卸载 tensorflow_hub: pip uninstall tensorflow_hub")


if __name__ == "__main__":
    main()
