from ultralytics import YOLO # pyright: ignore
import cv2 # type: ignore
import numpy as np
from pathlib import Path
import os

# =========================================
# YOLO セグメンテーションモデルの設定
# =========================================

import config

# =========================================
# YOLO セグメンテーションモデルの設定
# =========================================

SEG_WEIGHTS = config.SEGMENTATION_MODEL_PATH

# デバッグ用に前処理結果を保存するフォルダ（不要なら使わなくてもOK）
# Lambdaでは /tmp 以下しか保存できないので注意
SEG_OUTPUT_DIR = Path("/tmp/segmented")
SEG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# モデルはモジュールインポート時に一度だけロード
seg_model = YOLO(SEG_WEIGHTS)


def preprocess_with_segmentation(img, original_filename: str | None = None):
    """
    YOLO セグメンテーションでパズル領域を抽出し、
    最大領域のみ残して外側を白で塗りつぶした画像を返す。

    - img: OpenCV の BGR 画像 (np.ndarray, HxWx3)
    - original_filename: デバッグ用にファイル名を残したいときだけ指定

    マスクが検出されない / 有効な連結領域がない場合は、
    元画像をそのまま返す。
    """
    # YOLOv8 segmentation に numpy配列を直接渡す
    results = seg_model.predict(
        source=img,
        device="cpu",
        imgsz=960,
        verbose=False
    )
    r = results[0]

    orig = r.orig_img  # 元画像（通常は img と同じ）
    h, w = orig.shape[:2]

    if r.masks is None:
        print("[警告] セグメンテーションマスクが検出されませんでした。元画像を使用します。")
        return img

    # ---- 1. YOLO マスクを OR 結合 ----
    # r.masks.data: shape = (N, Hm, Wm)
    masks = r.masks.data.cpu().numpy()
    full_mask = np.any(masks > 0.5, axis=0).astype(np.uint8)  # (Hm, Wm)

    # ---- 2. 画像サイズにリサイズ ----
    full_mask = cv2.resize(full_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # ---- 3. 最大面積の連結成分のみ残す ----
    num_labels, labels = cv2.connectedComponents(full_mask)

    if num_labels <= 1:
        print("[警告] 有効な連結領域がありませんでした。元画像を使用します。")
        return img

    counts = np.bincount(labels.flatten())
    # 背景(0)を除いた中で最大のラベル
    largest_label = np.argmax(counts[1:]) + 1

    largest_mask = (labels == largest_label).astype(np.uint8)

    # ---- 4. マスクを少し膨張させて安全マージンを取る ----
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    largest_mask = cv2.dilate(largest_mask, kernel, iterations=1)

    # ---- 5. マスク外を白で塗りつぶす ----
    result = orig.copy()
    result[largest_mask == 0] = (255, 255, 255)

    # ---- 6. 任意で中間結果を保存（デバッグ用）----
    if original_filename is not None:
        stem, _ = os.path.splitext(os.path.basename(original_filename))
        out_path = SEG_OUTPUT_DIR / f"{stem}_segmented_yolo11.png"
        cv2.imwrite(str(out_path), result)
        print(f"[セグメント前処理画像を保存] {out_path}")

    return result
