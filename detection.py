"""
detection.py

ナンクロパズル用検出・認識モジュール
- バッチ化＋OpenCV前処理による高速OCR
- マージ機能を小関数に分割して可読性向上
"""
import os
import cv2  # type: ignore
import torch  # type: ignore
import numpy as np
from torchvision import transforms  # type: ignore
from PIL import Image  # type: ignore
import time  # ★ 追加：計測用

from preprocessing.transforms import transform_image
from utils.drawing_utils import draw_box_with_text
from utils.file_utils import get_timestamp


# ============================================================
# ★ 追加：transform_image() の出力整形
# transform_image() は (1,C,H,W) を返すため、stack 前に (C,H,W) に揃える
# ============================================================
def _ensure_chw(t: torch.Tensor, *, context: str = "") -> torch.Tensor:
    """
    期待:
      - (1,C,H,W) または (C,H,W)
    返す:
      - (C,H,W)
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(t)} {context}")

    # (1,C,H,W) -> (C,H,W)
    if t.dim() == 4 and t.size(0) == 1:
        t = t.squeeze(0)

    if t.dim() != 3:
        raise ValueError(f"Invalid tensor shape (expected CHW): {tuple(t.shape)} {context}")

    return t


# YOLO結果取得
def gather_raw_boxes(results, names):
    """
    YOLO推論結果から(x1,y1,x2,y2,conf,class)を抽出
    """
    raw = []
    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            cls = names[int(box.cls[0].item())]
            raw.append((x1, y1, x2, y2, conf, cls))
    return raw


# 重複フィルタ
def filter_overlapping_boxes(boxes, iou_threshold=0.0):
    """
    'kanji'のみIOUで重複排除
    """
    kanji = [b for b in boxes if b[5] == 'kanji']
    others = [b for b in boxes if b[5] != 'kanji']
    areas = []
    for x1, y1, x2, y2, conf, cls in kanji:
        areas.append((x1, y1, x2, y2, conf, cls, (x2 - x1) * (y2 - y1)))
    areas.sort(key=lambda x: x[6], reverse=True)

    keep = []
    for x1b, y1b, x2b, y2b, confb, clsb, _ in areas:
        ok = True
        for x1a, y1a, x2a, y2a, _, _, _ in keep:
            ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
            ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (x2a - x1a) * (y2a - y1a) * 1.0 + (x2b - x1b) * (y2b - y1b) - inter
                if inter / union > iou_threshold:
                    ok = False
                    break
        if ok:
            keep.append((x1b, y1b, x2b, y2b, confb, clsb, None))

    filtered_kanji = [(x1, y1, x2, y2, conf, cls) for x1, y1, x2, y2, conf, cls, _ in keep]
    return filtered_kanji + others


def process_filtered_boxes(
    image,
    filtered_boxes,
    number_model,
    number_classes,
    kanji_model,
    kanji_classes,
):
    """
    フィルタ済みボックスに対して文字認識を行う（バッチ推論化）。
    """
    detected_characters = []

    t0 = time.perf_counter()

    # バッチ処理用リスト
    kanji_crops = []
    kanji_meta = []  # (x1, y1, x2, y2, conf, cname)
    number_crops = []
    number_meta = []

    count_black = 0

    for (x1, y1, x2, y2, conf, cname) in filtered_boxes:
        if cname == "black_cell":
            detected_characters.append((x1, y1, x2, y2, "■", conf, cname))
            count_black += 1
            continue

        cropped_region = image[y1:y2, x1:x2]
        if cropped_region.size == 0:
            continue

        gray_region = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(gray_region).convert("L")

        if cname == "kanji":
            tensor = transform_image(pil_img, size=128)
            tensor = _ensure_chw(tensor, context="kanji")  # ★追加
            kanji_crops.append(tensor)
            kanji_meta.append((x1, y1, x2, y2, conf, cname))
        elif cname == "number":
            tensor = transform_image(pil_img, size=32)
            tensor = _ensure_chw(tensor, context="number")  # ★追加
            number_crops.append(tensor)
            number_meta.append((x1, y1, x2, y2, conf, cname))

    # Kanji Batch Inference
    if kanji_crops:
        batch = torch.stack(kanji_crops)  # (B, C, H, W) になる
        with torch.no_grad():
            out = kanji_model(batch)
            _, preds = torch.max(out, 1)

        for i, pred_idx in enumerate(preds):
            char = kanji_classes[pred_idx.item()]
            x1, y1, x2, y2, conf, cname = kanji_meta[i]
            detected_characters.append((x1, y1, x2, y2, char, conf, cname))

    # Number Batch Inference
    if number_crops:
        batch = torch.stack(number_crops)  # (B, C, H, W) になる
        with torch.no_grad():
            out = number_model(batch)
            _, preds = torch.max(out, 1)

        for i, pred_idx in enumerate(preds):
            char = number_classes[pred_idx.item()]
            x1, y1, x2, y2, conf, cname = number_meta[i]
            detected_characters.append((x1, y1, x2, y2, char, conf, cname))

    t1 = time.perf_counter()
    elapsed = t1 - t0
    total = len(filtered_boxes)
    avg = elapsed / total if total > 0 else 0.0

    print(
        f"[OCR] process_filtered_boxes (BATCH): total={elapsed:.3f}s "
        f"(avg={avg*1000:.1f}ms/box, "
        f"kanji={len(kanji_crops)}, number={len(number_crops)}, black={count_black}, "
        f"total_boxes={total})"
    )

    return detected_characters


# １）行グループ化：行の「中心Y」と「行高さ」の中央値を使ってクラスタリング
def group_rows(detected, center_thresh_ratio=0.25):
    """
    detected: [(x1,y1,x2,y2,char,conf,cls),...]
    center_thresh_ratio: 行高さ * この比率以内なら同じ行とみなす
    """
    # 各ボックスの中心Y を計算
    items = [(b, (b[1] + b[3]) / 2.0) for b in detected]
    # 行グループのリスト
    rows = []
    for box, cy in sorted(items, key=lambda x: x[1]):
        placed = False
        for row in rows:
            # 既存行の中心・高さ
            center = row['center']
            height = row['height']
            # ① 中心Yの差で同一行判定
            if abs(cy - center) <= height * center_thresh_ratio:
                row['boxes'].append(box)
                # 行の中央値を再計算
                cys = [(b[1] + b[3]) / 2.0 for b in row['boxes']]
                row['center'] = float(np.median(cys))
                heights = [b[3] - b[1] for b in row['boxes']]
                row['height'] = float(np.median(heights))
                placed = True
                break
        if not placed:
            # 新しい行グループを作る
            row_height = box[3] - box[1]
            rows.append({
                'boxes': [box],
                'center': cy,
                'height': float(row_height)
            })
    # 各行のボックスリストだけ返す
    return [r['boxes'] for r in rows]


# ２）～３）行内数字マージ：セル幅基準＆幅比チェックを追加
def merge_row_numbers(
    nums,
    cell_width,
    row_height,
    gap_thresh_ratio=0.2,
    cy_diff_ratio=0.125,
    width_ratio_thresh=0.7,
):
    """
    nums: 同一行内の数字ボックス [(x1,y1,x2,y2,char,conf,'number'),...]
    cell_width: 推定セル幅
    row_height: この行の高さ中央値
    """
    merged = []
    i = 0
    while i < len(nums):
        # 次のボックスとペアでマージできるかチェック
        if i < len(nums) - 1:
            x1, y1, x2, y2, c1, f1, _ = nums[i]
            nx1, ny1, nx2, ny2, c2, f2, _ = nums[i + 1]
            gap = nx1 - x2
            w1 = x2 - x1
            w2 = nx2 - nx1
            avgw = (w1 + w2) / 2.0
            cy1 = (y1 + y2) / 2.0
            cy2 = (ny1 + ny2) / 2.0

            # 条件①：セル幅を基準にした隙間
            cond_gap = gap < gap_thresh_ratio * cell_width
            # 条件②：行高さの一部以内の Yズレ
            cond_cy = abs(cy1 - cy2) < cy_diff_ratio * row_height
            # 条件③：幅比が似ているか
            cond_w = min(w1 / w2, w2 / w1) > width_ratio_thresh

            if cond_gap and cond_cy and cond_w:
                # マージ
                merged.append((
                    x1,
                    min(y1, ny1),
                    x1 + w1 + w2,
                    max(y2, ny2),
                    c1 + c2,
                    (f1 + f2) / 2.0,
                    'number'
                ))
                i += 2
                continue

        # マージできなければそのまま
        merged.append(nums[i])
        i += 1

    return merged


# 全体マージ統合
def merge_numbers(detected, **kw):
    """
    detected: 全ボックス [(x1,y1,x2,y2,char,conf,cls),...]
    kw:
      min_overlap_ratio: 行グループ化の重なり率（未使用）
      gap_thresh_ratio, cy_diff_ratio, width_ratio_thresh:
        merge_row_numbers に渡すパラメータ
    """
    # 全セル幅の推定（非数字ボックスの幅の中央値）
    widths = [b[2] - b[0] for b in detected if b[6] != 'number']
    cell_width = float(np.median(widths)) if widths else 40.0

    merged_all = []
    # ① 行グループ化
    t0 = time.perf_counter()
    rows = group_rows(detected)
    t1 = time.perf_counter()

    for row in rows:
        # この行の高さ中央値
        heights = [b[3] - b[1] for b in row]
        row_height = float(np.median(heights))

        # 数字／その他 に分ける
        nums = sorted([b for b in row if b[6] == 'number'], key=lambda b: b[0])
        others = [b for b in row if b[6] != 'number']

        # ② 行内数字マージ
        merged_nums = merge_row_numbers(
            nums,
            cell_width,
            row_height,
            gap_thresh_ratio=kw.get('gap_thresh_ratio', 0.2),
            cy_diff_ratio=kw.get('cy_diff_ratio', 0.125),
            width_ratio_thresh=kw.get('width_ratio_thresh', 0.7)
        )

        # マージ済数字＋その他 を戻す
        merged_all.extend(merged_nums + others)

    t2 = time.perf_counter()
    print(
        f"[OCR] merge_numbers: group_rows={t1-t0:.3f}s, "
        f"merge_row_numbers+collect={t2-t1:.3f}s, "
        f"rows={len(rows)}, detected={len(detected)}, merged={len(merged_all)}"
    )

    return merged_all


# 描画
def draw_detected_boxes(img, detected, font):
    t0 = time.perf_counter()
    for x1, y1, x2, y2, char, conf, cls in detected:
        draw_box_with_text(img, x1, y1, x2, y2, f"{char} {conf:.2f}", font)
    t1 = time.perf_counter()
    print(f"[OCR] draw_detected_boxes: {t1-t0:.3f}s (n={len(detected)})")
    return img


# メイン処理
def process_detections_y1(
    image,
    output_image,
    results_num,
    results_kji,
    num_model,
    num_cls,
    kan_model,
    kan_cls,
    font,
    names_num,
    names_kji,
    iou=0.0,
    profile: bool = True,  # ★ 追加：必要に応じて ON/
    draw: bool = True,
):
    """
    YOLO 検出結果から
    - 重複排除
    - OCR
    - 数字マージ
    - 描画
    をまとめて行う。
    profile=True のとき、各ステップの処理時間を stdout に出力する。
    """
    t_start = time.perf_counter()

    # 1. YOLO結果
    t0 = time.perf_counter()
    boxes_num = gather_raw_boxes(results_num, names_num)
    boxes_kji = gather_raw_boxes(results_kji, names_kji)
    boxes = boxes_num + boxes_kji
    t1 = time.perf_counter()

    # 2. 重複排除
    filtered = filter_overlapping_boxes(boxes, iou)
    t2 = time.perf_counter()

    # 3. 文字認識
    detected = process_filtered_boxes(image, filtered, num_model, num_cls, kan_model, kan_cls)
    t3 = time.perf_counter()

    # 4. 数字マージ
    merged = merge_numbers(detected)
    t4 = time.perf_counter()

    # 5. 描画
    if draw:
        out = draw_detected_boxes(output_image, merged, font)
    else:
        out = output_image  # or image.copy()
    t5 = time.perf_counter()

    if profile:
        print(
            "[OCR] process_detections_y1 timings:\n"
            f"  gather_raw_boxes : {t1-t0:.3f}s (n_num={len(boxes_num)}, n_kanji={len(boxes_kji)})\n"
            f"  filter_overlaps  : {t2-t1:.3f}s (n_filtered={len(filtered)})\n"
            f"  recognize (OCR)  : {t3-t2:.3f}s (n_recognized={len(detected)})\n"
            f"  merge_numbers    : {t4-t3:.3f}s\n"
            f"  draw_boxes       : {t5-t4:.3f}s\n"
            f"  TOTAL            : {t5-t_start:.3f}s"
        )

    return merged, out
