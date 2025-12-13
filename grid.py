"""
grid.py

ナンクロパズル用グリッド生成ユーティリティ（高速版）
- NumPyを用いてPythonループを極力排除
- 同グリッド判定されたボックス中心間を線で結び、画像内に描画（縦線：青, 横線：赤）
"""
import numpy as np
import pandas as pd
import cv2

# -----------------------------
# NumPyベースの1Dクラスタリング
# -----------------------------
def simple_1d_cluster_np(coords, thresh):
    """
    NumPyを使って1Dクラスタリングを高速化
    coords: 数値リスト or 1D array
    thresh: 同グループ判定の閾値
    return: 同長のラベル配列（0始まり）
    """
    arr = np.asarray(coords, dtype=float)
    n = arr.size
    if n == 0:
        return np.array([], dtype=int)
    order = np.argsort(arr)
    sorted_arr = arr[order]
    diffs = np.diff(sorted_arr)
    new_group = diffs > thresh
    labels_sorted = np.concatenate(([0], np.cumsum(new_group)))
    labels = np.empty(n, dtype=int)
    labels[order] = labels_sorted
    return labels

# -----------------------------
# 高速化版グリッド生成関数
# -----------------------------
def create_grid_with_threshold(detected_chars, image=None, thickness=2):
    """
    NumPyを用いた高速1Dクラスタリングによるグリッド生成。
    同グリッド判定されたボックス中心間を線で結び、画像に描画する（縦線：青, 横線：赤）。

    Args:
      detected_chars: List of tuples (x1,y1,x2,y2,char,conf,cls)
      image:         OpenCV画像（Noneなら描画なし）
      mask:          ユーザー描画の自由形状マスク（二値マスク ndarray:0/255 or False/True）
      thickness:     線の太さ
    Returns:
      pandas.DataFrame: グリッドDataFrame
      image:            描画用画像（マスク適用後）
    """
    # 以下、既存のセル検出・クラスタリング処理...
    sizes = np.array([b[2] - b[0] for b in detected_chars if b[6] != 'number'], dtype=float)
    if sizes.size == 0:
        raise ValueError('漢字または黒マスが検出されていません')
    box_avg = float(np.median(sizes))

    boxes = []
    for x1, y1, x2, y2, char, conf, cls in detected_chars:
        if cls == 'number':
            x2n, y2n = x1 + box_avg, y1 + box_avg
            boxes.append((x1, y1, x2n, y2n, char, conf, cls))
        else:
            boxes.append((x1, y1, x2, y2, char, conf, cls))

    cx = np.array([(x1 + x2) / 2 for x1, y1, x2, y2, *_ in boxes], dtype=float)
    cy = np.array([(y1 + y2) / 2 for x1, y1, x2, y2, *_ in boxes], dtype=float)

    thresh = box_avg * 0.3
    col_labels = simple_1d_cluster_np(cx, thresh)
    row_labels = simple_1d_cluster_np(cy, thresh)
    n_col = col_labels.max() + 1 if col_labels.size > 0 else 0
    n_row = row_labels.max() + 1 if row_labels.size > 0 else 0

    
    if image is not None:
        for lab in np.unique(col_labels):
            idxs = np.where(col_labels == lab)[0]
            pts = [(int(cx[i]), int(cy[i])) for i in idxs]
            pts.sort(key=lambda p: p[1])
            for p1, p2 in zip(pts, pts[1:]):
                cv2.line(image, p1, p2, (255, 0, 0), thickness)
        for lab in np.unique(row_labels):
            idxs = np.where(row_labels == lab)[0]
            pts = [(int(cx[i]), int(cy[i])) for i in idxs]
            pts.sort(key=lambda p: p[0])
            for p1, p2 in zip(pts, pts[1:]):
                cv2.line(image, p1, p2, (0, 0, 255), thickness)
    
    grid = [[None] * n_col for _ in range(n_row)]
    for (x1, y1, x2, y2, char, conf, cls), r, c in zip(boxes, row_labels, col_labels):
        if grid[r][c] is None:
            grid[r][c] = char

    return pd.DataFrame(grid), image