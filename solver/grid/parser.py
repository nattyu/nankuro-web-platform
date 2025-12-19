# -*- coding: utf-8 -*-
"""
盤面のセルを内部表現に正規化するモジュールです。

主な役割:
- pandas.DataFrame を numpy 配列に変換
- 各セルの値を「黒マス」「漢字」「数字記号(#n)」などに正規化
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# CJK 統合漢字の簡易的な判定用正規表現
KANJI_RE = re.compile(r"[一-龯]")


def is_kanji(ch: str) -> bool:
    """
    文字列が「漢字っぽいかどうか」をざっくり判定します。

    正確な漢字判定ではありませんが、ナンクロ用途には十分です。
    """
    return isinstance(ch, str) and KANJI_RE.match(ch) is not None


def normalize_cell(x: Any) -> str:
    """
    個々のセルの値を、内部表現に変換します。

    変換ルール（例）
    ----------------
    - 黒マス: "■" のまま
    - 数字: "3" → "#3" のように、先頭に "#" を付けた記号にする
    - 漢字: そのまま
    - それ以外: 一旦そのまま文字列として保持（将来拡張用）
    """
    if x is None:
        return ""

    s = str(x).strip()
    if not s:
        return ""

    # 黒マス
    if s == "■":
        return "■"

    # すでに "#3" のような形式なら、そのまま使う
    if s.startswith("#") and s[1:].isdigit():
        return f"#{int(s[1:])}"

    # 素の数字なら "#数字" に変換
    if s.isdigit():
        return f"#{int(s)}"

    # 漢字っぽい文字はそのまま
    if is_kanji(s):
        return s

    # それ以外は、そのまま返しておく（ほとんど使わない想定）
    return s


def normalize_grid(df: pd.DataFrame) -> np.ndarray:
    """
    DataFrame から 2次元 numpy 配列に変換し、
    各セルを :func:`normalize_cell` によって正規化します。

    Parameters
    ----------
    df : pandas.DataFrame
        入力の盤面データ。

    Returns
    -------
    numpy.ndarray
        shape = (rows, cols) の 2次元配列。
    """
    rows, cols = df.shape
    grid = np.empty((rows, cols), dtype=object)

    for i in range(rows):
        for j in range(cols):
            grid[i, j] = normalize_cell(df.iat[i, j])

    return grid
