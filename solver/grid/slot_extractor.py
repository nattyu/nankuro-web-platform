# -*- coding: utf-8 -*-
"""
黒マスで区切られた「語スロット」を抽出するモジュールです。

- 横方向（across）のスロット
- 縦方向（down）のスロット
をそれぞれ列挙します。
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..types import Slot


def extract_slots(grid: np.ndarray) -> List[Slot]:
    """
    黒マス "■" で区切られた横・縦のスロットを抽出します。

    Parameters
    ----------
    grid : numpy.ndarray
        :mod:`parser` で正規化された 2次元配列。

    Returns
    -------
    list of Slot
        見つかった全スロットのリスト。
    """
    slots: List[Slot] = []
    rows, cols = grid.shape
    sid = 0  # スロットIDの連番

    # --- 横方向のスロットを探す ---
    for i in range(rows):
        j = 0
        while j < cols:
            if grid[i, j] == "■":
                # 黒マスならスロットには含まれないのでスキップ
                j += 1
                continue

            positions = []
            pattern = []

            # 黒マスに当たるまで右方向に伸ばす
            while j < cols and grid[i, j] != "■":
                positions.append((i, j))
                pattern.append(grid[i, j])
                j += 1

            # 長さ 2 以上の場合のみスロットとして採用
            if len(positions) >= 2:
                slots.append(
                    Slot(
                        slot_id=sid,
                        direction="across",
                        positions=positions,
                        pattern=pattern,
                    )
                )
                sid += 1

    # --- 縦方向のスロットを探す ---
    for j in range(cols):
        i = 0
        while i < rows:
            if grid[i, j] == "■":
                i += 1
                continue

            positions = []
            pattern = []

            # 黒マスに当たるまで下方向に伸ばす
            while i < rows and grid[i, j] != "■":
                positions.append((i, j))
                pattern.append(grid[i, j])
                i += 1

            if len(positions) >= 2:
                slots.append(
                    Slot(
                        slot_id=sid,
                        direction="down",
                        positions=positions,
                        pattern=pattern,
                    )
                )
                sid += 1

    return slots
