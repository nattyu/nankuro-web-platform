# -*- coding: utf-8 -*-
"""
探索結果をもとに表示用の情報を構築するモジュールです。
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from ..types import Slot
from ..csp.domains import extract_var_id


def apply_assignment_to_grid(
    norm_grid: np.ndarray,
    assignment: Dict[int, str],
) -> np.ndarray:
    """
    正規化済みグリッドに、数字→漢字の割り当てを適用して
    「完成グリッド」を作ります。

    Parameters
    ----------
    norm_grid : numpy.ndarray
        数字が "#12" のような形で入った 2次元配列。
    assignment : dict[int, str]
        数字 → 漢字 の割り当て。

    Returns
    -------
    numpy.ndarray
        漢字が埋め込まれた完成グリッド。
    """
    rows, cols = norm_grid.shape
    out = norm_grid.copy()

    for i in range(rows):
        for j in range(cols):
            cell = out[i, j]
            vid = extract_var_id(cell)
            if vid is not None and vid in assignment:
                out[i, j] = assignment[vid]

    return out


def build_mapping_list(
    assignment: Dict[int, str],
    digit_conf: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    """
    数字→漢字＋信頼度の対応表を作る
    """
    items: List[Dict[str, Any]] = []
    digit_conf = digit_conf or {}

    for vid in sorted(assignment.keys()):
        items.append({
            "num": vid,
            "kanji": assignment[vid],
            "conf": float(digit_conf.get(vid, 1.0)),  # ★ デフォルト1.0
        })
    return items



def build_slot_texts(
    slots: List[Slot],
    assignment: Dict[int, str],
) -> List[Dict[str, Any]]:
    """
    各スロットについて、最終的にどんな語ができているかをまとめます。

    まだ決まっていない数字が含まれる場合は "□" を表示しておきます。
    """
    out: List[Dict[str, Any]] = []

    for slot in slots:
        chars: List[str] = []
        unresolved = False
        for pat in slot.pattern:
            vid = extract_var_id(pat)
            if vid is None:
                # 固定漢字（またはその他の記号）はそのまま
                chars.append(pat)
            else:
                if vid in assignment:
                    chars.append(assignment[vid])
                else:
                    chars.append("□")
                    unresolved = True

        out.append(
            {
                "slot_id": slot.slot_id,
                "direction": slot.direction,
                "positions": slot.positions,
                "text": "".join(chars),
                "unresolved": unresolved,
            }
        )

    return out


def build_result(
    original_df: pd.DataFrame,
    norm_grid: np.ndarray,
    assignment: Dict[int, str],
    slots: List[Slot],
    digit_conf: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:

    solved_grid = apply_assignment_to_grid(norm_grid, assignment)
    rows, cols = solved_grid.shape

    solved_df = pd.DataFrame(
        solved_grid,
        index=original_df.index,
        columns=original_df.columns,
    )

    return {
        "solved_board": solved_df.values.tolist(),  # ★ DataFrameを返さない
        "mapping": build_mapping_list(assignment, digit_conf),
        "shape": (rows, cols),
    }

