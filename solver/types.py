# -*- coding: utf-8 -*-
"""
ナンクロ solver で使う主なデータ構造（型）をまとめたモジュールです。

dataclass を使うことで、
「この構造体はどんなフィールドを持っているのか」を
分かりやすく表現しています。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

# グリッド上の座標を表す型 (row, col)
CellCoord = Tuple[int, int]


@dataclass
class Slot:
    """
    黒マスで区切られた「語スロット」を表すクラスです。

    Attributes
    ----------
    slot_id : int
        スロットの識別番号。0,1,2,... の連番など。
    direction : str
        "across"（横方向）または "down"（縦方向）を表します。
    positions : list of (row, col)
        このスロットに含まれるマスの座標のリスト。
    pattern : list of str
        各マスに格納されている文字列のリスト。
        - 漢字（例: "山"）
        - 数字記号（例: "#12"）
        などが入ります。
    """

    slot_id: int
    direction: str  # "across" or "down"
    positions: List[CellCoord]
    pattern: List[str]

    @property
    def length(self) -> int:
        """このスロットの長さ（マス数）を返します。"""
        return len(self.positions)


@dataclass
class CSPState:
    """
    Max-CSP 探索中の状態を表すクラスです。

    Attributes
    ----------
    assignment : dict[int, str]
        数字（例: 12）→ 漢字（例: "山"）への割り当てを表します。
    domains : dict[int, set[str]]
        各数字が取り得る候補漢字の集合。
    slot_candidates : dict[int, set[int]]
        各スロットに対して、辞書上の候補語IDの集合。
    score : float
        現在の状態のスコア（大きいほど良い）。
    """

    assignment: Dict[int, str]
    domains: Dict[int, Set[str]]
    slot_candidates: Dict[int, Set[int]]
    score: float
