# -*- coding: utf-8 -*-
"""
熟語辞書に対する検索用インデックスを作成するモジュールです。

インデックスを作ることで、
- 語の長さ
- ある位置に特定の漢字があるかどうか
などを使った絞り込みを高速に行えるようになります。
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

import pandas as pd


def build_indexes(
    df: pd.DataFrame,
) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, int, str], Set[int]]]:
    """
    辞書 DataFrame からインデックスを作成します。

    Parameters
    ----------
    df : pandas.DataFrame
        load_dictionary() で読み込んだ DataFrame。

    Returns
    -------
    len_index : dict[int, set[int]]
        {長さ L -> 辞書行IDの集合} のマップ。
    pos_char_index : dict[(int,int,str), set[int]]
        {(長さ L, 位置 pos, 漢字 ch) -> 辞書行ID集合} のマップ。
    """
    len_index: Dict[int, Set[int]] = {}
    pos_char_index: Dict[Tuple[int, int, str], Set[int]] = {}

    for entry_id, row in df.iterrows():
        length = int(row["length"])
        chars = list(row["word"])

        # entry_idがint型またはstr型の場合のみ変換を試みる
        if isinstance(entry_id, int):
            idx = entry_id
        elif isinstance(entry_id, str):
            try:
                idx = int(entry_id)
            except ValueError:
                idx = None
        else:
            idx = row.name if isinstance(row.name, int) else None
        if idx is None:
            continue  # インデックスがintでなければスキップ

        # 長さインデックス
        len_index.setdefault(length, set()).add(idx)

        # 位置＋文字インデックス
        for pos, ch in enumerate(chars):
            pos_char_index.setdefault((length, pos, ch), set()).add(idx)

    return len_index, pos_char_index
