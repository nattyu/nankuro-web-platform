# -*- coding: utf-8 -*-
"""
各スロットに対して、辞書上の候補語集合を計算するモジュールです。

ここでは、スロット内に含まれる「既知漢字」のみを使って、
辞書から候補を絞り込みます。

数字（例: "#12"）はこの段階では考慮しません。
（数字にどの漢字が入るかは、後のドメイン計算で行います）

また、計算量爆発を防ぐため、
1つのスロットが保持できる辞書候補数に上限を設けています。
上限を超える場合、そのスロットは「辞書候補なし」とみなし、
Max-CSP や BERT のフェーズに任せる方針を取ります。

今回の改修では、ナンクロから構築した
「漢字 → 熟語ID集合」（char_to_words）が渡されている場合、
候補が多すぎるスロットについては、その情報を使って
「ナンクロらしい語」を優先的に残すようにしています。
"""

from __future__ import annotations

from typing import Dict, Set, Tuple, List, Optional

import pandas as pd

from ..types import Slot
from ..grid.parser import is_kanji
from ..config import MAX_CANDIDATES_PER_SLOT


def build_initial_slot_candidates(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    len_index,
    pos_char_index,
    char_to_words: Optional[Dict[str, Set[int]]] = None,
) -> Tuple[Dict[int, Set[int]], Dict[int, bool]]:
    """
    各スロットに対して、辞書上の候補語ID集合を計算します。

    Parameters
    ----------
    slots : list[Slot]
        黒マスで区切られたスロットのリスト。
    dict_df : pandas.DataFrame
        熟語辞書 DataFrame。
        行インデックスが「語ID」として扱われる想定。
        少なくとも 'word' 列を持っている必要があります。
    len_index : dict[int, set[int]]
        長さ L ごとに、その長さを持つ辞書エントリID集合を持つインデックス。
        例: len_index[4] = {123, 456, ...}
    pos_char_index : dict[(int, int, str), set[int]]
        (長さ L, 位置 pos, 漢字 ch) をキーに、
        その位置に ch を持つ語ID集合を返すインデックス。
        例: pos_char_index[(4, 1, "学")] = {789, ...}
    char_to_words : dict[str, set[int]] or None, optional
        ナンクロ側から構築した
        「漢字 → 辞書エントリID集合」のインデックス。
        例: char_to_words["学"] = {語ID1, 語ID2, ...}
        渡されていれば、候補が多すぎるスロットに対して
        ナンクロ頻出語寄りに候補を絞るのに利用する。

    Returns
    -------
    slot_candidates : dict[int, set[int]]
        slot_id -> 辞書候補語ID集合。
    slot_has_dict : dict[int, bool]
        slot_id -> True/False。
        True の場合、そのスロットに辞書候補を使う（= スロットに辞書制約がある）。
        False の場合、そのスロットでは辞書候補を使わず、 Max-CSP / LM に任せる。
    """
    slot_candidates: Dict[int, Set[int]] = {}
    slot_has_dict: Dict[int, bool] = {}

    for slot in slots:
        L = len(slot.pattern)

        # その長さを持つ辞書語のID集合からスタート
        cand_ids: Set[int] = set(len_index.get(L, set()))
        if not cand_ids:
            # そもそもその長さの語が辞書に無ければ、辞書制約なし扱い
            slot_candidates[slot.slot_id] = set()
            slot_has_dict[slot.slot_id] = False
            continue

        # スロット内の「既知漢字」で候補を絞り込む
        known_kanji_positions: List[Tuple[int, str]] = []
        for pos, pat in enumerate(slot.pattern):
            if is_kanji(pat):
                known_kanji_positions.append((pos, pat))
                cand_ids &= pos_char_index.get((L, pos, pat), set())
                if not cand_ids:
                    break

        if not cand_ids:
            # 既知漢字で絞り込んだ結果、候補ゼロになった場合も
            # このスロットは辞書候補なしとして扱う
            slot_candidates[slot.slot_id] = set()
            slot_has_dict[slot.slot_id] = False
            continue

        # --- char_to_words による追加フィルタ（任意） ---
        # ナンクロから構築した「漢字→熟語ID」インデックスが渡されていれば、
        # スロット内の既知漢字を含む候補だけに絞り込むことで、
        # 大量の候補からナンクロらしい語を優先的に残す。
        if (
            cand_ids
            and len(cand_ids) > MAX_CANDIDATES_PER_SLOT
            and char_to_words is not None
            and known_kanji_positions
        ):
            narrowed: Set[int] = set()
            for _, ch in known_kanji_positions:
                # 辞書語ID集合の intersection を取ることで、
                # 「元の cand_ids にも含まれるナンクロ頻出語候補」だけを残す。
                narrowed |= (char_to_words.get(ch, set()) & cand_ids)

            # 絞り込みに成功した場合のみ採用（全消しや拡散は避ける）
            if narrowed and len(narrowed) < len(cand_ids):
                cand_ids = narrowed

        # フィルタ後の候補数が多すぎる場合は、
        # このスロットの辞書情報は使わない（候補なし扱い）
        if cand_ids and len(cand_ids) > MAX_CANDIDATES_PER_SLOT:
            slot_candidates[slot.slot_id] = set()
            slot_has_dict[slot.slot_id] = False
            continue

        slot_candidates[slot.slot_id] = cand_ids
        slot_has_dict[slot.slot_id] = len(cand_ids) > 0

    return slot_candidates, slot_has_dict
