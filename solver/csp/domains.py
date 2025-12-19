# -*- coding: utf-8 -*-
"""
数字（記号）ごとの初期ドメイン（候補漢字集合）を計算するモジュールです。

- 各スロットの辞書候補集合をもとに
- その数字が現れる位置に登場しうる漢字を集計し
- 共通集合や和集合をとることで初期ドメインとします。

ここでは、辞書が不完全な場合も想定し、
共通集合が空になった場合は和集合、
それすらなければ辞書全体の漢字集合を使うようにしています。

★ 本バージョンでは「候補となる漢字を常用漢字のみに制限」しています。
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import pandas as pd

from ..types import Slot
from solver.logging_utils import get_logger
from solver.jouyou_kanji import get_kanji_list  # 常用漢字リスト

# "#12" のような形式から数字部分を取り出すための正規表現
VAR_RE = re.compile(r"#(\d+)")

# 常用漢字セット（起動時に1回だけ作成）
JOUYOU_KANJI_SET: Set[str] = set(get_kanji_list())

logger = get_logger()


def extract_var_id(cell: str) -> int | None:
    """
    セル文字列から数字IDを取り出します。

    例:
    - "#12" -> 12
    - "山"   -> None
    """
    if not isinstance(cell, str):
        return None
    m = VAR_RE.match(cell)
    if not m:
        return None
    return int(m.group(1))


def build_initial_domains(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    slot_candidates: Dict[int, Set[int]],
    norm_grid,  # numpy.ndarray
):
    """
    初期ドメインと数字出現箇所情報を構築します。

    Parameters
    ----------
    slots : list[Slot]
        抽出済みのスロット一覧。
    dict_df : pandas.DataFrame
        熟語辞書。
        - 各行に "text" カラムがあり、熟語を構成する漢字列を保持している前提。
    slot_candidates : dict[int, set[int]]
        各スロットの辞書候補語ID集合。
    norm_grid : numpy.ndarray
        正規化された盤面グリッド。ヒント漢字を抽出するために使用。

    Returns
    -------
    domains : dict[int, set[str]]
        各数字ID -> 候補漢字集合（★常用漢字のみ、ヒント漢字を除外）
    var_occurrences : dict[int, list[(int,int)]]
        各数字ID -> [(slot_id, 位置pos), ...]
    global_char_set : set[str]
        辞書に登場する全ての「常用漢字」の集合。
    """
    # 盤面に既に存在する漢字（ヒント）を収集
    from ..grid.parser import is_kanji

    hint_kanji: Set[str] = set()
    for row in norm_grid:
        for cell in row:
            if isinstance(cell, str) and is_kanji(cell):
                hint_kanji.add(cell)

    if hint_kanji:
        logger.info(f"[INFO] ヒント漢字を除外: {sorted(hint_kanji)}")

    # まず、各数字が「どのスロットの何番目」に現れるかを集計
    var_occurrences: Dict[int, List[Tuple[int, int]]] = {}
    for slot in slots:
        for pos, pat in enumerate(slot.pattern):
            vid = extract_var_id(pat)
            if vid is not None:
                var_occurrences.setdefault(vid, []).append((slot.slot_id, pos))

    # 辞書に出てくる全漢字集合（★常用漢字のみ）
    global_char_set: Set[str] = set()
    for text in dict_df["text"]:
        if not isinstance(text, str):
            continue
        for ch in text:
            if ch in JOUYOU_KANJI_SET:
                global_char_set.add(ch)

    domains: Dict[int, Set[str]] = {}

    # 各数字ごとに初期ドメインを計算
    for vid, occs in var_occurrences.items():
        char_sets: List[Set[str]] = []

        for slot_id, pos in occs:
            cand_ids = slot_candidates.get(slot_id, set())
            if not cand_ids:
                # このスロットには辞書候補がない（辞書に載っていない語かもしれない）
                continue

            text_here: Set[str] = set()
            for eid in cand_ids:
                text = dict_df.at[eid, "text"]
                if not isinstance(text, str):
                    print("辞書の文字列が不正です:", text)
                    continue
                if pos < len(text):
                    ch = text[pos]
                    # ★ 常用漢字フィルタ：候補に入れるのは常用漢字のみ
                    if ch in JOUYOU_KANJI_SET:
                        text_here.add(ch)

            # このスロット・位置について常用漢字の候補が存在する場合のみ採用
            if text_here:
                char_sets.append(text_here)

        if not char_sets:
            # 辞書から何の情報も得られない場合：
            # ひとまず「辞書全体に登場する常用漢字集合」をドメインとする。
            domains[vid] = set(global_char_set)
            continue

        # 共通集合（スロットをまたいで共通する漢字）
        inter = set.intersection(*char_sets)
        if inter:
            domains[vid] = inter
            # logger.info(f"[INFO] 数字 #{vid} の初期ドメインサイズ: {len(domains[vid])} (共通集合)")
        else:
            # 共通集合が空：辞書に不備がある可能性があるので、
            # 和集合（どこかで登場する漢字の総集合）を使う。
            union: Set[str] = set()
            for s in char_sets:
                union.update(s)

            # 念のため、ここでも常用漢字のみに制限（理論上 union は既に常用漢字のみのはず）
            union = {ch for ch in union if ch in JOUYOU_KANJI_SET}

            domains[vid] = union if union else set(global_char_set)

    # ヒント漢字を全てのドメインから除外
    for vid in domains:
        original_size = len(domains[vid])
        domains[vid] = domains[vid] - hint_kanji
        # if len(domains[vid]) < original_size:
        #     logger.info(
        #         f"[INFO] 数字 #{vid}: ヒント漢字除外後のドメインサイズ {original_size} -> {len(domains[vid])}"
        #     )

    return domains, var_occurrences, global_char_set


# ======================================================================
# 追加関数: assignment を考慮してドメインを再計算する
# ======================================================================

def recompute_domains_with_assignment(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    slot_candidates: Dict[int, Set[int]],
    norm_grid,  # numpy.ndarray
    assignment: Dict[int, str],
):
    """
    assignment に基づいて、各スロットの辞書候補を再フィルタし、
    ドメインを再計算する関数です。

    「build_initial_domains の assignment 対応版」というイメージで、
    すでに確定している数字 -> 漢字 の割り当てと矛盾する熟語候補は除外したうえで、
    各数字のドメインを再構築します。

    Parameters
    ----------
    slots : list[Slot]
        抽出済みのスロット一覧。
    dict_df : pandas.DataFrame
        熟語辞書（"text" カラムを持つ前提）。
    slot_candidates : dict[int, set[int]]
        各スロットの辞書候補語ID集合（初期計算時と同じもの）。
    norm_grid : numpy.ndarray
        正規化された盤面グリッド。ヒント漢字を抽出するために使用。
    assignment : dict[int, str]
        すでに確定している数字ID -> 漢字 の割り当て。

    Returns
    -------
    domains : dict[int, set[str]]
        assignment と矛盾しない候補のみで再構成されたドメイン。
    var_occurrences : dict[int, list[(int,int)]]
        各数字ID -> [(slot_id, 位置pos), ...]（build_initial_domains と同様）。
    global_char_set : set[str]
        辞書に登場する全ての「常用漢字」の集合。
    """
    from ..grid.parser import is_kanji

    # 盤面に既に存在する漢字（ヒント）を収集
    hint_kanji: Set[str] = set()
    for row in norm_grid:
        for cell in row:
            if isinstance(cell, str) and is_kanji(cell):
                hint_kanji.add(cell)

    #if hint_kanji:
    #    logger.info(f"[RECOMPUTE] ヒント漢字を除外: {sorted(hint_kanji)}")

    # 各スロットの slot_id -> Slot マップを作っておく
    slot_map: Dict[int, Slot] = {slot.slot_id: slot for slot in slots}

    # 再度 var_occurrences を作る（スロット構造は変わらないので build_initial_domains と同様）
    var_occurrences: Dict[int, List[Tuple[int, int]]] = {}
    for slot in slots:
        for pos, pat in enumerate(slot.pattern):
            vid = extract_var_id(pat)
            if vid is not None:
                var_occurrences.setdefault(vid, []).append((slot.slot_id, pos))

    # 辞書に出てくる全漢字集合（★常用漢字のみ）: これは assignment に依存しない
    global_char_set: Set[str] = set()
    for text in dict_df["text"]:
        if not isinstance(text, str):
            continue
        for ch in text:
            if ch in JOUYOU_KANJI_SET:
                global_char_set.add(ch)

    domains: Dict[int, Set[str]] = {}

    # 各数字ごとにドメインを再計算
    for vid, occs in var_occurrences.items():
        char_sets: List[Set[str]] = []

        for slot_id, pos in occs:
            cand_ids = slot_candidates.get(slot_id, set())
            if not cand_ids:
                continue

            slot = slot_map.get(slot_id)
            if slot is None:
                continue
            pattern = slot.pattern

            text_here: Set[str] = set()
            for eid in cand_ids:
                text = dict_df.at[eid, "text"]
                if not isinstance(text, str):
                    print("辞書の文字列が不正です:", text)
                    continue
                if len(text) < len(pattern):
                    # 長さが合っていない候補はスキップ
                    continue

                # まず assignment との整合性チェック
                ok = True
                for j, cell in enumerate(pattern):
                    ch2 = text[j]
                    v2 = extract_var_id(cell)

                    if v2 is None:
                        # 固定漢字マスの場合
                        if isinstance(cell, str) and is_kanji(cell):
                            if ch2 != cell:
                                ok = False
                                break
                    else:
                        # 数字マスの場合: すでに assignment があれば、それと一致する必要がある
                        assigned = assignment.get(v2)
                        if assigned is not None and ch2 != assigned:
                            ok = False
                            break

                if not ok:
                    continue

                # このスロットでの vid の位置（pos）の文字を候補として集める
                if pos < len(text):
                    ch = text[pos]
                    if ch in JOUYOU_KANJI_SET:
                        text_here.add(ch)

            if text_here:
                char_sets.append(text_here)

        if not char_sets:
            # 何も情報がなくなった場合は、初期と同様に global_char_set を使う
            domains[vid] = set(global_char_set)
            continue

        inter = set.intersection(*char_sets)
        if inter:
            domains[vid] = inter
        else:
            union: Set[str] = set()
            for s in char_sets:
                union.update(s)
            union = {ch for ch in union if ch in JOUYOU_KANJI_SET}
            domains[vid] = union if union else set(global_char_set)

    # ヒント漢字を全てのドメインから除外
    for v in domains:
        original_size = len(domains[v])
        domains[v] = domains[v] - hint_kanji
        # if len(domains[v]) < original_size:
        #     logger.info(
        #         f"[RECOMPUTE] 数字 #{v}: ヒント漢字除外後のドメインサイズ {original_size} -> {len(domains[v])}"
        #     )

    return domains, var_occurrences, global_char_set
