# -*- coding: utf-8 -*-
"""
制約伝播（propagation）を行うモジュールです。

ここでの制約伝播は、Max-CSP を前提としているため
「矛盾を見つけたら即座に枝を捨てる」ような
強いものではありません。

代わりに、
- スロットの辞書候補集合を現在の割り当てに合わせてフィルタ
- その情報から、未割り当て変数のドメインを少しだけ絞る
という“緩めの”伝播を行います。

なお、本バージョンでは「縮んだ候補集合」を再利用するため、
propagate() には「現在のスロット候補」を渡すようにしています。
これにより、ノードが深くなるほど候補集合が小さくなり、
1ノードあたりの計算量が減ることを期待しています。

今回の修正では、漢字ナンクロのルール
「同じ漢字は2回使えない（all-different 制約）」を
ドメイン側で強制します。
"""

from __future__ import annotations

from typing import Dict, Set, Tuple, List

import pandas as pd

from ..types import Slot
from .domains import extract_var_id


def filter_slot_candidates_by_assignment(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    base_slot_candidates: Dict[int, Set[int]],
    assignment: Dict[int, str],
) -> Dict[int, Set[int]]:
    """
    現在の assignment（数字→漢字）に基づいて、
    各スロットの候補語をフィルタリングします。

    Parameters
    ----------
    base_slot_candidates : dict[int, set[int]]
        直前の状態（親ノードなど）でのスロット候補集合。

    候補がゼロになっても、ここでは「矛盾」とは扱わず、
    ゼロ集合のまま残します。（Max-CSP で後でペナルティとして評価）
    """
    new_slot_cands: Dict[int, Set[int]] = {}

    for slot in slots:
        slot_id = slot.slot_id
        # 直前の状態における候補集合をベースにする
        base_cands = base_slot_candidates.get(slot_id, set())
        if not base_cands:
            new_slot_cands[slot_id] = set()
            continue

        filtered: Set[int] = set()
        for eid in base_cands:
            chars = str(dict_df.at[eid, "word"])
            ok = True
            for pos, pat in enumerate(slot.pattern):
                vid = extract_var_id(pat)
                if vid is not None and vid in assignment:
                    # 割り当て済みの数字の位置に、同じ漢字が入るかどうかチェック
                    if pos >= len(chars) or chars[pos] != assignment[vid]:
                        ok = False
                        break
            if ok:
                filtered.add(eid)

        new_slot_cands[slot_id] = filtered

    return new_slot_cands


def shrink_domains_by_slots(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    slot_candidates: Dict[int, Set[int]],
    assignment: Dict[int, str],
    domains: Dict[int, Set[str]],
) -> Dict[int, Set[str]]:
    """
    スロットの候補集合をもとに、
    未割り当て変数のドメインを少しだけ縮小します。

    ただし、ドメインを空にしてしまうと探索できなくなるため、
    空になりそうな場合は縮小前のドメインを維持します。

    ※ all-different 制約は別途 propagate() 内で適用します。
    """
    new_domains: Dict[int, Set[str]] = {k: set(v) for k, v in domains.items()}

    # 変数ごとに、その変数が出る位置に「現れうる漢字」の集合を集計
    possible_chars_by_var: Dict[int, Set[str]] = {v: set() for v in domains.keys()}

    for slot in slots:
        slot_id = slot.slot_id
        cand_ids = slot_candidates.get(slot_id, set())
        if not cand_ids:
            # このスロットには辞書候補が残っていない
            continue

        for pos, pat in enumerate(slot.pattern):
            vid = extract_var_id(pat)
            if vid is None or vid in assignment:
                # 固定漢字 or 既に割り当て済みの数字は無視
                continue

            chars_here: Set[str] = set()
            for eid in cand_ids:
                chars = str(dict_df.at[eid, "word"])
                if pos < len(chars):
                    chars_here.add(chars[pos])

            if chars_here:
                if not possible_chars_by_var[vid]:
                    possible_chars_by_var[vid] = chars_here
                else:
                    # union をとる（このスロットの情報を追加）
                    possible_chars_by_var[vid].update(chars_here)

    # 集計した possible_chars_by_var を使ってドメインを縮小
    for vid, poss in possible_chars_by_var.items():
        if not poss:
            continue

        old_dom = new_domains.get(vid, set())
        if not old_dom:
            continue

        filtered = old_dom & poss
        # 空でなければ縮小を採用する（辞書制約は Max-CSP 的にソフト）
        if filtered:
            new_domains[vid] = filtered

    return new_domains


def propagate(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    base_slot_candidates: Dict[int, Set[int]],
    assignment: Dict[int, str],
    domains: Dict[int, Set[str]],
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[str]]]:
    """
    制約伝播をまとめて行うヘルパー関数です。

    1. all-different 制約をドメインに適用
       （既に使われた漢字を、未割り当て変数のドメインから除外）
    2. assignment をもとにスロット候補をフィルタ
       （直前の候補集合 base_slot_candidates からさらに絞る）
    3. その結果から未割り当て変数のドメインを縮小

    Returns
    -------
    new_slot_cands : dict[int, set[int]]
        フィルタ後のスロット候補集合。
    new_domains : dict[int, set[str]]
        縮小後のドメイン。
    """
    # 1. all-different 制約を適用（ナンクロの「同じ漢字は1回だけ」ルール）
    #    これはパズルのハード制約なので、ドメインが空になってもそのまま返す。
    #    （その枝は search 側で探索不能として自動的に打ち切られる）
    new_domains: Dict[int, Set[str]] = {k: set(v) for k, v in domains.items()}
    used_chars: Set[str] = set(assignment.values())

    if used_chars:
        for vid, dom in new_domains.items():
            # 既に割り当て済みの数字は対象外
            if vid in assignment:
                continue
            # 使われた漢字をドメインから除外
            new_domains[vid] = dom - used_chars

    # 2. assignment を反映してスロット候補をフィルタ
    new_slot_cands = filter_slot_candidates_by_assignment(
        slots, dict_df, base_slot_candidates, assignment
    )

    # 3. スロット候補から、さらにドメインを（ソフトに）縮小
    new_domains = shrink_domains_by_slots(
        slots, dict_df, new_slot_cands, assignment, new_domains
    )

    return new_slot_cands, new_domains

def has_hard_contradiction(
    slots: List[Slot],
    slot_candidates: Dict[int, Set[int]],
    slot_has_dict: Dict[int, bool],
) -> bool:
    """
    「さすがにこれは無理筋」とみなせるハードな矛盾があるかを判定する。

    現時点ではシンプルに:

    - もともと辞書候補が存在したスロット (slot_has_dict[slot_id] == True) なのに、
      現在の slot_candidates[slot_id] が空集合になっている

    場合をハード矛盾とする。

    Max-CSP の発想で「一部のスロットは諦めてでも全体スコアを上げたい」
    という場面もあるが、辞書信頼度の高いスロットについては
    「候補ゼロ＝復活の見込みなし」として枝を切る方針。
    """
    for slot in slots:
        sid = slot.slot_id
        if not slot_has_dict.get(sid, False):
            # そもそも辞書候補に期待していないスロットは無視
            continue

        cands = slot_candidates.get(sid, set())
        if not cands:
            # 「辞書候補があったはずなのに今はゼロ」→ 無理筋とみなす
            return True

    return False

