# -*- coding: utf-8 -*-
"""
Max-CSP の探索を行うモジュールです。

本バージョンでは、従来の再帰 DFS ではなく、
「スコアの高い状態から優先的に展開する Best-First Search」
を採用しています。

ざっくり流れ
------------
1. 初期状態（何も割り当てていない）を作り、スコアを計算してキューに入れる
2. ループのたびに、スコア最大の状態を取り出す
3. MRV で次に割り当てる変数を選び、LCV で値の順番を決める
4. 各値について制約伝播（propagate）し、新しい状態を作ってスコアを計算
5. その状態を再び優先度キューに入れる
6. 探索ノード数が max_nodes に達するか、キューが空になるまで続ける

こうすることで、同じノード数でも、
より「制約をたくさん満たしている」「スコアの高い」状態を優先的に深掘りできます。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, List, Optional

import heapq
import pandas as pd

from ..types import Slot, CSPState
from ..logging_utils import get_logger
from .propagation import propagate, has_hard_contradiction
from .scoring import evaluate_state_score

logger = get_logger()


@dataclass
class SearchContext:
    """
    探索全体で共有する情報をまとめたクラスです。
    """

    slots: List[Slot]
    dict_df: pd.DataFrame
    initial_slot_candidates: Dict[int, Set[int]]
    var_occurrences: Dict[int, List]
    slot_has_dict: Dict[int, bool]
    max_nodes: int
    frequency_map: Optional[Dict[str, float]] = None
    bigram_probs: Optional[Dict[str, Dict[str, float]]] = None

    nodes_visited: int = 0
    best_state: CSPState | None = None


def choose_next_var(
    assignment: Dict[int, str],
    domains: Dict[int, Set[str]],
    var_occurrences: Dict[int, List],
) -> int | None:
    """
    次に割り当てるべき数字（変数）を選びます。

    MRV（Minimum Remaining Values）＋ degree ヒューリスティック：
    - まだ割り当てられていない数字のうち、ドメインサイズが最も小さいもの
    - 同じなら、より多くのスロットに出現する変数を優先
    """
    candidates = [v for v in domains.keys() if v not in assignment]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda v: (
            len(domains[v]) if domains[v] else 999999,
            -len(var_occurrences.get(v, [])),
        )
    )


def order_values_lcv(
    ctx: SearchContext,
    state: CSPState,
    var: int,
    values: List[str],
) -> List[str]:
    """
    LCV（Least Constraining Value）で値の順序付けを行う。

    ヒューリスティック：
      - var が出現する各スロットについて、
        「その位置に value を置いても整合的な候補語が何個残るか」を数える。
      - 候補語をあまり減らさない value（= 影響が小さい）から試す。
    """
    dict_df = ctx.dict_df
    var_occs = ctx.var_occurrences.get(var, [])

    scored: List[tuple[float, str]] = []

    for val in values:
        impact = 0.0

        for slot_id, pos in var_occs:
            cands = state.slot_candidates.get(slot_id, set())
            if not cands:
                continue

            total_cands = len(cands)
            valid = 0

            for eid in cands:
                chars = str(dict_df.at[eid, "word"])
                if pos < len(chars) and chars[pos] == val:
                    valid += 1

            if valid == 0:
                # そのスロットの候補を全滅させるような値は強くペナルティ
                impact += total_cands
            else:
                impact += (total_cands - valid)

        scored.append((impact, val))

    scored.sort(key=lambda t: t[0])
    return [v for _, v in scored]


def max_csp_search(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    initial_domains: Dict[int, Set[str]],
    initial_slot_candidates: Dict[int, Set[int]],
    var_occurrences,
    slot_has_dict,
    max_nodes: int,
    frequency_map: Optional[Dict[str, float]] = None,
    bigram_probs: Optional[Dict[str, Dict[str, float]]] = None,
    initial_assignment: Optional[Dict[int, str]] = None,
) -> CSPState:
    """
    Max-CSP 探索のエントリポイント（Best-First Search 版）
    """

    ctx = SearchContext(
        slots=slots,
        dict_df=dict_df,
        initial_slot_candidates=initial_slot_candidates,
        var_occurrences=var_occurrences,
        slot_has_dict=slot_has_dict,
        max_nodes=max_nodes,
        frequency_map=frequency_map,
        bigram_probs=bigram_probs,
    )

    # 初期状態
    initial_state = CSPState(
        assignment=dict(initial_assignment) if initial_assignment else {},
        domains={k: set(v) for k, v in initial_domains.items()},
        slot_candidates={k: set(v) for k, v in initial_slot_candidates.items()},
        score=0.0,
    )
    initial_state.score = evaluate_state_score(
        slots, dict_df, initial_state.slot_candidates, initial_state.assignment, frequency_map or {}
    )
    ctx.best_state = initial_state

    # 優先度付きキュー：(-score, tie_breaker, state)
    heap: List[tuple[float, int, CSPState]] = []
    counter = 0
    heapq.heappush(heap, (-initial_state.score, counter, initial_state))
    counter += 1

    while heap and ctx.nodes_visited < ctx.max_nodes:
        neg_score, _, state = heapq.heappop(heap)
        ctx.nodes_visited += 1

        current_score = -neg_score  # state.score と一致するはずだが一応

        if ctx.nodes_visited % 1000 == 0:
            logger.info(
                "[search] nodes_visited = %d, best_score=%.3f, queue_size=%d",
                ctx.nodes_visited,
                ctx.best_state.score if ctx.best_state else 0.0,
                len(heap),
            )

        # ベスト更新
        if current_score > ctx.best_state.score:
            ctx.best_state = CSPState(
                assignment=dict(state.assignment),
                domains={k: set(v) for k, v in state.domains.items()},
                slot_candidates={k: set(v) for k, v in state.slot_candidates.items()},
                score=current_score,
            )

        # すべて割り当て済みなら展開不要
        if len(state.assignment) == len(state.domains):
            continue

        # 次に割り当てる変数を選択（MRV + degree）
        var = choose_next_var(state.assignment, state.domains, ctx.var_occurrences)
        if var is None:
            continue

        dom = list(state.domains.get(var, []))
        if not dom:
            continue

        # bigram フィルタがあれば適用
        if ctx.bigram_probs:
            from solver.dictionary.bigram import filter_domain_by_bigram
            filtered_domain_set = filter_domain_by_bigram(
                var=var,
                domain=set(dom),
                assignment=state.assignment,
                var_occurrences=ctx.var_occurrences,
                bigram_probs=ctx.bigram_probs,
                slots=ctx.slots,
            )
            dom = list(filtered_domain_set)

        if not dom:
            continue

        # LCV で値の順序付け
        dom = order_values_lcv(ctx, state, var, dom)

        # 各値で子状態を生成
        for val in dom:
            if ctx.nodes_visited >= ctx.max_nodes:
                break

            new_assignment = dict(state.assignment)
            new_assignment[var] = val

            new_domains = {k: set(v) for k, v in state.domains.items()}
            new_domains[var] = {val}

            new_slot_cands, shrunk_domains = propagate(
                ctx.slots,
                ctx.dict_df,
                state.slot_candidates,
                new_assignment,
                new_domains,
            )

            # 1) ドメインが空になった変数があれば即バックトラック
            if any((vid not in new_assignment) and (len(dom) == 0)
                for vid, dom in shrunk_domains.items()):
                continue

            # 2) 辞書的に「もう無理筋」なら即バックトラック
            if has_hard_contradiction(ctx.slots, new_slot_cands, ctx.slot_has_dict):
                continue

            next_state = CSPState(
                assignment=new_assignment,
                domains=shrunk_domains,
                slot_candidates=new_slot_cands,
                score=0.0,
            )
            next_state.score = evaluate_state_score(
                ctx.slots,
                ctx.dict_df,
                next_state.slot_candidates,
                next_state.assignment,
                ctx.frequency_map or {},
            )

            heapq.heappush(heap, (-next_state.score, counter, next_state))
            counter += 1

    assert ctx.best_state is not None

    # 探索終了後、ドメインサイズ1の変数が残っていたら埋める（念のため）
    for var, dom in ctx.best_state.domains.items():
        if var not in ctx.best_state.assignment and len(dom) == 1:
            (ch,) = dom
            ctx.best_state.assignment[var] = ch

    return ctx.best_state
