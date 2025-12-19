# solver/csp/search_sa.py
# -*- coding: utf-8 -*-
"""
Simulated Annealing（焼きなまし法）による探索を行うモジュールです。

焼きなまし法の特徴:
- 確率的に「悪い手」も受け入れることで局所最適から脱出
- 温度パラメータで探索の「大胆さ」を制御
- 時間とともに温度を下げて解を収束させる
"""

from __future__ import annotations

import math
import random
from typing import Dict, Set, List, Tuple
from dataclasses import dataclass

import pandas as pd

from ..types import Slot
from ..logging_utils import get_logger
from .scoring import evaluate_state_score
from .propagation import propagate

logger = get_logger()


def build_initial_assignment(
    var_domains: Dict[int, Set[str]],
    var_occurrences: Dict[int, List],
    slots: List[Slot],
    dict_df: pd.DataFrame,
    slot_candidates: Dict[int, Set[int]],
) -> Dict[int, str]:
    """
    初期割り当てを構築します。
    ランダムではなく、制約をなるべく満たすようにグリーディに構築します。
    
    戦略:
    1. ドメインサイズが小さい変数から順に処理（MRV）
    2. 各変数について、最も多くのスロットで辞書候補を残す値を選択
    """
    assignment = {}
    
    # ドメインサイズでソート（小さい順）
    vars_sorted = sorted(
        var_domains.keys(),
        key=lambda v: (len(var_domains[v]) if var_domains[v] else 9999, -len(var_occurrences.get(v, [])))
    )
    
    for var in vars_sorted:
        dom = var_domains.get(var, set())
        if not dom:
            continue
        
        if len(dom) == 1:
            # ドメインが1つなら即決定
            assignment[var] = list(dom)[0]
        else:
            # 各候補値について、「この値を割り当てたときに、
            # 関連するスロットの辞書候補が残る個数」を評価
            best_char = None
            best_remain_count = -1
            
            for char in dom:
                # 試しに割り当ててみる
                temp_assignment = dict(assignment)
                temp_assignment[var] = char
                
                # このvarが関わるスロットで、候補が残る数をカウント
                remain_count = 0
                for slot_id, _pos in var_occurrences.get(var, []):
                    cands = slot_candidates.get(slot_id, set())
                    if not cands:
                        continue
                    
                    # このスロットで temp_assignment と整合する候補が残るか
                    slot = next((s for s in slots if s.slot_id == slot_id), None)
                    if not slot:
                        continue
                    
                    for eid in cands:
                        chars_str = str(dict_df.at[eid, "word"])
                        ok = True
                        for pos, pat in enumerate(slot.pattern):
                            from .domains import extract_var_id
                            vid = extract_var_id(pat)
                            if vid is not None and vid in temp_assignment:
                                if pos >= len(chars_str) or chars_str[pos] != temp_assignment[vid]:
                                    ok = False
                                    break
                        if ok:
                            remain_count += 1
                            break  # このスロットは候補が残る
                
                if remain_count > best_remain_count:
                    best_remain_count = remain_count
                    best_char = char
            
            if best_char is None:
                # 評価できなかった場合はランダム
                best_char = random.choice(list(dom))
            
            assignment[var] = best_char
    
    return assignment


def generate_neighbor(
    current_assignment: Dict[int, str],
    var_domains: Dict[int, Set[str]],
    temperature: float,
) -> Dict[int, str]:
    """
    現在の割り当てから近傍状態を生成します。
    
    温度が高いときは大きな変化、低いときは小さな変化を許容します。
    
    操作:
    1. Change: 1つの変数の値を変更
    2. Swap: 2つの変数の値を交換
    3. Multi-change: 複数の変数を同時に変更（温度が高い時のみ）
    """
    neighbor = dict(current_assignment)
    vars_list = list(current_assignment.keys())
    
    # 温度が高いほど、大きな変化を許容
    if temperature > 50 and random.random() < 0.3:
        # Multi-change: 2-5個の変数を同時に変更
        num_changes = random.randint(2, min(5, len(vars_list)))
        for _ in range(num_changes):
            var = random.choice(vars_list)
            dom = var_domains.get(var, set())
            if dom:
                neighbor[var] = random.choice(list(dom))
    elif random.random() < 0.5:
        # Change: 1つの変数を変更
        var = random.choice(vars_list)
        dom = var_domains.get(var, set())
        if dom and len(dom) > 1:
            # 現在の値以外を選ぶ
            candidates = [c for c in dom if c != current_assignment[var]]
            if candidates:
                neighbor[var] = random.choice(candidates)
    else:
        # Swap: 2つの変数の値を交換
        if len(vars_list) >= 2:
            var1, var2 = random.sample(vars_list, 2)
            # ドメインチェック: 交換後も有効か確認
            if (var2 in current_assignment and 
                current_assignment[var2] in var_domains.get(var1, set()) and
                current_assignment[var1] in var_domains.get(var2, set())):
                neighbor[var1], neighbor[var2] = neighbor[var2], neighbor[var1]
    
    return neighbor


def simulated_annealing(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    initial_domains: Dict[int, Set[str]],
    initial_slot_candidates: Dict[int, Set[int]],
    var_occurrences: Dict[int, List],
    slot_has_dict: Dict[int, bool],
    frequency_map: Dict[str, float] = None,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.1,
    max_iterations: int = 50000,
) -> Tuple[Dict[int, str], float]:
    """
    Simulated Annealing によって最適な割り当てを探索します。
    
    戦略: DFS で良い初期状態を見つけてから、SAで改善する
    
    Returns
    -------
    tuple
        (best_assignment, best_score)
    """
    # 初期状態の構築: まずDFSで良い状態を見つける
    from .search import max_csp_search
    from ..types import CSPState
    
    logger.info("[SA] Building initial state with limited DFS...")
    initial_state = max_csp_search(
        slots=slots,
        dict_df=dict_df,
        initial_domains=initial_domains,
        initial_slot_candidates=initial_slot_candidates,
        var_occurrences=var_occurrences,
        slot_has_dict=slot_has_dict,
        max_nodes=1000,  # 少しだけDFSを実行
        frequency_map=frequency_map,
    )
    
    current_assignment = dict(initial_state.assignment)
    
    # 未割り当ての変数をランダムに埋める
    for var in initial_domains.keys():
        if var not in current_assignment:
            dom = initial_domains.get(var, set())
            if dom:
                current_assignment[var] = random.choice(list(dom))
    
    # 初期スコア計算
    current_slot_cands, _ = propagate(
        slots, dict_df, initial_slot_candidates, current_assignment, initial_domains
    )
    current_score = evaluate_state_score(
        slots, dict_df, current_slot_cands, current_assignment, frequency_map
    )
    
    best_assignment = dict(current_assignment)
    best_score = current_score
    
    temperature = initial_temp
    iteration = 0
    accept_count = 0
    reject_count = 0
    
    logger.info("[SA] Starting Simulated Annealing...")
    logger.info("[SA] Initial temp=%.2f, cooling_rate=%.4f, min_temp=%.2f", 
                initial_temp, cooling_rate, min_temp)
    logger.info("[SA] Initial score=%.3f", current_score)
    
    while temperature > min_temp and iteration < max_iterations:
        iteration += 1
        
        # 近傍状態を生成
        neighbor_assignment = generate_neighbor(current_assignment, initial_domains, temperature)
        
        # 近傍状態のスコア計算
        neighbor_slot_cands, _ = propagate(
            slots, dict_df, initial_slot_candidates, neighbor_assignment, initial_domains
        )
        neighbor_score = evaluate_state_score(
            slots, dict_df, neighbor_slot_cands, neighbor_assignment, frequency_map
        )
        
        # スコア差を計算
        delta = neighbor_score - current_score
        
        # 受理判定
        accept = False
        if delta > 0:
            # 改善する場合は必ず受理
            accept = True
        else:
            # 悪化する場合も確率的に受理
            accept_prob = math.exp(delta / temperature)
            accept = random.random() < accept_prob
        
        if accept:
            current_assignment = neighbor_assignment
            current_score = neighbor_score
            accept_count += 1
            
            # ベスト更新チェック
            if current_score > best_score:
                best_assignment = dict(current_assignment)
                best_score = current_score
        else:
            reject_count += 1
        
        # 温度を冷却
        temperature *= cooling_rate
        
        # 進捗ログ
        if iteration % 1000 == 0:
            accept_rate = accept_count / (accept_count + reject_count) if (accept_count + reject_count) > 0 else 0
            logger.info(
                "[SA] iter=%d, T=%.3f, current=%.3f, best=%.3f, accept_rate=%.2f",
                iteration, temperature, current_score, best_score, accept_rate
            )
            accept_count = 0
            reject_count = 0
    
    logger.info("[SA] Finished. Best score=%.3f (iterations=%d)", best_score, iteration)
    
    return best_assignment, best_score
