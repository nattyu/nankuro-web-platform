# solver/csp/search_ga.py
# -*- coding: utf-8 -*-
"""
Genetic Algorithm（遺伝的アルゴリズム）による探索を行うモジュールです。

GAの特徴:
- 複数の解候補（個体）を並列的に進化させる
- 交叉と突然変異で新しい解を生成
- 局所最適に陥りにくい
"""

from __future__ import annotations

import random
from typing import Dict, Set, List, Tuple
from dataclasses import dataclass

import pandas as pd

from ..types import Slot, CSPState
from ..logging_utils import get_logger
from .scoring import evaluate_state_score
from .search import max_csp_search

logger = get_logger()


@dataclass
class Individual:
    """遺伝的アルゴリズムの個体"""
    assignment: Dict[int, str]
    score: float


def create_initial_population(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    initial_domains: Dict[int, Set[str]],
    initial_slot_candidates: Dict[int, Set[int]],
    var_occurrences: Dict[int, List],
    slot_has_dict: Dict[int, bool],
    frequency_map: Dict[str, float],
    population_size: int,
    dfs_nodes: int,
) -> List[Individual]:
    """
    初期個体群を生成します。
    DFSを短時間実行して、多様な初期状態を作ります。
    """
    population = []
    
    logger.info("[GA] Creating initial population (size=%d)...", population_size)
    
    for i in range(population_size):
        # ランダムシードを変えて多様性を確保
        random.seed(random.randint(0, 1000000))
        
        # DFSで個体を生成
        state = max_csp_search(
            slots=slots,
            dict_df=dict_df,
            initial_domains=initial_domains,
            initial_slot_candidates=initial_slot_candidates,
            var_occurrences=var_occurrences,
            slot_has_dict=slot_has_dict,
            max_nodes=dfs_nodes,
            frequency_map=frequency_map,
        )
        
        individual = Individual(
            assignment=dict(state.assignment),
            score=state.score,
        )
        population.append(individual)
        
        if (i + 1) % 10 == 0:
            logger.info("[GA] Created %d/%d individuals...", i + 1, population_size)
    
    # スコア順にソート
    population.sort(key=lambda x: x.score, reverse=True)
    logger.info("[GA] Best initial score: %.3f", population[0].score)
    
    return population


def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
    """
    トーナメント選択: ランダムに選んだ個体の中で最良のものを返す
    """
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.score)


def crossover(
    parent1: Individual,
    parent2: Individual,
    var_domains: Dict[int, Set[str]],
) -> Individual:
    """
    2点交叉: 2つの親から子個体を生成
    """
    child_assignment = {}
    
    all_vars = set(parent1.assignment.keys()) | set(parent2.assignment.keys())
    
    for var in all_vars:
        # 50%の確率でどちらかの親から遺伝子を継承
        if random.random() < 0.5:
            if var in parent1.assignment:
                child_assignment[var] = parent1.assignment[var]
        else:
            if var in parent2.assignment:
                child_assignment[var] = parent2.assignment[var]
    
    # ドメインチェック: 無効な割り当てを削除
    valid_assignment = {}
    for var, char in child_assignment.items():
        dom = var_domains.get(var, set())
        if char in dom:
            valid_assignment[var] = char
    
    return Individual(assignment=valid_assignment, score=0.0)


def mutate(
    individual: Individual,
    var_domains: Dict[int, Set[str]],
    mutation_rate: float,
) -> Individual:
    """
    突然変異: 確率的にランダムな変数の値を変更
    """
    mutated_assignment = dict(individual.assignment)
    
    for var in list(mutated_assignment.keys()):
        if random.random() < mutation_rate:
            dom = var_domains.get(var, set())
            if dom and len(dom) > 1:
                # 現在の値以外からランダムに選択
                candidates = [c for c in dom if c != mutated_assignment[var]]
                if candidates:
                    mutated_assignment[var] = random.choice(candidates)
    
    # 追加の突然変異: 未割り当て変数をランダムに追加
    if random.random() < mutation_rate:
        unassigned_vars = [v for v in var_domains.keys() if v not in mutated_assignment]
        if unassigned_vars:
            var = random.choice(unassigned_vars)
            dom = var_domains.get(var, set())
            if dom:
                mutated_assignment[var] = random.choice(list(dom))
    
    return Individual(assignment=mutated_assignment, score=0.0)


def evaluate_population(
    population: List[Individual],
    slots: List[Slot],
    dict_df: pd.DataFrame,
    initial_slot_candidates: Dict[int, Set[int]],
    initial_domains: Dict[int, Set[str]],
    frequency_map: Dict[str, float],
) -> None:
    """
    個体群のスコアを評価（インプレース更新）
    """
    from .propagation import propagate
    
    for individual in population:
        slot_cands, _ = propagate(
            slots, dict_df, initial_slot_candidates, individual.assignment, initial_domains
        )
        individual.score = evaluate_state_score(
            slots, dict_df, slot_cands, individual.assignment, frequency_map
        )


def genetic_algorithm(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    initial_domains: Dict[int, Set[str]],
    initial_slot_candidates: Dict[int, Set[int]],
    var_occurrences: Dict[int, List],
    slot_has_dict: Dict[int, bool],
    frequency_map: Dict[str, float] = None,
    population_size: int = 30,
    generations: int = 100,
    mutation_rate: float = 0.15,
    crossover_rate: float = 0.8,
    elite_size: int = 2,
    dfs_nodes: int = 300,
) -> Tuple[Dict[int, str], float]:
    """
    Genetic Algorithm によって最適な割り当てを探索します。
    
    Returns
    -------
    tuple
        (best_assignment, best_score)
    """
    # 初期個体群の生成
    population = create_initial_population(
        slots, dict_df, initial_domains, initial_slot_candidates,
        var_occurrences, slot_has_dict, frequency_map, population_size, dfs_nodes
    )
    
    best_individual = population[0]
    no_improvement_count = 0
    
    logger.info("[GA] Starting evolution for %d generations...", generations)
    
    for generation in range(generations):
        # エリート保存
        population.sort(key=lambda x: x.score, reverse=True)
        elite = population[:elite_size]
        
        # 新しい個体群を生成
        new_population = list(elite)
        
        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                # 交叉: エリートとトーナメント選択を組み合わせる
                parent1 = elite[random.randint(0, len(elite) - 1)]  # エリートから選択
                parent2 = tournament_selection(population)
                child = crossover(parent1, parent2, initial_domains)
            else:
                # 複製: トーナメント選択から直接コピー
                parent = tournament_selection(population)
                child = Individual(
                    assignment=dict(parent.assignment),
                    score=parent.score,  # スコアもコピー
                )
            
            # 突然変異: より保守的に
            if random.random() < mutation_rate * 0.5:  # 確率を半分に
                # 1変数のみ変更
                if child.assignment:
                    var = random.choice(list(child.assignment.keys()))
                    dom = initial_domains.get(var, set())
                    if dom and len(dom) > 1:
                        candidates = [c for c in dom if c != child.assignment[var]]
                        if candidates:
                            child.assignment[var] = random.choice(candidates)
                            child.score = 0.0  # 再評価が必要
            
            new_population.append(child)
        
        # スコア評価（スコアが0.0の個体のみ）
        to_evaluate = [ind for ind in new_population if ind.score == 0.0]
        if to_evaluate:
            evaluate_population(
                to_evaluate, slots, dict_df, initial_slot_candidates,
                initial_domains, frequency_map
            )
        
        population = new_population
        population.sort(key=lambda x: x.score, reverse=True)
        
        # ベスト更新チェック
        if population[0].score > best_individual.score:
            best_individual = population[0]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # 進捗ログ
        if (generation + 1) % 10 == 0 or generation == 0:
            logger.info(
                "[GA] Generation %d/%d: best=%.3f, avg=%.3f, no_improvement=%d",
                generation + 1, generations,
                population[0].score,
                sum(ind.score for ind in population) / len(population),
                no_improvement_count
            )
        
        # 早期終了: 15世代連続で改善なし
        if no_improvement_count >= 15:
            logger.info("[GA] Early stopping: no improvement for 15 generations")
            break
    
    logger.info("[GA] Evolution complete. Best score=%.3f", best_individual.score)
    
    return best_individual.assignment, best_individual.score
