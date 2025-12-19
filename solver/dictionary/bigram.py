# -*- coding: utf-8 -*-
"""
文字レベルのbigram統計を使ったドメインフィルタリングモジュール
"""
from typing import Dict, Set, List
import pandas as pd
from collections import defaultdict
from solver.logging_utils import get_logger

logger = get_logger()


def extract_bigrams(csv_path: str, min_prob: float = 0.01) -> Dict[str, Dict[str, float]]:
    """
    wiki_compounds.csv から文字bigramを抽出し、条件付き確率を計算
    
    Parameters
    ----------
    csv_path : str
        wiki_compounds.csvのパス
    min_prob : float
        最小確率閾値（これ未満は除外してメモリ節約）
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        bigram_probs[前の文字][次の文字] = P(次|前)
        
    Example
    -------
    >>> probs = extract_bigrams("data/wiki_compounds.csv")
    >>> probs['日']['本']  # P(本|日)
    0.25
    >>> probs['日']['石']  # 存在しない場合はKeyError
    KeyError
    """
    logger.info(f"Extracting bigrams from {csv_path}...")
    
    # カウント用の辞書
    # bigram_counts[前][次] = 出現回数
    bigram_counts = defaultdict(lambda: defaultdict(int))
    
    try:
        # CSVを読み込み
        df = pd.read_csv(csv_path)
        
        if "word" not in df.columns:
            logger.warning("No 'word' column found in CSV")
            return {}
        
        # 各単語からbigramを抽出
        processed = 0
        for _, row in df.iterrows():
            word = str(row["word"])
            
            # 単語内の連続する文字ペアを抽出
            for i in range(len(word) - 1):
                prev_char = word[i]
                next_char = word[i + 1]
                bigram_counts[prev_char][next_char] += 1
            
            processed += 1
            if processed % 50000 == 0:
                logger.info(f"Processed {processed} words...")
        
        logger.info(f"Total words processed: {processed}")
        
    except Exception as e:
        logger.error(f"Failed to extract bigrams: {e}")
        return {}
    
    # 条件付き確率を計算
    bigram_probs = {}
    total_bigrams = 0
    filtered_bigrams = 0
    
    for prev_char, next_counts in bigram_counts.items():
        total = sum(next_counts.values())
        probs = {}
        
        for next_char, count in next_counts.items():
            prob = count / total
            
            # 閾値フィルタリング
            if prob >= min_prob:
                probs[next_char] = prob
                total_bigrams += 1
            else:
                filtered_bigrams += 1
        
        if probs:  # 空でなければ追加
            bigram_probs[prev_char] = probs
    
    logger.info(f"Extracted {len(bigram_probs)} unique first characters")
    logger.info(f"Total bigrams (>={min_prob*100}%): {total_bigrams}")
    logger.info(f"Filtered low-prob bigrams: {filtered_bigrams}")
    
    return bigram_probs


def filter_domain_by_bigram(
    var: int,
    domain: Set[str],
    assignment: Dict[int, str],
    var_occurrences: Dict[int, List],
    bigram_probs: Dict[str, Dict[str, float]],
    slots: List,  # 追加：スロット情報が必要
) -> Set[str]:
    """
    変数varのドメインを、bigram確率に基づいてフィルタリング
    
    Parameters
    ----------
    var : int
        フィルタリング対象の変数ID
    domain : Set[str]
        現在のドメイン
    assignment : Dict[int, str]
        現在の割り当て
    var_occurrences : Dict[int, List[Tuple[int, int]]]
        変数の出現箇所情報（slot_id, position）のリスト
    bigram_probs : Dict[str, Dict[str, float]]
        bigram確率マップ
    slots : List[Slot]
        スロットのリスト
        
    Returns
    -------
    Set[str]
        フィルタリング後のドメイン
    """
    if not bigram_probs:
        return domain  # bigramデータがない場合はフィルタリングしない
    
    # スロットIDからスロットオブジェクトへのマップを作成
    slot_map = {slot.slot_id: slot for slot in slots}
    
    filtered = set()
    
    for candidate in domain:
        valid = True
        
        # 各出現箇所で確認（タプル形式：(slot_id, position)）
        for slot_id, position in var_occurrences.get(var, []):
            slot = slot_map.get(slot_id)
            if not slot:
                continue
            
            pattern = slot.pattern
            
            # 左隣の文字を取得
            if position > 0:
                left_pattern = pattern[position - 1]
                left_char = None
                
                # 左隣が変数の場合
                if isinstance(left_pattern, str) and left_pattern.startswith('#'):
                    left_var = int(left_pattern[1:])
                    if left_var in assignment:
                        left_char = assignment[left_var]
                # 左隣が確定文字の場合
                else:
                    left_char = str(left_pattern)
                
                # 左隣が確定していて、bigramチェック
                if left_char:
                    # P(candidate | left_char) が存在するか
                    if left_char not in bigram_probs:
                        valid = False
                        break
                    if candidate not in bigram_probs[left_char]:
                        valid = False
                        break
            
            # 右隣の文字を取得
            if position < len(pattern) - 1:
                right_pattern = pattern[position + 1]
                right_char = None
                
                # 右隣が変数の場合
                if isinstance(right_pattern, str) and right_pattern.startswith('#'):
                    right_var = int(right_pattern[1:])
                    if right_var in assignment:
                        right_char = assignment[right_var]
                # 右隣が確定文字の場合
                else:
                    right_char = str(right_pattern)
                
                # 右隣が確定していて、bigramチェック
                if right_char:
                    # P(right_char | candidate) が存在するか
                    if candidate not in bigram_probs:
                        valid = False
                        break
                    if right_char not in bigram_probs[candidate]:
                        valid = False
                        break
            
            if not valid:
                break
        
        if valid:
            filtered.add(candidate)
    
    # フィルタリング結果が空の場合は元のドメインを返す（安全策）
    return filtered if filtered else domain
