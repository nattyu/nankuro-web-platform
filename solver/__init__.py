# solver/__init__.py
# -*- coding: utf-8 -*-
"""
solver パッケージの入口となるモジュールです。

route/solver.py などから:

    from solver import solve

と呼び出されることを想定しています。

ここでは、盤面（pandas.DataFrame）を受け取り、
1. 盤面の前処理
2. 語スロットの抽出
3. 熟語辞書の読み込みとインデックス作成
4. Max-CSP による探索（数字→漢字割り当て）
5. BERT による最終調整（任意）
6. 表示用の結果構築
を順番に呼び出します。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Set
BigramProbs = Dict[str, Dict[str, float]]

from numpy import var
import pandas as pd

from .config import (
    DEFAULT_JUKUGO_PATH, MAX_SEARCH_NODES, BERT_ENABLED, NGRAM_LM_JSON_PATH, NGRAM_PROB_LOW_LIMIT,
    NANKURO_DICT_PATH, MERGED_DICT_PATH, NANKURO_LOCK_THRESHOLD,
    USE_STAGE1_NANKURO, USE_STAGE2_MERGED, USE_STAGE3_LM
)
from .logging_utils import get_logger
from .grid.parser import normalize_grid
from .grid.slot_extractor import extract_slots
from .dictionary.loader import load_dictionary
from .dictionary.indexer import build_indexes
from .csp.slots_candidates import build_initial_slot_candidates
from .csp.domains import build_initial_domains, recompute_domains_with_assignment
from .csp.search import max_csp_search
from .llm.refine_symbol import refine_with_bigram_and_bert
from .postprocess.render_result import build_result
from .dictionary.frequency import load_frequency_map
from .csp.scoring import get_ngram_lm
from .eval.confidence import evaluate_word_confidences

logger = get_logger()


def run_solver_with_dictionary(
    slots: list,
    dict_df: pd.DataFrame,
    norm_grid,
    frequency_map: Optional[Dict[str, float]] = None,
    bigram_probs: Optional[BigramProbs] = None,
    initial_assignment: Optional[Dict[int, str]] = None,
    max_nodes: int = MAX_SEARCH_NODES,
) -> Tuple[Dict[int, str], Dict[int, Any], float]:
    """
    指定された辞書を使って CSP 探索を1回実行するヘルパー関数。
    """
    if initial_assignment is None:
        initial_assignment = {}

    # 1) インデックス構築
    len_index, pos_char_index = build_indexes(dict_df)

    # 2) スロット候補計算
    slot_candidates, slot_has_dict = build_initial_slot_candidates(
        slots, dict_df, len_index, pos_char_index
    )

    # 3) ドメイン構築
    # initial_assignment がある場合は、それを考慮してドメインを絞り込む
    if initial_assignment:
        var_domains, var_occurrences, global_char_set = recompute_domains_with_assignment(
            slots=slots,
            dict_df=dict_df,
            slot_candidates=slot_candidates,
            norm_grid=norm_grid,
            assignment=initial_assignment,
        )
        # 念のため、initial_assignment にある変数のドメインは固定しておく
        for v, ch in initial_assignment.items():
            var_domains[v] = {ch}
    else:
        var_domains, var_occurrences, global_char_set = build_initial_domains(
            slots=slots,
            dict_df=dict_df,
            slot_candidates=slot_candidates,
            norm_grid=norm_grid,
        )

    logger.info("Run solver: Vars=%d, DictSize=%d", len(var_domains), len(dict_df))

    # 4) CSP探索
    # ここではシンプルに max_csp_search (DFS) を使う
    best_state = max_csp_search(
        slots=slots,
        dict_df=dict_df,
        initial_domains=var_domains,
        initial_slot_candidates=slot_candidates,
        var_occurrences=var_occurrences,
        slot_has_dict=slot_has_dict,
        max_nodes=max_nodes,
        frequency_map=frequency_map,
        bigram_probs=bigram_probs,
        initial_assignment=initial_assignment,
    )

    assignment = best_state.assignment
    score = best_state.score if hasattr(best_state, 'score') else 0.0
    
    # 探索終了時のドメイン（もしくは探索で絞り込まれた結果）を返したいが、
    # max_csp_search は state を返すだけなので、
    # ここでは簡易的に「assignmentで確定したもの」をドメインとするか、
    # あるいは再度 recompute して返すのが丁寧。
    # 次のステージのために「まだ候補が複数あるもの」を知りたいので、
    # recompute して返す。
    final_domains, _, _ = recompute_domains_with_assignment(
        slots=slots,
        dict_df=dict_df,
        slot_candidates=slot_candidates,
        norm_grid=norm_grid,
        assignment=assignment,
    )

    return assignment, final_domains, score


def solve(
    df: pd.DataFrame,
    output_path: str = "output.png",
    jukugo_path: str = DEFAULT_JUKUGO_PATH,  # 互換性のため残すが、内部ではステージ制御に従う
    frequency_path: str = "data/merged_dict.csv",
    max_search_nodes: int = MAX_SEARCH_NODES,
    _bert: bool = BERT_ENABLED,
) -> Dict[str, Any]:
    """
    ナンクロを解くメイン関数（3ステージ構成）。
    """
    logger.info("=== solve() START ===")
    logger.info("Grid shape: %s", df.shape)

    # 1) 盤面パース（共通）
    norm_grid = normalize_grid(df)
    slots = extract_slots(norm_grid)
    logger.info("Extracted %d slots.", len(slots))

    # 共通リソースの読み込み（頻度・Bigram）
    # ステージ間で共通して使う
    logger.info("Loading frequency/bigram from %s...", frequency_path)
    freq_map = load_frequency_map(frequency_path)
    from .dictionary.bigram import extract_bigrams
    bigram_probs = extract_bigrams(frequency_path, min_prob=NGRAM_PROB_LOW_LIMIT)

    # 現在の確定割り当て
    current_assignment: Dict[int, str] = {}
    current_domains: Dict[int, Set[str]] = {}

    # --- Stage 1: Nankuro Dictionary ---
    if USE_STAGE1_NANKURO:
        logger.info("--- Stage 1: Nankuro Dictionary ---")
        dict_df_nk = load_dictionary(NANKURO_DICT_PATH)
        logger.info("Nankuro Dict loaded: %d entries.", len(dict_df_nk))

        assign_s1, domains_s1, score_s1 = run_solver_with_dictionary(
            slots=slots,
            dict_df=dict_df_nk,
            norm_grid=norm_grid,
            frequency_map=freq_map, # 頻度はMergedのものを使う
            bigram_probs=bigram_probs,
            initial_assignment={},
            max_nodes=max_search_nodes,
        )
        
        # 確定したものを抽出
        locked_assignment = {}
        for v, ch in assign_s1.items():
            # ドメインサイズが閾値以下なら確定とみなす
            # (assign_s1 にある時点で1つ選ばれているが、ドメインが1になっているか確認)
            dom_size = len(domains_s1.get(v, set()))
            if dom_size <= NANKURO_LOCK_THRESHOLD:
                locked_assignment[v] = ch
        
        logger.info("Stage 1 locked %d vars (threshold=%d).", len(locked_assignment), NANKURO_LOCK_THRESHOLD)
        current_assignment = locked_assignment
        current_domains = domains_s1 # 参考用
    else:
        logger.info("Stage 1 skipped.")

    # --- Stage 2: Merged Dictionary ---
    if USE_STAGE2_MERGED:
        logger.info("--- Stage 2: Merged Dictionary ---")
        dict_df_merged = load_dictionary(MERGED_DICT_PATH)
        logger.info("Merged Dict loaded: %d entries.", len(dict_df_merged))

        # Stage 1 の確定値を initial_assignment として渡す
        assign_s2, domains_s2, score_s2 = run_solver_with_dictionary(
            slots=slots,
            dict_df=dict_df_merged,
            norm_grid=norm_grid,
            frequency_map=freq_map,
            bigram_probs=bigram_probs,
            initial_assignment=current_assignment,
            max_nodes=max_search_nodes,
        )

        # ここでの結果を基本とする
        current_assignment = assign_s2
        current_domains = domains_s2
        
        # 辞書として使うのは Merged Dict (後続の処理用)
        final_dict_df = dict_df_merged
    else:
        logger.info("Stage 2 skipped.")
        # Stage 1 の辞書をそのまま使う場合（Stage 2 オフなら）
        if USE_STAGE1_NANKURO:
             final_dict_df = dict_df_nk # type: ignore
        else:
             # 両方オフはありえないが、フォールバック
             final_dict_df = load_dictionary(jukugo_path)

    # --- Stage 3: LM / BERT Refinement ---
    # ここでは「まだドメインが大きい（候補が絞りきれていない）数字」に対して
    # BERT/LM でのスコアリングを行い、最終決定する
    
    final_assignment = current_assignment
    digit_conf = {}

    if USE_STAGE3_LM and _bert:
        logger.info("--- Stage 3: LM / BERT Refinement ---")
        
        # 未確定（ドメインサイズ > 1）の変数を特定
        unresolved_vars = [v for v, d in current_domains.items() if len(d) > 1]
        logger.info("Unresolved vars: %d", len(unresolved_vars))
        
        # 既存の refine_with_bigram_and_bert を呼び出す
        # ただし、すでに確定しているものは変更しないようにしたいが、
        # refine_... は現状全体を見直す作りになっている。
        # ここでは「Stage 2の結果」を初期値として、全体をリファインする形で呼ぶ。
        # (unresolved だけを対象にするロジックは refine 側で制御するか、
        #  あるいは refine 側が「確信度が高いものは変えない」挙動をすることを期待)
        
        # インデックス再構築（Stage 2 の辞書で）
        len_index, pos_char_index = build_indexes(final_dict_df)
        slot_candidates, _ = build_initial_slot_candidates(slots, final_dict_df, len_index, pos_char_index)

        # var_occurrences を計算（refine_with_bigram_and_bert に渡すため）
        _, var_occurrences, _ = recompute_domains_with_assignment(
            slots, final_dict_df, slot_candidates, norm_grid, current_assignment
        )

        result_tuple = refine_with_bigram_and_bert(
            slots=slots,
            assignment=current_assignment,
            domains=current_domains,
            var_occurrences=var_occurrences,
            bigram_probs=bigram_probs,
            dict_df=final_dict_df,
            slot_candidates=slot_candidates,
            norm_grid=norm_grid,
            used_chars=None,
            return_confidences=True,
        )
        
        if isinstance(result_tuple, tuple):
            final_assignment, digit_conf = result_tuple
        else:
            final_assignment = result_tuple
            digit_conf = {}
    else:
        logger.info("Stage 3 skipped.")


    # ログ出力：数字→漢字マッピング
    logger.info("--- 数字 → 漢字 マッピング（mapping） ---")
    for v, ch in sorted(final_assignment.items()):
        conf = digit_conf.get(v, 1.0)
        logger.info("  #%d -> %s (conf=%.3f)", v, ch, conf)

    # 8) 熟語ごとの信頼度評価
    lm = get_ngram_lm()
    word_confs = evaluate_word_confidences(
        slots=slots,
        assignment=final_assignment,
        dict_df=final_dict_df,
        lm=lm,
    )

    logger.info("--- 熟語ごとの信頼度（上位30件） ---")
    for slot_id, word, conf, in_dict in sorted(word_confs, key=lambda x: x[2]):
        src = "dict" if in_dict else "LM"
        logger.info("  slot_id=%3d word=%s conf=%.3f source=%s", slot_id, word, conf, src)

    # 9) 完成盤面と表示用情報を構築
    result = build_result(
        original_df=df,
        norm_grid=norm_grid,
        assignment=final_assignment,
        slots=slots,
        digit_conf=digit_conf,
    )

    # 重複チェック（All-Different制約の最終確認）
    used_kanjis = {}
    for v, ch in final_assignment.items():
        if ch in used_kanjis:
            logger.warning(
                "[WARNING] Duplicate assignment detected! %s is assigned to #%d and #%d",
                ch, used_kanjis[ch], v
            )
        used_kanjis[ch] = v

    logger.info("=== solve() END ===")
    return result
