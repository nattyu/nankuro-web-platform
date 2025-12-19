# scoring.py
# -*- coding: utf-8 -*-
"""
CSP 状態のスコアリングを行うモジュール。

スコアの内訳
-----------
1. 辞書スコア
   - 各スロットに対して、候補熟語（slot_candidates）が 1 つ以上残っていれば +1.0
   - 候補ゼロなら -0.5 のペナルティ（Max-CSP 的な「満たせていない制約」）

2. n-gram LM スコア（任意）
   - assignment から完全に決まるスロットの熟語を取り出す
   - そのスロットの辞書候補が空（= 辞書に無いとみなす）場合のみ、
     文字 n-gram LM で log p(w) を計算
   - その平均 log-prob を NGRAM_LM_WEIGHT 倍して加算

3. 割り当て数ボーナス
   - 0.01 * len(assignment)
   - （同じ制約充足度なら、より多くの数字を決めている状態を優先）

注意
----
- n-gram LM は config.NGRAM_LM_JSON_PATH が存在し、かつ
  NGRAM_LM_WEIGHT > 0 のときのみ使用されます。
- LM ファイルが見つからない場合や JSON が壊れている場合は、
  辞書スコア＋割り当て数ボーナスだけでスコアリングします。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
import json
import math

import pandas as pd

from ..types import Slot
from ..grid.parser import is_kanji
from ..csp.domains import extract_var_id
from ..logging_utils import get_logger

logger = get_logger()

# ----------------------------------------------------------------------
# コンフィグ値の読み込み（config に無ければデフォルト値）
# ----------------------------------------------------------------------
try:
    from ..config import (
        NGRAM_LM_JSON_PATH,
        NGRAM_LM_WEIGHT,
        DICT_POSITIVE_SLOT_SCORE,
        DICT_NEGATIVE_SLOT_SCORE,
        # ここから追加
        SINGLE_CANDIDATE_BONUS, # type: ignore
        FEW_CANDIDATES_BONUS,# type: ignore
        MANY_CANDIDATES_PENALTY,# type: ignore
        FULL_MATCH_BONUS,# type: ignore
        COMPLETION_BONUS,# type: ignore
        KANJI_REUSE_BONUS,# type: ignore
    )
except Exception:
    NGRAM_LM_JSON_PATH = None          # LM を使わない
    NGRAM_LM_WEIGHT = 0.3              # 0 なら無効
    DICT_POSITIVE_SLOT_SCORE = 1.0
    DICT_NEGATIVE_SLOT_SCORE = -0.5

    # デフォルト値（お好みで調整してください）
    SINGLE_CANDIDATE_BONUS = 0.6       # 候補1個のときの追加ボーナス
    FEW_CANDIDATES_BONUS = 0.3         # 候補2〜5個のときの追加ボーナス
    MANY_CANDIDATES_PENALTY = -0.2     # 候補が多すぎるときのペナルティ
    FULL_MATCH_BONUS = 1.5             # 完成熟語が辞書にピッタリ一致したとき
    COMPLETION_BONUS = 0.5             # completion_ratio に掛ける係数
    KANJI_REUSE_BONUS = 0.1            # 同じ漢字が複数スロットで活躍するとき



# ----------------------------------------------------------------------
# n-gram LM 実装
# ----------------------------------------------------------------------


@dataclass
class NGramLanguageModel:
    n: int
    log_probs: Dict[str, float]
    unk_log_prob: float
    bos: str = "<s>"
    eos: str = "</s>"

    @classmethod
    def from_json(cls, data: Dict) -> "NGramLanguageModel":
        return cls(
            n=int(data["n"]),
            log_probs=dict(data["log_probs"]),
            unk_log_prob=float(data["unk_log_prob"]),
            bos=data.get("bos", "<s>"),
            eos=data.get("eos", "</s>"),
        )

    def _tokens(self, text: str) -> List[str]:
        """文字列 → BOS/EOS 付きトークン列"""
        chars = list(text)
        return [self.bos] * (self.n - 1) + chars + [self.eos]

    def score(self, text: str) -> float:
        """
        1 つの熟語に対するスコア（平均 log-prob）を返す。
        値が大きいほど自然な語とみなす。
        """
        tokens = self._tokens(text)
        if len(tokens) < self.n:
            return 0.0

        logs: List[float] = []
        for i in range(len(tokens) - self.n + 1):
            ng = tokens[i : i + self.n]
            key = " ".join(ng)
            logs.append(self.log_probs.get(key, self.unk_log_prob))

        if not logs:
            return 0.0
        # 長さに依存しないよう、平均 log-p にしておく
        return sum(logs) / len(logs)

    def word_confidence(self, text: str) -> float:
        """
        熟語の信頼度を 0.0〜1.0 で返す。
        簡易的に、平均 log-prob をシグモイド風に変換するか、
        あるいは単に確率空間での相対値っぽく返す。
        ここではユーザー要望に合わせて 0.73 くらいが出るような
        ヒューリスティックな変換を行う。
        """
        s = self.score(text)
        # s は log probability (例: -5.0 とか -15.0 とか)
        # これを 0~1 に潰す。
        # 経験的に、良い語は -5 ~ -8 くらい、悪い語は -15 以下とかになる。
        # -10 を中心にシグモイド関数にかけるイメージで実装。
        
        # log_p = -10 -> 0.5 になるように調整
        # log_p = -5  -> ほぼ 1.0
        # log_p = -20 -> ほぼ 0.0
        
        import math
        # sigmoid(x) = 1 / (1 + exp(-x))
        # x = (s - center) * scale
        center = -12.0
        scale = 0.5
        
        x = (s - center) * scale
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0


# lazy load 用のグローバル
_NGRAM_LM: Optional[NGramLanguageModel] = None
_NGRAM_LM_LOADED = False


def get_ngram_lm() -> Optional[NGramLanguageModel]:
    """
    n-gram LM を lazy load して返す。
    利用不可/失敗時は None。
    """
    global _NGRAM_LM, _NGRAM_LM_LOADED

    # すでに試行済み
    if _NGRAM_LM_LOADED:
        return _NGRAM_LM

    _NGRAM_LM_LOADED = True  # この関数を複数回呼んでも 1 回だけ試す

    if not NGRAM_LM_JSON_PATH or NGRAM_LM_WEIGHT <= 0.0:
        logger.info("[scoring] N-gram LM disabled (no path or weight=0).")
        _NGRAM_LM = None
        return None

    path = Path(NGRAM_LM_JSON_PATH)
    if not path.is_file():
        logger.warning("[scoring] N-gram LM file not found: %s", path)
        _NGRAM_LM = None
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        _NGRAM_LM = NGramLanguageModel.from_json(data)
        logger.info(
            "[scoring] N-gram LM loaded: n=%d, |ngrams|=%d",
            _NGRAM_LM.n,
            len(_NGRAM_LM.log_probs),
        )
    except Exception as e:
        logger.exception("[scoring] Failed to load N-gram LM: %s", e)
        _NGRAM_LM = None

    return _NGRAM_LM


# ----------------------------------------------------------------------
# スロットから「現在確定している熟語」を取り出すヘルパー
# ----------------------------------------------------------------------


def build_slot_word(slot: Slot, assignment: Dict[int, str]) -> Optional[str]:
    """
    slot.pattern と assignment から、熟語が完全に決まっている場合は文字列を返す。
    まだ未確定の数字が含まれている場合は None。
    """
    chars: List[str] = []

    for pat in slot.pattern:
        vid = extract_var_id(pat)
        if vid is None:
            # 固定漢字ならそのまま、それ以外（記号など）は「未確定扱い」
            if is_kanji(pat):
                chars.append(pat)
            else:
                return None
        else:
            # 数字マス
            if vid not in assignment:
                return None
            chars.append(assignment[vid])

    if not chars:
        return None
    return "".join(chars)


# ----------------------------------------------------------------------
# メインのスコアリング関数
# ----------------------------------------------------------------------


def evaluate_state_score(
    slots: List[Slot],
    dict_df: pd.DataFrame,
    slot_candidates: Dict[int, Set[int]],
    assignment: Dict[int, str],
    frequency_map: Optional[Dict[str, float]] = None,
) -> float:
    """
    現在の CSP 状態に対してスコアを計算する。

    スコアの内訳:
    1. 辞書スコア（各スロットに対して）
       - 候補が残っている:
         - 基本点: DICT_POSITIVE_SLOT_SCORE
         - 候補数が少ないほどボーナス（1個/2〜5個など）
         - completion_ratio（割り当て完了度）に応じたボーナス
         - 完全一致する辞書語があれば FULL_MATCH_BONUS
         - 頻度に基づくボーナス（frequency_map）
       - 候補が全て消えた:
         - DICT_NEGATIVE_SLOT_SCORE
         - ただし n-gram LM が使える場合は、そのスコアで少し緩和
    2. N-gram LM スコア（辞書候補が無い場合のフォールバック）
    3. 割り当て数ボーナス: 0.01 * len(assignment)
    4. 漢字再利用ボーナス:
       - 同じ漢字が複数スロットで「整合的なスロット」に使われていれば
         KANJI_REUSE_BONUS を加算
    """
    score = 0.0
    lm = get_ngram_lm()

    # 「良いスロット」で使われた漢字の出現回数（再利用ボーナス用）
    kanji_good_usage: Dict[str, int] = {}

    for slot in slots:
        sid = slot.slot_id
        cands = slot_candidates.get(sid, set())

        # 現在の assignment から、このスロットが完全に決まっているなら文字列を作る
        slot_word = build_slot_word(slot, assignment)

        # -------- 1) 候補がゼロのスロットの処理 --------
        if not cands:
            # N-gram LM でフォールバック評価（辞書に無い熟語用）
            if lm is not None and NGRAM_LM_WEIGHT > 0.0 and slot_word:
                w_score = lm.score(slot_word)
                score += NGRAM_LM_WEIGHT * w_score
            else:
                score += DICT_NEGATIVE_SLOT_SCORE
            continue

        # -------- 2) 割り当て完了度（completion_ratio） --------
        assigned_count = 0
        total_vars = 0
        for pat in slot.pattern:
            vid = extract_var_id(pat)
            if vid is not None:
                total_vars += 1
                if vid in assignment:
                    assigned_count += 1

        completion_ratio = assigned_count / total_vars if total_vars > 0 else 1.0

        # -------- 3) 現在の assignment と整合的な候補語を数える --------
        ok_exists = False
        num_valid = 0
        max_freq_score = 0.0
        has_exact_match = False  # 完全一致する辞書語があるか

        for eid in cands:
            chars = str(dict_df.at[eid, "word"])
            ok = True
            for pos, pat in enumerate(slot.pattern):
                vid = extract_var_id(pat)
                if vid is not None and vid in assignment:
                    if pos >= len(chars) or chars[pos] != assignment[vid]:
                        ok = False
                        break
            if not ok:
                continue

            ok_exists = True
            num_valid += 1

            # 完全一致チェック
            if slot_word and chars == slot_word:
                has_exact_match = True

            # 頻度ボーナス（部分割り当てにも緩く効かせる）
            if frequency_map:
                text = str(dict_df.at[eid, "text"])
                freq = frequency_map.get(text, 0.0)
                freq_score = min(freq / 7.0, 1.0)
                freq_score *= completion_ratio
                if freq_score > max_freq_score:
                    max_freq_score = freq_score

        if ok_exists:
            # ---- 3-1) 辞書スロットの基本点 ----
            slot_score = DICT_POSITIVE_SLOT_SCORE

            # ---- 3-2) 候補数に応じたボーナス／ペナルティ ----
            if num_valid == 1:
                slot_score += SINGLE_CANDIDATE_BONUS
            elif 2 <= num_valid <= 5:
                slot_score += FEW_CANDIDATES_BONUS
            elif num_valid > 20:
                slot_score += MANY_CANDIDATES_PENALTY

            # ---- 3-3) 割り当て完了度ボーナス ----
            slot_score += COMPLETION_BONUS * completion_ratio

            # ---- 3-4) 頻度ボーナス ----
            slot_score += max_freq_score

            # ---- 3-5) 完全一致ボーナス ----
            if has_exact_match:
                slot_score += FULL_MATCH_BONUS

            score += slot_score

            # 「良いスロット」で使われている漢字を記録（再利用ボーナス用）
            slot_used_chars = set()
            for pat in slot.pattern:
                vid = extract_var_id(pat)
                if vid is not None and vid in assignment:
                    slot_used_chars.add(assignment[vid])

            for ch in slot_used_chars:
                kanji_good_usage[ch] = kanji_good_usage.get(ch, 0) + 1

        else:
            # もともと候補があったのに、現在の割り当てでは
            # すべて消えてしまった → 減点
            score += DICT_NEGATIVE_SLOT_SCORE

    # -------- 4) 漢字再利用ボーナス --------
    for ch, cnt in kanji_good_usage.items():
        if cnt > 1:
            # 1スロット目は基準、2スロット目以降にボーナス
            score += KANJI_REUSE_BONUS * (cnt - 1)

    # -------- 5) 割り当て数ボーナス --------
    score += 0.01 * len(assignment)

    return score


