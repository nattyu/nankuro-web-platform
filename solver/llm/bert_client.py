# -*- coding: utf-8 -*-
"""
BERT モデルを使って、
「[MASK] を含む語パターン」に対する候補漢字のスコアを計算するモジュールです。

例:
    pattern   = "登[MASK]場"
    candidates = ["山", "場", "校"]

のような入力に対し、各候補漢字が [MASK] に入る自然さをスコアとして返します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForMaskedLM  # type: ignore

from ..config import BERT_MODEL_NAME, BERT_DEVICE
from ..logging_utils import get_logger, get_bert_debug_logger

# 通常ログ（モデルロードなど）
logger = get_logger()
# BERT スコア可視化専用ログ
bert_logger = get_bert_debug_logger()


@dataclass
class BertCandidateScorer:
    """
    BERT を使って候補漢字のスコアを計算するクラスです。

    Attributes
    ----------
    model_name : str
        使用する BERT モデル名。
    device : str
        使用するデバイス（"cpu" や "cuda" など）。
    """

    model_name: str = BERT_MODEL_NAME
    device: str = BERT_DEVICE

    def __post_init__(self) -> None:
        """
        インスタンス生成後に、自動的にモデルとトークナイザをロードします。
        """
        logger.info("[BERT] Loading model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        if self.mask_token is None or self.mask_token_id is None:
            logger.warning(
                "[BERT] mask_token / mask_token_id が取得できません。"
                "スコアリングは 0 として扱われます。"
            )

    @torch.no_grad()
    def score_pattern(self, pattern: str, candidates: List[str]) -> Dict[str, float]:
        """
        1つのパターン文字列に対し、候補漢字それぞれのスコアを計算します。

        Parameters
        ----------
        pattern : str
            例: "登[MASK]場" のような、[MASK] を 1 つ含む文字列。
        candidates : list[str]
            候補漢字のリスト。

        Returns
        -------
        dict[str, float]
            {漢字: スコア} のマップ。スコアが大きいほど自然とみなされます。
        """
        if self.mask_token is None or self.mask_token_id is None:
            bert_logger.warning(
                "[BERT-SCORE] mask_token が None のため、全候補を 0.0 として返します pattern=%s candidates=%s",
                pattern,
                candidates,
            )
            return {c: 0.0 for c in candidates}

        # ユーザー側の "[MASK]" を、実際の BERT のマスクトークンに置き換える
        text = pattern.replace("[MASK]", self.mask_token)

        bert_logger.debug(
            "[BERT-SCORE] 入力パターン='%s' -> 実際の BERT 入力='%s' candidates=%s",
            pattern,
            text,
            candidates,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # マスクトークンの位置を探す
        mask_positions = (inputs["input_ids"][0] == self.mask_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(mask_positions) == 0:
            bert_logger.warning(
                "[BERT-SCORE] マスク位置が見つかりませんでした text='%s'",
                text,
            )
            return {c: 0.0 for c in candidates}
        mask_idx = mask_positions[0].item()

        scores: Dict[str, float] = {}
        for c in candidates:
            tokens = self.tokenizer.tokenize(c)
            if len(tokens) != 1:
                # 1文字1トークンでない場合は扱いづらいので 0 にしておく
                bert_logger.debug(
                    "[BERT-SCORE] 候補 '%s' はトークン数=%d -> 0.0 扱い",
                    c,
                    len(tokens),
                )
                scores[c] = 0.0
                continue
            tid = self.tokenizer.convert_tokens_to_ids(tokens[0])
            score = float(logits[0, mask_idx, tid].item())
            scores[c] = score

        # ★ logit の降順でログ出力
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        bert_logger.debug(
            "[BERT-SCORE] pattern='%s' の候補スコア logits(sorted)=%s",
            pattern,
            sorted_scores,
        )

        return scores

    @torch.no_grad()
    def score_patterns(
        self, patterns: List[str], candidates: List[str]
    ) -> Dict[str, float]:
        """
        複数のパターンについてスコアを合計します。

        1つの数字が複数の語スロットにまたがっている場合、
        それぞれのスロットから得られるパターンを総合して評価します。
        """
        total: Dict[str, float] = {c: 0.0 for c in candidates}
        if not candidates or not patterns:
            bert_logger.debug(
                "[BERT-SCORE] score_patterns 呼び出し: patterns=%s candidates=%s -> 何もしない",
                patterns,
                candidates,
            )
            return total

        bert_logger.info(
            "[BERT-SCORE] 複数パターンでスコア計算開始: patterns=%s candidates=%s",
            patterns,
            candidates,
        )

        for p in patterns:
            s = self.score_pattern(p, candidates)
            for c in candidates:
                total[c] += s.get(c, 0.0)

        # ★ 合計 logit も降順でログ
        sorted_total = sorted(total.items(), key=lambda kv: kv[1], reverse=True)
        bert_logger.info(
            "[BERT-SCORE] パターン合計スコア total_logits(sorted)=%s",
            sorted_total,
        )

        return total


# グローバルに 1 インスタンスだけ持っておき、何度もロードしないようにする
_global_scorer: BertCandidateScorer | None = None


def get_bert_scorer() -> BertCandidateScorer:
    """
    グローバルな BertCandidateScorer インスタンスを返します。
    （初回呼び出し時にモデルをロードします）
    """
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = BertCandidateScorer()
    return _global_scorer
