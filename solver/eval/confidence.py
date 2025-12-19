# solver/eval/confidence.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from ..types import Slot
from ..grid.parser import is_kanji
from ..csp.domains import extract_var_id
from ..csp.scoring import NGramLanguageModel


def build_word_from_slot(slot: Slot, assignment: Dict[int, str]) -> str | None:
    """
    Slot.pattern と assignment（数字→漢字）から、完成した熟語を生成する。
    □ や ■ が混ざる場合は None を返す。
    """
    chars: List[str] = []
    for pat in slot.pattern:
        vid = extract_var_id(pat)
        if vid is None:
            # 固定漢字ならそのまま、それ以外（■ や # など）は熟語ではないとみなして None
            if is_kanji(pat):
                chars.append(pat)
            else:
                return None
        else:
            ch = assignment.get(vid)
            if ch is None:
                return None
            chars.append(ch)
    if not chars:
        return None
    return "".join(chars)


def evaluate_word_confidences(
    slots: List[Slot],
    assignment: Dict[int, str],
    dict_df: pd.DataFrame,
    lm: NGramLanguageModel | None,
) -> List[Tuple[int, str, float, bool]]:
    """
    各スロットについて、熟語とその信頼度を計算する。

    Returns
    -------
    list of (slot_id, word, confidence, in_dict)
    """
    # dict_df の "chars" 列に完成熟語が入っている前提
    dict_words = set(dict_df["word"].astype(str).tolist())

    results: List[Tuple[int, str, float, bool]] = []

    for slot in slots:
        word = build_word_from_slot(slot, assignment)
        if not word:
            continue

        in_dict = word in dict_words
        if in_dict:
            conf = 1.0
        elif lm is not None:
            conf = lm.word_confidence(word)
        else:
            conf = 0.5  # LM が無い場合はとりあえず中間

        results.append((slot.slot_id, word, conf, in_dict))

    return results
