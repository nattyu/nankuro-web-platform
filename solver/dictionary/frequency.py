# -*- coding: utf-8 -*-
"""
単語の出現頻度データを読み込むモジュールです。
merged_dict.csv / wiki_compounds.csv のどちらにも対応。
"""

import pandas as pd
from typing import Dict
from math import log10
from solver.logging_utils import get_logger

logger = get_logger()


def load_frequency_map(path: str) -> Dict[str, float]:
    """
    単語頻度データを読み込み、対数変換したスコアのマップを返します。

    対応フォーマット:
    - wiki_compounds.csv : word, count
    - merged_dict.csv    : word, freq_total, freq_jukugo, freq_all_words, ...

    Parameters
    ----------
    path : str
        CSVファイルのパス。

    Returns
    -------
    Dict[str, float]
        単語 -> 頻度スコア (log10)
    """
    logger.info(f"Loading frequency data from {path}...")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")

        # ---------- 優先的に使う頻度列を判断 ----------
        freq_col = None

        if "freq_total" in df.columns:
            freq_col = "freq_total"   # merged_dict 用
            logger.info("Using 'freq_total' as frequency source.")

        elif "count" in df.columns:
            freq_col = "count"        # wiki_compounds 用
            logger.info("Using 'count' as frequency source.")

        elif "freq_jukugo" in df.columns and "freq_all_words" in df.columns:
            # freq_total が無い場合は合算できる
            logger.info("Using 'freq_jukugo + freq_all_words' as frequency source.")
            df["freq_total"] = df["freq_jukugo"].fillna(0) + df["freq_all_words"].fillna(0)
            freq_col = "freq_total"

        else:
            logger.warning(
                "No valid frequency column found (expected freq_total, count, or freq_jukugo+freq_all_words)."
            )
            return {}

        # ---------- スコア計算 ----------
        freq_map = {}
        for _, row in df.iterrows():
            word = str(row["word"])

            try:
                val = float(row[freq_col])
            except Exception:
                continue

            if val > 0:
                freq_map[word] = log10(val)
            # 0 の場合はスコアなし（入れない）

        logger.info(f"Loaded frequency data for {len(freq_map)} words.")
        return freq_map

    except Exception as e:
        logger.error(f"Failed to load frequency data: {e}")
        return {}
