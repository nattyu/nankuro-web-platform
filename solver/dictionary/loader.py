# -*- coding: utf-8 -*-
"""
熟語辞書（CSV）を読み込むモジュールです。

今回の仕様：
- CSV に必ず 'word' 列がある（例：登山）
- これを text / chars / length に統一変換する

戻り値：
- text   : 熟語文字列
- chars  : 熟語の各漢字のリスト
- length : 文字数
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dictionary(path: str | Path) -> pd.DataFrame:
    """
    熟語辞書 CSV を読み込み、統一フォーマットの DataFrame にして返します。

    Parameters
    ----------
    path : str or Path
        CSV ファイルのパス。

    Returns
    -------
    pandas.DataFrame
        'text', 'chars', 'length' 列を持つ DataFrame。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dictionary CSV not found: {p}")

    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)

    # --- 新仕様: 'word' 列から熟語を読み込む ---
    if "word" not in df.columns:
        raise ValueError("Dictionary CSV must have a 'word' column.")
    
    # dfの重複を削除
    df = df.drop_duplicates(subset=["word"], keep="first")

    # text 列としてコピー
    df["text"] = df["word"].astype(str)

    # chars: 1文字ずつ分解してリスト化
    df["chars"] = df["text"].apply(lambda s: list(str(s)))

    # length 列は現状の方法でOK（文字数）
    df["length"] = df["chars"].apply(len)

    # index を 0 から振り直しておくと扱いやすい
    df = df.reset_index(drop=True)

    return df
