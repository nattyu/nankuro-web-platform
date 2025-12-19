# -*- coding: utf-8 -*-
"""
ログ出力の設定を行うモジュールです。

初学者向けポイント:
- 「ログ」とは、プログラムの実行状況を記録するメッセージのことです。
- 開発中やデバッグ時に「どこまで処理が進んだか」「何が起きたか」を
  確認するのに役立ちます。
"""

from __future__ import annotations

import logging
import os

# solver パッケージ共通で使うロガー名
LOGGER_NAME = "solver"


def get_logger() -> logging.Logger:
    """
    solver 全体で共通して使う logger を返します。

    すでに handler（出力先）が設定されていない場合は、
    標準出力（コンソール）に INFO レベルのログを表示するように設定します。
    """
    logger = logging.getLogger(LOGGER_NAME)

    # まだハンドラが設定されていなければ、簡単な設定を行う
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

def get_bert_debug_logger():
    logger = logging.getLogger("bert_debug")

    if logger.handlers:
        return logger  # すでに初期化済み

    logger.setLevel(logging.DEBUG)

    # ログファイル保存場所
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "bert_debug.log")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    # 他ロガーへの伝播禁止（stdout に出さない）
    logger.propagate = False

    return logger
