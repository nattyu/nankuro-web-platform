# -*- coding: utf-8 -*-
"""
solver 全体で共通して使う設定値をまとめたモジュールです。

実運用時には、ここを編集することで
- 辞書ファイルの場所
- 探索の深さ（ノード数上限）
- スロットごとの辞書候補上限
- BERT モデル名
などを簡単に変更できます。
"""

from __future__ import annotations

from pathlib import Path

# ==== 辞書ファイル関連 =====================================================

# 熟語辞書 CSV のパス
# 例: data/jukugo.csv
DEFAULT_JUKUGO_PATH: str = "data/merged_dict.csv"
NANKURO_DICT_PATH: str = "data/nankuro_dictionary.csv"
MERGED_DICT_PATH: str = "data/merged_dict.csv"

# ==== ステージ設定 =========================================================
# ステージ境界の閾値（ドメインサイズがこれ以下なら確定とみなす）
NANKURO_LOCK_THRESHOLD: int = 1

# 各ステージの有効/無効
USE_STAGE1_NANKURO: bool = True
USE_STAGE2_MERGED: bool = True
USE_STAGE3_LM: bool = True

# ==== N-gram LM 関連 =======================================================

# 文字 N-gram LM の JSON パス
NGRAM_LM_JSON_PATH: str = "data/ngram_lm.json"

# N-gram LM スコアの重み (0.0 で無効化)
NGRAM_LM_WEIGHT: float = 0.3

# N-gram モデルの下限確率
NGRAM_PROB_LOW_LIMIT: float = 0.10

# 辞書候補があるスロットのベーススコア
DICT_POSITIVE_SLOT_SCORE: float = 1.0

# 辞書候補がないスロットのペナルティ
DICT_NEGATIVE_SLOT_SCORE: float = -0.5

# ---- スロットごとの詳細スコア調整 ----------------------------------------
# 候補が 1 個だけのスロットに与えるボーナス
SINGLE_CANDIDATE_BONUS: float = 0.6

# 候補が 2〜5 個のスロットに与えるボーナス
FEW_CANDIDATES_BONUS: float = 0.3

# 候補が 20 個より多いスロットへのペナルティ
MANY_CANDIDATES_PENALTY: float = -0.4

# 完全に割り当てが決まり、その熟語が辞書にピッタリ一致した場合のボーナス
FULL_MATCH_BONUS: float = 1.5

# completion_ratio（そのスロット内でどれだけ数字が埋まっているか）に掛ける係数
# 例: completion_ratio=0.8 なら 0.8 * COMPLETION_BONUS が加点
COMPLETION_BONUS: float = 0.8

# 同じ漢字が複数の「良さげなスロット」で再利用されているときのボーナス
# 2回目以降の出現ごとに KANJI_REUSE_BONUS を加算
KANJI_REUSE_BONUS: float = 0.1


# ==== 探索関連 =============================================================

# Max-CSP の探索で、何ノードまで探索するかの上限。
# 大きくすると精度は上がりやすいが、時間も増えます。
MAX_SEARCH_NODES: int = 10000

# 1つのスロットに対して保持する「辞書候補語ID」の最大数。
# これを超える場合、そのスロットは「辞書候補なし」とみなします。
# （候補が多すぎるスロットは計算量爆発の原因になるため、
# 　一旦辞書には頼らず、別のスロットや BERT に任せる方針）
MAX_CANDIDATES_PER_SLOT: int = 200

# ==== BERT / LLM 関連 ======================================================

# 使用する日本語 BERT モデル名（Hugging Face のモデル名）
BERT_MODEL_NAME: str = "cl-tohoku/bert-base-japanese-v2"

# BERT を動かすデバイス名。
# CPU の場合は "cpu"、GPU が使える場合は "cuda" などを指定します。
BERT_DEVICE: str = "cpu"

# 1つの数字（記号）について、
# BERT に投げる「語パターン（スロット）」の最大数。
BERT_MAX_PATTERNS_PER_SYMBOL: int = 20

# BERT を使った最終調整を行うかどうか。
BERT_ENABLED: bool = True

# ==== Simulated Annealing 関連 =============================================

# 探索アルゴリズムの選択: "dfs", "sa", "ga"
SEARCH_ALGORITHM: str = "dfs"

# Simulated Annealing の初期温度
SA_INITIAL_TEMP: float = 100.0

# Simulated Annealing の冷却率
SA_COOLING_RATE: float = 0.995

# Simulated Annealing の終了温度
SA_MIN_TEMP: float = 0.1

# Simulated Annealing の最大反復回数
SA_MAX_ITERATIONS: int = 50000

# ==== Genetic Algorithm 関連 ============================================

# Genetic Algorithm の個体数
GA_POPULATION_SIZE: int = 30

# Genetic Algorithm の世代数
GA_GENERATIONS: int = 100

# Genetic Algorithm の突然変異率
GA_MUTATION_RATE: float = 0.15

# Genetic Algorithm の交叉率
GA_CROSSOVER_RATE: float = 0.8

# Genetic Algorithm のエリート個体数（次世代に必ず残す）
GA_ELITE_SIZE: int = 2

# Genetic Algorithm の初期個体生成時のDFSノード数
GA_INITIAL_DFS_NODES: int = 300


