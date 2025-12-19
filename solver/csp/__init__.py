# -*- coding: utf-8 -*-
"""
solver.csp パッケージ

Max-CSP（制約充足 + スコア最大化）に関する処理をまとめています。

主に以下の役割を持つモジュールから構成されています。
- slots_candidates.py : スロットごとの辞書候補集合の初期計算
- domains.py          : 数字のドメイン（候補漢字集合）の初期計算
- propagation.py      : 制約伝播（ドメインの絞り込み）
- scoring.py          : 状態のスコアリング
- search.py           : 深さ優先探索による Max-CSP 解の探索
"""
