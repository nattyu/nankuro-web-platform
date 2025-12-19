# -*- coding: utf-8 -*-
"""
BERT + bigram によって数字（記号）ごとの候補漢字を絞り込むモジュールです。

改善内容：
- 辞書整合性 (recompute_domains_with_assignment)
- bigram フィルタ (filter_domain_by_bigram)
- BERT スコアリング
- bigram/BERT 統合 (0.7 : 0.3)
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

from ..types import Slot
from ..logging_utils import get_logger, get_bert_debug_logger
from ..grid.parser import is_kanji
from ..csp.domains import extract_var_id, recompute_domains_with_assignment
from .bert_client import get_bert_scorer
from ..dictionary.bigram import filter_domain_by_bigram

logger = get_logger()
bert_logger = get_bert_debug_logger()

# ================================================================
# パターン生成
# ================================================================
def build_patterns_for_var(var, slots, var_occurrences, assignment):
    pats = []
    occs = var_occurrences.get(var, [])
    for slot_id, _pos in occs:
        slot = next((s for s in slots if s.slot_id == slot_id), None)
        if slot is None:
            continue
        elems = []
        for pat in slot.pattern:
            vid = extract_var_id(pat)
            if vid is None:
                elems.append(pat if is_kanji(pat) else "□")
            else:
                elems.append("[MASK]" if vid == var else assignment.get(vid, "□"))
        pats.append("".join(elems))
    return list(dict.fromkeys(pats))


# ================================================================
# bigram スコア
# ================================================================
def score_bigram(var, candidate, slots, var_occurrences, assignment, bigram_probs):
    slot_map = {slot.slot_id: slot for slot in slots}
    total, count = 0.0, 0
    for slot_id, pos in var_occurrences.get(var, []):
        slot = slot_map.get(slot_id)
        if slot is None:
            continue
        pattern = slot.pattern
        # 左文脈
        if pos > 0:
            left = pattern[pos - 1]
            lv = extract_var_id(left)
            lc = assignment.get(lv) if lv else (left if is_kanji(left) else None)
            if lc and candidate in bigram_probs.get(lc, {}):
                total += math.log(bigram_probs[lc][candidate])
                count += 1
        # 右文脈
        if pos < len(pattern) - 1:
            right = pattern[pos + 1]
            rv = extract_var_id(right)
            rc = assignment.get(rv) if rv else (right if is_kanji(right) else None)
            if rc and rc in bigram_probs.get(candidate, {}):
                total += math.log(bigram_probs[candidate][rc])
                count += 1
    return total / count if count else 0.0


def z_normalize(d):
    if not d:
        return {}
    vals = list(d.values())
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    s = math.sqrt(v) if v > 0 else 1e-9
    return {k: (v - m) / s for k, v in d.items()}

# ================================================================
# bigram + BERT refine 本体（完全統合版）
# ================================================================
def refine_with_bigram_and_bert(
    slots,
    assignment,
    domains,
    var_occurrences,
    bigram_probs,
    dict_df,
    slot_candidates,
    norm_grid,
    used_chars=None,
    return_confidences=False,
):
    scorer = get_bert_scorer()
    new_assignment = dict(assignment)
    # used の初期化：現在の assignment に含まれる漢字を登録
    used = dict(used_chars) if used_chars else {}
    for ch in new_assignment.values():
        used[ch] = used.get(ch, 0) + 1
    
    bert_conf = {}

    # ============================================================
    # 1) singleton を確定 → 辞書整合性で domain 再計算
    # ============================================================
    changed = True
    while changed:
        changed = False
        for v, dom in list(domains.items()):
            if len(dom) == 1 and v not in new_assignment:
                ch = next(iter(dom))
                # すでに使われている漢字ならスキップ（矛盾だが、ここでは割り当てない）
                if used.get(ch, 0) > 0:
                    continue
                
                new_assignment[v] = ch
                used[ch] = used.get(ch, 0) + 1
                changed = True

        if changed:
            domains, var_occurrences, _ = recompute_domains_with_assignment(
                slots, dict_df, slot_candidates, norm_grid, new_assignment
            )

    # ============================================================
    # 2) ドメイン小さい順で処理
    # ============================================================
    all_vars = sorted(domains.keys(), key=lambda x: len(domains[x]))

    BAD_VARS = {5, 6, 9, 11, 13, 14, 15, 17, 18, 19, 22, 24, 27}

    for var in all_vars:
        dom = domains.get(var, set())

        if var in BAD_VARS:
            bert_logger.info(
                "[DEBUG-DOM] var=%d 初期ドメインサイズ=%d 内容=%s",
                var,
                len(dom),
                "".join(sorted(dom)),
            )

        # 一意なら確定
        if var in new_assignment:
            # すでに割り当て済みならスキップ
            bert_conf[var] = 1.0
            continue

        if not dom:
            continue

        # unused を優先（というか used は除外）
        candidates = [c for c in dom if used.get(c, 0) == 0]
        
        if not candidates:
            # 候補がすべて使用済みなら、この変数は割り当て不能
            bert_logger.warning("[REFINE] var=%d: All candidates used. Skipping.", var)
            continue

        # ========================================================
        # 2-1) 辞書整合性で domain は最新
        # ========================================================

        # ========================================================
        # 2-2) ★ bigram フィルタを使ってさらに削る
        # ========================================================
        filtered_dom = filter_domain_by_bigram(
            var=var,
            domain=set(candidates),
            assignment=new_assignment,
            var_occurrences=var_occurrences,
            bigram_probs=bigram_probs,
            slots=slots,
        )
        if var in BAD_VARS:
            bert_logger.info(
                "[DEBUG-DOM] var=%d bigram後ドメインサイズ=%d 内容=%s",
                var,
                len(filtered_dom),
                "".join(sorted(filtered_dom)),
            )
        if len(filtered_dom) > 0:
            candidates = list(filtered_dom)

        # ========================================================
        # 2-3) パターン生成（□なしだけ BERT に使う）
        # ========================================================
        raw_patterns = build_patterns_for_var(var, slots, var_occurrences, new_assignment)
        bert_patterns = [p for p in raw_patterns if "□" not in p]

        # ========================================================
        # 2-4) bigram スコア
        # ========================================================
        ngram_scores = {
            c: score_bigram(var, c, slots, var_occurrences, new_assignment, bigram_probs)
            for c in candidates
        }

        # ========================================================
        # 2-5) BERT スコア
        # ========================================================
        if bert_patterns:
            bert_scores = scorer.score_patterns(bert_patterns, candidates)
        else:
            bert_scores = {c: 0.0 for c in candidates}

        # ========================================================
        # 2-6) 統合スコア
        # ========================================================
        zn = z_normalize(ngram_scores)
        zb = z_normalize(bert_scores)
        combined = {c: 0.7 * zn[c] + 0.3 * zb[c] for c in candidates}

        best = max(combined, key=lambda c: combined[c])
        new_assignment[var] = best
        used[best] = used.get(best, 0) + 1

        # softmax confidence
        logits = list(combined.values())
        maxlog = max(logits)
        exps = [math.exp(x - maxlog) for x in logits]
        conf = exps[candidates.index(best)] / sum(exps)
        bert_conf[var] = conf

        # ========================================================
        # 2-7) ドメイン更新 → 辞書整合性で再計算（超重要）
        # ========================================================
        domains[var] = {best}
        for ov, od in domains.items():
            if ov != var and best in od:
                od.discard(best)

        # 🔥再計算（辞書整合性チェック）
        domains, var_occurrences, _ = recompute_domains_with_assignment(
            slots, dict_df, slot_candidates, norm_grid, new_assignment
        )

    if return_confidences:
        return new_assignment, bert_conf
    return new_assignment
