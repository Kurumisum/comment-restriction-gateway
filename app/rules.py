"""
规则检测模块 - 基于关键词列表对文本进行风险检测
Rule-based Detection Module
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import normalize, normalize_with_mapping

# ── 数据目录路径 ─────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"

# ── 词表文件名映射 ───────────────────────────────────────────────────────────
_WORD_LIST_FILES: dict[str, str] = {
    "abuse": "abuse_words.txt",
    "fraud": "fraud_words.txt",
    "sex": "sex_words.txt",
}

# ── 全局词表缓存，避免重复读取磁盘 ─────────────────────────────────────────
_word_lists: dict[str, list[str]] | None = None
# 预计算的规范化关键词缓存：{category: [(original, normalized), ...]}
_normalized_kw_cache: dict[str, list[tuple[str, str]]] | None = None


@dataclass
class RuleHit:
    """单个规则命中结果"""
    category: str           # 风险类别："abuse"、"fraud"、"sex"
    keyword: str            # 命中的关键词
    start: int              # 在原始文本中的起始位置
    end: int                # 在原始文本中的结束位置（不含）
    normalized_hit: bool    # 是否在规范化后文本中命中（True=绕过手段被识破）


def _load_word_list(filepath: Path) -> list[str]:
    """
    从文件加载词表，忽略空行和以 # 开头的注释行，
    按词语长度从长到短排序（优先匹配较长的关键词）
    """
    words: list[str] = []
    if not filepath.exists():
        return words
    with filepath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    # 长词优先，避免短词掩盖长词的命中
    words.sort(key=len, reverse=True)
    return words


def get_word_lists() -> dict[str, list[str]]:
    """获取所有类别的词表（带缓存）"""
    global _word_lists
    if _word_lists is None:
        _word_lists = {}
        for category, filename in _WORD_LIST_FILES.items():
            filepath = _DATA_DIR / filename
            _word_lists[category] = _load_word_list(filepath)
    return _word_lists


def _get_normalized_keywords() -> dict[str, list[tuple[str, str]]]:
    """
    获取预规范化的关键词对缓存。
    每个条目为 (original_keyword, normalized_keyword)。
    在首次调用时计算并缓存，避免每次请求重复规范化。
    """
    global _normalized_kw_cache
    if _normalized_kw_cache is None:
        _normalized_kw_cache = {}
        word_lists = get_word_lists()
        for category, keywords in word_lists.items():
            pairs: list[tuple[str, str]] = []
            for kw in keywords:
                pairs.append((kw, normalize(kw)))
            _normalized_kw_cache[category] = pairs
    return _normalized_kw_cache


def reload_word_lists() -> None:
    """强制重新加载词表（用于热更新场景）"""
    global _word_lists, _normalized_kw_cache
    _word_lists = None
    _normalized_kw_cache = None
    get_word_lists()


def _find_all_occurrences(text: str, keyword: str) -> list[tuple[int, int]]:
    """
    在 text 中找出所有 keyword 的出现位置，返回 (start, end) 列表。
    使用非重叠匹配。
    """
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(keyword, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(keyword)))
        start = idx + len(keyword)  # 非重叠：从命中末尾继续
    return spans


def _map_normalized_span_to_original(
    norm_start: int,
    norm_end: int,
    pos_map: list[int],
    original_len: int,
) -> tuple[int, int]:
    """
    将规范化文本中的位置区间映射回原始文本位置区间。

    pos_map[i] 记录规范化文本第 i 个字符对应原始文本的字符索引。
    映射规则：
    - orig_start = pos_map[norm_start]
    - orig_end   = pos_map[norm_end - 1] + 1
    """
    if not pos_map or norm_start >= len(pos_map):
        return (0, original_len)

    orig_start = pos_map[norm_start]
    last_idx = min(norm_end - 1, len(pos_map) - 1)
    orig_end = pos_map[last_idx] + 1
    return (orig_start, orig_end)


def check_rules(text: str) -> list[RuleHit]:
    """
    对文本进行规则检测，返回所有命中结果。

    检测策略：
    1. 先在原始文本中直接匹配关键词（直接命中）
    2. 再在规范化文本中匹配规范化后的关键词，将位置映射回原始文本（绕过命中）
    已在原始文本中命中的位置不会重复报告。
    """
    if not text or not text.strip():
        return []

    # 使用预规范化的关键词缓存，避免每次请求重复计算
    norm_kw_map = _get_normalized_keywords()
    hits: list[RuleHit] = []

    # 规范化文本及位置映射
    normalized_text, pos_map = normalize_with_mapping(text)
    normalized_lower = normalized_text.lower()
    original_lower = text.lower()

    for category, kw_pairs in norm_kw_map.items():
        # 记录已命中的原始文本区间，用于去重
        covered_spans: list[tuple[int, int]] = []

        for keyword, kw_normalized in kw_pairs:
            kw_lower = keyword.lower()
            kw_normalized_lower = kw_normalized.lower()

            # ── 策略一：在原始文本中直接匹配 ──
            for start, end in _find_all_occurrences(original_lower, kw_lower):
                # 检查是否与已命中区间重叠
                if not _overlaps(start, end, covered_spans):
                    hits.append(
                        RuleHit(
                            category=category,
                            keyword=keyword,
                            start=start,
                            end=end,
                            normalized_hit=False,
                        )
                    )
                    covered_spans.append((start, end))

            # ── 策略二：在规范化文本中匹配（使用规范化后的关键词） ──
            for norm_start, norm_end in _find_all_occurrences(normalized_lower, kw_normalized_lower):
                orig_start, orig_end = _map_normalized_span_to_original(
                    norm_start, norm_end, pos_map, len(text)
                )
                # 只报告原始文本中未被直接命中的区间
                if not _overlaps(orig_start, orig_end, covered_spans):
                    hits.append(
                        RuleHit(
                            category=category,
                            keyword=keyword,
                            start=orig_start,
                            end=orig_end,
                            normalized_hit=True,
                        )
                    )
                    covered_spans.append((orig_start, orig_end))

    return hits


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    """判断区间 [start, end) 是否与已有区间列表中的任何区间重叠"""
    for s, e in spans:
        if start < e and end > s:
            return True
    return False


def summarize_hits(hits: list[RuleHit]) -> dict[str, list[str]]:
    """
    汇总命中结果，返回每个类别的命中关键词列表（去重），
    便于上层模块快速获取风险摘要。
    """
    summary: dict[str, list[str]] = {}
    for hit in hits:
        if hit.category not in summary:
            summary[hit.category] = []
        if hit.keyword not in summary[hit.category]:
            summary[hit.category].append(hit.keyword)
    return summary
