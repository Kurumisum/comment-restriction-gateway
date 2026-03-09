"""
证据片段提取模块 - 从规则命中和 LLM 判定中提取精确的证据位置
Evidence Span Extraction Module
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvidenceSpan:
    """单个证据片段"""
    start: int      # 在原始文本中的起始字符索引（含）
    end: int        # 在原始文本中的结束字符索引（不含）
    text: str       # 证据文本内容
    source: str     # 证据来源："rule"（规则引擎）或 "llm"（LLM 判定）
    category: str   # 风险类别："abuse"、"fraud"、"sex"


def _find_text_position(original: str, fragment: str) -> tuple[int, int] | None:
    """
    在原始文本中查找 fragment 的位置，返回 (start, end) 或 None。
    支持部分匹配：若 fragment 较长，尝试在 original 中查找前缀片段。
    """
    if not fragment:
        return None

    # 直接查找
    idx = original.find(fragment)
    if idx != -1:
        return (idx, idx + len(fragment))

    # 忽略大小写查找
    idx = original.lower().find(fragment.lower())
    if idx != -1:
        return (idx, idx + len(fragment))

    # 对于较长的证据文本，尝试查找前 10 个字符的子串
    if len(fragment) > 10:
        short = fragment[:10]
        idx = original.find(short)
        if idx != -1:
            end = min(idx + len(fragment), len(original))
            return (idx, end)

    return None


def _merge_overlapping_spans(spans: list[EvidenceSpan]) -> list[EvidenceSpan]:
    """
    合并重叠或相邻的证据片段（同一 category 和 source 的片段合并）。
    不同类别/来源的片段不合并，以保留完整信息。
    """
    if not spans:
        return spans

    # 按起始位置排序
    sorted_spans = sorted(spans, key=lambda s: (s.category, s.source, s.start))
    merged: list[EvidenceSpan] = []

    for span in sorted_spans:
        if (
            merged
            and merged[-1].category == span.category
            and merged[-1].source == span.source
            and span.start <= merged[-1].end  # 重叠或相邻
        ):
            # 扩展上一个片段
            last = merged[-1]
            new_end = max(last.end, span.end)
            merged[-1] = EvidenceSpan(
                start=last.start,
                end=new_end,
                text=span.text[: new_end - last.start] if new_end > last.start else last.text,
                source=last.source,
                category=last.category,
            )
        else:
            merged.append(span)

    return merged


def extract_evidence(
    original_text: str,
    rule_hits: list,      # list[RuleHit]，避免循环导入
    llm_judgment=None,    # LLMJudgment | None
) -> list[EvidenceSpan]:
    """
    从规则命中和 LLM 判定中提取证据片段。

    参数：
    - original_text: 原始输入文本
    - rule_hits: 规则引擎的命中结果列表（RuleHit 对象）
    - llm_judgment: LLM 的审核结果（LLMJudgment 对象），可为 None

    返回：
    - 去重、排序后的 EvidenceSpan 列表
    """
    spans: list[EvidenceSpan] = []

    # ── 来自规则引擎的证据 ──────────────────────────────────────────────────
    for hit in rule_hits:
        # 验证位置区间的合法性
        start = max(0, hit.start)
        end = min(len(original_text), hit.end)
        if start >= end:
            # 位置无效，尝试重新在原文中定位关键词
            result = _find_text_position(original_text, hit.keyword)
            if result is None:
                continue
            start, end = result

        snippet = original_text[start:end]
        spans.append(
            EvidenceSpan(
                start=start,
                end=end,
                text=snippet,
                source="rule",
                category=hit.category,
            )
        )

    # ── 来自 LLM 的证据 ─────────────────────────────────────────────────────
    if llm_judgment is not None and llm_judgment.is_violation:
        for evidence_text in llm_judgment.evidence_texts:
            evidence_text = evidence_text.strip()
            if not evidence_text:
                continue

            result = _find_text_position(original_text, evidence_text)
            if result is None:
                # 无法定位到原文，创建一个"全文"级别的证据（位置 0, len）
                # 并将文本截断为前 50 字
                snippet = original_text[:50] + ("..." if len(original_text) > 50 else "")
                spans.append(
                    EvidenceSpan(
                        start=0,
                        end=len(original_text),
                        text=snippet,
                        source="llm",
                        category=llm_judgment.labels[0] if llm_judgment.labels else "abuse",
                    )
                )
                continue

            start, end = result
            # 确定该证据所属类别（尝试与 LLM 返回的 labels 对应）
            category = llm_judgment.labels[0] if llm_judgment.labels else "abuse"

            spans.append(
                EvidenceSpan(
                    start=start,
                    end=end,
                    text=original_text[start:end],
                    source="llm",
                    category=category,
                )
            )

    # ── 去重与合并 ──────────────────────────────────────────────────────────
    # 去除完全重复的片段（相同 start/end/category）
    seen: set[tuple[int, int, str]] = set()
    unique_spans: list[EvidenceSpan] = []
    for span in spans:
        key = (span.start, span.end, span.category)
        if key not in seen:
            seen.add(key)
            unique_spans.append(span)

    # 按位置排序
    unique_spans.sort(key=lambda s: s.start)

    return unique_spans


def spans_to_dict(spans: list[EvidenceSpan]) -> list[dict]:
    """将 EvidenceSpan 列表转换为可序列化的字典列表"""
    return [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "source": s.source,
            "category": s.category,
        }
        for s in spans
    ]
