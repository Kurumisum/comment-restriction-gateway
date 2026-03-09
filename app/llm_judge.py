"""
LLM 审核模块 - 调用 OpenAI 兼容 API（支持 DeepSeek）进行内容审核
LLM Judge Module - uses OpenAI-compatible API for content moderation
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

# LLM API 调用超时时间（秒），支持通过环境变量覆盖
_LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "15.0"))

logger = logging.getLogger(__name__)

# ── 系统提示词（中文"审核官"角色） ─────────────────────────────────────────
_SYSTEM_PROMPT = """你是一名专业的内容安全审核官，负责对用户评论进行安全审核。
你需要判断以下文本是否违规，并识别以下风险类别：
- abuse（辱骂/仇恨言论）：包含针对个人或群体的侮辱、谩骂、人身攻击等
- fraud（诈骗/欺诈）：包含引导转账、诈骗话术、虚假中奖、非法集资等内容
- sex（色情/成人内容）：包含露骨的性描写、色情诱导、裸露等内容

请以严格的 JSON 格式返回审核结果，不要包含任何其他内容，格式如下：
{
  "is_violation": true或false,
  "labels": ["abuse", "fraud", "sex"],  // 仅包含实际命中的类别
  "confidence": 0.95,  // 判断置信度，0到1之间的浮点数
  "rationale": "违规原因简述（不超过100字）",
  "evidence_texts": ["直接引用的违规片段1", "违规片段2"]  // 最多5个
}

注意事项：
1. 只有在有充分证据时才判定为违规，避免误判正常言论
2. confidence 需真实反映你的判断把握程度
3. evidence_texts 必须是原文中实际存在的文本片段
4. 对于模糊情况，is_violation 应为 false，confidence 设为较低值
"""

# ── 默认回退响应（API 不可用时使用） ────────────────────────────────────────
_FALLBACK_JUDGMENT = None  # 无 API key 时返回 None，由上层决定是否降级


@dataclass
class LLMJudgment:
    """LLM 审核结果"""
    is_violation: bool          # 是否违规
    labels: list[str]           # 命中的风险类别列表
    confidence: float           # 判断置信度（0-1）
    rationale: str              # 判断理由
    evidence_texts: list[str]   # 证据文本片段列表


def _get_api_client() -> AsyncOpenAI | None:
    """
    根据环境变量创建 OpenAI 兼容客户端。
    优先使用 DEEPSEEK_API_KEY，其次 OPENAI_API_KEY。
    如果均未配置，返回 None。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # DeepSeek 使用 OpenAI 兼容接口
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _get_model_name() -> str:
    """获取模型名称，支持通过环境变量覆盖"""
    return os.getenv("LLM_MODEL", "deepseek-chat")


def _parse_llm_response(content: str) -> LLMJudgment:
    """
    解析 LLM 返回的 JSON 内容，容错处理格式异常。
    """
    # 尝试从 markdown 代码块中提取 JSON
    if "```" in content:
        match_start = content.find("```json")
        if match_start == -1:
            match_start = content.find("```")
        if match_start != -1:
            content = content[match_start:]
            content = content.split("```", 1)[-1]
            end = content.find("```")
            if end != -1:
                content = content[:end]

    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取 JSON 对象片段
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(content[start:end])
            except json.JSONDecodeError:
                logger.warning("无法解析 LLM 返回的 JSON 内容: %s", content[:200])
                return LLMJudgment(
                    is_violation=False,
                    labels=[],
                    confidence=0.0,
                    rationale="LLM 返回内容解析失败",
                    evidence_texts=[],
                )
        else:
            return LLMJudgment(
                is_violation=False,
                labels=[],
                confidence=0.0,
                rationale="LLM 返回内容格式异常",
                evidence_texts=[],
            )

    # 提取各字段，处理缺失或类型错误
    is_violation = bool(data.get("is_violation", False))
    labels = [str(l) for l in data.get("labels", []) if isinstance(l, str)]
    # 置信度限制在 [0, 1] 区间
    raw_confidence = data.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(raw_confidence)))
    except (TypeError, ValueError):
        confidence = 0.5
    rationale = str(data.get("rationale", ""))[:500]  # 截断防止过长
    evidence_texts = [str(e) for e in data.get("evidence_texts", [])[:5]]

    return LLMJudgment(
        is_violation=is_violation,
        labels=labels,
        confidence=confidence,
        rationale=rationale,
        evidence_texts=evidence_texts,
    )


async def llm_judge(
    text: str,
    rule_hits: list,
    skip_if_no_key: bool = True,
) -> LLMJudgment | None:
    """
    调用 LLM 进行审核判定，返回结构化判定结果。

    参数：
    - text: 待审核的原始文本
    - rule_hits: 规则检测的命中结果（list[RuleHit]），用于构造增强提示
    - skip_if_no_key: 若为 True，在无 API key 时返回 None 而非报错

    返回：
    - LLMJudgment 对象，或 None（API 不可用且 skip_if_no_key=True）
    """
    client = _get_api_client()
    if client is None:
        if skip_if_no_key:
            logger.debug("未配置 LLM API key，跳过 LLM 审核")
            return None
        raise RuntimeError("未配置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")

    # 构造用户提示，附加规则命中信息作为参考
    user_content = f"请对以下评论内容进行安全审核：\n\n{text}"

    if rule_hits:
        # 将规则命中结果整理为辅助信息，引导 LLM 关注重点
        hit_summary = []
        for hit in rule_hits[:10]:  # 最多传入 10 条规则命中
            hit_summary.append(f"- [{hit.category}] 关键词「{hit.keyword}」")
        rule_hint = "\n".join(hit_summary)
        user_content += f"\n\n【规则预检提示】以下关键词已被规则引擎标记，请重点审查：\n{rule_hint}"

    model = _get_model_name()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,    # 低温度保证判断一致性
            max_tokens=512,
            timeout=_LLM_TIMEOUT,
        )
        content = response.choices[0].message.content or ""
        return _parse_llm_response(content)

    except APITimeoutError:
        logger.warning("LLM 审核超时（model=%s）", model)
        return LLMJudgment(
            is_violation=False,
            labels=[],
            confidence=0.0,
            rationale="LLM 审核超时，降级为规则引擎结果",
            evidence_texts=[],
        )
    except RateLimitError:
        logger.warning("LLM API 调用频率超限")
        return LLMJudgment(
            is_violation=False,
            labels=[],
            confidence=0.0,
            rationale="LLM API 频率限制，降级为规则引擎结果",
            evidence_texts=[],
        )
    except APIError as e:
        logger.error("LLM API 错误: %s", e)
        return LLMJudgment(
            is_violation=False,
            labels=[],
            confidence=0.0,
            rationale=f"LLM API 错误：{type(e).__name__}",
            evidence_texts=[],
        )
    except Exception as e:
        logger.error("LLM 审核发生未知错误: %s", e, exc_info=True)
        return LLMJudgment(
            is_violation=False,
            labels=[],
            confidence=0.0,
            rationale="LLM 审核异常，降级为规则引擎结果",
            evidence_texts=[],
        )
