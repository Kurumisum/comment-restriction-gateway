"""
评论内容安全审核网关 - FastAPI 主应用
Comment Moderation Gateway - Main FastAPI Application
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .evidence import EvidenceSpan, extract_evidence, spans_to_dict
from .llm_judge import LLMJudgment, llm_judge
from .normalize import normalize
from .rules import RuleHit, check_rules, get_word_lists

# 加载 .env 文件中的环境变量（开发环境使用）
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── 风险等级阈值配置（可通过环境变量覆盖） ──────────────────────────────────
# 规则命中数量 >= 该值时直接 block
try:
    RULE_BLOCK_THRESHOLD = int(os.getenv("RULE_BLOCK_THRESHOLD", "1"))
except ValueError:
    raise ValueError("环境变量 RULE_BLOCK_THRESHOLD 必须为整数（例如：1）")
# LLM 判断 block 所需最低置信度
try:
    LLM_BLOCK_CONFIDENCE = float(os.getenv("LLM_BLOCK_CONFIDENCE", "0.80"))
except ValueError:
    raise ValueError("环境变量 LLM_BLOCK_CONFIDENCE 必须为 0-1 之间的浮点数（例如：0.80）")
# LLM 判断 review 所需最低置信度
try:
    LLM_REVIEW_CONFIDENCE = float(os.getenv("LLM_REVIEW_CONFIDENCE", "0.50"))
except ValueError:
    raise ValueError("环境变量 LLM_REVIEW_CONFIDENCE 必须为 0-1 之间的浮点数（例如：0.50）")


# ── Pydantic 模型（OpenAPI schema） ─────────────────────────────────────────

class ModerateRequest(BaseModel):
    """审核请求体"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="待审核的评论文本",
        examples=["这个商品真的很好用，值得购买！"],
    )


class EvidenceSpanSchema(BaseModel):
    """证据片段 schema"""
    start: int = Field(..., description="在原始文本中的起始字符索引（含）")
    end: int = Field(..., description="在原始文本中的结束字符索引（不含）")
    text: str = Field(..., description="证据文本内容")
    source: Literal["rule", "llm"] = Field(..., description="证据来源")
    category: Literal["abuse", "fraud", "sex"] = Field(..., description="风险类别")


class ModerateResponse(BaseModel):
    """审核响应体"""
    risk_level: Literal["allow", "review", "block"] = Field(
        ...,
        description="风险等级：allow=通过，review=人工复审，block=拦截",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="命中的风险类别列表（abuse/fraud/sex）",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="综合置信度（0-1）",
    )
    evidence_spans: list[EvidenceSpanSchema] = Field(
        default_factory=list,
        description="证据片段列表，标注原文中违规内容的位置",
    )
    rationale: str = Field(
        ...,
        description="审核理由说明",
    )
    action: str = Field(
        ...,
        description="建议执行的动作",
    )
    normalized_text: str = Field(
        ...,
        description="规范化处理后的文本（用于调试绕过检测）",
    )


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    word_list_sizes: dict[str, int]
    llm_enabled: bool


# ── 风险等级决策逻辑 ─────────────────────────────────────────────────────────

def _decide_risk_level(
    rule_hits: list[RuleHit],
    llm_judgment: LLMJudgment | None,
) -> tuple[Literal["allow", "review", "block"], float, str, list[str]]:
    """
    根据规则命中和 LLM 判定，综合决策风险等级。

    返回：(risk_level, confidence, rationale, labels)

    决策规则：
    1. 规则引擎有命中 → block（规则命中是明确证据）
    2. LLM 判为违规且置信度 >= LLM_BLOCK_CONFIDENCE → block
    3. LLM 判为违规且置信度 >= LLM_REVIEW_CONFIDENCE → review
    4. LLM 判为不违规 / 无 LLM 结果 → allow
    """
    labels: set[str] = set()
    rationale_parts: list[str] = []
    confidence = 0.0

    has_rule_hits = len(rule_hits) >= RULE_BLOCK_THRESHOLD

    if has_rule_hits:
        # 汇总规则命中类别
        for hit in rule_hits:
            labels.add(hit.category)
        category_str = "、".join(
            {"abuse": "辱骂", "fraud": "诈骗", "sex": "色情"}.get(c, c) for c in sorted(labels)
        )
        bypass_count = sum(1 for h in rule_hits if h.normalized_hit)
        hint = f"（其中 {bypass_count} 处通过绕过手段检出）" if bypass_count else ""
        rationale_parts.append(
            f"规则引擎命中 {len(rule_hits)} 处风险词汇，类别：{category_str}{hint}"
        )
        # 规则命中置信度：命中越多越高，最高 0.99
        confidence = min(0.99, 0.85 + 0.05 * len(rule_hits))

    if llm_judgment is not None:
        if llm_judgment.is_violation:
            for label in llm_judgment.labels:
                labels.add(label)
            rationale_parts.append(
                f"LLM 审核判定违规（置信度 {llm_judgment.confidence:.0%}）：{llm_judgment.rationale}"
            )
            # 综合置信度取规则和 LLM 的最大值
            confidence = max(confidence, llm_judgment.confidence)
        else:
            rationale_parts.append(
                f"LLM 审核未发现明显违规（置信度 {llm_judgment.confidence:.0%}）"
            )
            if not has_rule_hits:
                # 仅 LLM 无违规，降低总体置信度
                confidence = max(confidence, 1.0 - llm_judgment.confidence)

    # ── 最终裁定 ──
    if has_rule_hits:
        risk_level: Literal["allow", "review", "block"] = "block"
    elif llm_judgment is not None and llm_judgment.is_violation:
        if llm_judgment.confidence >= LLM_BLOCK_CONFIDENCE:
            risk_level = "block"
        elif llm_judgment.confidence >= LLM_REVIEW_CONFIDENCE:
            risk_level = "review"
        else:
            risk_level = "review"
    else:
        risk_level = "allow"
        if not rationale_parts:
            rationale_parts.append("文本内容未发现风险")
        confidence = confidence if confidence > 0 else 0.05  # 最低噪声值

    rationale = "；".join(rationale_parts) if rationale_parts else "无违规内容"

    return risk_level, confidence, rationale, sorted(labels)


def _get_action_description(risk_level: str) -> str:
    """根据风险等级返回建议执行的动作说明"""
    actions = {
        "allow": "放行评论，正常展示",
        "review": "评论进入人工复审队列，暂不展示",
        "block": "拦截评论，禁止发布，通知用户内容违规",
    }
    return actions.get(risk_level, "未知操作")


# ── FastAPI 应用初始化 ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时预加载词表"""
    logger.info("正在预加载风险词表...")
    word_lists = get_word_lists()
    sizes = {cat: len(words) for cat, words in word_lists.items()}
    logger.info("词表加载完成：%s", sizes)
    yield
    logger.info("应用关闭")


app = FastAPI(
    title="评论内容安全审核网关",
    description=(
        "Comment Moderation Gateway - 基于规则引擎 + LLM 的多层次评论内容安全审核服务。\n\n"
        "支持辱骂、诈骗、色情三大风险类别检测，内置文本规范化处理（防绕过）。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS 中间件配置 ──────────────────────────────────────────────────────────
_allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 路由定义 ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查服务状态，返回词表大小和 LLM 可用性",
)
async def health_check() -> HealthResponse:
    """健康检查接口"""
    word_lists = get_word_lists()
    sizes = {cat: len(words) for cat, words in word_lists.items()}
    llm_enabled = bool(
        os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    return HealthResponse(
        status="ok",
        word_list_sizes=sizes,
        llm_enabled=llm_enabled,
    )


@app.post(
    "/moderate",
    response_model=ModerateResponse,
    summary="评论内容审核",
    description=(
        "对提交的评论文本进行多层次安全审核，返回风险等级、证据片段和建议动作。\n\n"
        "审核流程：文本规范化 → 规则引擎检测 → LLM 深度判定 → 综合裁定"
    ),
)
async def moderate(request: ModerateRequest) -> ModerateResponse:
    """
    评论内容审核主接口。

    审核流程：
    1. 对输入文本进行规范化（处理全角、谐音、拆字等绕过手段）
    2. 使用规则引擎在原始文本和规范化文本中检测风险词汇
    3. 调用 LLM 进行深度语义审核（如已配置 API key）
    4. 综合两层结果，决策风险等级并提取证据片段
    """
    t_start = time.perf_counter()
    text = request.text

    # 步骤 1：文本规范化
    normalized = normalize(text)
    logger.debug("规范化完成: %r → %r", text[:50], normalized[:50])

    # 步骤 2：规则引擎检测
    rule_hits = check_rules(text)
    logger.info(
        "规则检测完成，命中 %d 条，文本长度 %d",
        len(rule_hits), len(text),
    )

    # 步骤 3：LLM 深度审核
    judgment: LLMJudgment | None = None
    try:
        judgment = await llm_judge(text, rule_hits, skip_if_no_key=True)
    except Exception as e:
        logger.error("LLM 审核调用失败: %s", e)

    # 步骤 4：综合决策
    risk_level, confidence, rationale, labels = _decide_risk_level(rule_hits, judgment)

    # 步骤 5：提取证据片段
    evidence_spans = extract_evidence(text, rule_hits, judgment)
    evidence_dicts = spans_to_dict(evidence_spans)

    action = _get_action_description(risk_level)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "审核完成: risk_level=%s, confidence=%.2f, labels=%s, 耗时=%.1fms",
        risk_level, confidence, labels, elapsed_ms,
    )

    return ModerateResponse(
        risk_level=risk_level,
        labels=labels,
        confidence=round(confidence, 4),
        evidence_spans=[EvidenceSpanSchema(**span) for span in evidence_dicts],
        rationale=rationale,
        action=action,
        normalized_text=normalized,
    )
