# 评论内容安全审核网关

**Comment Moderation Gateway** | 基于规则引擎 + LLM 的多层次评论内容安全审核服务

---

## 简介 / Introduction

本项目实现了一套评论内容安全审核网关，采用 **规则引擎（关键词检测）+ LLM 深度语义审核** 的双层架构，支持以下风险类别的自动化审核：

| 类别 | 说明 |
|------|------|
| `abuse` | 辱骂 / 仇恨言论 |
| `fraud` | 诈骗 / 欺诈内容 |
| `sex` | 色情 / 成人内容 |

内置**文本规范化**处理，可识别常见的绕过审核手段（全角字符、谐音字、拆字、leet-speak 等）。

---

## 功能特性 / Features

- 🔍 **多层检测**：规则引擎 + LLM 语义审核，互补提升覆盖率
- 🛡️ **防绕过**：文本规范化处理（全角转半角、繁简转换、谐音字、拆字重组、leet-speak）
- 📍 **精确定位**：返回违规内容在原文中的精确字符位置（`evidence_spans`）
- ⚡ **异步高性能**：基于 FastAPI + asyncio，支持高并发
- 🔌 **LLM 可选**：配置 API key 后启用 LLM 深度审核，未配置时自动降级为纯规则模式
- 📊 **离线评估**：内置评估脚本，支持精确率/召回率/F1 计算

---

## 项目结构 / Project Structure

```
comment-restriction-gateway/
├── app/
│   ├── __init__.py          # 包初始化
│   ├── main.py              # FastAPI 应用，/moderate 和 /health 接口
│   ├── normalize.py         # 文本规范化（防绕过处理）
│   ├── rules.py             # 规则引擎（关键词词表检测）
│   ├── llm_judge.py         # LLM 审核模块（OpenAI 兼容接口）
│   └── evidence.py          # 证据片段提取与位置定位
├── eval/
│   ├── __init__.py
│   └── offline_eval.py      # 离线评估脚本
├── data/
│   ├── abuse_words.txt      # 辱骂词汇词表
│   ├── fraud_words.txt      # 诈骗词汇词表
│   ├── sex_words.txt        # 色情词汇词表
│   └── test_cases.json      # 评估测试用例
├── requirements.txt
└── README.md
```

---

## API 定义 / API Reference

### `POST /moderate` — 评论审核

**请求体：**

```json
{
  "text": "待审核的评论文本（最长 10000 字符）"
}
```

**响应体：**

```json
{
  "risk_level": "block",
  "labels": ["abuse"],
  "confidence": 0.95,
  "evidence_spans": [
    {
      "start": 5,
      "end": 7,
      "text": "傻逼",
      "source": "rule",
      "category": "abuse"
    }
  ],
  "rationale": "规则引擎命中 1 处风险词汇，类别：辱骂",
  "action": "拦截评论，禁止发布，通知用户内容违规",
  "normalized_text": "你真是个傻逼，滚你妈的！"
}
```

**风险等级说明：**

| `risk_level` | 含义 | 建议动作 |
|---|---|---|
| `allow` | 未检测到风险 | 正常展示 |
| `review` | 存在疑似风险 | 进入人工复审队列 |
| `block` | 确认违规 | 拦截，禁止发布 |

**`source` 字段说明：**

| `source` | 含义 |
|---|---|
| `rule` | 规则引擎（关键词）命中 |
| `llm` | LLM 语义审核命中 |

---

### `GET /health` — 健康检查

```json
{
  "status": "ok",
  "word_list_sizes": {
    "abuse": 30,
    "fraud": 25,
    "sex": 22
  },
  "llm_enabled": true
}
```

---

## 环境变量 / Environment Variables

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API 密钥（优先使用） |
| `OPENAI_API_KEY` | — | OpenAI API 密钥（备选） |
| `LLM_BASE_URL` | `https://api.deepseek.com` | LLM API 基础 URL |
| `LLM_MODEL` | `deepseek-chat` | 使用的模型名称 |
| `RULE_BLOCK_THRESHOLD` | `1` | 规则命中数 ≥ 该值时直接 block |
| `LLM_BLOCK_CONFIDENCE` | `0.80` | LLM block 判定最低置信度 |
| `LLM_REVIEW_CONFIDENCE` | `0.50` | LLM review 判定最低置信度 |
| `CORS_ORIGINS` | `*` | 允许的 CORS 来源（逗号分隔） |

---

## 快速开始 / Quick Start

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量（可选，用于启用 LLM 审核）

```bash
# 创建 .env 文件
cat > .env << EOF
DEEPSEEK_API_KEY=your_deepseek_api_key_here
LLM_MODEL=deepseek-chat
EOF
```

### 3. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：
- API 文档：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc
- 健康检查：http://localhost:8000/health

### 4. 测试接口

```bash
# 测试正常内容
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "这个商品质量很好，推荐购买！"}'

# 测试违规内容
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "你真是个傻逼！"}'
```

---

## LLM 调用说明 / LLM Integration

本项目使用 **DeepSeek API**（与 OpenAI API 完全兼容）。若需更换模型：

```bash
# 使用 DeepSeek
export DEEPSEEK_API_KEY=sk-xxxxx
export LLM_BASE_URL=https://api.deepseek.com
export LLM_MODEL=deepseek-chat

# 使用 OpenAI GPT-4
export OPENAI_API_KEY=sk-xxxxx
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
```

**未配置 API key 时**，服务会自动降级为纯规则模式运行，不影响基础功能。

---

## 离线评估 / Offline Evaluation

```bash
# 使用默认测试集评估
python eval/offline_eval.py

# 使用自定义测试集
python eval/offline_eval.py --data path/to/test_cases.json

# 保存评估报告为 JSON
python eval/offline_eval.py --output report.json
```

**测试用例格式：**

```json
[
  {
    "text": "待评估的评论文本",
    "expected_risk_level": "block",
    "expected_labels": ["abuse"],
    "description": "测试用例描述（可选）"
  }
]
```

---

## 词表扩展 / Word List Extension

词表文件位于 `data/` 目录，格式为每行一个词，`#` 开头为注释行：

```
# 辱骂词汇
傻逼
废物
# 更多词汇...
```

修改词表后重启服务即可生效（词表在启动时缓存至内存）。

---

## 优化建议 / Improvement Suggestions

1. **词表精细化**：根据业务场景持续维护词表，定期分析误判/漏判案例
2. **白名单机制**：为正常词语添加白名单，减少误判（如"草莓"中的"草"字）
3. **上下文感知**：引入滑动窗口上下文分析，避免孤立词语误判
4. **用户画像**：结合用户历史行为，对高风险用户提高审核灵敏度
5. **多语言支持**：扩展英文、粤语等多语种词表
6. **异步批处理**：实现批量审核接口，提升吞吐量
7. **审计日志**：记录每次审核的完整链路，支持事后溯源
8. **A/B 测试**：支持多版本规则并行，通过评估指标选择最优配置
9. **向量检索**：使用语义向量库替代精确词表，提升泛化能力
10. **联邦学习**：在保护隐私的前提下，利用多方数据联合优化模型

---

## License

MIT
