"""
离线评估脚本 - 使用测试数据集评估审核系统的准确性
Offline Evaluation Script - evaluates moderation system accuracy using test dataset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# 确保能找到 app 包（脚本可从项目根目录或 eval/ 目录运行）
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.evidence import extract_evidence
from app.llm_judge import llm_judge
from app.normalize import normalize
from app.rules import check_rules


# ── 默认测试数据路径 ─────────────────────────────────────────────────────────
_DEFAULT_TEST_DATA = _PROJECT_ROOT / "data" / "test_cases.json"

# ── 允许的风险等级 ──────────────────────────────────────────────────────────
_VALID_RISK_LEVELS = {"allow", "review", "block"}

# ── 支持的标签类别 ──────────────────────────────────────────────────────────
_ALL_CATEGORIES = ["abuse", "fraud", "sex"]


def _decide_risk_level_sync(rule_hits, judgment) -> tuple[str, float, list[str]]:
    """
    同步版风险等级决策（与 main.py 中的逻辑保持一致，便于离线评估独立运行）
    """
    from app.main import _decide_risk_level
    risk_level, confidence, rationale, labels = _decide_risk_level(rule_hits, judgment)
    return risk_level, confidence, labels


async def _evaluate_single(case: dict) -> dict:
    """
    对单条测试用例执行审核，返回评估结果字典。
    """
    text = case["text"]
    expected_risk = case.get("expected_risk_level", "allow")
    expected_labels = set(case.get("expected_labels", []))

    t_start = time.perf_counter()

    # 规则检测
    rule_hits = check_rules(text)

    # LLM 审核（如未配置 API key 则跳过）
    judgment = await llm_judge(text, rule_hits, skip_if_no_key=True)

    # 综合决策
    risk_level, confidence, labels = _decide_risk_level_sync(rule_hits, judgment)
    predicted_labels = set(labels)

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # ── 计算各维度正确性 ──
    risk_correct = risk_level == expected_risk

    # 标签命中判断（预测标签是否覆盖了期望标签）
    if expected_labels:
        label_correct = expected_labels.issubset(predicted_labels)
    else:
        # 期望无标签时，预测也不应有标签（allow 情况）
        label_correct = len(predicted_labels) == 0

    return {
        "text": text,
        "expected_risk": expected_risk,
        "predicted_risk": risk_level,
        "risk_correct": risk_correct,
        "expected_labels": sorted(expected_labels),
        "predicted_labels": sorted(predicted_labels),
        "label_correct": label_correct,
        "confidence": confidence,
        "rule_hits": len(rule_hits),
        "llm_used": judgment is not None,
        "elapsed_ms": elapsed_ms,
        "description": case.get("description", ""),
    }


async def run_evaluation(test_data_path: Path) -> dict:
    """
    对整个测试集进行评估，返回汇总报告。
    """
    print(f"\n📂 加载测试数据：{test_data_path}")
    with test_data_path.open(encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"📊 共 {len(test_cases)} 条测试用例\n")

    results = []
    for i, case in enumerate(test_cases, 1):
        result = await _evaluate_single(case)
        results.append(result)
        status = "✅" if result["risk_correct"] else "❌"
        print(
            f"  [{i:02d}/{len(test_cases)}] {status} "
            f"期望={result['expected_risk']:6s} 预测={result['predicted_risk']:6s} "
            f"耗时={result['elapsed_ms']:.0f}ms  {result['text'][:40]}"
        )

    return _compute_metrics(results)


def _compute_metrics(results: list[dict]) -> dict:
    """
    计算整体和分类别的评估指标（精确率/召回率/F1）。
    """
    total = len(results)
    risk_correct = sum(1 for r in results if r["risk_correct"])
    overall_accuracy = risk_correct / total if total else 0.0

    # ── 风险等级混淆矩阵 ────────────────────────────────────────────────────
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r["expected_risk"]][r["predicted_risk"]] += 1

    # ── 标签级别的精确率/召回率/F1 ─────────────────────────────────────────
    label_metrics: dict[str, dict] = {}
    for category in _ALL_CATEGORIES:
        tp = fp = fn = 0
        for r in results:
            pred_has = category in r["predicted_labels"]
            exp_has = category in r["expected_labels"]
            if pred_has and exp_has:
                tp += 1
            elif pred_has and not exp_has:
                fp += 1
            elif not pred_has and exp_has:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        label_metrics[category] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # ── 假正例和假负例列表 ──────────────────────────────────────────────────
    false_positives = [
        r for r in results
        if r["predicted_risk"] in ("review", "block") and r["expected_risk"] == "allow"
    ]
    false_negatives = [
        r for r in results
        if r["predicted_risk"] == "allow" and r["expected_risk"] in ("review", "block")
    ]

    return {
        "total": total,
        "overall_accuracy": round(overall_accuracy, 4),
        "risk_level_correct": risk_correct,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "label_metrics": label_metrics,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "results": results,
    }


def _print_report(report: dict) -> None:
    """打印人类可读的评估报告"""
    print("\n" + "═" * 60)
    print("  📋 评估报告 / Evaluation Report")
    print("═" * 60)

    print(f"\n整体准确率（风险等级）：{report['overall_accuracy']:.1%}")
    print(f"  正确 {report['risk_level_correct']} / 总计 {report['total']}")

    print("\n── 标签级别指标（Precision / Recall / F1）──")
    print(f"  {'类别':<8} {'Precision':>10} {'Recall':>8} {'F1':>8}  (TP/FP/FN)")
    for cat, m in report["label_metrics"].items():
        cat_cn = {"abuse": "辱骂", "fraud": "诈骗", "sex": "色情"}.get(cat, cat)
        print(
            f"  {cat_cn}({cat:<5}) "
            f"{m['precision']:>10.1%} "
            f"{m['recall']:>8.1%} "
            f"{m['f1']:>8.1%}  "
            f"({m['tp']}/{m['fp']}/{m['fn']})"
        )

    print("\n── 混淆矩阵（期望 → 预测）──")
    levels = ["allow", "review", "block"]
    cm = report["confusion_matrix"]
    header = f"  {'期望\\预测':<10}" + "".join(f"{l:>8}" for l in levels)
    print(header)
    for exp in levels:
        row = f"  {exp:<10}" + "".join(
            f"{cm.get(exp, {}).get(pred, 0):>8}" for pred in levels
        )
        print(row)

    fps = report["false_positives"]
    fns = report["false_negatives"]

    if fps:
        print(f"\n── 假正例（{len(fps)} 条，应放行但被拦截）──")
        for r in fps[:5]:
            print(f"  [{r['predicted_risk']}] {r['text'][:60]}")
        if len(fps) > 5:
            print(f"  ... 共 {len(fps)} 条，显示前 5 条")

    if fns:
        print(f"\n── 假负例（{len(fns)} 条，应拦截但被放行）──")
        for r in fns[:5]:
            print(f"  [{r['expected_risk']}] {r['text'][:60]}")
        if len(fns) > 5:
            print(f"  ... 共 {len(fns)} 条，显示前 5 条")

    print("\n" + "═" * 60 + "\n")


def main() -> None:
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="评论内容安全审核网关 - 离线评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python eval/offline_eval.py
  python eval/offline_eval.py --data data/my_test_cases.json
  python eval/offline_eval.py --output report.json
        """,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_TEST_DATA,
        help=f"测试数据文件路径（默认：{_DEFAULT_TEST_DATA}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="将报告保存为 JSON 文件（可选）",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.60,
        help="整体准确率通过阈值，低于此值脚本以非零状态退出（默认：0.60）",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"❌ 测试数据文件不存在：{args.data}")
        sys.exit(1)

    # 运行评估
    report = asyncio.run(run_evaluation(args.data))

    # 打印报告
    _print_report(report)

    # 保存 JSON 报告
    if args.output:
        # 移除不可序列化的详细结果（可选择保留）
        output_data = {k: v for k, v in report.items() if k != "results"}
        output_data["results"] = report["results"]
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"📄 报告已保存到：{args.output}")

    # 根据整体准确率设置退出码
    threshold = float(args.accuracy_threshold)
    if report["overall_accuracy"] < threshold:
        print(f"⚠️  整体准确率 {report['overall_accuracy']:.1%} 低于阈值 {threshold:.0%}，请检查词表配置")
        sys.exit(1)


if __name__ == "__main__":
    main()
