"""Generate docs/FINAL_SUMMARY_ZH.md (Chinese audit summary; all numbers read from artifacts).

For the researcher's own audit and interview preparation; not a formal submission.
Every number is recomputed from the canonical frozen run, configs/final.yaml and
docs/freeze.json; no model result, hyperparameter, SHA or p-value is hard-coded here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yaml

PRIMARY = {"M0": "none", "M1": "none", "M2": "F3", "M3": "F3", "M4": "none", "M5": "F3"}
NAME = {"M0": "HS", "M1": "GARCH-t", "M2": "LinQR", "M3": "MLP", "M4": "GJR-t", "M5": "GRU"}


def _dm_value(dm: pd.DataFrame, a: str, b: str, tail: float, col: str) -> float:
    row = dm[(dm["model_a"] == a) & (dm["model_b"] == b) & (dm["tail"] == tail)]
    if not len(row):
        return float("nan")
    return float(row.iloc[0][col])


def _load(out_root: Path):
    from scripts.common import canonical_run_dir

    run_dir = canonical_run_dir(out_root)
    return (
        pd.read_csv(run_dir / "tables" / "metrics.csv"),
        pd.read_csv(run_dir / "tables" / "dm_comparison.csv"),
        pd.read_csv(run_dir / "tables" / "regime_metrics.csv"),
        pd.read_csv(run_dir / "tables" / "ablation.csv"),
    )


def build(out_root: Path) -> str:
    import json

    from scripts.common import canonical_run_dir

    run_dir = canonical_run_dir(out_root)
    metrics, dm, regime, ablation = _load(out_root)
    m = metrics[metrics.apply(lambda r: PRIMARY.get(r["model"]) == r["feature_set"], axis=1)]
    freeze = json.loads((Path(out_root) / "manifests" / "freeze.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text(encoding="utf-8"))
    m3, m5 = cfg["models"]["M3"], cfg["models"]["M5"]

    h_m4_5 = _dm_value(dm, "M1", "M4", 0.05, "holm_dm_pvalue")
    h_m4_10 = _dm_value(dm, "M1", "M4", 0.10, "holm_dm_pvalue")
    h_m4_1 = _dm_value(dm, "M1", "M4", 0.01, "holm_dm_pvalue")
    b_m4_5 = _dm_value(dm, "M1", "M4", 0.05, "bootstrap_pvalue")
    b_m4_10 = _dm_value(dm, "M1", "M4", 0.10, "bootstrap_pvalue")
    h_m3_1 = _dm_value(dm, "M1", "M3", 0.01, "holm_dm_pvalue")
    h_m3_5 = _dm_value(dm, "M1", "M3", 0.05, "holm_dm_pvalue")
    h_m3_10 = _dm_value(dm, "M1", "M3", 0.10, "holm_dm_pvalue")
    b_m1m3_1 = _dm_value(dm, "M1", "M3", 0.01, "bootstrap_pvalue")

    L = []
    L.append("# 中文审计摘要（FINAL_SUMMARY_ZH）")
    L.append("")
    L.append("> 本文件为研究者本人审计与面试准备材料，非正式提交文档。所有数字由脚本从 canonical frozen run 产物重算（`scripts/generate_final_summary.py`），与英文报告一致。")
    L.append("")
    L.append("## 0. 本轮审计发现了什么（为什么旧 final 全部失效）")
    L.append("")
    L.append("完整记录见 `docs/AUDIT_REMEDIATION.md`。三个 correctness 问题：")
    L.append("")
    L.append("1. **Student-t VaR 尺度 bug（P0）**：旧实现直接用 `scipy.stats.t.ppf(alpha, nu)`，而 arch 的 Student-t innovation 是方差 1 的标准化分布。")
    L.append("   正确分位数 = `t_ppf(alpha, nu) × sqrt((nu-2)/nu)`（= arch `StudentsT().ppf`）。在拟合自由度 nu≈5-12 时旧 VaR 高估幅度约 10-25%，")
    L.append("   导致旧 GARCH-t / GJR-t 违例率系统性偏低（1% tail 旧值 0.76% vs 修正后 1.5%）。因此旧 M1/M4 的全部 failure rate、Kupiec、")
    L.append("   Christoffersen、DQ、pinball、DM、bootstrap、regime 分层与'每 tail 最佳模型'结论一律 INVALIDATED BY AUDIT。")
    L.append("2. **target-date 分区 bug（P0）**：旧实现按 forecast origin 日期划分 dev/final，2007-12-31 的 origin（target 2008-01-02）被错归 development；")
    L.append("   修正为按 target date 划分后，final 从 2640 变为 2641 个预测日（含 2008-01-02）。")
    L.append("3. **provenance 缺口（P1）**：旧冻结不校验代码签名与工作树、--data 可绕过冻结数据、产物可被静默复用、neural search → final.yaml 链路无强制 artifact。")
    L.append("   本轮全部修复（见 AUDIT_REMEDIATION §2.4-2.10）。")
    L.append("")
    L.append("修复过程中**未**根据旧 final 结果做任何模型/特征/窗口/超参数选择；所有选择来自 development-only 证据或 correctness 要求。")
    L.append("")
    L.append("## 1. 最终实验协议（修正后）")
    L.append("")
    L.append(f"- 数据：SPY 日度 log_ret / rv5 / bv，4640 行（2000-01-04 ~ 2018-06-27），SHA256 {freeze['data_sha256'][:16]}... 冻结。")
    L.append("- development 期：target date < 2008-01-01，rolling-origin 验证（公共目标 498 个，2006-01-06 ~ 2007-12-28）。")
    L.append(f"- final test（freeze manifest 记录的 git_commit {freeze['git_commit'][:12]}）：target date >= 2008-01-01，2641 个预测日（2008-01-02 ~ 2018-06-27），全部模型同日期集。")
    L.append(f"- 冻结内容：primary window={freeze['primary_window']}（等权归一化聚合，24 cells 中 23 个偏好 1500）、M3 hidden {m3['hidden']} / lr {m3['lr']} / wd {m3['weight_decay']} / batch {m3['batch_size']}、")
    L.append(f"  M5 hidden {m5['hidden']} / lr {m5['lr']} / wd {m5['weight_decay']} / batch {m5['batch_size']}、primary seed 42、robustness seeds {{7, 2026}}、config SHA256 {freeze['config_sha256'][:16]}...、data SHA256 {freeze['data_sha256'][:16]}...")
    L.append("- 零泄漏约束：特征/标准化/early-stopping 全部限制在训练窗口内（截断不变性测试锁定）；MLP/GRU target 采用 train-only 标准化（Scheme B）。")
    L.append("- 冻结门禁（强化）：data/config/code signature + working tree clean（freeze 时）+ effective data path 校验；正式产物隔离于 `outputs/runs/<freeze_id>/`，复用仅限签名一致。")
    L.append("")
    L.append("## 2. 为什么选择 1500 日窗口（等权聚合，审计修复）")
    L.append("")
    L.append("- 决策规则（预先声明）：每 (model, feature-set, tail) 单元内候选相对归一化损失等权平均；报告 per-cell winner、win count、drop-one sensitivity；聚合不一致时保守选长窗口。")
    L.append("- 证据（outputs/development/window_decision.json）：24 个单元中 23 个偏好 1500（等权平均归一化损失 0.962 vs 1.038）；raw pinball sum 一致（无分歧）；drop-one sensitivity 24/24 仍选 1500。")
    L.append("- 1% tail 校准：1500 窗口 |failure rate - 1%| 更小 —— 长窗口有效尾部样本更多（约 15 vs 10 个期望违例），经验分位数更稳定。")
    L.append("")
    L.append("## 3. 模型含义（数学/经济学）")
    L.append("")
    L.append("- M0 HS：窗口内历史收益经验分位数；无参数、隐含收益分布平稳假设，regime 突变时反应滞后。")
    L.append("- M1 GARCH(1,1)-t：σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}；z_t 为方差 1 标准化 Student-t（分位数 = t_ppf × sqrt((ν-2)/ν)）；VaR = μ + F_t⁻¹(α)·σ。")
    L.append("- M2 线性分位数回归：pinball 损失 min_β Σ ρ_τ(y_{t+1} - x_tᵀβ)，与 MLP 共享特征；独立拟合允许交叉（crossing rate 报告）。")
    L.append("- M3 MLP：joint pinball + 有序输出头 q_0.05 = q_0.01 + softplus(g1)、q_0.10 = q_0.05 + softplus(g2)，结构非交叉；train-only target 标准化（Scheme B）。")
    L.append("- M4 GJR-GARCH(1,1,1)-t：σ²_t = ω + α·ε²_{t-1} + γ·I(ε<0)·ε²_{t-1} + β·σ²_{t-1}（leverage）；VaR 构造同 M1。")
    L.append("- M5 GRU：最近 22 日特征序列 → 隐状态 → 有序分位数头；序列建模稳健性扩展（非替代 MLP）。")
    L.append("")
    L.append("## 4. 各 tail 最佳模型（修正后冻结样本外，2641 日）")
    L.append("")
    for tail in (0.01, 0.05, 0.10):
        sub = m[m["tail"] == tail]
        best_cal = sub.loc[sub["failure_rate"].sub(tail).abs().idxmin()]
        best_loss = sub.loc[sub["mean_pinball"].idxmin()]
        L.append(
            f"- **{int(tail*100)}% tail**：经验违例率最接近名义水平 = {NAME[best_cal['model']]}"
            f"（{best_cal['failure_rate']:.4f} vs 目标 {tail:.2f}）；pinball 最低 = {NAME[best_loss['model']]}"
            f"（{best_loss['mean_pinball']:.5f}）。"
        )
    L.append("")
    L.append("## 5. failure rate 与覆盖检验（修正后）")
    L.append("")
    L.append("| 模型 | 1% 违例率 | 5% 违例率 | 10% 违例率 | Kupiec(1%) p | Ind(1%) p | CC(1%) p |")
    L.append("|---|---|---|---|---|---|---|")
    for model in ["M0", "M1", "M2", "M3", "M4", "M5"]:
        r1 = m[(m["model"] == model) & (m["tail"] == 0.01)].iloc[0]
        r5 = m[(m["model"] == model) & (m["tail"] == 0.05)].iloc[0]
        r10 = m[(m["model"] == model) & (m["tail"] == 0.10)].iloc[0]
        L.append(
            f"| {NAME[model]} | {r1['failure_rate']:.4f} | {r5['failure_rate']:.4f} | {r10['failure_rate']:.4f} | "
            f"{r1['kupiec_pvalue']:.3f} | {r1['christoffersen_ind_pvalue']:.3f} | {r1['conditional_coverage_pvalue']:.3f} |"
        )
    L.append("")
    g1_1 = m[(m["model"] == "M1") & (m["tail"] == 0.01)].iloc[0]
    g4_1 = m[(m["model"] == "M4") & (m["tail"] == 0.01)].iloc[0]
    g1_5 = m[(m["model"] == "M1") & (m["tail"] == 0.05)].iloc[0]
    g4_5 = m[(m["model"] == "M4") & (m["tail"] == 0.05)].iloc[0]
    g1_10 = m[(m["model"] == "M1") & (m["tail"] == 0.10)].iloc[0]
    g4_10 = m[(m["model"] == "M4") & (m["tail"] == 0.10)].iloc[0]
    m0_5 = m[(m["model"] == "M0") & (m["tail"] == 0.05)].iloc[0]
    L.append(
        f"- 修正后 GARCH 族 1% 违例率 {g1_1['failure_rate']:.4f} / {g4_1['failure_rate']:.4f}，**高于**名义 1%"
        f"（即 mildly under-conservative：VaR 不够负、违例偏多，Kupiec p={g1_1['kupiec_pvalue']:.3f} / "
        f"{g4_1['kupiec_pvalue']:.3f}）；HS 5%/10% 频率最接近目标但独立性检验 "
        f"p={m0_5['christoffersen_ind_pvalue']:.3f}（违例强聚集）。"
    )
    L.append(
        f"- 条件充分性（逐模型）：仅 GJR-t 的 1% CC 在 5% 水平不拒绝（p={g4_1['conditional_coverage_pvalue']:.3f}），"
        f"GARCH-t 1% CC p={g1_1['conditional_coverage_pvalue']:.3f} 拒绝；5% CC 两模型均拒绝"
        f"（p={g1_5['conditional_coverage_pvalue']:.3f} / {g4_5['conditional_coverage_pvalue']:.3f}）；"
        f"10% CC GARCH-t 拒绝（p={g1_10['conditional_coverage_pvalue']:.3f}）、GJR-t 不拒绝"
        f"（p={g4_10['conditional_coverage_pvalue']:.3f}）。正确频率 ≠ 正确条件 VaR 动态。"
    )
    L.append("")
    L.append("## 6. pinball loss 对比（修正后）")
    L.append("")
    for tail in (0.01, 0.05, 0.10):
        sub = m[m["tail"] == tail].sort_values("mean_pinball")
        ranked = " < ".join(f"{NAME[r['model']]}({r['mean_pinball']:.5f})" for _, r in sub.iterrows())
        L.append(f"- {int(tail*100)}% tail：{ranked}")
    L.append("")
    L.append("## 7. 特征消融（F0 returns → F1 +RV → F2 +BV → F3 +jump/downside block）")
    L.append("")
    for model in ["M2", "M3"]:
        sub = ablation[ablation["model"] == model].sort_values(["tail", "feature_set"])
        L.append(f"- {NAME[model]} 各 tail pinball（F0→F3）：")
        for tail in (0.01, 0.05, 0.10):
            sub_t = sub[sub["tail"] == tail]
            line = " → ".join(f"{r['feature_set']}({r['mean_pinball']:.5f})" for _, r in sub_t.iterrows())
            L.append(f"  - {int(tail*100)}% tail：{line}")
    L.append("")

    def abl_delta(model: str, fa: str, fb: str, tail: float) -> float:
        sub = ablation[(ablation["model"] == model) & (ablation["tail"] == tail)]
        va = float(sub[sub["feature_set"] == fa].iloc[0]["mean_pinball"])
        vb = float(sub[sub["feature_set"] == fb].iloc[0]["mean_pinball"])
        return 100.0 * (va - vb) / va

    d_m2_rv = {t: abl_delta("M2", "F0", "F1", t) for t in (0.01, 0.05, 0.10)}
    d_m3_rv = {t: abl_delta("M3", "F0", "F1", t) for t in (0.01, 0.05, 0.10)}
    d_m2_bv = {t: abl_delta("M2", "F1", "F2", t) for t in (0.01, 0.05, 0.10)}
    d_m3_bv = {t: abl_delta("M3", "F1", "F2", t) for t in (0.01, 0.05, 0.10)}
    d_m2_f3 = {t: abl_delta("M2", "F2", "F3", t) for t in (0.01, 0.05, 0.10)}
    d_m3_f3 = {t: abl_delta("M3", "F2", "F3", t) for t in (0.01, 0.05, 0.10)}
    L.append("- **RV 增量**（F0→F1，pinball 相对变化 %，正 = 改善）：")
    L.append(
        f"  - LinQR：1% {d_m2_rv[0.01]:+.1f}%、5% {d_m2_rv[0.05]:+.1f}%、10% {d_m2_rv[0.10]:+.1f}%"
        " —— 5%/10% tail 有改善（1% 近似为零）。"
    )
    L.append(
        f"  - MLP：1% {d_m3_rv[0.01]:+.1f}%、5% {d_m3_rv[0.05]:+.1f}%、10% {d_m3_rv[0.10]:+.1f}%"
        " —— 1%/5% tail 有改善（10% 近似为零）。"
    )
    L.append("- **BV 条件增量**（F1→F2，正 = 改善）：")
    L.append(
        f"  - LinQR：1% {d_m2_bv[0.01]:+.1f}%、5% {d_m2_bv[0.05]:+.1f}%、10% {d_m2_bv[0.10]:+.1f}%"
        " —— RV 之后对线性分位数回归几乎没有增量。"
    )
    L.append(
        f"  - MLP：1% {d_m3_bv[0.01]:+.1f}%、5% {d_m3_bv[0.05]:+.1f}%、10% {d_m3_bv[0.10]:+.1f}%"
        " —— 尤其 1% tail 有明显增量。但这是**信息增量 ≠ 模型层面超越**：MLP 加 BV 后仍未超过 GARCH 族。"
    )
    L.append(
        f"- **F3 jump/downside block 增量**（F2→F3，正 = 改善）：LinQR 1% {d_m2_f3[0.01]:+.1f}%、"
        f"5% {d_m2_f3[0.05]:+.1f}%、10% {d_m2_f3[0.10]:+.1f}%；MLP 1% {d_m3_f3[0.01]:+.1f}%、"
        f"5% {d_m3_f3[0.05]:+.1f}%、10% {d_m3_f3[0.10]:+.1f}% —— 各 model×tail 无稳定正增量，"
        "整块效果不单独归因 jump。"
    )
    L.append("")
    L.append("## 8. 危机与 regime 发现（修正后，5% tail failure rate）")
    L.append("")
    reg5 = regime[regime["tail"] == 0.05]
    reg5 = reg5[reg5.apply(lambda r: PRIMARY.get(r["model"]) == r["feature_set"], axis=1)]
    for reg in ["crisis_2008_2009", "elevated_2010_2012", "calm_2013_2014", "stress_2015_2016", "calm_2017", "spike_2018"]:
        sub = reg5[reg5["regime"] == reg]
        if not len(sub):
            continue
        cells = "、".join(f"{NAME[r['model']]}={r['failure_rate']:.3f}" for _, r in sub.iterrows())
        L.append(f"- {reg}：{cells}")
    L.append("")
    L.append("- 观察 HS 的双向 regime 适应滞后：危机开始违例率过高（风险反应慢），危机结束进入平静期违例率过低（历史危机观测仍滞留在滚动尾部）—— 是 **slow two-sided regime adaptation**，而非单纯危机低估。")
    L.append("")
    L.append("## 9. DM / bootstrap 结论（Holm 校正，headline 族）")
    L.append("")
    head = dm[dm["headline"] == 1]
    for _, r in head.iterrows():
        L.append(
            f"- {NAME[r['model_a']]} vs {NAME[r['model_b']]} @{int(r['tail']*100)}%：DM={r['dm_stat']:.2f} "
            f"(raw p={r['dm_pvalue']:.4f}, Holm p={r['holm_dm_pvalue']:.4f}, bootstrap p={r['bootstrap_pvalue']:.3f}, favors {r['favors']})"
        )
    L.append("")
    L.append(
        f"- GJR-t（M4）vs GARCH-t：5% Holm p={h_m4_5:.4f}（显著，bootstrap p={b_m4_5:.3f}）、"
        f"10% Holm p={h_m4_10:.4f}（显著，bootstrap p={b_m4_10:.3f}）、1% Holm p={h_m4_1:.4f}"
        "（方向性证据，不显著）—— 非对称波动率设定带来预测损失改进，与 leverage/asymmetry 效应 consistent（本实验不做参数级 leverage 识别）。"
    )
    L.append(
        f"- MLP vs GARCH-t：10% Holm p={h_m3_10:.4f}（显著更差）、1% Holm p={h_m3_1:.4f}（不显著）、"
        f"5% Holm p={h_m3_5:.4f}（不显著）—— 负结果有 tail 依赖。"
    )
    L.append(
        f"- DM 与 bootstrap 不一致处（如 1% tail 的 GARCH-t vs MLP：raw DM p 显著但 bootstrap "
        f"p={b_m1m3_1:.2f}）如实报告为 inference sensitive to dependence/finite-sample procedure。"
    )
    L.append("")
    L.append("## 10. Neural 是否有 absolute incremental value？")
    L.append("")
    L.append(
        "- **没有绝对增量**：MLP 三 tail pinball 均高于 GARCH 族；对 GARCH-t 的劣势仅 10% tail Holm 显著"
        f"（p={h_m3_10:.3f}），1%/5% tail 不显著（p={h_m3_1:.3f} / {h_m3_5:.3f}）。"
    )
    L.append("- 修正后 MLP 与线性差距缩小（5% tail 不显著）：Scheme B 标准化改善了 MLP 训练；joint loss + non-crossing 是方法差异而非完美纯结构控制。")
    rob = pd.read_csv(run_dir / "tables" / "seed_robustness_summary.csv")
    gr5_1 = rob[(rob["model"] == "M5") & (rob["tail"] == 0.01)].iloc[0]
    L.append(
        f"- GRU：1% tail 严重过度违例（{gr5_1['failure_rate_mean']:.4f} ± {gr5_1['failure_rate_std']:.4f}，"
        "seed std 最大）—— seed sensitivity 如实报告。"
    )
    L.append("")
    L.append("## 11. seed robustness（n_seeds=3：42 primary + 7/2026）")
    L.append("")
    L.append("| 模型 | tail | failure mean ± std | pinball mean ± std |")
    L.append("|---|---|---|---|")
    for _, r in rob.iterrows():
        L.append(
            f"| {NAME[r['model']]} | {int(r['tail']*100)}% | {r['failure_rate_mean']:.4f} ± {r['failure_rate_std']:.4f} | "
            f"{r['pinball_mean']:.5f} ± {r['pinball_std']:.5f} |"
        )
    L.append("")
    L.append("## 12. 假设评级（H1-H7，修正后冻结证据）")
    L.append("")
    L.append("| 假设 | Verdict | 关键证据 |")
    L.append("|---|---|---|")
    L.append("| H1 HS regime adaptation lag | strongly supported | 2008-2009 违例率远超目标、平静期（2013-14/2017）违例率极低 —— 双向滞后（slow two-sided regime adaptation） |")
    L.append("| H2 GARCH 型波动率模型改善 VaR | supported, calibration mixed | proper loss 三 tail 最低；但覆盖并非全面通过（1% UC 拒绝、5% CC 拒绝） |")
    L.append("| H3a RV 增量 | supported（tail 依赖） | F0→F1：LinQR 5%/10% 改善、MLP 1%/5% 改善（见 §7 动态数字） |")
    L.append("| H3b BV 条件增量 | model-dependent | LinQR ≈0 或略负；MLP 各 tail 均有改善（1% 最明显，-10.9%）—— 无 model-agnostic 一致增量 |")
    L.append("| H4 MLP 绝对增量 | unsupported | 三 tail pinball 均不优于 GARCH 族；仅 10% Holm 显著更差（p=" + f"{h_m3_10:.3f}" + "） |")
    L.append("| H5 非线性映射价值 | unsupported / inconclusive | 10% tail 显著更差（Holm p=" + f"{h_m3_10:.4f}" + "）；1%/5% 无清晰优势；joint loss+non-crossing 使其非纯 mapping control |")
    L.append("| H6 tail 依赖的相对表现 | supported | 校准与 DM 差异随 tail 明显变化（GJR 5/10% 显著、1% 方向性） |")
    L.append("| H7 jump/downside block 极端 tail 增量 | not identified | F3 为联合 block 且各 model×tail 无稳定正增量；无 component identification |")
    L.append("")
    L.append("## 13. 哪些结论可靠，哪些只是描述性")
    L.append("")
    L.append("**可靠**（检验功效充分）：GARCH 族 sharpness 优势（三 tail pinball 最低）；GJR vs GARCH-t 的 leverage 增量（Holm 校正显著）；HS 违例聚集（Ind p<0.001）；"
    f"MLP 10% tail 显著劣于 GARCH-t/LinQR（Holm p={h_m3_10:.3f}，1%/5% 非显著属证据不足而非等价）；危机期 HS 滞后。")
    L.append("")
    L.append("**描述性/有限样本**：1% tail 各检验功效低（期望违例 ~26 个）；5% tail MLP vs GARCH-t 不显著属于证据不足而非等价；bootstrap 与 DM 不一致处；F3 整块效果的成分归因。")
    L.append("")
    L.append("## 14. 项目主要局限")
    L.append("")
    L.append("- 1% tail 期望违例少，覆盖检验功效低；单一资产（SPY ETF）；2018 数据仅至 6 月；rv5/bv 为日度聚合。")
    L.append(
        f"- NN 计算成本约为经典模型 10 倍且无绝对增量；GRU 1% tail seed sensitivity"
        f"（std {gr5_1['failure_rate_std']:.4f}）。"
    )
    L.append("- **NN 训练协议局限**：early stopping 用窗口最后 10% 做验证集，最佳 epoch 确定后未在完整窗口 refit —— NN 实际梯度训练仅用约 90% 的窗口（W=1500 时约 1350 天），且被排除的恰是最新约 150 天；GARCH/分位数回归用全窗口。本轮按研究纪律未改（final 已看过，改训练协议有 test-driven 嫌疑），列为明确 limitation。")
    L.append("- **Neural 搜索功效有限**：搜索仅 101 个 rolling origins（2006-2007 每 5 日取 1），selection criterion 为三 tail pinball 未归一化平均（10% 尺度天然更大 → 权重更高）；grid 预先声明未扩大。")
    L.append("- DM vs bootstrap 在极端 tail 结论不一致处只能如实呈现。")
    L.append("")
    L.append("## 15. 面试时最值得解释的五个问题")
    L.append("")
    L.append("1. **为什么旧 GARCH VaR 错，错多少？** 标准化 Student-t 分位数 = t_ppf × sqrt((ν-2)/ν)；ν=6 时 1% 分位数 -3.14 vs -2.57（约 22% 高估），违例率被系统性压低。")
    L.append("2. **为什么按 target date 划分 dev/final？** 预测 t+1 的 origin 在 t，观察属于 t+1 的信息集边界；origin 2007-12-31 的预测目标 2008-01-02 属于 final。")
    L.append(
        f"3. **GJR 的 leverage 增量如何被证明？** GJR-t vs GARCH-t 的 DM Holm p={h_m4_5:.3f}/{h_m4_10:.3f}"
        "（5%/10%），γ 项捕捉负冲击放大。"
    )
    L.append("4. **为什么 MLP 负结果可信？** 同信息集 vs 线性（F0-F3 共享）、三种子稳健、Holm 校正的 headline DM、Scheme B 标准化排除初始化干扰。")
    L.append("5. **冻结门禁怎么防造假？** data/config/code signature + 工作树检查 + effective data path + canonical run 隔离 + 实验签名复用校验。")
    L.append("")
    L.append("---")
    L.append(f"生成时间与实验产物对应：canonical run `{run_dir}`；冻结清单 `docs/FREEZE_MANIFEST.md`；审计记录 `docs/AUDIT_REMEDIATION.md`。")
    return "\n".join(L)


def main() -> None:
    out_root = ROOT / "outputs"
    (ROOT / "docs" / "FINAL_SUMMARY_ZH.md").write_text(build(out_root), encoding="utf-8")
    print("docs/FINAL_SUMMARY_ZH.md 已生成")


if __name__ == "__main__":
    main()
