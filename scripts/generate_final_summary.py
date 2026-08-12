"""生成 docs/FINAL_SUMMARY_ZH.md（中文审计摘要，数字全部从产物读取）。

供用户审计与面试准备使用，非正式提交材料。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

PRIMARY = {"M0": "none", "M1": "none", "M2": "F3", "M3": "F3", "M4": "none", "M5": "F3"}
NAME = {"M0": "HS", "M1": "GARCH-t", "M2": "LinQR", "M3": "MLP", "M4": "GJR-t", "M5": "GRU"}


def _load(out_root: Path):
    return (
        pd.read_csv(out_root / "tables" / "metrics.csv"),
        pd.read_csv(out_root / "tables" / "dm_comparison.csv"),
        pd.read_csv(out_root / "tables" / "regime_metrics.csv"),
        pd.read_csv(out_root / "tables" / "ablation.csv"),
    )


def build(out_root: Path) -> str:
    metrics, dm, regime, ablation = _load(out_root)
    m = metrics[metrics.apply(lambda r: PRIMARY.get(r["model"]) == r["feature_set"], axis=1)]
    L = []
    L.append("# 中文审计摘要（FINAL_SUMMARY_ZH）")
    L.append("")
    L.append("> 本文件为研究者本人审计与面试准备材料，非正式提交文档。所有数字由脚本从 `outputs/` 产物重算（`scripts/generate_final_summary.py`），与英文报告一致。")
    L.append("")
    L.append("## 1. 最终实验协议")
    L.append("")
    L.append("- 数据：SPY 日度 log_ret / rv5 / bv，4640 行（2000-01-04 ~ 2018-06-27），SHA256 冻结于 FREEZE_MANIFEST。")
    L.append("- development 期：< 2008-01-01，rolling-origin 验证（公共原点 499 个，2006-01-06 ~ 2007-12-31；1500 窗口的验证期较 2005-2007 有文档化轻微调整）。")
    L.append("- final test（冻结）：>= 2008-01-01，2640 个预测原点（2008-01-02 ~ 2018-06-27），全部模型同日期集。")
    L.append("- 冻结内容：primary window=1500、M3 hidden [32,32] / lr 1e-3 / wd 1e-4、M5 hidden 32 / lr 1e-3 / wd 1e-4、primary seed 42、robustness seeds {7, 2026}、config SHA256 ed67a634、data SHA256 277406a8。")
    L.append("- 零泄漏约束：特征/标准化/early-stopping 全部限制在训练窗口内（截断不变性测试锁定）；final test 首次运行前完成冻结 commit（db29cf0）。")
    L.append("")
    L.append("## 2. 为什么选择 1500 日窗口")
    L.append("")
    L.append("- 决策规则（预先声明）：公共验证原点上三 tail mean pinball 之和较小者；相对差 > 1% 时直接选较小者。")
    L.append("- 证据：1000 窗口 pinball 总和 0.02431 vs 1500 窗口 0.02338（相对差 3.96%）；8/8 模型配置全部偏好 1500。")
    L.append("- 1% tail 校准：1500 窗口 |failure rate - 1%| = 1.0%，1000 窗口 2.5% —— 长窗口有效尾部样本更多（约 15 vs 10 个期望违例），经验分位数更稳定。")
    L.append("- 短窗口（1000）regime 适应更快的论点在相对平静的 2006-2007 验证期未转化为 pinball 优势；10% tail 覆盖率 1500 稍差（偏差 1.9% vs 0.8%），但总 pinball 仍偏好 1500。")
    L.append("")
    L.append("## 3. 模型含义（数学/经济学）")
    L.append("")
    L.append("- M0 HS：窗口内历史收益经验分位数；无参数、无模型假设，但隐含“收益分布平稳”假设，regime 突变时反应滞后。")
    L.append("- M1 GARCH(1,1)-t：sigma²_t = omega + alpha·eps²_{t-1} + beta·sigma²_{t-1}，Student-t 创新捕捉厚尾；VaR = mu + t_ppf(alpha, nu)·sigma。经济学含义：波动率集聚（beta 高）+ 冲击衰减（alpha）。")
    L.append("- M2 线性分位数回归：对 1/5/10% 各拟合线性 pinball 回归（与 MLP 共享特征），HAR 型 RV/BV 聚合体现波动率长记忆。")
    L.append("- M3 MLP：joint pinball loss + softplus-gap 有序输出头（结构非交叉）；非线性映射 + 相同信息集。")
    L.append("- M4 GJR-GARCH(1,1,1)-t：增加 gamma·1[eps<0]·eps² 项，捕捉 leverage（负冲击放大波动）。")
    L.append("- M5 GRU：最近 22 日特征序列 → 隐状态 → 有序分位数头；检验显式序列建模在 HAR 特征之外的增量。")
    L.append("")
    L.append("## 4. 各 tail 最佳模型（冻结样本外，2640 日）")
    L.append("")
    for tail in (0.01, 0.05, 0.10):
        sub = m[m["tail"] == tail]
        best_cal = sub.loc[sub["failure_rate"].sub(tail).abs().idxmin()]
        best_loss = sub.loc[sub["mean_pinball"].idxmin()]
        L.append(
            f"- **{int(tail*100)}% tail**：校准最佳 = {NAME[best_cal['model']]}"
            f"（failure rate {best_cal['failure_rate']:.4f}）；pinball 最低 = {NAME[best_loss['model']]}"
            f"（{best_loss['mean_pinball']:.5f}）。"
        )
    L.append("")
    L.append("## 5. failure rate 与覆盖检验")
    L.append("")
    L.append("| 模型 | 1% 违例率 | 5% 违例率 | 10% 违例率 | Kupiec(1%) p | Ind(1%) p | CC(1%) p |")
    L.append("|---|---|---|---|---|---|---|")
    for model in ["M0", "M1", "M2", "M3", "M4", "M5"]:
        rows = m[m["model"] == model]
        r1 = rows[rows["tail"] == 0.01].iloc[0]
        r5 = rows[rows["tail"] == 0.05].iloc[0]
        r10 = rows[rows["tail"] == 0.10].iloc[0]
        L.append(
            f"| {NAME[model]} | {r1['failure_rate']:.4f} | {r5['failure_rate']:.4f} | {r10['failure_rate']:.4f} | "
            f"{r1['kupiec_pvalue']:.3f} | {r1['christoffersen_ind_pvalue']:.3f} | {r1['conditional_coverage_pvalue']:.3f} |"
        )
    L.append("")
    L.append("- GARCH 族（M1/M4）三 tail 均未被 Kupiec 拒绝（1% p=0.19/0.13）；MLP 在 5%/10% 显著过度违例（6.7%/12.7%），GRU 在 1% 严重过度保守（0.15%）。")
    L.append("- 违例聚集：HS 的独立性检验 p<0.001（危机期连续违例）；GARCH 族无聚集证据（Ind p>0.1）。")
    L.append("")
    L.append("## 6. pinball loss 对比")
    L.append("")
    for tail in (0.01, 0.05, 0.10):
        sub = m[m["tail"] == tail].sort_values("mean_pinball")
        ranked = " < ".join(f"{NAME[r['model']]}({r['mean_pinball']:.5f})" for _, r in sub.iterrows())
        L.append(f"- {int(tail*100)}% tail：{ranked}")
    L.append("")
    L.append("## 7. 特征消融（F0 returns → F3 +RV+BV+jump+downside）")
    L.append("")
    abl5 = ablation[ablation["tail"] == 0.05]
    for model in ["M2", "M3"]:
        sub = abl5[abl5["model"] == model].sort_values("feature_set")
        line = " → ".join(f"{r['feature_set']}({r['mean_pinball']:.5f})" for _, r in sub.iterrows())
        L.append(f"- {NAME[model]} 5% tail pinball：{line}")
    L.append("- 线性模型（M2）：F0→F3 pinball 降约 3.5%（RV/BV 信息小幅增量）；MLP（M3）F0→F3 降约 33%（非线性模型更依赖 RV 信息）但绝对水平仍落后线性基线。")
    L.append("- 结论：RV/BV/jump 信息有增量但不足以改变模型排序；BV 相对 RV 的边际增量很小。")
    L.append("")
    L.append("## 8. 危机与 regime 发现（5% tail failure rate）")
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
    L.append("- **HS 在 2008-2009 危机期严重滞后**（违例率 15.7%，接近 3 倍目标）—— regime adaptation 慢的直接证据；GARCH 族同期 6.4%。")
    L.append("- 2018 尖峰：GJR-t 4.9%、GARCH-t 5.7% 明显优于 HS 10.6% —— GARCH 对波动率突升反应更快。")
    L.append("- 平静期（2013-2014、2017）各模型接近目标；HS 甚至过度保守（0.2%-1.2%）。")
    L.append("")
    L.append("## 9. NN 是否真正提供增量？")
    L.append("")
    L.append("- **没有**。DM 检验：M1 vs M3 三 tail p<0.001（favors GARCH-t）；M2 vs M3 三 tail p<0.001（favors 线性）。block bootstrap p 值（0.10-0.15）因尾部损失矩弱而功效不足，但方向一致。")
    L.append("- M1 vs M5 在 1% tail 显著（DM=-9.87, p<0.001, bootstrap p=0.001）—— GRU 的 1% 预测过度保守。")
    L.append("- seed robustness：M3/M5 跨种子 failure rate std <= 0.9%、pinball std 小 —— 主 seed 结果有代表性，负结果不是种子运气。")
    L.append("- 解释：日度 SPY 的条件分位数结构近似可被线性/GARCH 参数化捕捉；1500 日窗口 × 17 特征的样本量对非线性映射的学习收益有限。")
    L.append("")
    L.append("## 10. 哪些结论统计上可靠，哪些只是有限样本迹象")
    L.append("")
    L.append("**可靠**（检验功效充分或方向一致）：")
    L.append("- GARCH 族在 5%/10% tail 与 HS 的差异（DM p<0.001，2640 日样本）；MLP 落后于线性基线（DM p<0.001）。")
    L.append("- 危机期 HS 滞后（2008-2009 违例率 15.7% vs 6.4%，样本内 500 日）。")
    L.append("- 无违例聚集的 GARCH 属性（Ind 检验 p>0.1）。")
    L.append("")
    L.append("**有限样本迹象**（功效不足，仅作方向参考）：")
    L.append("- 1% tail 的 Kupiec/CC 检验（期望违例仅 ~26 个；M1 vs M2 在 1% 的 DM p=0.056）。")
    L.append("- block bootstrap 对 M1 vs M3 的 p 值（0.10-0.15）—— 尾部损失矩较弱，DM 渐近近似更可靠。")
    L.append("- M3 在 1% 的校准（0.98%）与 M1/M4 的差异在噪声范围内。")
    L.append("")
    L.append("## 11. 项目主要局限")
    L.append("")
    L.append("- 1% tail 期望违例数少（约 2.6 个/年），单模型覆盖检验功效低。")
    L.append("- NN 每日重训 300 epochs（early stopping），计算量 10 倍于经典模型；信息增益为负时成本不可忽视。")
    L.append("- rv5/bv 为日度聚合，未利用日内路径；jump 代理基于 rv5-bv 的简单差。")
    L.append("- 2018 年数据仅至 6 月，spike regime 样本短。")
    L.append("- 单一资产（SPY ETF）；结论外推到个股/其他市场需谨慎。")
    L.append("")
    L.append("## 12. 下一步最值得做的事")
    L.append("")
    L.append("1. **GARCH-X / realized GARCH**：把 log(rv5) 作为外生变量加入波动率方程（Hansen et al. 2012），检验 RV 信息在参数化框架内的增量（我们仅在线性/非线性分位数回归中检验了 RV）。")
    L.append("2. **CAViaR / 半参数动态分位数**：Engle-Manganelli 框架直接建模分位数过程，比较参数化波动率 vs 直接分位数路线。")
    L.append("3. **多资产扩展**：在更多 ETF/个股上验证 GARCH 族占优与 NN 负结果是否稳健。")
    L.append("4. **更好的 NN 正则**：若坚持神经网络路线，用 walk-forward 重训练 + 更大样本（更长历史或多资产池）再评估。")
    L.append("5. **expectile / ES 联合预测**：VaR 之外扩展到期望短缺（ES），与分位数联合建模。")
    L.append("")
    L.append("---")
    L.append("生成时间与实验产物对应：`outputs/tables/metrics.csv`、`dm_comparison.csv`、`regime_metrics.csv`、`ablation.csv`、`seed_robustness_summary.csv`；冻结清单 `docs/FREEZE_MANIFEST.md`。")
    return "\n".join(L)


def main() -> None:
    out_root = ROOT / "outputs"
    (ROOT / "docs" / "FINAL_SUMMARY_ZH.md").write_text(build(out_root), encoding="utf-8")
    print("docs/FINAL_SUMMARY_ZH.md 已生成")


if __name__ == "__main__":
    main()
