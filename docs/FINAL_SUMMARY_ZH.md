# 中文审计摘要（FINAL_SUMMARY_ZH）

> 本文件为研究者本人审计与面试准备材料，非正式提交文档。所有数字由脚本从 `outputs/` 产物重算（`scripts/generate_final_summary.py`），与英文报告一致。

## 1. 最终实验协议

- 数据：SPY 日度 log_ret / rv5 / bv，4640 行（2000-01-04 ~ 2018-06-27），SHA256 冻结于 FREEZE_MANIFEST。
- development 期：< 2008-01-01，rolling-origin 验证（公共原点 499 个，2006-01-06 ~ 2007-12-31；1500 窗口的验证期较 2005-2007 有文档化轻微调整）。
- final test（冻结）：>= 2008-01-01，2640 个预测原点（2008-01-02 ~ 2018-06-27），全部模型同日期集。
- 冻结内容：primary window=1500、M3 hidden [32,32] / lr 1e-3 / wd 1e-4、M5 hidden 32 / lr 1e-3 / wd 1e-4、primary seed 42、robustness seeds {7, 2026}、config SHA256 ed67a634、data SHA256 277406a8。
- 零泄漏约束：特征/标准化/early-stopping 全部限制在训练窗口内（截断不变性测试锁定）；final test 首次运行前完成冻结 commit（db29cf0）。

## 2. 为什么选择 1500 日窗口

- 决策规则（预先声明）：公共验证原点上三 tail mean pinball 之和较小者；相对差 > 1% 时直接选较小者。
- 证据：1000 窗口 pinball 总和 0.02431 vs 1500 窗口 0.02338（相对差 3.96%）；8/8 模型配置全部偏好 1500。
- 1% tail 校准：1500 窗口 |failure rate - 1%| = 1.0%，1000 窗口 2.5% —— 长窗口有效尾部样本更多（约 15 vs 10 个期望违例），经验分位数更稳定。
- 短窗口（1000）regime 适应更快的论点在相对平静的 2006-2007 验证期未转化为 pinball 优势；10% tail 覆盖率 1500 稍差（偏差 1.9% vs 0.8%），但总 pinball 仍偏好 1500。

## 3. 模型含义（数学/经济学）

- M0 HS：窗口内历史收益经验分位数；无参数、无模型假设，但隐含“收益分布平稳”假设，regime 突变时反应滞后。
- M1 GARCH(1,1)-t：sigma²_t = omega + alpha·eps²_{t-1} + beta·sigma²_{t-1}，Student-t 创新捕捉厚尾；VaR = mu + t_ppf(alpha, nu)·sigma。经济学含义：波动率集聚（beta 高）+ 冲击衰减（alpha）。
- M2 线性分位数回归：对 1/5/10% 各拟合线性 pinball 回归（与 MLP 共享特征），HAR 型 RV/BV 聚合体现波动率长记忆。
- M3 MLP：joint pinball loss + softplus-gap 有序输出头（结构非交叉）；非线性映射 + 相同信息集。
- M4 GJR-GARCH(1,1,1)-t：增加 gamma·1[eps<0]·eps² 项，捕捉 leverage（负冲击放大波动）。
- M5 GRU：最近 22 日特征序列 → 隐状态 → 有序分位数头；检验显式序列建模在 HAR 特征之外的增量。

## 4. 各 tail 最佳模型（冻结样本外，2640 日）

- **1% tail**：校准最佳 = MLP（failure rate 0.0098）；pinball 最低 = GJR-t（0.00035）。
- **5% tail**：校准最佳 = HS（failure rate 0.0530）；pinball 最低 = GJR-t（0.00126）。
- **10% tail**：校准最佳 = HS（failure rate 0.0962）；pinball 最低 = GJR-t（0.00205）。

## 5. failure rate 与覆盖检验

| 模型 | 1% 违例率 | 5% 违例率 | 10% 违例率 | Kupiec(1%) p | Ind(1%) p | CC(1%) p |
|---|---|---|---|---|---|---|
| HS | 0.0155 | 0.0530 | 0.0962 | 0.008 | 0.000 | 0.000 |
| GARCH-t | 0.0076 | 0.0428 | 0.0814 | 0.191 | 0.580 | 0.365 |
| LinQR | 0.0216 | 0.0610 | 0.1125 | 0.000 | 0.041 | 0.000 |
| MLP | 0.0098 | 0.0674 | 0.1269 | 0.937 | 0.000 | 0.000 |
| GJR-t | 0.0072 | 0.0424 | 0.0822 | 0.128 | 0.600 | 0.273 |
| GRU | 0.0015 | 0.0455 | 0.1205 | 0.000 | 0.003 | 0.000 |

- GARCH 族（M1/M4）三 tail 均未被 Kupiec 拒绝（1% p=0.19/0.13）；MLP 在 5%/10% 显著过度违例（6.7%/12.7%），GRU 在 1% 严重过度保守（0.15%）。
- 违例聚集：HS 的独立性检验 p<0.001（危机期连续违例）；GARCH 族无聚集证据（Ind p>0.1）。

## 6. pinball loss 对比

- 1% tail：GJR-t(0.00035) < GARCH-t(0.00036) < LinQR(0.00052) < GRU(0.00053) < HS(0.00062) < MLP(0.00104)
- 5% tail：GJR-t(0.00126) < GARCH-t(0.00130) < LinQR(0.00133) < GRU(0.00137) < HS(0.00169) < MLP(0.00259)
- 10% tail：GJR-t(0.00205) < LinQR(0.00208) < GARCH-t(0.00208) < GRU(0.00225) < HS(0.00244) < MLP(0.00425)

## 7. 特征消融（F0 returns → F3 +RV+BV+jump+downside）

- LinQR 5% tail pinball：F0(0.00138) → F1(0.00132) → F2(0.00132) → F3(0.00133)
- MLP 5% tail pinball：F0(0.00387) → F1(0.00330) → F2(0.00344) → F3(0.00259)
- 线性模型（M2）：F0→F3 pinball 降约 3.5%（RV/BV 信息小幅增量）；MLP（M3）F0→F3 降约 33%（非线性模型更依赖 RV 信息）但绝对水平仍落后线性基线。
- 结论：RV/BV/jump 信息有增量但不足以改变模型排序；BV 相对 RV 的边际增量很小。

## 8. 危机与 regime 发现（5% tail failure rate）

- crisis_2008_2009：HS=0.157、GARCH-t=0.063、LinQR=0.077、MLP=0.081、GJR-t=0.063、GRU=0.062
- elevated_2010_2012：HS=0.034、GARCH-t=0.042、LinQR=0.066、MLP=0.056、GJR-t=0.045、GRU=0.052
- calm_2013_2014：HS=0.002、GARCH-t=0.042、LinQR=0.058、MLP=0.071、GJR-t=0.042、GRU=0.032
- stress_2015_2016：HS=0.036、GARCH-t=0.034、LinQR=0.050、MLP=0.056、GJR-t=0.030、GRU=0.040
- calm_2017：HS=0.012、GARCH-t=0.016、LinQR=0.024、MLP=0.076、GJR-t=0.016、GRU=0.028
- spike_2018：HS=0.106、GARCH-t=0.057、LinQR=0.098、MLP=0.098、GJR-t=0.049、GRU=0.057

- **HS 在 2008-2009 危机期严重滞后**（违例率 15.7%，接近 3 倍目标）—— regime adaptation 慢的直接证据；GARCH 族同期 6.4%。
- 2018 尖峰：GJR-t 4.9%、GARCH-t 5.7% 明显优于 HS 10.6% —— GARCH 对波动率突升反应更快。
- 平静期（2013-2014、2017）各模型接近目标；HS 甚至过度保守（0.2%-1.2%）。

## 9. NN 是否真正提供增量？

- **没有**。DM 检验：M1 vs M3 三 tail p<0.001（favors GARCH-t）；M2 vs M3 三 tail p<0.001（favors 线性）。block bootstrap p 值（0.10-0.15）因尾部损失矩弱而功效不足，但方向一致。
- M1 vs M5 在 1% tail 显著（DM=-9.87, p<0.001, bootstrap p=0.001）—— GRU 的 1% 预测过度保守。
- seed robustness：M3/M5 跨种子 failure rate std <= 0.9%、pinball std 小 —— 主 seed 结果有代表性，负结果不是种子运气。
- 解释：日度 SPY 的条件分位数结构近似可被线性/GARCH 参数化捕捉；1500 日窗口 × 17 特征的样本量对非线性映射的学习收益有限。

## 10. 哪些结论统计上可靠，哪些只是有限样本迹象

**可靠**（检验功效充分或方向一致）：
- GARCH 族在 5%/10% tail 与 HS 的差异（DM p<0.001，2640 日样本）；MLP 落后于线性基线（DM p<0.001）。
- 危机期 HS 滞后（2008-2009 违例率 15.7% vs 6.4%，样本内 500 日）。
- 无违例聚集的 GARCH 属性（Ind 检验 p>0.1）。

**有限样本迹象**（功效不足，仅作方向参考）：
- 1% tail 的 Kupiec/CC 检验（期望违例仅 ~26 个；M1 vs M2 在 1% 的 DM p=0.056）。
- block bootstrap 对 M1 vs M3 的 p 值（0.10-0.15）—— 尾部损失矩较弱，DM 渐近近似更可靠。
- M3 在 1% 的校准（0.98%）与 M1/M4 的差异在噪声范围内。

## 11. 项目主要局限

- 1% tail 期望违例数少（约 2.6 个/年），单模型覆盖检验功效低。
- NN 每日重训 300 epochs（early stopping），计算量 10 倍于经典模型；信息增益为负时成本不可忽视。
- rv5/bv 为日度聚合，未利用日内路径；jump 代理基于 rv5-bv 的简单差。
- 2018 年数据仅至 6 月，spike regime 样本短。
- 单一资产（SPY ETF）；结论外推到个股/其他市场需谨慎。

## 12. 下一步最值得做的事

1. **GARCH-X / realized GARCH**：把 log(rv5) 作为外生变量加入波动率方程（Hansen et al. 2012），检验 RV 信息在参数化框架内的增量（我们仅在线性/非线性分位数回归中检验了 RV）。
2. **CAViaR / 半参数动态分位数**：Engle-Manganelli 框架直接建模分位数过程，比较参数化波动率 vs 直接分位数路线。
3. **多资产扩展**：在更多 ETF/个股上验证 GARCH 族占优与 NN 负结果是否稳健。
4. **更好的 NN 正则**：若坚持神经网络路线，用 walk-forward 重训练 + 更大样本（更长历史或多资产池）再评估。
5. **expectile / ES 联合预测**：VaR 之外扩展到期望短缺（ES），与分位数联合建模。

---
生成时间与实验产物对应：`outputs/tables/metrics.csv`、`dm_comparison.csv`、`regime_metrics.csv`、`ablation.csv`、`seed_robustness_summary.csv`；冻结清单 `docs/FREEZE_MANIFEST.md`。