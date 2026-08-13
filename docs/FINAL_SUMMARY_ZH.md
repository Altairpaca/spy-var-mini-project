# 中文审计摘要（FINAL_SUMMARY_ZH）

> 本文件为研究者本人审计与面试准备材料，非正式提交文档。所有数字由脚本从 canonical frozen run 产物重算（`scripts/generate_final_summary.py`），与英文报告一致。

## 0. 本轮审计发现了什么（为什么旧 final 全部失效）

> 本节为历史审计记录（旧实现修复前的事实），数字引用自 `docs/AUDIT_REMEDIATION.md`，非当前实验结果。

完整记录见 `docs/AUDIT_REMEDIATION.md`。三个 correctness 问题：

1. **Student-t VaR 尺度 bug（P0）**：旧实现直接用 `scipy.stats.t.ppf(alpha, nu)`，而 arch 的 Student-t innovation 是方差 1 的标准化分布。
   正确分位数 = `t_ppf(alpha, nu) × sqrt((nu-2)/nu)`（= arch `StudentsT().ppf`）。在拟合自由度 nu≈5-12 时旧 VaR 高估幅度约 10-25%，
   导致旧 GARCH-t / GJR-t 违例率系统性偏低（1% tail 旧值 0.76% vs 修正后 1.5%）。因此旧 M1/M4 的全部 failure rate、Kupiec、
   Christoffersen、DQ、pinball、DM、bootstrap、regime 分层与'每 tail 最佳模型'结论一律 INVALIDATED BY AUDIT。
2. **target-date 分区 bug（P0）**：旧实现按 forecast origin 日期划分 dev/final，2007-12-31 的 origin（target 2008-01-02）被错归 development；
   修正为按 target date 划分后，final 从 2640 变为 2641 个预测日（含 2008-01-02）。
3. **provenance 缺口（P1）**：旧冻结不校验代码签名与工作树、--data 可绕过冻结数据、产物可被静默复用、neural search → final.yaml 链路无强制 artifact。
   本轮全部修复（见 AUDIT_REMEDIATION §2.4-2.10）。

修复过程中**未**根据旧 final 结果做任何模型/特征/窗口/超参数选择；所有选择来自 development-only 证据或 correctness 要求。

## 1. 最终实验协议（修正后）

- 数据：SPY 日度 log_ret / rv5 / bv，4640 行（2000-01-04 ~ 2018-06-27），SHA256 277406a832c1418d... 冻结。
- development 期：target date < 2008-01-01，rolling-origin 验证（公共目标 498 个，2006-01-06 ~ 2007-12-28）。
- final test（freeze manifest 记录的 git_commit e34928235ed6）：target date >= 2008-01-01，2641 个预测日（2008-01-02 ~ 2018-06-27），全部模型同日期集。
- 冻结内容：primary window=1500（等权归一化聚合，24 cells 中 23 个偏好 1500）、M3 hidden [16] / lr 0.001 / wd 0.0001 / batch 128、
  M5 hidden 16 / lr 0.001 / wd 0.0 / batch 64、primary seed 42、robustness seeds {7, 2026}、config SHA256 058d5aea59044889...、data SHA256 277406a832c1418d...
- 零泄漏约束：特征/标准化/early-stopping 全部限制在训练窗口内（截断不变性测试锁定）；MLP/GRU target 采用 train-only 标准化（Scheme B）。
- 冻结门禁（强化）：data/config/code signature + working tree clean（freeze 时）+ effective data path 校验；正式产物隔离于 `outputs/runs/<freeze_id>/`，复用仅限签名一致。

## 2. 为什么选择 1500 日窗口（等权聚合，审计修复）

- 决策规则（预先声明）：每 (model, feature-set, tail) 单元内候选相对归一化损失等权平均；报告 per-cell winner、win count、drop-one sensitivity；聚合不一致时保守选长窗口。
- 证据（outputs/development/window_decision.json）：24 个单元中 23 个偏好 1500（等权平均归一化损失 0.962 vs 1.038）；raw pinball sum 一致（无分歧）；drop-one sensitivity 24/24 仍选 1500。
- 1% tail 校准：1500 窗口 |failure rate - 1%| 更小 —— 长窗口有效尾部样本更多（约 15 vs 10 个期望违例），经验分位数更稳定。

## 3. 模型含义（数学/经济学）

- M0 HS：窗口内历史收益经验分位数；无参数、隐含收益分布平稳假设，regime 突变时反应滞后。
- M1 GARCH(1,1)-t：σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}；z_t 为方差 1 标准化 Student-t（分位数 = t_ppf × sqrt((ν-2)/ν)）；VaR = μ + F_t⁻¹(α)·σ。
- M2 线性分位数回归：pinball 损失 min_β Σ ρ_τ(y_{t+1} - x_tᵀβ)，与 MLP 共享特征；独立拟合允许交叉（crossing rate 报告）。
- M3 MLP：joint pinball + 有序输出头 q_0.05 = q_0.01 + softplus(g1)、q_0.10 = q_0.05 + softplus(g2)，结构非交叉；train-only target 标准化（Scheme B）。
- M4 GJR-GARCH(1,1,1)-t：σ²_t = ω + α·ε²_{t-1} + γ·I(ε<0)·ε²_{t-1} + β·σ²_{t-1}（leverage）；VaR 构造同 M1。
- M5 GRU：最近 22 日特征序列 → 隐状态 → 有序分位数头；序列建模稳健性扩展（非替代 MLP）。

## 4. 各 tail 最佳模型（修正后冻结样本外，2641 日）

- **1% tail**：经验违例率最接近名义水平 = GJR-t（0.0144 vs 目标 0.01）；pinball 最低 = GJR-t（0.00034）。
- **5% tail**：经验违例率最接近名义水平 = HS（0.0530 vs 目标 0.05）；pinball 最低 = GJR-t（0.00126）。
- **10% tail**：经验违例率最接近名义水平 = MLP（0.0969 vs 目标 0.10）；pinball 最低 = GJR-t（0.00204）。

## 5. failure rate 与覆盖检验（修正后）

| 模型 | 1% 违例率 | 5% 违例率 | 10% 违例率 | Kupiec(1%) p | Ind(1%) p | CC(1%) p |
|---|---|---|---|---|---|---|
| HS | 0.0155 | 0.0530 | 0.0966 | 0.008 | 0.000 | 0.000 |
| GARCH-t | 0.0151 | 0.0629 | 0.1136 | 0.014 | 0.148 | 0.017 |
| LinQR | 0.0216 | 0.0610 | 0.1125 | 0.000 | 0.041 | 0.000 |
| MLP | 0.0231 | 0.0576 | 0.0969 | 0.000 | 0.065 | 0.000 |
| GJR-t | 0.0144 | 0.0632 | 0.1079 | 0.033 | 0.577 | 0.089 |
| GRU | 0.0363 | 0.0674 | 0.1352 | 0.000 | 0.000 | 0.000 |

- 修正后 GARCH 族 1% 违例率 0.0151 / 0.0144，**高于**名义 1%（即 mildly under-conservative：VaR 不够负、违例偏多，Kupiec p=0.014 / 0.033）；HS 5%/10% 频率最接近目标但独立性检验 p=0.000（违例强聚集）。
- 条件充分性（逐模型）：仅 GJR-t 的 1% CC 在 5% 水平不拒绝（p=0.089），GARCH-t 1% CC p=0.017 拒绝；5% CC 两模型均拒绝（p=0.007 / 0.008）；10% CC GARCH-t 拒绝（p=0.036）、GJR-t 不拒绝（p=0.074）。正确频率 ≠ 正确条件 VaR 动态。

## 6. pinball loss 对比（修正后）

- 1% tail：GJR-t(0.00034) < GARCH-t(0.00036) < MLP(0.00043) < LinQR(0.00052) < HS(0.00062) < GRU(0.00068)
- 5% tail：GJR-t(0.00126) < GARCH-t(0.00129) < MLP(0.00131) < LinQR(0.00133) < GRU(0.00150) < HS(0.00169)
- 10% tail：GJR-t(0.00204) < GARCH-t(0.00207) < LinQR(0.00208) < MLP(0.00219) < GRU(0.00224) < HS(0.00244)

## 7. 特征消融（F0 returns → F1 +RV → F2 +BV → F3 +jump/downside block）

- LinQR 各 tail pinball（F0→F3）：
  - 1% tail：F0(0.00045) → F1(0.00045) → F2(0.00045) → F3(0.00052)
  - 5% tail：F0(0.00138) → F1(0.00132) → F2(0.00132) → F3(0.00133)
  - 10% tail：F0(0.00216) → F1(0.00206) → F2(0.00207) → F3(0.00208)
- MLP 各 tail pinball（F0→F3）：
  - 1% tail：F0(0.00050) → F1(0.00046) → F2(0.00041) → F3(0.00043)
  - 5% tail：F0(0.00137) → F1(0.00134) → F2(0.00130) → F3(0.00131)
  - 10% tail：F0(0.00221) → F1(0.00222) → F2(0.00219) → F3(0.00219)

- **RV 增量**（F0→F1，pinball 相对变化 %，正 = 改善）：
  - LinQR：1% +0.1%、5% +4.4%、10% +4.8% —— 5%/10% tail 有改善（1% 近似为零）。
  - MLP：1% +8.2%、5% +2.0%、10% -0.6% —— 1%/5% tail 有改善（10% 近似为零）。
- **BV 条件增量**（F1→F2，正 = 改善）：
  - LinQR：1% -1.3%、5% +0.1%、10% -0.5% —— RV 之后对线性分位数回归几乎没有增量。
  - MLP：1% +10.9%、5% +2.8%、10% +1.5% —— 尤其 1% tail 有明显增量。但这是**信息增量 ≠ 模型层面超越**：MLP 加 BV 后仍未超过 GARCH 族。
- **F3 jump/downside block 增量**（F2→F3，正 = 改善）：LinQR 1% -14.7%、5% -1.0%、10% -0.4%；MLP 1% -3.9%、5% -0.7%、10% -0.1% —— 各 model×tail 无稳定正增量，整块效果不单独归因 jump。

## 8. 危机与 regime 发现（修正后，5% tail failure rate）

- crisis_2008_2009：HS=0.156、GARCH-t=0.077、LinQR=0.077、MLP=0.071、GJR-t=0.079、GRU=0.133
- elevated_2010_2012：HS=0.034、GARCH-t=0.069、LinQR=0.066、MLP=0.054、GJR-t=0.065、GRU=0.050
- calm_2013_2014：HS=0.002、GARCH-t=0.056、LinQR=0.058、MLP=0.046、GJR-t=0.056、GRU=0.054
- stress_2015_2016：HS=0.036、GARCH-t=0.060、LinQR=0.050、MLP=0.065、GJR-t=0.067、GRU=0.048
- calm_2017：HS=0.012、GARCH-t=0.024、LinQR=0.024、MLP=0.020、GJR-t=0.020、GRU=0.028
- spike_2018：HS=0.106、GARCH-t=0.089、LinQR=0.098、MLP=0.114、GJR-t=0.089、GRU=0.122

- 观察 HS 的双向 regime 适应滞后：危机开始违例率过高（风险反应慢），危机结束进入平静期违例率过低（历史危机观测仍滞留在滚动尾部）—— 是 **slow two-sided regime adaptation**，而非单纯危机低估。

## 9. DM / bootstrap 结论（Holm 校正，headline 族）

- GARCH-t vs MLP @1%：DM=-2.06 (raw p=0.0395, Holm p=0.0790, bootstrap p=0.090, favors a)
- GARCH-t vs GJR-t @1%：DM=2.26 (raw p=0.0235, Holm p=0.0705, bootstrap p=0.042, favors b)
- LinQR vs MLP @1%：DM=1.34 (raw p=0.1803, Holm p=0.1803, bootstrap p=0.472, favors b)
- MLP vs GRU @1%：DM=-2.88 (raw p=0.0040, Holm p=0.0160, bootstrap p=0.227, favors a)
- GARCH-t vs MLP @5%：DM=-0.50 (raw p=0.6148, Holm p=1.0000, bootstrap p=0.601, favors a)
- GARCH-t vs GJR-t @5%：DM=3.16 (raw p=0.0016, Holm p=0.0063, bootstrap p=0.001, favors b)
- LinQR vs MLP @5%：DM=0.52 (raw p=0.6054, Holm p=1.0000, bootstrap p=0.697, favors b)
- MLP vs GRU @5%：DM=-2.22 (raw p=0.0266, Holm p=0.0799, bootstrap p=0.305, favors a)
- GARCH-t vs MLP @10%：DM=-2.60 (raw p=0.0093, Holm p=0.0227, bootstrap p=0.012, favors a)
- GARCH-t vs GJR-t @10%：DM=2.77 (raw p=0.0057, Holm p=0.0227, bootstrap p=0.003, favors b)
- LinQR vs MLP @10%：DM=-2.73 (raw p=0.0064, Holm p=0.0227, bootstrap p=0.025, favors a)
- MLP vs GRU @10%：DM=-0.69 (raw p=0.4886, Holm p=0.4886, bootstrap p=0.699, favors a)

- GJR-t（M4）vs GARCH-t：5% Holm p=0.0063（显著，bootstrap p=0.001）、10% Holm p=0.0227（显著，bootstrap p=0.003）、1% Holm p=0.0705（方向性证据，不显著）—— 非对称波动率设定带来预测损失改进，与 leverage/asymmetry 效应 consistent（本实验不做参数级 leverage 识别）。
- MLP vs GARCH-t：10% Holm p=0.0227（显著更差）、1% Holm p=0.0790（不显著）、5% Holm p=1.0000（不显著）—— 负结果有 tail 依赖。
- DM 与 bootstrap 不一致处（如 1% tail 的 GARCH-t vs MLP：raw DM p 显著但 bootstrap p=0.09）如实报告为 inference sensitive to dependence/finite-sample procedure。

## 10. Neural 是否有 absolute incremental value？

- **没有绝对增量**：MLP 三 tail pinball 均高于 GARCH 族；对 GARCH-t 的劣势仅 10% tail Holm 显著（p=0.023），1%/5% tail 不显著（p=0.079 / 1.000）。
- 修正后 MLP 与线性差距缩小（5% tail 不显著）：Scheme B 标准化改善了 MLP 训练；joint loss + non-crossing 是方法差异而非完美纯结构控制。
- GRU：1% tail 严重过度违例（0.0398 ± 0.0034，seed std 最大）—— seed sensitivity 如实报告。

## 11. seed robustness（n_seeds=3：42 primary + 7/2026）

| 模型 | tail | failure mean ± std | pinball mean ± std |
|---|---|---|---|
| MLP | 1% | 0.0321 ± 0.0078 | 0.00046 ± 0.00003 |
| MLP | 5% | 0.0579 ± 0.0017 | 0.00132 ± 0.00002 |
| MLP | 10% | 0.0972 ± 0.0023 | 0.00220 ± 0.00003 |
| GRU | 1% | 0.0398 ± 0.0034 | 0.00070 ± 0.00003 |
| GRU | 5% | 0.0690 ± 0.0017 | 0.00149 ± 0.00001 |
| GRU | 10% | 0.1279 ± 0.0067 | 0.00224 ± 0.00001 |

## 12. 假设评级（H1-H7，修正后冻结证据）

| 假设 | Verdict | 关键证据 |
|---|---|---|
| H1 HS regime adaptation lag | strongly supported | 2008-2009 违例率远超目标、平静期（2013-14/2017）违例率极低 —— 双向滞后（slow two-sided regime adaptation） |
| H2 GARCH 型波动率模型改善 VaR | supported, calibration mixed | proper loss 三 tail 最低；但覆盖并非全面通过（1% UC 拒绝、5% CC 拒绝） |
| H3a RV 增量 | supported（tail 依赖） | F0→F1：LinQR 5%/10% 改善、MLP 1%/5% 改善（见 §7 动态数字） |
| H3b BV 条件增量 | model-dependent | LinQR ≈0 或略负；MLP 各 tail 均有改善（1% 最明显，+10.9%）—— 无 model-agnostic 一致增量 |
| H4 MLP 绝对增量 | unsupported | 三 tail pinball 均不优于 GARCH 族；仅 10% Holm 显著更差（p=0.023） |
| H5 非线性映射价值 | unsupported / inconclusive | 10% tail 显著更差（Holm p=0.0227）；1%/5% 无清晰优势；joint loss+non-crossing 使其非纯 mapping control |
| H6 tail 依赖的相对表现 | supported | 校准与 DM 差异随 tail 明显变化（GJR 5/10% 显著、1% 方向性） |
| H7 jump/downside block 极端 tail 增量 | not identified | F3 为联合 block 且各 model×tail 无稳定正增量；无 component identification |

## 13. 哪些结论可靠，哪些只是描述性

**可靠**（检验功效充分）：GARCH 族分位数预测损失优势（三 tail proper quantile loss 最低）；GJR vs GARCH-t 的 leverage 增量（Holm 校正显著）；HS 违例聚集（Ind p<0.001）；MLP 10% tail 显著劣于 GARCH-t/LinQR（Holm p=0.023，1%/5% 非显著属证据不足而非等价）；危机期 HS 滞后。

**描述性/有限样本**：1% tail 各检验功效低（期望违例 ~26 个）；5% tail MLP vs GARCH-t 不显著属于证据不足而非等价；bootstrap 与 DM 不一致处；F3 整块效果的成分归因。

## 14. 项目主要局限

- 1% tail 期望违例少，覆盖检验功效低；单一资产（SPY ETF）；2018 数据仅至 6 月；rv5/bv 为日度聚合。
- NN 计算成本约为经典模型 10 倍且无绝对增量；GRU 1% tail seed sensitivity（std 0.0034）。
- **NN 训练协议局限**：early stopping 用窗口最后 10% 做验证集，最佳 epoch 确定后未在完整窗口 refit —— NN 实际梯度训练仅用约 90% 的窗口（W=1500 时约 1350 天），且被排除的恰是最新约 150 天；GARCH/分位数回归用全窗口。本轮按研究纪律未改（final 已看过，改训练协议有 test-driven 嫌疑），列为明确 limitation。
- **Neural 搜索功效有限**：搜索使用 2006-2007 每 5 个交易日取 1 的稀疏验证原点，selection criterion 为三 tail pinball 未归一化平均（10% 尺度天然更大 → 权重更高）；grid 预先声明未扩大。
- DM vs bootstrap 在极端 tail 结论不一致处只能如实呈现。

## 15. 面试时最值得解释的五个问题

1. **为什么旧 GARCH VaR 错，错多少？** 标准化 Student-t 分位数 = t_ppf × sqrt((ν-2)/ν)；ν=6 时 1% 分位数 -3.14 vs -2.57（约 22% 高估），违例率被系统性压低。
2. **为什么按 target date 划分 dev/final？** 预测 t+1 的 origin 在 t，观察属于 t+1 的信息集边界；origin 2007-12-31 的预测目标 2008-01-02 属于 final。
3. **GJR 的 leverage 增量如何被证明？** GJR-t vs GARCH-t 的 DM Holm p=0.006/0.023（5%/10%），γ 项捕捉负冲击放大。
4. **为什么 MLP 负结果可信？** 同信息集 vs 线性（F0-F3 共享）、三种子稳健、Holm 校正的 headline DM、Scheme B 标准化排除初始化干扰。
5. **冻结门禁怎么防造假？** data/config/code signature + 工作树检查 + effective data path + canonical run 隔离 + 实验签名复用校验。

---
生成时间与实验产物对应：canonical run `outputs/runs/e34928235ed6-058d5aea`；冻结清单 `docs/FREEZE_MANIFEST.md`；审计记录 `docs/AUDIT_REMEDIATION.md`。