# Experiment Specification

Status: **FROZEN — 与实现一致；修改需走 RESEARCH_LOG 记录**

本文件记录已定稿的实验协议。所有开放决策已由用户任务协议 + development 前证据确定，
全部决策历史见 `RESEARCH_LOG.md`。首次 final test 前，最终配置与数据哈希在
`configs/final.yaml` + `docs/FREEZE_MANIFEST.md` 中冻结，gate 由测试与脚本强制。

## 1. Research question
Can conditional volatility information and nonlinear sequence models improve one-day-ahead SPY VaR forecasts relative to simple and classical statistical baselines at alpha = 1%, 5%, and 10%?

Secondary questions:
- Does realized volatility (`rv5`) add value beyond return history?
- Does bipower variation (`bv`) and a jump proxy add incremental information beyond realized volatility?
- Are gains, if any, stable across tail levels and market regimes?
- Does a neural model improve calibrated quantile forecasts, rather than merely changing failure rates?

## 2. Target
For each forecast origin t, predict the conditional return quantile for t+1:

`q_alpha,t+1` such that `P(r_t+1 <= q_alpha,t+1 | F_t) = alpha`, alpha in {0.01, 0.05, 0.10}.

Violation indicator:
`I_alpha,t+1 = 1[r_t+1 <= q_alpha,t+1]`（VaR 是收益分位数本身，左尾通常为负）。

## 3. Dataset
Daily SPY observations with `log_ret`, `rv5`, `bv`（只读输入；SHA256 冻结）。
数据审计（`docs/DATA_AUDIT.md`）在建模前由 `scripts/audit_data.py` 自动完成：
行数、日期范围、重复/单调、缺失/inf、描述统计、偏度/峰度、经验分位数、
rv5/bv 尺度、sqrt(rv5) 合理性、RV/BV 相关、自相关、jump 分布、负收益与未来波动关系。

## 4. Information set（F0-F3，因果构造，截断不变性测试锁定）
- F0: lag_ret_{1,2,5,22}, abs_ret_1, sq_ret_1, abs_ret_5d
- F1: F0 + log_rv5, rv5_scale(sqrt), log_rv5_5d, log_rv5_22d
- F2: F1 + log_bv, log_bv_5d, log_bv_22d
- F3: F2 + jump=max(rv5-bv,0), jump_rel=jump/rv5, down_mean_5d（负收益冲击强度）

## 5. Data split（已冻结）
- development: 所有日期 < 2008-01-01（rolling-origin validation，主验证年份 2005/2006/2007；
  1500 窗口最早可行原点约 2005-12，其有效验证期为 2006-01~2007-12 —— 文档化轻微调整）；
- frozen final test: >= 2008-01-01（覆盖 2008-2009 危机、2010-2012 高波动、2013-2014 平静、
  2015-2016 压力、2017 平静、2018 尖峰）。

## 6. Rolling protocol（已冻结）
- 候选窗口仅 {1000, 1500}（用户协议固定）。选择规则（预先声明）：
  公共验证原点上三 tail mean pinball 之和较小者；相对差 < 1% 时选 1500
  （长窗口 1% tail 有效尾部样本更多、经验分位数更稳定；短窗口 regime 适应更快）。
- 统一引擎：窗口 = [t-W+1, t]；特征行 s 的标签 = r_{s+1}；origin 行只用于预测；
  所有预处理在窗口训练行上拟合；每日全量重拟合；fit_status 记录失败。
- NN early stopping：窗口最后 10% 行（时间顺序）作验证集，不参与 scaler 与梯度。

## 7. Model ladder（全部实现）
- M0 Historical Simulation（经验分位数，无参数基准）
- M1 GARCH(1,1)-t（核心波动率基线；Gaussian 仅 development 诊断 M1_gauss）
- M2 Linear/HAR Quantile（statsmodels QuantReg，独立拟合 3 tail，交叉率报告）
- M3 Multi-Quantile MLP（joint pinball；softplus-gap 有序输出头，结构非交叉）
- M4 GJR-GARCH(1,1,1)-t（leverage 检验）
- M5 小型 GRU（1 层 hidden=16，seq_len=22，F3 特征序列；序列建模稳健性扩展）

## 8. Neural ablations（已冻结）
- M2/M3 共用 F0-F3，Linear vs MLP 归因于映射非线性而非信息集。
- primary seed 42 预先声明；robustness seeds {7, 2026}（M3-F3 / M5-F3）。
- 超参搜索：development-only 小空间（hidden {16,32,{32,16},{32,32}} × lr {1e-3,3e-4} ×
  wd {0,1e-4}），验证原点 2006-2007 每 5 日取 1，窗口 = window_decision.json 选定值
  （1500，窗口选择先于搜索）；选择验证期 mean pinball。
- 主结果 = primary seed 的面板；robustness 以 mean/std 报告。

## 9. Evaluation（全部实现，公式级测试锁定）
- 覆盖：n、违例数、期望违例数、经验失败率
- 检验：Kupiec LR_uc、Christoffersen LR_ind/LR_cc（chi2 近似）、
  DQ（Engle-Manganelli，附加诊断）、DM（Newey-West HAC）、移动块 bootstrap（配对 t）
- 损失：mean pinball（逐 tail）
- 结构：quantile crossing rate、violation runs、regime 分段（预定义 6 个区间）
- 有限样本警示：1% tail 期望违例数极少，检验功效低，禁止夸大数值差异。

## 10. Primary comparison rules
- 所有模型在同一冻结日期集上比较（common-panel 测试强制）。
- 逐 tail 排名；区分校准与分位数预测损失（proper quantile loss，pinball）。
- 负结果有效：不要求神经网络胜出。

## 11. Final-test freeze checklist（完成即冻结）
- [x] split dates（用户协议）
- [x] rolling window 候选与选择规则
- [x] feature definitions（F0-F3）
- [x] model families 与默认超参
- [x] retraining schedule（每日全量重拟合）
- [x] seeds（42 primary；7/2026 robustness）
- [x] evaluation metrics/tests
- [x] primary window 数值（development 后写入 final.yaml：1500，见 freeze.json）
- [x] NN 最终超参（development 搜索后写入 final.yaml：M3 hidden [16] lr 1e-3 wd 1e-4；M5 hidden 16 lr 1e-3 wd 0）
- [x] freeze_final.py 执行 + 冻结 commit（7a6359d，manifest git_commit e34928235ed6；首次 final test 前完成）
