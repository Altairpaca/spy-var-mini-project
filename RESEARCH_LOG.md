# Research Log

研究决策历史（非终端输出）。每个实质性方法论变更记录理由，以及是否在 final-test 结果可见之前做出。

## 2026-08-12 — 框架搭建与协议冻结（final test 之前）

### 阻塞：数据文件缺失
- 仓库中不存在 `assignment/readme.pdf` 与 `data/raw/spy_data.csv`（git 历史、远程分支、文件系统均无）。
- 已向用户报告；在等待数据期间完成全部不依赖真实数据的框架（src/tests/configs/scripts），并以合成数据验证管线。
- 数据到达后：运行 `scripts/run_all.py` 完成审计与全部实验；数据 SHA256 在冻结时记录。
- **禁止伪造数据**：所有审计数字必须来自真实 `spy_data.csv`。

### 数据划分（用户协议优先，覆盖 EXPERIMENT_SPEC 的候选方案 A/B）
- development: 所有日期 < 2008-01-01；final test: >= 2008-01-01。
- 理由：用户任务协议为最高约束；覆盖 2008 危机、2010-2012 高波动、2015-2016、2018 尖峰，final test 兼具普通样本外与压力测试作用。

### Rolling 窗口
- 候选仅 {1000, 1500} trading days（用户协议固定，不扩展网格；EXPERIMENT_SPEC 中的 2000 候选被覆盖）。
- 验证折叠：2005/2006/2007 为主验证年份。窗口 1500 的最早可行原点约在 2005-12（1500 个交易日 ≈ 6 年），因此 1500 窗口的有效验证期为 2006-01 ~ 2007-12 —— 这是文档化的轻微调整（用户协议允许"轻微调整并记录理由"）。两窗口在公共原点集上比较。
- 选择规则（预先声明，development 证据驱动）：三 tail mean pinball 之和较小者；相对差 < 1% 时选 1500（长窗口 1% tail 有效尾部样本更多，经验分位数更稳定；短窗口 regime 适应更快）。

### 特征工程（F0-F3）
- F0: 收益滞后 1/2/5/22 天、|r|、r²、5 日平均 |r|。
- F1: + log(rv5)、sqrt(rv5)、log 尺度 5d/22d HAR 聚合。
- F2: + log(bv)、bv 的 5d/22d HAR 聚合。
- F3: + jump = max(rv5-bv, 0)、jump/rv5、5 日负收益冲击强度 down_mean_5d。
- 因果性：特征在窗口切片上计算（min_periods 强制），训练行从窗口起点 + max_lag(22) 天开始 —— 不存在"窗口外历史"参与拟合的模糊地带；截断不变性测试锁定。

### 统一滚动引擎语义
- 预测原点 t 的窗口 = [t-W+1, t]（严格 W 行）；特征行 s 的标签 = r_{s+1}；窗口最后一行（origin）只用于预测。
- 所有模型每日全量重拟合；NN 的 scaler 仅用训练子集（窗口最后 10% 行作 early-stopping 验证集，不参与 scaler/梯度）。
- GRU 序列完全落在窗口内（训练行 s >= 窗口起点 + pad + L - 1）—— 曾发现 `feat[-L+1:1]` 负索引取到窗口尾部数据的实现 bug（由测试捕获），已修复。
- 失败记录：fit_status 列（non_convergence/nan_output/failed:*），绝不静默。

### 统计检验实现
- Kupiec LR_uc、Christoffersen LR_ind/LR_cc、DQ（Engle-Manganelli，附加诊断）、DM（Newey-West HAC）、移动块 bootstrap（配对 t 统计量，中心化处理）。
- 验证：已知数值手算对照 + H0 下 p 值均匀性蒙特卡洛（n=2000 保证渐近近似；n=600 时 Kupiec LR 有已知有限样本正偏，文档化）。
- bootstrap 曾缺中心化导致 p 值失效（测试捕获），已修复。
- 有限样本警告：1% tail 在 1000/1500 窗口中的期望违例数仅 10/15，覆盖检验功效低 —— 报告中将明确说明。

### 性能
- 本机 88 核；OMP/MKL/OPENBLAS 线程统一限制为 1；并行经 joblib 进程池（默认 16 workers，可配置）。
- 合成 smoke test 验证 8 workers 全链路；真实数据上按用户要求 benchmark 16/32/64。

### GPU
- 当前节点 nvidia-smi 报 "Driver/library version mismatch"，CUDA 不可用；NN 全部 CPU 训练（模型极小，参数量 ~10³-10⁴，CPU 足够）。已记录；若环境修复可切换 device=cuda。

### 首次 final test 前的冻结清单（对应任务 §13）
- [x] 统一引擎与 schema
- [x] 全部自动测试（leakage/alignment/rolling/gate/统计检验）
- [ ] development 实验（窗口对比、NN 搜索）—— 数据到达后
- [ ] 窗口/特征/超参选择记录（本文件追加）
- [ ] configs/final.yaml 定稿 + freeze_final.py + 冻结 commit

### 2026-08-13 — 审计修复决策记录（final test 之前）
- Student-t VaR 尺度 bug：arch innovation 为方差 1 标准化 t；分位数用 StudentsT().ppf
  （等价 scipy t × sqrt((nu-2)/nu)）。旧 M1/M4 全部 final/dev 结果 INVALIDATED。
- target-date 分区：dev/final 按被预测日期划分（target_date < 2008-01-01 为 dev）。
- MLP/GRU 方案 B：train-only target 标准化（初始 softplus gap ~0.693 远超日收益
  分位差量级）；仿射保序，不改变 non-crossing 结构。未扩大网络规模。
- 窗口选择聚合：每 (model, feature-set, tail) 候选相对归一化损失等权平均；
  报告 per-cell winner / win count / drop-one sensitivity；不一致时保守选长窗口。
- DM headline 族（M1-M3, M2-M3, M3-M5, M1-M4）逐 tail Holm 校正；全矩阵为附录。
- seed robustness：primary 42 + {7, 2026} => n_seeds=3，mean/std 汇总。
- DQ 保留为 auxiliary diagnostic（已有 power/size 测试）。
- 冻结门禁：data/config/code signature + working tree clean + effective data path。
- 正式 final 输出隔离到 outputs/runs/<freeze_id>/；artifact 复用仅限签名一致。
### 2026-08-12 — 仓库初始状态
- main 分支两个初始 commit；README/AGENTS/EXPERIMENT_SPEC/ENVIRONMENT/RESULTS_SCHEMA/RESEARCH_LOG 等 bootstrap 文档齐全。
- 本日志初始段由 bootstrap 写入（2026-08-11 条目为模板遗留，保留作为历史）。

## 2026-08-11 — Project initialization（bootstrap 遗留）
### Assignment interpretation
The task is one-day-ahead SPY VaR forecasting at 1%, 5%, and 10% tails with a strict rolling-window evaluation. At least one non-neural and one neural method are required.

### Initial methodological stance
- Treat leakage prevention and temporal alignment as first-class research requirements.
- Use a model ladder: Historical Simulation -> GARCH-t -> direct conditional quantile model -> small neural quantile model.
- Evaluate calibration and quantile loss; do not rank models by failure rate alone.
- Prefer feature/information ablations over a large architecture zoo.
- Do not require neural methods to win.

### Compute decision
Primary environment: altair-server (256 CPU cores, 128 GB RAM, RTX 3060, Codex available). SUPER-26 is optional batch compute only.

### Open decisions before coding the final experiment
- final train/validation/test split;
- primary rolling-window length;
- direct-quantile baseline choice;
- neural architecture (MLP-QR vs GRU-QR);
- neural retraining schedule;
- exact statistical test implementation and significance reporting.
