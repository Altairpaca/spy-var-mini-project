# SPY VaR Mini-Project

PhD pre-screening mini-project：SPY 下一交易日对数收益的 1%/5%/10% 条件 VaR 预测，
严格 rolling-window 框架，经典方法与小型神经网络方法的公平比较。

## 研究协议摘要

- **零未来信息泄漏**：预测 t+1 只用截至 t 的信息；scaler/early-stopping/超参选择全部限制在训练窗口内。
- **development**（< 2008-01-01）：rolling-origin validation，验证年份 2005/2006/2007（1500 窗口为 2006-2007，见 RESEARCH_LOG）。
- **final test**（>= 2008-01-01）：首次运行前必须冻结（`configs/final.yaml` + `docs/FREEZE_MANIFEST.md`，gate 由测试与脚本强制）。
- **模型**：M0 Historical Simulation、M1 GARCH(1,1)-t、M2 Linear/HAR Quantile、M3 Multi-Quantile MLP（结构非交叉）、M4 GJR-GARCH-t、M5 GRU。
- **特征消融**：F0（returns）→ F1（+RV）→ F2（+BV）→ F3（+jump + downside asymmetry），M2/M3 共享信息集。
- **seeds**：primary 42（预先声明），robustness {7, 2026}；主结果对应 primary seed，非 ensemble。
- **评价**：failure rate、Kupiec、Christoffersen ind/cc、pinball loss、crossing rate、violation clustering、regime 分段、DM + block bootstrap；DQ 为附加诊断。

## 环境

```bash
uv sync                 # python 3.11 + 全部依赖（含 CPU torch）
```

要求：Python 3.11、约 4 GB 内存、CPU 即可（NN 模型极小）；GPU 可选。
BLAS 线程限制已在脚本内自动设置（OMP/MKL/OPENBLAS=1）。

## 复现

```bash
# 全量（数据审计 -> development -> NN 搜索 -> 窗口选择 -> 冻结 -> final test
#   -> seed 稳健性 -> 统一评估 -> 图表 -> PDF 报告）
python scripts/run_all.py --config configs/final.yaml

# 报告再生（不重训模型，从 outputs 产物直接生成）
python scripts/make_report.py --config configs/final.yaml
```

自动测试：

```bash
python -m pytest tests/
```

## 产物

```text
outputs/predictions/   每实验预测面板（parquet）+ manifest（json）
outputs/tables/        指标/对比/审计表（csv/json）
outputs/figures/       论文级图表（png）
outputs/manifests/     冻结清单与数据哈希
docs/DATA_AUDIT.md     数据审计（自动生成）
docs/FREEZE_MANIFEST.md final-test 冻结清单（自动生成）
docs/FINAL_SUMMARY_ZH.md 中文审计摘要
report/final_report.pdf 英文正式报告（自动生成）
```

所有报告数字由脚本从产物重算，禁止手工录入。

## 仓库布局

```text
src/spyvar/      数据层/特征/滚动引擎/模型/评价/冻结机制
scripts/         可复现实验入口（run_all.py 统一调度）
tests/           泄漏/对齐/滚动/符号约定/统计检验正确性测试
configs/         实验协议（development.yaml / final.yaml）
docs/            审计、冻结、中文摘要、任务摘要
report/          英文 PDF 报告
```

## 数据

`data/raw/spy_data.csv`（列：date, log_ret, rv5, bv）为只读输入；
加载器校验完整性，冻结时记录 SHA256。数据文件由任务方提供。
