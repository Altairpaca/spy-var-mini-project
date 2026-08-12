# 数据审计报告

由 `scripts/audit_data.py` 自动生成；所有数字来自 `outputs/tables/data_audit.json`。

## 基本结构

- 行数: 4640
- 日期范围: 2000-01-04 ~ 2018-06-27
- 重复日期: 0
- 日期单调递增: True
- 缺失/inf: {'log_ret': 0, 'rv5': 0, 'bv': 0}

## log_ret 描述统计

| 统计量 | 值 |
|---|---|
| mean | 0.000133341 |
| std | 0.0120128 |
| min | -0.0968836 |
| max | 0.10642 |
| skewness | -0.207852 |
| excess_kurtosis | 8.2189 |
| q001 | -0.0345454 |
| q005 | -0.0188074 |
| q010 | -0.0128854 |

经验左尾样本数（`realized <= quantile`）:

- n_below_q001: 47
- n_below_q005: 232
- n_below_q010: 464

## rv5 / bv 尺度

| 统计量 | rv5 | bv |
|---|---|---|
| mean | 0.000109597 | 8.81694e-05 |
| median | 4.952e-05 | 3.93467e-05 |
| std | 0.000248284 | 0.000206689 |
| min | 1.21806e-06 | 1.05604e-06 |
| max | 0.00774774 | 0.00601815 |
| pct_zeros | 0 | 0 |

## sqrt(rv5) 合理性（应接近日波动率量级并与 |r| 相关）

- mean: 0.00857291
- corr_with_abs_ret: 0.63245
- corr_with_log_ret2: 0.555919

## RV/BV 相关

- corr: 0.964019
- corr_log: 0.968572

## 自相关

| 序列 | lag1 | lag5 |
|---|---|---|
| log_ret | -0.07054 | - |
| abs_ret | 0.2598 | 0.3385 |
| sq_ret | 0.2121 | - |
| log_rv5 | 0.8175 | 0.7139 |
| log_bv | 0.8486 | - |

## jump proxy `max(rv5-bv, 0)` 分布

- mean: 2.25601e-05
- median: 5.21923e-06
- pct_positive: 0.83125
- corr_with_future_rv5: 0.340973

## 负收益与未来波动的关系

- corr_negind_with_next_abs_ret: 0.0911095
- corr_negind_with_next_rv5: 0.0947889
- corr_negret_with_next_rv5: -0.447139
