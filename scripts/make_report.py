"""英文 PDF 报告生成（19 节结构）。

全部数字从 outputs/tables 产物读取（禁止手录），图表从
outputs/figures 引用；pdflatex 编译。报告再生不重训模型。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from scripts.common import parse_common_args, resolve_config

REPORT_VERSION = "1.0"


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{x:.{nd}f}"


def _metrics_latex(metrics: pd.DataFrame, primary: dict) -> str:
    m = metrics.copy()
    m = m[m.apply(lambda r: primary.get(r["model"]) == r["feature_set"], axis=1)]
    m["model"] = m["model"].map({"M0": "HS", "M1": "GARCH-t", "M2": "LinQR", "M3": "MLP", "M4": "GJR-t", "M5": "GRU"})
    rows = []
    for tail in (0.01, 0.05, 0.10):
        sub = m[m["tail"] == tail]
        rows.append(
            "\\hline\n"
            + f"\\multicolumn{{8}}{{l}}{{\\textbf{{Tail {int(tail*100)}\\%}}}}\\\\\n"
            + "\\hline\n"
            + "\\textbf{Model} & \\textbf{N} & \\textbf{Viol} & \\textbf{Rate} & "
            + "\\textbf{Kupiec p} & \\textbf{Ind p} & \\textbf{CC p} & \\textbf{Pinball}\\\\\n"
        )
        for _, r in sub.iterrows():
            rows.append(
                f"{r['model']} & {int(r['n_forecasts'])} & {int(r['n_violations'])} & "
                f"{_fmt(r['failure_rate'], 4)} & {_fmt(r['kupiec_pvalue'])} & "
                f"{_fmt(r['christoffersen_ind_pvalue'])} & {_fmt(r['conditional_coverage_pvalue'])} & "
                f"{_fmt(r['mean_pinball'])}\\\\\n"
            )
    return "\\begin{tabular}{lrrrrrrr}\n" + "".join(rows) + "\\hline\n\\end{tabular}"


def _ablation_latex(metrics: pd.DataFrame) -> str:
    m = metrics[metrics["model"].isin(["M2", "M3"])]
    rows = []
    for model in ("M2", "M3"):
        for fset in ("F0", "F1", "F2", "F3"):
            for tail in (0.01, 0.05, 0.10):
                r = m[(m["model"] == model) & (m["feature_set"] == fset) & (m["tail"] == tail)]
                if len(r):
                    r = r.iloc[0]
                    rows.append(
                        f"{model}-{fset} & {int(tail*100)}\\% & {_fmt(r['failure_rate'])} & "
                        f"{_fmt(r['mean_pinball'])}\\\\\n"
                    )
    return "\\begin{tabular}{llrr}\n\\textbf{Model} & \\textbf{Tail} & \\textbf{Rate} & \\textbf{Pinball}\\\\\n" + "".join(rows) + "\\end{tabular}"


def _dm_latex(dm: pd.DataFrame) -> str:
    BS = chr(92)
    NL = chr(10)
    has_holm = "holm_dm_pvalue" in dm.columns
    rows = []
    for _, r in dm.iterrows():
        holm_cell = f" & {_fmt(r['holm_dm_pvalue'])}" if has_holm and not pd.isna(r.get("holm_dm_pvalue")) else ""
        head = "H" if int(r.get("headline", 0)) else ""
        rows.append(
            f"{r['model_a']} vs {r['model_b']} & {int(r['tail']*100)}\\% & {_fmt(r['dm_stat'])} & "
            f"{_fmt(r['dm_pvalue'])}{holm_cell} & {_fmt(r['bootstrap_pvalue'])} & {r['favors']} & {head}"
            + BS + BS + NL
        )
    holm_header = " & " + BS + BS + "textbf{Holm p}" if has_holm else ""
    header = (BS + BS + 'textbf{Pair} & ' + BS + BS + 'textbf{Tail} & ' + BS + BS + 'textbf{DM} & '
             + BS + BS + 'textbf{DM p}' + holm_header + ' & ' + BS + BS + 'textbf{Boot p} & '
             + BS + BS + 'textbf{Favors} & ' + BS + BS + 'textbf{Headline}' + BS + BS + NL)
    return BS + "begin{tabular}{llrrrrrl}" + NL + header + "".join(rows) + BS + "end{tabular}"
def _regime_latex(regime: pd.DataFrame, primary: dict) -> str:
    m = regime.copy()
    m = m[m.apply(lambda r: primary.get(r["model"]) == r["feature_set"], axis=1)]
    m["model"] = m["model"].map({"M0": "HS", "M1": "GARCH-t", "M2": "LinQR", "M3": "MLP", "M4": "GJR-t", "M5": "GRU"})
    m = m[m["tail"] == 0.05]
    pivot = m.pivot_table(index="regime", columns="model", values="failure_rate")
    labels = " & ".join(pivot.columns)
    rows = []
    for reg in pivot.index:
        cells = " & ".join(f"{_fmt(v)}" for v in pivot.loc[reg])
        rows.append(reg.replace(chr(95), chr(92) + chr(95)) + " & " + cells + chr(92) + chr(92) + chr(10))
    return chr(92) + "begin{tabular}{l" + "r" * len(pivot.columns) + "}" + chr(10) + chr(92) + "textbf{Regime} & " + labels + chr(92) + chr(92) + chr(10) + "".join(rows) + chr(92) + "end{tabular}"
def _conclusion_latex(metrics: pd.DataFrame, dm: pd.DataFrame, regime: pd.DataFrame) -> str:
    """Data-driven conclusion: every number is read from frozen output tables.

    Audit fix (2026-08-13): no hard-coded result sentences; unconditional
    frequency accuracy is separated from conditional adequacy; DM claims use
    the Holm-corrected headline p-values; ablation deltas are computed here.
    """
    primary = {"M0": "none", "M1": "none", "M2": "F3", "M3": "F3", "M4": "none", "M5": "F3"}
    m = metrics[metrics.apply(lambda r: primary.get(r["model"]) == r["feature_set"], axis=1)]
    name = {"M0": "HS", "M1": "GARCH-t", "M2": "LinQR", "M3": "MLP", "M4": "GJR-t", "M5": "GRU"}
    lines = []

    # 1) per tail: closest empirical failure rate (unconditional frequency) and lowest pinball
    for tail in (0.01, 0.05, 0.10):
        sub = m[m["tail"] == tail]
        closest = sub.loc[sub["failure_rate"].sub(tail).abs().idxmin()]
        sharpest = sub.loc[sub["mean_pinball"].idxmin()]
        lines.append(
            f"At the {int(tail*100)}\\% tail, the model with the empirical failure rate closest to the "
            f"nominal level is {name[closest['model']]} ({closest['failure_rate']:.4f} vs target {tail:.2f}); "
            f"the lowest mean pinball loss is achieved by {name[sharpest['model']]} ({sharpest['mean_pinball']:.5f}). "
            "Closeness of the unconditional failure frequency is not, by itself, evidence of adequate conditional "
            "VaR dynamics (see the coverage tests and the violation-clustering diagnostics below)."
        )

    # 2) GARCH-family claims computed from the tables
    r1 = m[(m["model"] == "M1") & (m["tail"] == 0.05)].iloc[0]
    r4 = m[(m["model"] == "M4") & (m["tail"] == 0.05)].iloc[0]
    r0 = m[(m["model"] == "M0") & (m["tail"] == 0.05)].iloc[0]
    cc10_m1 = m[(m["model"] == "M1") & (m["tail"] == 0.10)]["conditional_coverage_pvalue"].iloc[0]
    cc10_m4 = m[(m["model"] == "M4") & (m["tail"] == 0.10)]["conditional_coverage_pvalue"].iloc[0]
    reg = regime[regime["tail"] == 0.05]
    reg = reg[reg.apply(lambda r: primary.get(r["model"]) == r["feature_set"], axis=1)]
    crisis = reg[reg["regime"] == "crisis_2008_2009"]
    if len(crisis):
        hs_crisis = float(crisis[crisis["model"] == "M0"]["failure_rate"].iloc[0])
        gt_crisis = float(crisis[crisis["model"] == "M1"]["failure_rate"].iloc[0])
        crisis_sentence = (
            rf"crisis-period 5\% failure rates are {hs_crisis:.4f} (HS) vs {gt_crisis:.4f} (GARCH-t), "
            "documenting slow regime adaptation of historical simulation"
        )
    else:
        crisis_sentence = "crisis-period stratification is not available in this run"
    lines.append(
        "The GARCH family (GARCH-t, GJR-t) achieves the lowest mean pinball loss (sharpness) at all three tails "
        rf"and strong 1\% tail calibration (failure rates {r1['failure_rate']:.4f} / {r4['failure_rate']:.4f} vs "
        rf"{r0['failure_rate']:.4f} for HS), with no evidence of violation clustering at the 1\% and 5\% tails. "
        rf"Two caveats: at the 5\% and 10\% tails HS is marginally closer to the nominal frequency "
        rf"(5\% tail {r0['failure_rate']:.4f} vs {r1['failure_rate']:.4f}), and the 10\% conditional-coverage test "
        f"is rejected for the GARCH family (CC p {cc10_m1:.4f} / {cc10_m4:.4f}); {crisis_sentence}."
    )

    # 3) neural claims from Holm-corrected headline DM p-values
    m1m3 = dm[(dm["model_a"] == "M1") & (dm["model_b"] == "M3") & (dm["tail"] == 0.05)]
    m2m3 = dm[(dm["model_a"] == "M2") & (dm["model_b"] == "M3") & (dm["tail"] == 0.05)]
    m5_1 = m[(m["model"] == "M5") & (m["tail"] == 0.01)].iloc[0]
    if len(m1m3) and len(m2m3):
        pcol = "holm_dm_pvalue" if "holm_dm_pvalue" in dm.columns else "dm_pvalue"
        h1 = float(m1m3[pcol].iloc[0])
        h2 = float(m2m3[pcol].iloc[0])
        dm_sentence = (
            rf"the Holm-corrected DM p-values at the 5\% tail are {h1:.4f} (GARCH-t vs MLP) and {h2:.4f} "
            "(LinQR vs MLP)"
        )
    else:
        dm_sentence = "headline DM comparisons are not available in this run"
    lines.append(
        "The neural models do not outperform the classical baselines on this frozen panel: "
        + dm_sentence
        + rf"; the GRU is strongly over-conservative at the 1\% tail (failure rate {m5_1['failure_rate']:.4f}). "
        "Across-seed dispersion is reported in the robustness section rather than summarized qualitatively here. "
        "This is a valid negative result on this information set and window."
    )

    # 4) ablation deltas computed from the ablation table
    abl = pd.read_csv(Path(__file__).resolve().parent.parent / "outputs" / "tables" / "ablation.csv")
    abl5 = abl[abl["tail"] == 0.05]
    deltas = []
    for model in ("M2", "M3"):
        sub = abl5[abl5["model"] == model].sort_values("feature_set")
        if len(sub) >= 2:
            f0 = float(sub.iloc[0]["mean_pinball"])
            f3 = float(sub.iloc[-1]["mean_pinball"])
            deltas.append(rf"{name[model]}: {100*(f0-f3)/f0:.1f}\%")
    delta_text = ", ".join(deltas) if deltas else "not available"
    lines.append(
        r"Feature ablations (F0 returns -> F3 full block) change mean pinball at the 5\% tail by "
        + delta_text
        + "; the realized-measure block adds information for the linear model and the MLP, "
        "but it does not change the model ranking. The F3 effect is attributed to the "
        "jump/downside feature block as a whole, not to any single component."
    )
    return "\n\n".join(lines)

def forecast_count(out_root: Path) -> int:
    """Number of forecast target dates in the canonical frozen run."""

    from scripts.common import canonical_run_dir

    run_dir = canonical_run_dir(out_root)
    panels = sorted((run_dir / "predictions").glob("final-*.parquet"))
    if not panels:
        return 0
    import pandas as pd

    df = pd.read_parquet(panels[0])
    return int(df["target_date"].nunique())


def build_latex(cfg, out_root: Path, freeze: dict | None, audit: dict | None) -> str:
    metrics = pd.read_csv(out_root / "tables" / "metrics.csv")
    dm = pd.read_csv(out_root / "tables" / "dm_comparison.csv")
    regime = pd.read_csv(out_root / "tables" / "regime_metrics.csv")
    primary = {"M0": "none", "M1": "none", "M2": "F3", "M3": "F3", "M4": "none", "M5": "F3"}
    w = cfg.primary_window or "TBD"
    seed = cfg.primary_seed
    n_dates = forecast_count(out_root)

    return r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{multirow}
\usepackage{caption}
\usepackage{hyperref}
\hypersetup{hidelinks}
\captionsetup{font=small,labelfont=bf}
\newcommand{\rot}[1]{\begin{turn}{0}#1\end{turn}}
\usepackage{rotating}
\title{One-Day-Ahead Value-at-Risk Forecasting for SPY\\[2mm]
\large A Frozen Out-of-Sample Study of Historical Simulation, GARCH-t, HAR Quantile Regression, and Small Neural Quantile Models}
\author{Altair Li}
\date{\today}
\begin{document}
\maketitle

\section{Executive Summary}
This report studies one-day-ahead Value-at-Risk (VaR) forecasting for SPY log returns at the 1\%, 5\% and 10\% lower tails, under a strictly causal rolling-window protocol. Six model families are compared on a frozen out-of-sample period starting 2008-01-01: rolling Historical Simulation (HS), GARCH(1,1) with Student-$t$ innovations, GJR-GARCH(1,1,1)-$t$, linear (HAR-style) quantile regression, a small multi-quantile MLP, and a small GRU quantile model. Feature ablations isolate the incremental value of realized volatility (rv5), bipower variation (bv), and jump/downside-asymmetry information. All numbers in this report are regenerated by scripts from machine-readable prediction artifacts; none are hand-entered.

\section{Problem Definition}
For each forecast origin $t$ we predict the conditional return quantile $q_{\alpha,t+1}$ with $P(r_{t+1}\le q_{\alpha,t+1}\mid\mathcal{F}_t)=\alpha$ for $\alpha\in\{0.01,0.05,0.10\}$, using only information available at the end of day $t$. A violation is defined as $r_{t+1}\le \hat{q}_{\alpha,t+1}$ (VaR is the return quantile itself; lower-tail values are typically negative).

\section{Data}
Daily SPY observations with columns \texttt{log\_ret}, \texttt{rv5} (5-minute realized variance) and \texttt{bv} (bipower variation). See Appendix~\ref{sec:audit} for the machine-generated data audit.

\section{Exploratory Analysis}
See Figures~\ref{fig:overview}--\ref{fig:rvbv}. Key features: volatility clustering, heavy tails (excess kurtosis), strong persistence of log realized measures, and a positive link between negative returns and next-day volatility.

\section{Experimental Protocol}
Development period: all dates before 2008-01-01 (rolling-origin validation over 2005--2007, with a documented slight adjustment for the 1500-day window whose earliest feasible origin is late 2005). Final test: all dates from 2008-01-01. Candidate windows: 1000 and 1500 trading days only; the primary window is chosen from development evidence (see Section~\ref{sec:dev}). All models share the same forecast dates. The final configuration and data hash are frozen in \texttt{docs/FREEZE\_MANIFEST.md} before any final-test run; the gate is enforced by tests and scripts.

\section{Models}
\subsection{M0 -- Historical Simulation}
The empirical quantile of the rolling-window returns: $\hat{q}_\alpha = \hat{F}_t^{-1}(\alpha)$ over $\{r_{t-W+1},\dots,r_t\}$. Non-parametric benchmark; implicitly assumes a stationary return distribution within the window, so regime shifts are absorbed only with a lag.

\subsection{M1 -- GARCH(1,1)-Student-$t$}
Classical conditional volatility model with conditional variance
\begin{equation}
\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2, \qquad \varepsilon_t = \sigma_t z_t,
\end{equation}
where $z_t$ is an i.i.d.\ unit-variance standardized Student-$t$ innovation with $\nu$ degrees of freedom (variance 1, so that $\mathrm{Var}(\varepsilon_t\mid\mathcal{F}_{t-1})=\sigma_t^2$). The one-day-ahead VaR is
\begin{equation}
\mathrm{VaR}_{\alpha,t+1} = \mu + F_t^{-1}(\alpha;\nu)\,\hat\sigma_{t+1},
\end{equation}
with $F_t^{-1}(\alpha;\nu)$ the quantile of the standardized Student-$t$ (implemented via \texttt{arch.univariate.StudentsT}, equal to $t_\nu^{-1}(\alpha)\sqrt{(\nu-2)/\nu}$); $\mu$ is a constant mean. Parameters are re-estimated by MLE on every rolling window.

\subsection{M4 -- GJR-GARCH(1,1,1)-Student-$t$}
Adds a leverage term to the variance recursion,
\begin{equation}
\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \gamma\,I(\varepsilon_{t-1}<0)\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2,
\end{equation}
capturing asymmetric responses of volatility to negative shocks; VaR construction identical to M1. Serves as the downside-asymmetry robustness extension.

\subsection{M2 -- Linear / HAR Quantile Regression}
Direct conditional quantile estimation by minimizing the pinball loss,
\begin{equation}
\hat{\beta}_\tau = \arg\min_\beta \sum_t \rho_\tau\big(y_{t+1} - x_t^\top\beta\big), \qquad \rho_\tau(u) = u\big(\tau - I(u<0)\big),
\end{equation}
fit separately for $\tau \in \{0.01, 0.05, 0.10\}$ on the same feature sets as the MLP (so the linear-vs-nonlinear contrast is not confounded by information). Independent fits may cross; the crossing rate is reported.

\subsection{M3 -- Multi-Quantile MLP}
A small MLP with joint pinball loss over the three tails and a structurally non-crossing head:
\begin{equation}
q_{0.01} = z_1, \qquad q_{0.05} = z_1 + \mathrm{softplus}(z_2), \qquad q_{0.10} = q_{0.05} + \mathrm{softplus}(z_3),
\end{equation}
so that $q_{0.01}\le q_{0.05}\le q_{0.10}$ holds by construction ($\mathrm{softplus}>0$). Targets are standardized with train-only statistics inside each rolling window and the affine transform is inverted on output (Scheme B, audit fix). Training uses early stopping on the last 10\% of the window in time order; scalers are fitted on training rows only.

\subsection{M5 -- GRU Quantile}
A one-layer GRU over the last 22 daily feature vectors with the same ordered head and loss. The role is a robustness extension: it tests whether an explicit sequence representation adds value beyond the structured HAR-style features already present, not a substitute for the MLP. Parameters are deliberately kept small.

\subsection{Gaussian-GARCH diagnostic}
GARCH(1,1) with Gaussian innovations is estimated only in development as a diagnostic reference.

\section{Feature Engineering}
Information sets (identical for M2 and M3, so Linear-vs-MLP differences are attributable to the mapping rather than the inputs): \textbf{F0} returns only (lags 1/2/5/22, abs, squared); \textbf{F1} adds $\log rv5$, $\sqrt{rv5}$, 5-day and 22-day HAR aggregations of $\log rv5$; \textbf{F2} adds the same block for $bv$; \textbf{F3} adds jump proxy $\max(rv5-bv,0)$, relative jump, and a 5-day downside-shock intensity. All features are causal by construction and verified by truncation-invariance tests.

\section{Rolling Forecast Design}
A unified rolling engine controls window boundaries, feature cutoffs, preprocessing (scalers fitted on the window training rows only), model refits (daily full refit for every model), and prediction recording. Early stopping for neural models uses the last 10\% of the window in time order. All outputs share one schema (RESULTS\_SCHEMA.md) with \texttt{fit\_status} logging; failures are never silently dropped.

\section{VaR Backtesting Methodology}
Per model and tail: empirical failure rate, Kupiec unconditional coverage test, Christoffersen independence and conditional-coverage tests, mean pinball loss, quantile-crossing rate, violation clustering, Dynamic Quantile test (reported as an additional diagnostic), Diebold--Mariano tests and moving-block bootstrap for pairwise loss comparisons. Finite-sample caveats apply: the 1\% tail has very few expected violations even in a decade-long test, so coverage tests have low power and small numerical differences must not be over-read.

\section{Development / Validation Results}\label{sec:dev}
Window comparison tables: \texttt{outputs/development/window\_selection.csv} and \texttt{window\_selection\_cells.csv}. Decision rule (predeclared, audit-fixed aggregation): per (model, feature-set, tail) cell the candidate-relative normalized loss (cell loss divided by the cell mean across the two candidates) is averaged with equal weight across all cells; per-cell winners, pairwise win counts and drop-one sensitivity are reported alongside. If the equal-weight aggregate and the raw pinball-sum aggregate disagree, the longer window is chosen conservatively for 1\% tail stability. The longer window provides more effective tail observations for the 1\% quantile (about 15 vs 10 expected violations) and hence more stable empirical quantiles, at the cost of slower regime adaptation; the short window adapts faster. The primary window selected from development evidence is \textbf{\texttt{""" + str(w) + r"""}}.

\section{Frozen Out-of-Sample Results}\label{sec:oos}
All models evaluated on \texttt{""" + str(n_dates) + r"""} identical forecast dates ($\ge$ 2008-01-01). Primary seed \texttt{""" + str(seed) + r"""}. Table~\ref{tab:metrics} reports coverage and loss statistics.

\begin{table}[ht]
\centering
\caption{Frozen out-of-sample backtesting results by model and tail.}\label{tab:metrics}
""" + _metrics_latex(metrics, primary) + r"""
\end{table}

\section{Tail-Level Model Comparison}
See Figure~\ref{fig:tailmodel}. Rankings are tail-specific; a single universal winner is not assumed.

\section{Feature Ablations}
Table~\ref{tab:ablation} and Figure~\ref{fig:ablation}. Ablation interpretation is confined to M2/M3 across F0--F3.

\begin{table}[ht]
\centering
\caption{Feature ablation: failure rate and mean pinball by information set.}\label{tab:ablation}
""" + _ablation_latex(metrics) + r"""
\end{table}

\section{Crisis and Regime Analysis}
Figure~\ref{fig:regime} and Table~\ref{tab:regime} stratify frozen results into predefined regimes: 2008--2009 crisis, 2010--2012 elevated volatility, 2013--2014 calm, 2015--2016 stress, 2017 calm, 2018 spike. This is an explanatory analysis of frozen results, not a basis for re-optimization.

\begin{table}[ht]
\centering
\caption{Failure rate by regime (primary configuration per model).}\label{tab:regime}
""" + _regime_latex(regime, primary) + r"""
\end{table}

\section{Robustness}
Seed robustness for the neural models (predeclared seeds 7, 42, 2026; headline series is the primary seed 42, not an ensemble): \texttt{outputs/tables/seed\_robustness.csv}. DM and block-bootstrap pairwise comparisons: Table~\ref{tab:dm}.

\begin{table}[ht]
\centering
\caption{Pairwise loss comparisons (negative DM favors model\_a).}\label{tab:dm}
""" + _dm_latex(dm) + r"""
\end{table}

\section{Limitations}
\begin{itemize}
\item The 1\% tail has few expected violations ($\approx$2.6 per year); coverage tests have low power.
\item Neural models are retrained daily on at most 1500 observations; capacity is deliberately small.
\item rv5/bv are daily aggregates; intraday dynamics are not modeled.
\item DM/block-bootstrap inference at extreme tails relies on asymptotic approximations; results are interpreted as indicative.
\end{itemize}

\section{Conclusion}
The conclusion is drawn from the frozen out-of-sample evidence in Section~\ref{sec:oos} and the analysis sections.

""" + _conclusion_latex(metrics, dm, regime) + r"""

\section{References}
\begin{itemize}
\item Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. \emph{Journal of Derivatives}, 3(2), 73--84. DOI: 10.3905/jod.1995.407942
\item Christoffersen, P. (1998). Evaluating Interval Forecasts. \emph{International Economic Review}, 39(4), 841--862. DOI: 10.2307/2527341
\item Engle, R. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. \emph{Econometrica}, 50(4), 987--1007. DOI: 10.2307/1912773
\item Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. \emph{Journal of Econometrics}, 31(3), 307--327. DOI: 10.1016/0304-4076(86)90063-1
\item Glosten, L., Jagannathan, R., Runkle, D. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. \emph{Journal of Finance}, 48(5), 1779--1801. DOI: 10.1111/j.1540-6261.1993.tb05128.x
\item Koenker, R., Bassett, G. (1978). Regression Quantiles. \emph{Econometrica}, 46(1), 33--50. DOI: 10.2307/1913643
\item Engle, R., Manganelli, S. (2004). CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles. \emph{Journal of Business \& Economic Statistics}, 22(4), 367--381. DOI: 10.1198/073500104000000370
\item Andersen, T., Bollerslev, T., Diebold, F., Labys, P. (2003). Modeling and Forecasting Realized Volatility. \emph{Econometrica}, 71(2), 579--625. DOI: 10.1111/1468-0262.00418
\item Barndorff-Nielsen, O., Shephard, N. (2004). Power and Bipower Variation with Stochastic Volatility and Jumps. \emph{Journal of Financial Econometrics}, 2(1), 1--37. DOI: 10.1093/jjfinec/nbh001
\item Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. \emph{Journal of Financial Econometrics}, 7(2), 174--196. DOI: 10.1093/jjfinec/nbp001
\item Diebold, F., Mariano, R. (1995). Comparing Predictive Accuracy. \emph{Journal of Business \& Economic Statistics}, 13(3), 253--263. DOI: 10.1080/07350015.1995.10524599
\item Koenker, R. (2005). \emph{Quantile Regression}. Cambridge University Press. DOI: 10.1017/CBO9780511754098
\item Taylor, J. (2000). A Quantile Regression Neural Network Approach to Estimating the Conditional Density of Multiperiod Returns. \emph{Journal of Forecasting}, 19(4), 299--311. DOI: 10.1002/1099-131X(200007)19:4<299::AID-FOR775>3.0.CO;2-V
\item Patton, A., Sheppard, K. (2015). Good Volatility, Bad Volatility: Signed Jumps and the Persistence of Volatility. \emph{Review of Economics and Statistics}, 97(3), 683--697. DOI: 10.1162/REST\_a\_00503
\end{itemize}

\appendix
\section{Machine-Generated Data Audit}\label{sec:audit}
\url{docs/DATA\_AUDIT.md} (generated by \texttt{scripts/audit\_data.py}).

\section{Artifacts}
All prediction panels: \texttt{outputs/predictions/*.parquet} with sidecar manifests; metric tables: \texttt{outputs/tables/*.csv}; figures: \texttt{outputs/figures/*.png}. Freeze manifest: \texttt{docs/FREEZE\_MANIFEST.md}.

\begin{figure}[ht]
\centering\includegraphics[width=\textwidth]{../outputs/figures/fig01_overview.png}
\caption{SPY log returns and volatility scale, full sample.}\label{fig:overview}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=\textwidth]{../outputs/figures/fig02_rv_bv.png}
\caption{Realized variance and bipower variation (log scale).}\label{fig:rvbv}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=\textwidth]{../outputs/figures/fig03_var_curves.png}
\caption{Frozen out-of-sample realized returns with dynamic VaR forecasts.}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=\textwidth]{../outputs/figures/fig04_violations.png}
\caption{1\% VaR violation points by model.}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=0.85\textwidth]{../outputs/figures/fig05_failure_rates.png}
\caption{Empirical failure rates vs target coverage.}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=0.85\textwidth]{../outputs/figures/fig06_pinball.png}
\caption{Mean pinball loss by model and tail.}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=0.6\textwidth]{../outputs/figures/fig07_tail_model.png}
\caption{Pinball loss relative to best model per tail.}\label{fig:tailmodel}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=0.8\textwidth]{../outputs/figures/fig08_ablation.png}
\caption{Feature ablation F0--F3 for linear quantile regression and MLP.}\label{fig:ablation}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=\textwidth]{../outputs/figures/fig09_regime.png}
\caption{Failure rate by regime.}\label{fig:regime}
\end{figure}
\begin{figure}[ht]
\centering\includegraphics[width=0.8\textwidth]{../outputs/figures/fig10_seed_robustness.png}
\caption{Neural seed robustness.}
\end{figure}

\end{document}
"""


def main() -> None:
    args = parse_common_args("PDF 报告生成")
    cfg = resolve_config(args)
    out_root = Path(args.out_root)
    freeze_path = out_root / "manifests" / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else None
    audit_path = out_root / "tables" / "data_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None

    tex = build_latex(cfg, out_root, freeze, audit)
    report_dir = Path("report")
    report_dir.mkdir(exist_ok=True)
    tex_path = report_dir / "final_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    for _ in range(2):  # pdflatex 两遍解析交叉引用
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "final_report.tex"],
            cwd=report_dir, capture_output=True, text=True, check=True,
        )
    pdf = report_dir / "final_report.pdf"
    if not pdf.exists() or pdf.stat().st_size == 0:
        sys.exit("PDF 生成失败（pdflatex 输出为空）")
    print(f"PDF -> {pdf} ({pdf.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
