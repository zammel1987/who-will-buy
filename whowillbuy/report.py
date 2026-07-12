# -*- coding: utf-8 -*-
"""报告：受众 + 定价 → 一份可直接转发的 Markdown。

硬口径（不可去除）：AI 仿真生成标注 / 方向排序可信、绝对值未校准 / 波动带含义。
"""
from __future__ import annotations

import datetime

import pandas as pd

from .product_spec import ProductSpec

# 交流入口
CONTACT = "GitHub Issues（https://github.com/zammel1987/who-will-buy/issues）"

_DISCLAIMER = """\
## 方法与口径（必读）

- 本报告由 **AI 仿真生成**（合成人群 × 大模型角色扮演），不是真人调研。
- 人群 = 依据**公开普查聚合边际**（七普 2020 等，来源见仓库 `data/census_marginals.json`）
  IPF 构建的纯合成人群，画像为人口学属性，未经真实微数据富化。
- **怎么读数字**：分层之间的**相对排序、价位之间的方向**（谁高谁低、涨价掉多少的方向）有参考价值；
  **绝对值未经真实数据校准，不可直接当作市场渗透率**。
- 「意向率」的口径是**购买兴趣/考虑集**（软化问法，为拉开人群区分度），比"实际会购买"宽松，
  绝对值必然显著高于真实转化率——请只用它比较分层与价位的相对高低。
- 波动带 = 抽样噪声的 5%-95% 区间（bootstrap），不含模型本身的系统误差。
- mock 模式的数字来自演示用启发式，仅用于验证流程，**没有任何预测含义**。

> 作者正在真实调查微数据与下游校准的方向上做更完整的版本（绝对值可校准的那种）。
> 对这个方向有兴趣或想交流方法的，见 {contact}。
"""


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _bar(x: float, width: int = 24) -> str:
    n = max(0, min(width, round(x * width)))
    return "█" * n + "░" * (width - n)


def build_report(spec: ProductSpec, provider: str, n_pop: int, n_sample: int, k: int,
                 aud_table: pd.DataFrame, overall: float, tops: pd.DataFrame,
                 curve: pd.DataFrame, answer_rate: float) -> str:
    ts = datetime.date.today().isoformat()
    L: list[str] = []
    L.append(f"# 谁会买「{spec.name}」？ —— 受众与价格接受度仿真报告\n")
    L.append(f"> 生成日期 {ts} · 引擎 who-will-buy · 后端 `{provider}` · "
             f"合成人群 {n_pop} 人（抽样 {n_sample} 人 × {k} 次采样，作答率 {_fmt_pct(answer_rate)}）\n")
    L.append(f"**商品**：【{spec.name}】（{spec.category}）{spec.description}\n")

    # ---- 摘要 ----
    main_p = spec.resolved_main_price()
    L.append("## 摘要\n")
    L.append(f"- 主价位 {main_p:g} {spec.currency} 下，全体购买意向率（top2box）≈ **{_fmt_pct(overall)}**（绝对值未校准，看相对结构）；")
    if len(tops):
        seg = tops.iloc[0]
        L.append(f"- 意向最强人群：**{seg['维度']}={seg['分层']}**（TGI {seg['TGI']:.0f}）；")
    if len(curve) >= 2:
        drop = curve.iloc[0]["意向率"] - curve.iloc[-1]["意向率"]
        L.append(f"- 价位从 {curve.iloc[0]['价位']:g} 提到 {curve.iloc[-1]['价位']:g} {spec.currency}，"
                 f"意向率变化 {-drop * 100:+.1f} 个百分点"
                 + ("；" if curve.attrs.get("monotonic") else
                    "（⚠️ 曲线非单调，多为抽样噪声，建议加大 --sample 复跑）；"))
    L.append("")

    # ---- 高意向人群 ----
    L.append("## 一、谁会买（高意向人群 TGI 榜）\n")
    L.append("| 维度 | 分层 | 意向率 | TGI | 样本数 |")
    L.append("|---|---|---:|---:|---:|")
    for _, r in tops.iterrows():
        L.append(f"| {r['维度']} | {r['分层']} | {_fmt_pct(r['意向率'])} | {r['TGI']:.0f} | {r['样本数']} |")
    L.append("\n<details><summary>全部分层明细</summary>\n")
    L.append("| 维度 | 分层 | 意向率 | TGI | 样本数 |")
    L.append("|---|---|---:|---:|---:|")
    for _, r in aud_table.iterrows():
        L.append(f"| {r['维度']} | {r['分层']} | {_fmt_pct(r['意向率'])} | {r['TGI']:.0f} | {r['样本数']} |")
    L.append("\n</details>\n")

    # ---- 价格接受度 ----
    L.append("## 二、卖多少钱（价格接受度曲线）\n")
    L.append(f"| 价位（{spec.currency}） | 意向率 | 波动带(5%-95%) | 曲线 |")
    L.append("|---:|---:|---|---|")
    for _, r in curve.iterrows():
        L.append(f"| {r['价位']:g} | {_fmt_pct(r['意向率'])} "
                 f"| {_fmt_pct(r['波动带下'])} ~ {_fmt_pct(r['波动带上'])} | `{_bar(r['意向率'])}` |")
    L.append("")

    L.append(_DISCLAIMER.format(contact=CONTACT))
    return "\n".join(L)
