# -*- coding: utf-8 -*-
"""聚合层：受众画像（分层意向 + TGI）与价格接受度曲线（bootstrap 波动带）。

口径：
- top2box = 每个 persona 的 k 次采样中落在「肯定会买/可能会买」的比例（分布级，不取众数）；
- TGI = 分层意向率 / 全体意向率 × 100（>120 视为显著高倾向）；
- 波动带 = 对 persona 重采样 500 次的 5%-95% 分位（衡量抽样噪声，不含模型系统误差）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import _DEMO_DIMS, _MAPS

DIM_LABELS = {
    "age": "年龄", "gender": "性别", "city_tier": "城市层级", "urban_rural": "城乡",
    "education": "学历", "income_band": "收入档", "occupation": "职业",
}


def _persona_top2(sim: pd.DataFrame) -> pd.DataFrame:
    """长表 → persona×场景 粒度的 top2box 率（按 agent_i 分组，人口学重复的个体不合并）。"""
    sim = sim.copy()
    sim["top2"] = (sim["intent_idx"] <= 1).astype(float)
    keys = ["agent_i"] + _DEMO_DIMS + ["weight", "scenario", "price"]
    grp = sim.groupby(keys, observed=True)["top2"].mean().reset_index()
    return grp


def _wmean(values: np.ndarray, weights: np.ndarray) -> float:
    s = weights.sum()
    return float((values * weights).sum() / s) if s > 0 else float("nan")


def audience_table(sim: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """受众分层表（只用 audience 场景）。返回 (分层表, 全体意向率)。"""
    p = _persona_top2(sim)
    p = p[p["scenario"] == "audience"]
    overall = _wmean(p["top2"].to_numpy(), p["weight"].to_numpy())
    rows = []
    for dim in _DEMO_DIMS:
        for cat, g in p.groupby(dim, observed=True):
            rate = _wmean(g["top2"].to_numpy(), g["weight"].to_numpy())
            rows.append({
                "维度": DIM_LABELS[dim], "dim": dim,
                "分层": _MAPS.get(dim, {}).get(cat, cat),   # 中文分层名
                "意向率": rate, "TGI": (rate / overall * 100) if overall > 0 else float("nan"),
                "样本数": len(g),
            })
    out = pd.DataFrame(rows).sort_values(["维度", "TGI"], ascending=[True, False]).reset_index(drop=True)
    return out, overall


def top_segments(table: pd.DataFrame, min_n: int = 20, topn: int = 5) -> pd.DataFrame:
    """全维度里 TGI 最高的分层（样本量达标才上榜）。"""
    ok = table[table["样本数"] >= min_n]
    return ok.sort_values("TGI", ascending=False).head(topn).reset_index(drop=True)


def price_curve(sim: pd.DataFrame, n_boot: int = 500, seed: int = 7) -> pd.DataFrame:
    """各价位 top2box 率 + bootstrap 5%-95% 波动带 + 单调性提示。"""
    p = _persona_top2(sim)
    p = p[p["scenario"] != "audience"]
    rng = np.random.default_rng(seed)
    rows = []
    for price, g in sorted(p.groupby("price", observed=True), key=lambda kv: kv[0]):
        vals, w = g["top2"].to_numpy(), g["weight"].to_numpy()
        point = _wmean(vals, w)
        idx = np.arange(len(vals))
        boots = [_wmean(vals[s], w[s]) for s in (rng.choice(idx, size=len(idx)) for _ in range(n_boot))]
        rows.append({"价位": price, "意向率": point,
                     "波动带下": float(np.percentile(boots, 5)),
                     "波动带上": float(np.percentile(boots, 95)),
                     "样本数": len(g)})
    out = pd.DataFrame(rows)
    # 单调性提示：价升意向不降 → 大概率是噪声/样本量不足，提示加大 sample
    out.attrs["monotonic"] = bool((out["意向率"].diff().dropna() <= 1e-9).all())
    return out
