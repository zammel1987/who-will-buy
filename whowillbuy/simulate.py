# -*- coding: utf-8 -*-
"""仿真：合成人群逐人作答购买意向问题。

- mock：启发式作答器（收入可负担性 × 人群亲和度 + 噪声），零成本跑通全链路，输出仅供演示；
- 真实档：每个 persona × 每个场景 × k 次采样问 LLM（分布级采样，禁单次贪心当唯一答案）；
- 选项顺序逐次随机（对冲位置偏置），作答率低于阈值告警。
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import random

import pandas as pd

from . import cost as cost_mod
from . import llm as llm_mod
from .product_spec import INTENT_LEVELS, ProductSpec, build_scenarios, render_question

# ---------------------------------------------------------------------------
# persona 渲染（人口学 → 中文自然描述）
# ---------------------------------------------------------------------------
_MAPS = {
    "gender": {"male": "男性", "female": "女性"},
    "age": {"18-24": "18-24 岁", "25-34": "25-34 岁", "35-44": "35-44 岁",
            "45-54": "45-54 岁", "55-64": "55-64 岁", "65+": "65 岁以上"},
    "city_tier": {"tier1": "一线/新一线城市", "tier2": "二线城市", "tier3plus": "三线及以下城市"},
    "urban_rural": {"urban": "城镇", "rural": "农村"},
    "education": {"highschool_below": "高中及以下学历", "college": "大专学历",
                  "bachelor": "本科学历", "postgrad": "研究生学历"},
    "income_band": {"<5k": "家庭人均月收入 5 千元以下", "5-10k": "家庭人均月收入 5 千到 1 万元",
                    "10-20k": "家庭人均月收入 1 到 2 万元", "20-40k": "家庭人均月收入 2 到 4 万元",
                    "40k+": "家庭人均月收入 4 万元以上"},
    "occupation": {"student": "学生", "blue_collar": "产业/体力劳动者", "service": "服务业从业者",
                   "white_collar": "企业职员", "professional": "专业技术人员", "retired": "退休人员"},
}


def render_persona(row: pd.Series) -> str:
    return (f"你是一位{_MAPS['age'][row['age']]}的{_MAPS['gender'][row['gender']]}，"
            f"{_MAPS['education'][row['education']]}，{_MAPS['occupation'][row['occupation']]}，"
            f"生活在{_MAPS['city_tier'][row['city_tier']]}（{_MAPS['urban_rural'][row['urban_rural']]}），"
            f"{_MAPS['income_band'][row['income_band']]}。"
            f"请完全代入这个身份，按这个人真实的消费能力和习惯作答，不要迎合提问者。")


# ---------------------------------------------------------------------------
# mock 作答器：可负担性 × 亲和度 启发式（仅演示管线，绝非预测）
# ---------------------------------------------------------------------------
_INCOME_MID = {"<5k": 3500, "5-10k": 7500, "10-20k": 15000, "20-40k": 30000, "40k+": 60000}
_AFFINITY = {
    ("age", "18-24"): 0.03, ("age", "25-34"): 0.06, ("age", "35-44"): 0.03,
    ("age", "55-64"): -0.05, ("age", "65+"): -0.10,
    ("city_tier", "tier1"): 0.05, ("city_tier", "tier3plus"): -0.03,
    ("education", "bachelor"): 0.03, ("education", "postgrad"): 0.04,
    ("occupation", "white_collar"): 0.04, ("occupation", "professional"): 0.04,
    ("occupation", "retired"): -0.06, ("occupation", "student"): -0.02,
    ("urban_rural", "rural"): -0.04,
}


def _mock_top2_prob(row: pd.Series, price: float) -> float:
    affordability = price / _INCOME_MID[row["income_band"]]
    p = 0.42 - 1.4 * affordability
    for dim in ("age", "city_tier", "education", "occupation", "urban_rural"):
        p += _AFFINITY.get((dim, row[dim]), 0.0)
    return max(0.02, min(0.75, p))


def _mock_answer(row: pd.Series, price: float, rng: random.Random) -> int:
    """返回意向档位索引（0=肯定会买 … 3=肯定不会买）。"""
    p2 = _mock_top2_prob(row, price)
    r = rng.random()
    if r < p2 * 0.3:
        return 0
    if r < p2:
        return 1
    return 2 if rng.random() < 0.55 else 3


# ---------------------------------------------------------------------------
# 跑批
# ---------------------------------------------------------------------------
def _stable_seed(*parts) -> int:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def _apply_market_filter(df: pd.DataFrame, market: dict) -> pd.DataFrame:
    for col, allowed in (market or {}).items():
        if col not in df.columns:
            raise SystemExit(f"market 过滤字段不存在：{col}")
        df = df[df[col].isin(allowed)]
    if len(df) == 0:
        raise SystemExit("market 过滤后人群为空，请放宽条件")
    return df


def run_simulation(spec: ProductSpec, population: pd.DataFrame, *,
                   provider: str = "mock", model: str | None = None,
                   n_sample: int = 300, k: int = 3, workers: int = 8,
                   budget_usd: float | None = None, seed: int = 2026,
                   temperature: float = 0.8) -> pd.DataFrame:
    """返回长表：每行 = persona × 场景 × 采样，含 intent_idx（0-3）。"""
    df = _apply_market_filter(population, spec.market)
    n_sample = min(n_sample, len(df))
    sampled = df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    scenarios = build_scenarios(spec)

    rows: list[dict] = []
    if provider == "mock":
        for sc in scenarios:
            for i, row in sampled.iterrows():
                rng = random.Random(_stable_seed(seed, sc["id"], i))
                for j in range(k):
                    rows.append({**_demo_cols(row), "agent_i": i,
                                 "scenario": sc["id"], "price": sc["price"],
                                 "sample_no": j, "intent_idx": _mock_answer(row, sc["price"], rng)})
        out = pd.DataFrame(rows)
        out.attrs["answer_rate"] = 1.0
        return out

    # ---- 真实档：先估价再放行（拦截在任何网络调用之前） ----
    cfg = llm_mod.resolve(provider, model)
    n_calls = n_sample * len(scenarios) * k
    est = cost_mod.estimate_usd(n_calls, tokens_in=420, tokens_out=4,
                                price_in=cfg["price_in"], price_out=cfg["price_out"])
    cost_mod.guard(budget_usd, est, n_calls)

    # 预检：单次试调用，Key 无效/余额不足等账号问题立刻报明白，不浪费整批调用
    try:
        llm_mod.chat(cfg, [{"role": "user", "content": "回复字母A"}],
                     temperature=0.0, max_tokens=4, retries=1)
    except llm_mod.LLMError as e:
        raise SystemExit(f"[预检失败] {e}\n（整批调用未发出，未产生批量费用）") from e

    tasks = []
    for sc in scenarios:
        for i, row in sampled.iterrows():
            for j in range(k):
                tasks.append((sc, i, row, j))

    def _ask(task):
        sc, i, row, j = task
        rng = random.Random(_stable_seed(seed, sc["id"], i, j))
        order = list(range(4))
        rng.shuffle(order)                       # 选项位置随机化
        q = render_question(spec, sc["price"], order)
        messages = [{"role": "system", "content": render_persona(row)},
                    {"role": "user", "content": q}]
        try:
            text, _ti, _to = llm_mod.chat(cfg, messages, temperature=temperature)
        except llm_mod.LLMError:
            return None
        letter = next((ch for ch in text.upper() if ch in "ABCD"), None)
        if letter is None:
            return None
        return {**_demo_cols(row), "agent_i": i,
                "scenario": sc["id"], "price": sc["price"],
                "sample_no": j, "intent_idx": order[ord(letter) - 65]}

    done, failed = [], 0
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for res in ex.map(_ask, tasks):
            if res is None:
                failed += 1
            else:
                done.append(res)
    total = len(tasks)
    rate = (total - failed) / total if total else 0.0
    if rate < 0.99:
        print(f"[警告] 作答率 {rate:.1%} < 99%——请检查 Key/限流/模型输出格式，带病数据勿直接出报告。")
    out = pd.DataFrame(done)
    out.attrs["answer_rate"] = rate
    return out


_DEMO_DIMS = ["age", "gender", "city_tier", "urban_rural", "education", "income_band", "occupation"]


def _demo_cols(row: pd.Series) -> dict:
    d = {c: row[c] for c in _DEMO_DIMS}
    d["weight"] = float(row["weight"]) if "weight" in row and pd.notna(row["weight"]) else 1.0
    return d


__all__ = ["run_simulation", "render_persona", "INTENT_LEVELS", "_DEMO_DIMS"]
