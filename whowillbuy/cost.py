# -*- coding: utf-8 -*-
"""预算守门：估价发生在任何网络调用之前，超预算直接拦下。"""
from __future__ import annotations


def estimate_usd(n_calls: int, tokens_in: int, tokens_out: int,
                 price_in: float, price_out: float) -> float:
    """粗估一次跑批的美元成本。price_* 单位 = USD / 1M tokens。"""
    return n_calls * (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def guard(budget_usd: float | None, est_usd: float, n_calls: int) -> None:
    """预算拦截：budget 为 None 时只提示不拦。"""
    line = f"[估价] 约 {n_calls} 次调用，预估 ${est_usd:.2f}（约 ¥{est_usd * 7.3:.1f}，按官网价可能有出入）"
    if budget_usd is None:
        print(line + "  ——未设 --budget，仅提示不拦截。")
        return
    if est_usd > budget_usd:
        raise SystemExit(
            f"{line}\n[拦截] 超出预算 --budget {budget_usd}。可调小 --sample/--k，或提高预算。"
            "（拦截发生在任何网络调用之前，未产生费用）")
    print(line + f"  ——预算 ${budget_usd}，放行。")
