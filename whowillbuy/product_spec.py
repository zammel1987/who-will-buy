# -*- coding: utf-8 -*-
"""商品 spec：product.yaml → 仿真场景（受众 + 价位扫描）。

设计纪律（继承自主引擎的可靠度清单）：
- 价格只走 price 字段单通道注入，商品描述文本里禁止出现价格数字（防多通道打架）；
- 问法中性，不暴露"想要的答案"（不写"这么优惠你会买吗"）；
- 选项完备（四档意向覆盖全空间），每次作答随机排列选项对冲位置偏置。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

# 四档购买兴趣（固定语义，展示顺序每次随机）。
# 口径说明：用「兴趣/考虑集」而非「未来N个月实购」——硬购买问法下对齐大模型会塌缩到
# 「肯定不会买」（实测 76% 选最末档、区分度归零），软化问法才能拉开分层结构；
# 代价是绝对值高于真实转化率，报告口径节已作说明。
INTENT_LEVELS = ["很感兴趣，会认真考虑购买", "有点兴趣，可能会买", "兴趣不大", "完全没兴趣"]
TOP2 = {0, 1}   # 前两档 = 有购买兴趣（top2box）

_PRICE_IN_TEXT = re.compile(r"(\d[\d,.]*\s*(元|块|¥|￥|rmb|RMB))|([¥￥]\s*\d)")


@dataclass
class ProductSpec:
    name: str
    category: str
    description: str                      # 一句话卖点（禁含价格）
    price_points: list[float]
    main_price: float | None = None       # 受众场景用的主价位，缺省取中位
    market: dict = field(default_factory=dict)   # 可选过滤，如 {"city_tier": ["tier1","tier2"]}
    currency: str = "元"

    def resolved_main_price(self) -> float:
        if self.main_price is not None:
            return float(self.main_price)
        pts = sorted(self.price_points)
        return float(pts[len(pts) // 2])


def load_spec(path: str) -> ProductSpec:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    missing = [k for k in ("name", "category", "description", "price_points") if not raw.get(k)]
    if missing:
        raise SystemExit(f"product.yaml 缺少必填字段：{', '.join(missing)}")
    pts = [float(p) for p in raw["price_points"]]
    if not (2 <= len(pts) <= 6):
        raise SystemExit("price_points 需要 2~6 个候选价位")
    if any(p <= 0 for p in pts):
        raise SystemExit("price_points 必须为正数")
    if _PRICE_IN_TEXT.search(str(raw["description"])) or _PRICE_IN_TEXT.search(str(raw["name"])):
        raise SystemExit(
            "商品名称/描述里不要写价格——价格只通过 price_points 单通道注入，"
            "否则问题文本与扫描价位打架，定价曲线不可信。")
    spec = ProductSpec(
        name=str(raw["name"]).strip(),
        category=str(raw["category"]).strip(),
        description=str(raw["description"]).strip(),
        price_points=pts,
        main_price=raw.get("main_price"),
        market=raw.get("market") or {},
        currency=str(raw.get("currency", "元")),
    )
    return spec


def build_scenarios(spec: ProductSpec) -> list[dict]:
    """受众场景（主价位）在前，价位扫描场景随后。"""
    scenarios = [{"id": "audience", "price": spec.resolved_main_price(), "role": "audience"}]
    for p in sorted(spec.price_points):
        scenarios.append({"id": f"price_{p:g}", "price": float(p), "role": "price"})
    return scenarios


def render_question(spec: ProductSpec, price: float, option_order: list[int]) -> str:
    """中性问法 + 单通道价格注入 + 按给定顺序排列选项。"""
    opts = "\n".join(f"{chr(65 + i)}. {INTENT_LEVELS[idx]}" for i, idx in enumerate(option_order))
    return (
        f"下面是一款商品的介绍：\n"
        f"【{spec.name}】（{spec.category}）{spec.description}\n"
        f"售价：{price:g} {spec.currency}。\n\n"
        f"请根据你自己的实际情况、消费能力和习惯判断：你对这款商品的购买兴趣是？\n"
        f"{opts}\n\n"
        f"只回答一个字母（A/B/C/D），不要解释。"
    )


TEMPLATE_YAML = """\
# who-will-buy 商品信息模板 —— 填完跑：python -m whowillbuy run product.yaml
name: 便携手冲咖啡机          # 商品名（不要含价格）
category: 小家电 / 咖啡器具    # 品类
description: 一分钟出杯的便携手冲咖啡机，机身 500g，USB-C 充电，适合通勤和出差。  # 一句话卖点（不要含价格）
price_points: [199, 299, 399]  # 候选价位（2~6 个）
# main_price: 299              # 受众分析用的主价位，缺省取中位数
# market:                      # 可选：目标市场过滤
#   city_tier: [tier1, tier2]  # 只看一二线
"""
