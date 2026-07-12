# -*- coding: utf-8 -*-
"""离线自测：IPF 边际 / spec 编译纪律 / mock 全链路 / 报告口径，零成本零网络。"""
from __future__ import annotations

import json
import os
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, "data", "census_marginals.json")

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    print(f"  [{'✓' if cond else '✗'}] {name}" + (f"  — {detail}" if detail else ""))
    if cond:
        _passed += 1
    else:
        _failed += 1


def main() -> int:
    from .audience import audience_table, price_curve, top_segments
    from .product_spec import TEMPLATE_YAML, load_spec
    from .report import CONTACT, build_report
    from .simulate import run_simulation
    from .skeleton import generate_skeleton, load_config

    print("=" * 56)
    print("who-will-buy 离线自测（零成本零网络）")
    print("=" * 56)

    with tempfile.TemporaryDirectory() as tmp:
        # 1) IPF 骨架：边际大体贴合普查目标
        pop_csv = os.path.join(tmp, "pop.csv")
        generate_skeleton(CENSUS, n=2000, out_csv=pop_csv, verbose=False)
        pop = pd.read_csv(pop_csv)
        check("IPF 骨架生成 2000 人", len(pop) == 2000)
        cfg = load_config(CENSUS)
        tgt = cfg["variables"]["gender"]["target_proportions"]
        share_male = (pop["gender"] == "male").mean()
        check("性别边际贴合普查目标（±5pp）", abs(share_male - tgt[0]) < 0.05,
              f"male={share_male:.3f} vs 目标 {tgt[0]:.3f}")

        # 2) spec 编译纪律
        spec_path = os.path.join(tmp, "product.yaml")
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(TEMPLATE_YAML)
        spec = load_spec(spec_path)
        check("模板 spec 可加载", spec.name != "" and len(spec.price_points) == 3)
        bad = spec_path + ".bad"
        with open(bad, "w", encoding="utf-8") as f:
            f.write(TEMPLATE_YAML.replace("一分钟出杯", "只要 199 元，一分钟出杯"))
        try:
            load_spec(bad)
            check("描述含价格被拦截（单通道纪律）", False)
        except SystemExit:
            check("描述含价格被拦截（单通道纪律）", True)

        # 3) mock 全链路 + 结果结构
        sim = run_simulation(spec, pop, provider="mock", n_sample=200, k=2, seed=11)
        check("mock 仿真出长表", len(sim) == 200 * 2 * (1 + 3),
              f"rows={len(sim)}")
        table, overall = audience_table(sim)
        tops = top_segments(table)
        curve = price_curve(sim, n_boot=100)
        check("受众表覆盖全部 7 维", table["dim"].nunique() == 7)
        check("全体意向率在合理区间", 0.02 < overall < 0.8, f"{overall:.3f}")
        check("定价曲线首价位意向 ≥ 末价位（mock 弹性方向）",
              curve.iloc[0]["意向率"] >= curve.iloc[-1]["意向率"],
              f"{curve.iloc[0]['意向率']:.3f} vs {curve.iloc[-1]['意向率']:.3f}")
        check("波动带包住点估计", bool(((curve["波动带下"] <= curve["意向率"])
                                        & (curve["意向率"] <= curve["波动带上"])).all()))

        # 4) 报告硬口径
        md = build_report(spec, "mock", 2000, 200, 2, table, overall, tops, curve, 1.0)
        for kw in ("AI 仿真生成", "绝对值未经真实数据校准", "波动带", CONTACT):
            check(f"报告含硬口径：{kw[:12]}…", kw in md)

        # 5) 数据文件干净（不含真实微数据引用）
        raw = json.dumps(json.load(open(CENSUS, encoding="utf-8")), ensure_ascii=False)
        check("边际数据为公开聚合口径（含来源说明）", "_data_sources" in raw and "普查" in raw)

    print("=" * 56)
    print(f"结果：通过 {_passed} / 失败 {_failed}")
    print("=" * 56)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
