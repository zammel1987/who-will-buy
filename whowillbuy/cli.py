# -*- coding: utf-8 -*-
"""who-will-buy 命令行入口。

  python -m whowillbuy init                 # 生成 product.yaml 模板
  python -m whowillbuy run product.yaml     # mock 免费跑通
  python -m whowillbuy run product.yaml --provider deepseek --budget 2
  python -m whowillbuy selftest             # 离线自测
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, "data", "census_marginals.json")


def _ensure_population(n: int, seed: int, out_dir: str, quiet: bool = True) -> pd.DataFrame:
    """构建（或读缓存）纯合成人群骨架。"""
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, f"population_n{n}_seed{seed}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    from .skeleton import generate_skeleton
    print(f"[人群] 首次构建合成人群 n={n}（公开普查聚合边际 + IPF），约几秒…")
    generate_skeleton(CENSUS, n=n, out_csv=cache, verbose=not quiet)
    return pd.read_csv(cache)


def cmd_init(args) -> int:
    from .product_spec import TEMPLATE_YAML
    path = args.path or "product.yaml"
    if os.path.exists(path) and not args.force:
        print(f"{path} 已存在（--force 覆盖）")
        return 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE_YAML)
    print(f"已生成 {path}，填好后：python -m whowillbuy run {path}")
    return 0


def cmd_run(args) -> int:
    from .audience import audience_table, price_curve, top_segments
    from .product_spec import load_spec
    from .report import build_report
    from .simulate import run_simulation

    spec = load_spec(args.spec)
    out_dir = args.out or os.path.join("out", spec.name.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    pop = _ensure_population(args.n, args.seed, out_dir=os.path.join("out", "_population"))
    sim = run_simulation(
        spec, pop, provider=args.provider, model=args.model,
        n_sample=args.sample, k=args.k, workers=args.workers,
        budget_usd=args.budget, seed=args.seed, temperature=args.temperature)
    if len(sim) == 0:
        print("没有有效作答，请检查 Key/网络后重试。")
        return 1
    sim.to_csv(os.path.join(out_dir, "raw_answers.csv"), index=False, encoding="utf-8-sig")

    table, overall = audience_table(sim)
    tops = top_segments(table)
    curve = price_curve(sim)
    md = build_report(spec, args.provider, n_pop=args.n, n_sample=min(args.sample, len(pop)),
                      k=args.k, aud_table=table, overall=overall, tops=tops, curve=curve,
                      answer_rate=float(sim.attrs.get("answer_rate", 1.0)))
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n[完成] 报告：{report_path}")
    print(f"       原始作答：{os.path.join(out_dir, 'raw_answers.csv')}")
    print(f"       全体意向率 {overall * 100:.1f}%（{'mock 演示值' if args.provider == 'mock' else '未校准仿真值'}，"
          "相对结构可参考，绝对值勿直接引用）")
    return 0


def cmd_selftest(_args) -> int:
    from .selftest import main as st_main
    return st_main()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="whowillbuy", description="谁会买？——商品受众与价格接受度仿真")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="生成 product.yaml 模板")
    p_init.add_argument("path", nargs="?", default=None)
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(fn=cmd_init)

    p_run = sub.add_parser("run", help="跑受众+定价仿真")
    p_run.add_argument("spec", help="product.yaml 路径")
    p_run.add_argument("--provider", default="mock",
                       choices=["mock", "deepseek", "qwen", "glm", "custom"])
    p_run.add_argument("--model", default=None, help="覆盖默认模型名（custom 必填）")
    p_run.add_argument("--n", type=int, default=3000, help="合成人群规模（默认 3000）")
    p_run.add_argument("--sample", type=int, default=300, help="每场景抽样人数（默认 300）")
    p_run.add_argument("--k", type=int, default=3, help="每人采样次数（默认 3）")
    p_run.add_argument("--workers", type=int, default=8)
    p_run.add_argument("--budget", type=float, default=None,
                       help="预算上限 USD；超出在任何网络调用前拦截")
    p_run.add_argument("--temperature", type=float, default=0.8)
    p_run.add_argument("--seed", type=int, default=2026)
    p_run.add_argument("--out", default=None, help="产物目录（默认 out/<商品名>）")
    p_run.set_defaults(fn=cmd_run)

    p_st = sub.add_parser("selftest", help="离线自测（零成本）")
    p_st.set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
