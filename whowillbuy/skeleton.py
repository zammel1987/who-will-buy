#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合成人群骨架生成（IPF / Raking）

思路：
  1) 用 build_seed_sample() 程序化生成合成种子（内置常识性的属性间关联结构，
     如年龄×学历、收入×城市层级）。有自有调研微数据的用户可自行替换种子来源。
  2) 由种子构建多维列联表（contingency table），保留关联结构。
  3) 用 IPF 迭代调整该列联表，使其各维边际分布对齐【公开普查聚合边际】，得到校准后的联合分布。
  4) 从校准联合分布中抽样 N 个个体，作为人口骨架。
  5) 对实际抽样做 raking，赋予事后分层权重 weight（修正有限样本的边际漂移）。

为什么不能让 LLM 直接造骨架：仅凭边际分布无法还原联合分布；LLM 直接生成会众数坍塌、
分布失真、组合不自洽。骨架必须由统计方法（IPF + 种子关联）保证。
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))  # 默认路径锚定脚本目录，可在任意 cwd 运行


# --------------------------------------------------------------------------- #
# 载入配置
# --------------------------------------------------------------------------- #
def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 种子微数据（代表真实样本的关联结构）
#   正式使用：删除本函数，改为 pd.read_csv("your_microdata.csv")，
#   并确保列名与 var_order 一致、取值落在各变量 categories 内。
# --------------------------------------------------------------------------- #
def build_seed_sample(cfg, n_seed=8000, seed=42):
    rng = np.random.default_rng(seed)
    cats = {v: cfg["variables"][v]["categories"] for v in cfg["var_order"]}

    def pick(options, probs):
        probs = np.array(probs, dtype=float)
        probs /= probs.sum()
        return rng.choice(options, p=probs)

    rows = []
    for _ in range(n_seed):
        # 潜在社会经济得分，制造 教育/收入/职业/城市 之间的相关性
        ses = rng.normal(0, 1)

        age = pick(cats["age"], [0.14, 0.24, 0.20, 0.16, 0.14, 0.12])
        gender = pick(cats["gender"], [0.5, 0.5])

        # 城市分级：ses 越高越偏一线
        tier = pick(cats["city_tier"],
                    [0.18 + 0.12 * (ses > 0.5), 0.35, 0.47 - 0.12 * (ses > 0.5)])

        # 城乡：与城市分级强相关（一二线几乎全城镇，三线及以下含较多乡村）；
        # 整体占比由 IPF 拉到七普 63.89/36.11，这里只负责制造与 tier 的真实关联结构
        if tier == "tier1":
            urban_rural = pick(cats["urban_rural"], [0.97, 0.03])
        elif tier == "tier2":
            urban_rural = pick(cats["urban_rural"], [0.85, 0.15])
        else:
            urban_rural = pick(cats["urban_rural"], [0.45, 0.55])

        # 教育：受 ses 与年龄影响（年轻 + 高 ses → 学历更高）
        young = age in ["18-24", "25-34"]
        edu_score = ses + (0.6 if young else 0) + (0.4 if tier == "tier1" else 0)
        if edu_score > 1.2:
            education = pick(cats["education"], [0.05, 0.15, 0.45, 0.35])
        elif edu_score > 0.2:
            education = pick(cats["education"], [0.25, 0.30, 0.35, 0.10])
        else:
            education = pick(cats["education"], [0.70, 0.20, 0.09, 0.01])

        # 收入：受 教育 / 年龄 / ses 影响
        edu_rank = cats["education"].index(education)
        prime_age = age in ["25-34", "35-44", "45-54"]
        inc_score = 0.8 * edu_rank + ses + (0.6 if prime_age else -0.3) \
            + (0.5 if tier == "tier1" else 0)
        if inc_score > 3.0:
            income = pick(cats["income_band"], [0.02, 0.10, 0.30, 0.38, 0.20])
        elif inc_score > 1.8:
            income = pick(cats["income_band"], [0.08, 0.27, 0.40, 0.20, 0.05])
        elif inc_score > 0.6:
            income = pick(cats["income_band"], [0.30, 0.42, 0.22, 0.05, 0.01])
        else:
            income = pick(cats["income_band"], [0.62, 0.30, 0.07, 0.01, 0.00])

        # 职业：受 年龄 / 教育 影响
        if age == "18-24" and education in ["bachelor", "postgrad"]:
            occupation = pick(cats["occupation"], [0.55, 0.05, 0.15, 0.18, 0.07, 0.00])
        elif age == "65+":
            occupation = pick(cats["occupation"], [0.00, 0.08, 0.10, 0.05, 0.02, 0.75])
        elif edu_rank >= 2:
            occupation = pick(cats["occupation"], [0.03, 0.05, 0.12, 0.40, 0.38, 0.02])
        else:
            occupation = pick(cats["occupation"], [0.05, 0.40, 0.30, 0.18, 0.05, 0.02])

        rows.append(dict(age=age, gender=gender, city_tier=tier,
                         urban_rural=urban_rural,
                         education=education, income_band=income,
                         occupation=occupation))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 列联表 + IPF
# --------------------------------------------------------------------------- #
def seed_to_table(df_seed, cfg, smoothing=0.5):
    var_order = cfg["var_order"]
    cats = {v: cfg["variables"][v]["categories"] for v in var_order}
    code = {v: {c: i for i, c in enumerate(cats[v])} for v in var_order}
    dims = [len(cats[v]) for v in var_order]
    table = np.full(dims, smoothing, dtype=float)  # Laplace 平滑，避免空格
    idxs = np.stack([df_seed[v].map(code[v]).values for v in var_order], axis=1)
    for row in idxs:
        table[tuple(row)] += 1.0
    apply_structural_zeros(table, cfg)  # 平滑之后立刻清零不可能组合，避免给它们留概率底
    return table


def apply_structural_zeros(table, cfg, verbose=False):
    """把逻辑上不可能的组合（structural zeros）对应的单元格强制置 0 且不平滑。

    规则来自 cfg["structural_zeros"]：每条 conditions 是「变量→允许取值列表」的字典，
    同一单元格【所有变量都命中各自列表】(AND) 才被清零；单变量内多取值是 OR。
    置 0 后 IPF 的乘法更新不会再把零格抬起来，从而从联合分布中物理删除不自洽组合。
    """
    rules = cfg.get("structural_zeros") or []
    if not rules:
        return table
    var_order = cfg["var_order"]
    cats = {v: cfg["variables"][v]["categories"] for v in var_order}
    zeroed_before = int((table == 0).sum())
    for rule in rules:
        conds = rule.get("conditions", {})
        # 为每个 ndim 维构造该规则命中的下标集合；未在 conditions 出现的维度=全选
        idx_per_dim = []
        for d, v in enumerate(var_order):
            if v in conds:
                allowed = [cats[v].index(c) for c in conds[v] if c in cats[v]]
                idx_per_dim.append(np.array(allowed, dtype=int))
            else:
                idx_per_dim.append(np.arange(len(cats[v])))
        table[np.ix_(*idx_per_dim)] = 0.0  # 笛卡尔积区域整体清零
    if verbose:
        zeroed = int((table == 0).sum()) - zeroed_before
        print(f"[结构性零] 应用 {len(rules)} 条规则，本次清零单元格 {zeroed} 个")
    return table


def structural_zero_violations(df, cfg):
    """统计抽样人群中落入结构性零组合的个体数（应为 0）。"""
    rules = cfg.get("structural_zeros") or []
    n = 0
    for rule in rules:
        conds = rule.get("conditions", {})
        mask = np.ones(len(df), dtype=bool)
        for v, allowed in conds.items():
            mask &= df[v].isin(allowed).to_numpy()
        n += int(mask.sum())
    return n


def ipf(seed_table, targets, max_iter=2000, tol=1e-7):
    """targets[d]: 长度 = 第 d 维类别数，已按 population_total 缩放，各维总和相等。"""
    table = seed_table.astype(float).copy()
    T = float(np.sum(targets[0]))
    table *= T / table.sum()
    history = []
    for it in range(max_iter):
        for d, tg in enumerate(targets):
            axes = tuple(i for i in range(table.ndim) if i != d)
            cur = table.sum(axis=axes)
            factor = np.where(cur > 0, tg / cur, 1.0)
            shape = [1] * table.ndim
            shape[d] = len(tg)
            table *= factor.reshape(shape)
        err = max(
            float(np.abs(table.sum(axis=tuple(i for i in range(table.ndim) if i != d)) - tg).max())
            for d, tg in enumerate(targets)
        )
        history.append(err)
        if err < tol:
            break
    return table, it + 1, err, history


# --------------------------------------------------------------------------- #
# 抽样 + raking 权重
# --------------------------------------------------------------------------- #
def sample_population(table, cfg, n, seed=7):
    rng = np.random.default_rng(seed)
    var_order = cfg["var_order"]
    cats = {v: cfg["variables"][v]["categories"] for v in var_order}
    probs = table.flatten()
    probs = probs / probs.sum()
    flat_idx = rng.choice(len(probs), size=n, p=probs)
    multi = np.array(np.unravel_index(flat_idx, table.shape)).T  # n x ndim
    data = {v: [cats[v][multi[i, d]] for i in range(n)] for d, v in enumerate(var_order)}
    return pd.DataFrame(data)


def rake_weights(df, cfg, max_iter=80, tol=1e-7):
    var_order = cfg["var_order"]
    cats = {v: cfg["variables"][v]["categories"] for v in var_order}
    targets = {v: np.array(cfg["variables"][v]["target_proportions"], float) for v in var_order}
    code = {v: df[v].map({c: i for i, c in enumerate(cats[v])}).values for v in var_order}
    w = np.ones(len(df))
    for _ in range(max_iter):
        maxerr = 0.0
        for v in var_order:
            tg = targets[v] / targets[v].sum()
            for ci in range(len(cats[v])):
                mask = code[v] == ci
                cur = w[mask].sum() / w.sum()
                if cur > 0:
                    w[mask] *= tg[ci] / cur
                    maxerr = max(maxerr, abs(tg[ci] - cur))
        if maxerr < tol:
            break
    w *= len(w) / w.sum()  # 归一到均值 1
    return w


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def generate_skeleton(config_path, n, out_csv, n_seed=8000, seed_df=None, verbose=True):
    cfg = load_config(config_path)
    var_order = cfg["var_order"]
    total = cfg.get("population_total", n)

    # seed_df 不为空时用调用方自备的种子微数据，否则用内置合成种子
    df_seed = seed_df.copy() if seed_df is not None else build_seed_sample(cfg, n_seed=n_seed)
    seed_violations = structural_zero_violations(df_seed, cfg)  # 种子里本可能存在的不自洽组合
    table = seed_to_table(df_seed, cfg)

    targets = [np.array(cfg["variables"][v]["target_proportions"], float) for v in var_order]
    targets = [t / t.sum() * total for t in targets]  # 缩放到统一总量

    fitted, iters, err, _ = ipf(table, targets)
    if verbose:
        print(f"[IPF] 收敛于 {iters} 次迭代，最大边际误差 = {err:.3e}")

    df = sample_population(fitted, cfg, n)
    df["weight"] = rake_weights(df, cfg)
    df.insert(0, "agent_id", [f"cn-{i:06d}" for i in range(len(df))])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_violations = structural_zero_violations(df, cfg)
    if verbose and cfg.get("structural_zeros"):
        print(f"[结构性零] 种子中不自洽组合 {seed_violations} 个 → 骨架抽样中 {out_violations} 个"
              f"（应为 0，如『45+在校学生 / 45 岁以下退休』已被物理删除）")

    if verbose:
        print(f"[OUT] 已写出 {len(df)} 个骨架个体 -> {out_csv}")
        print("\n[拟合检查] 目标 vs 实际(加权) 边际分布：")
        for v in var_order:
            cats = cfg["variables"][v]["categories"]
            tgt = np.array(cfg["variables"][v]["target_proportions"], float)
            tgt = tgt / tgt.sum()
            real = (df.groupby(v)["weight"].sum() / df["weight"].sum()).reindex(cats).fillna(0)
            line = "  ".join(f"{c}:{t:.2f}/{r:.2f}" for c, t, r in zip(cats, tgt, real.values))
            print(f"  - {v:11s} {line}")
        print("  （格式 类别:目标/实际）")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="生成合成人群骨架（IPF，合成种子）")
    ap.add_argument("--config", default=os.path.join(HERE, "data", "census_marginals.json"))
    ap.add_argument("--n", type=int, default=3000, help="生成个体数")
    ap.add_argument("--n_seed", type=int, default=8000, help="合成种子样本量")
    ap.add_argument("--out", default="skeleton.csv")
    args = ap.parse_args()
    generate_skeleton(args.config, args.n, args.out, n_seed=args.n_seed)
