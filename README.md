# who-will-buy

填一份商品信息,预测"谁会买"和"卖多少钱能接受"。

起因很简单:产品上线前最想知道的两件事——受众是谁、定价多少——传统答案是问卷调研,
贵且慢,大多数早期想法根本轮不到"值得花几万块调研"的待遇,最后就是拍脑袋。
学术圈这几年有一条线叫 silicon sampling(用大模型扮演不同人口学背景的人来近似调查),
我想验证它在消费决策这个具体场景里能不能工程化成一个可用的工具。这个仓库就是结果。

## 用法

```bash
git clone https://github.com/zammel1987/who-will-buy && cd who-will-buy
pip install -r requirements.txt

# 不花钱先跑通(mock 模式,不调用任何 API)
python -m whowillbuy run examples/coffee_maker.yaml

# 生成模板,填自己的商品
python -m whowillbuy init my_product.yaml

# 接真实大模型(自备 Key)
export DEEPSEEK_API_KEY=sk-xxx
python -m whowillbuy run my_product.yaml --provider deepseek --budget 2
```

product.yaml 只有四个必填字段:

```yaml
name: 便携手冲咖啡机
category: 小家电 / 咖啡器具
description: 一分钟出杯的便携手冲咖啡机,机身 500g,USB-C 充电,适合通勤和出差场景。
price_points: [199, 299, 399]
```

跑完在 `out/<商品名>/report.md` 拿到一份报告:七个人口学维度的购买兴趣分层(TGI 排序)、
各价位的接受度曲线(带 bootstrap 波动带)、以及完整的口径说明。

[examples/sample_output/](examples/sample_output/) 里放了 DeepSeek 真实档和 mock 各一份。
真实档那份值得一看:收入梯度完全单调(家庭人均月收入 <5k 的兴趣率 2.4%,10-20k 是 8.3%,
20-40k 到 50%),价格曲线方向也对——这些结构没有任何硬编码,是模型角色扮演聚合出来的。

## 它是怎么工作的

两步,统计的归统计,语言模型的归语言模型:

**第一步,用 IPF 造人群骨架。** 从公开普查聚合数据(七普公报的年龄/性别/学历/城乡边际,
详见 `whowillbuy/data/census_marginals.json`,逐项附来源)出发,迭代比例拟合(IPF)出一个
边际对齐真实人口结构的联合分布,从中抽 3000 个虚拟个体。这一步不用大模型——
只凭边际分布还原不了联合分布,让 LLM 直接"生成 3000 个中国人"会众数坍塌、
组合不自洽(会出现大量"18 岁退休研究生"),骨架必须由统计方法保证。

**第二步,逐人角色扮演作答。** 把每个虚拟个体的人口学画像写成 system prompt
("你是一位 25-34 岁的女性,本科学历,生活在二线城市……"),问它对这个商品、
这个价位的购买兴趣,四档选项。所有回答聚合起来,就是受众结构和价格弹性。

为什么不直接问大模型"分析一下这个产品的受众"?因为那样只会得到一段谁都能写的车轱辘话。
逐人作答的意义在于输出是**可分层、可交叉、可对比价位的结构化分布**,
而且每一层的样本量、波动带都是明确的。

## 踩过的坑(本项目最有信息量的部分)

这些问题不修,跑出来的数字会以很隐蔽的方式错掉。欢迎在 Issues 里聊这类问题。

**购买问法塌缩。** 最早的问法是"未来 6 个月内你购买这款商品的可能性",结果 1600 个回答里
76% 塌缩到"肯定不会买",零个"肯定会买"——对齐过的大模型在扮演普通人时,对"承诺花钱"
极端保守,整个分布的区分度直接归零。把问法软化成"你对这款商品的购买兴趣是?"之后,
分层结构立刻拉开(就是上面那个单调的收入梯度)。所以本项目的口径明确定义为
**兴趣/考虑集**而非实购转化,报告里也这么写。elicitation 的措辞对结果的影响,
比大多数人想象的大一个数量级。

**选项位置偏置。** 大模型对选项顺序不是中性的,固定顺序会系统性偏向某些位置。
每次作答选项随机排列,解析时再映射回语义。

**价格多通道打架。** 如果商品描述里写了"仅需 199 元",而价格扫描又在问 399 元的接受度,
两个价格信号会互相污染。所以价格只允许通过 `price_points` 单通道注入,
描述里出现价格数字会直接报错拒跑。

**单次输出不是答案。** 每人采样 k 次取分布(temperature 采样),
不拿一次贪心解码当作"这个人的唯一答案"。

**成本要在调用前拦住。** `--budget` 的估价和拦截发生在任何网络调用之前;
真实档跑批前还会先发一次预检调用,Key 无效或余额不足立刻报明白,不会烧完一整批才发现。

## 边界与已知局限

诚实地说,这个工具能给你的是**方向和排序**:哪层人群兴趣高、涨价之后掉多少的方向。
绝对值未经任何真实数据校准,3.2% 的兴趣率不等于 3.2% 的市场渗透率,请不要把它写进 BP。

已知局限,也是想继续做的方向:

- **persona 只有人口学骨架**。没有消费习惯、生活方式这些"血肉",子群差异全靠模型对
  人口学标签的先验。加一层画像富化能提升多少区分度,是个值得量化的问题。
- **绝对值校准是真正的难题**。要让数字可引用,需要拿真实调查数据做下游校准
  (分层加权、响应纠偏、分布校准),这超出了一个开源 demo 的数据条件。
- **大模型后端非平稳**。同配置隔一小时跑,数值可能漂移;跨时段对比请同窗复跑,看波动带。
- **样本量**。默认每场景 200-300 人,细分层(比如高收入段)样本很小,TGI 榜设了
  最小样本量门槛,但小分层的数字仍要谨慎。

我自己正在真实调查微数据和校准层的方向上做一个更完整的版本(数据授权、
富化、校准这套东西比开源 demo 复杂不少)。对这个方向有兴趣、或者手里有类似问题的,
欢迎开 Issue 交流。

## 模型后端与成本

| provider | 环境变量 | 默认模型 |
|---|---|---|
| `mock` | 无 | — |
| `deepseek` | `DEEPSEEK_API_KEY` | deepseek-chat |
| `qwen` | `DASHSCOPE_API_KEY` | qwen-plus |
| `glm` | `ZHIPU_API_KEY` | glm-4-flash |
| `custom` | `OPENAI_API_KEY` + `OPENAI_BASE_URL` | `--model` 指定 |

默认配置一次真实跑批约 1600-4500 次调用,deepseek 大概 ¥1.5-5。
所有输出带"AI 仿真生成"标注;自部署接入大模型 API 的服务义务由使用者自行承担。

## License

Apache-2.0

---

*who-will-buy simulates "who would buy" (audience segments with TGI index) and "at what price"
(price acceptance curves with bootstrap bands) for any product, by IPF-constructing a synthetic
population from public census aggregates and having an LLM role-play each individual.
Directional insights only — absolute values are uncalibrated by design. Issues welcome.*
