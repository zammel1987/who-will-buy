# -*- coding: utf-8 -*-
"""LLM 接入层：OpenAI 兼容 chat 接口（deepseek / qwen / glm / custom），零 SDK 依赖。

- mock 模式不经此模块（见 simulate.py 的启发式作答器）。
- API Key 一律走环境变量，绝不写进代码或配置文件。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

# 价格为估算用（USD / 1M tokens，2026-07 参考值）；跑批前请以各官网现价为准，可用 --price-in/--price-out 覆盖。
PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "base": "https://api.deepseek.com/v1",
        "env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "price_in": 0.27, "price_out": 1.10,
    },
    "qwen": {
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env": "DASHSCOPE_API_KEY",
        "model": "qwen-plus",
        "price_in": 0.40, "price_out": 1.20,
    },
    "glm": {
        "base": "https://open.bigmodel.cn/api/paas/v4",
        "env": "ZHIPU_API_KEY",
        "model": "glm-4-flash",
        "price_in": 0.10, "price_out": 0.10,
    },
    "custom": {  # 任意 OpenAI 兼容端点：OPENAI_BASE_URL + OPENAI_API_KEY，--model 必填
        "base": None,
        "env": "OPENAI_API_KEY",
        "model": None,
        "price_in": 1.00, "price_out": 2.00,
    },
}


class LLMError(RuntimeError):
    pass


def resolve(provider: str, model: str | None = None) -> dict:
    """解析 provider 配置；返回 {base, key, model, price_in, price_out}。"""
    if provider not in PROVIDERS:
        raise LLMError(f"未知 provider: {provider}（可选 {'/'.join(PROVIDERS)} 或 mock）")
    cfg = dict(PROVIDERS[provider])
    if provider == "custom":
        cfg["base"] = os.environ.get("OPENAI_BASE_URL") or ""
        if not cfg["base"]:
            raise LLMError("custom 模式需设置环境变量 OPENAI_BASE_URL")
    key = os.environ.get(cfg["env"], "")
    if not key:
        raise LLMError(f"缺少环境变量 {cfg['env']}（{provider} 的 API Key）")
    cfg["key"] = key
    cfg["model"] = model or cfg["model"]
    if not cfg["model"]:
        raise LLMError("custom 模式需用 --model 指定模型名")
    return cfg


def chat(cfg: dict, messages: list[dict], temperature: float = 0.8,
         max_tokens: int = 8, timeout: int = 60, retries: int = 3) -> tuple[str, int, int]:
    """单次 chat 调用。返回 (文本, 输入tokens, 输出tokens)。指数退避重试。"""
    payload = json.dumps({
        "model": cfg["model"], "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        cfg["base"].rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['key']}"})
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
            return text.strip(), int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "ignore")[:200]
            except OSError:
                pass
            if e.code in (400, 401, 402, 403, 404, 422):   # 账号/配置类错误：重试无意义，快速失败
                hint = {401: "Key 无效或未授权", 402: "账号余额不足，请前往平台充值",
                        403: "无权限", 404: "端点或模型名不存在"}.get(e.code, "请求被拒绝")
                raise LLMError(f"HTTP {e.code}（{hint}）：{body}") from e
            last_err = e
            time.sleep(1.5 ** attempt)
        except (urllib.error.URLError, KeyError,
                json.JSONDecodeError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(1.5 ** attempt)
    raise LLMError(f"LLM 调用失败（重试 {retries} 次）：{last_err}")
