# -*- coding: utf-8 -*-
"""大模型客户端：读 .env 里的密钥，走 OpenAI 兼容协议调用国产大模型。

没有配置密钥、或者网络/额度有问题时，抛 ``LLMUnavailable``，
上层会降级成本地模板日报，保证界面永远有内容可看。
"""
from __future__ import annotations

import os
from typing import Iterator, List, Optional

from ..config import Config


class LLMUnavailable(RuntimeError):
    """没有密钥 / 调用失败时抛出，调用方据此降级。"""


_client_cache = {}


def load_api_key(cfg: Config) -> Optional[str]:
    """先看环境变量，再读项目根目录的 .env 文件。"""
    key = os.getenv(cfg.llm.api_key_env)
    if key:
        return key.strip()
    path = cfg.llm.env_path
    if os.path.exists(path):
        try:
            from dotenv import load_dotenv
            # 显式传路径：避免 find_dotenv() 在某些运行方式下找不到文件
            load_dotenv(dotenv_path=path, override=False)
        except ImportError:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith(cfg.llm.api_key_env + "="):
                        return line.split("=", 1)[1].strip()
        key = os.getenv(cfg.llm.api_key_env)
    return key.strip() if key else None


def has_api_key(cfg: Config) -> bool:
    return bool(load_api_key(cfg))


def get_client(cfg: Config):
    """返回 OpenAI 兼容客户端（同一份代码可对接 DeepSeek / GLM / 千问 / Kimi）。"""
    key = load_api_key(cfg)
    if not key:
        raise LLMUnavailable(
            "没有找到 API 密钥。请在项目根目录的 .env 里填 %s=你的密钥"
            % cfg.llm.api_key_env)
    cached = _client_cache.get(cfg.llm.base_url)
    if cached is not None:
        return cached
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMUnavailable("缺少 openai 包，请先 pip install openai") from exc
    client = OpenAI(api_key=key, base_url=cfg.llm.base_url, timeout=cfg.llm.timeout)
    _client_cache[cfg.llm.base_url] = client
    return client


def _friendly_error(exc: Exception) -> str:
    """把 SDK 抛出的异常翻译成人能看懂的话。"""
    name = type(exc).__name__
    text = str(exc)
    if "AuthenticationError" in name or "401" in text:
        return "API 密钥无效或已过期，请检查 .env 里的密钥"
    if "InsufficientBalance" in text or "402" in text:
        return "账户余额不足，请到服务商控制台充值"
    if "RateLimit" in name or "429" in text:
        return "请求太频繁（触发限流），稍等几秒再试"
    if "Connection" in name or "Timeout" in name or "APIConnection" in name:
        return "网络连接失败，请检查网络（国产模型不需要代理）"
    return "%s: %s" % (name, text[:200])


def chat(cfg: Config, messages: List[dict], tools: Optional[list] = None,
         temperature: Optional[float] = None):
    """调用一次对话补全，返回 message 对象（可能带 tool_calls）。"""
    client = get_client(cfg)
    kwargs = {
        "model": cfg.llm.model,
        "messages": messages,
        "temperature": cfg.llm.temperature if temperature is None else temperature,
        "max_tokens": cfg.llm.max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:      # 统一翻译成可读错误
        raise LLMUnavailable(_friendly_error(exc)) from exc
    return resp.choices[0].message, getattr(resp, "usage", None)


def chat_stream(cfg: Config, messages: List[dict],
                temperature: Optional[float] = None) -> Iterator[str]:
    """流式输出，逐段 yield 文本（界面里可以边生成边显示）。"""
    client = get_client(cfg)
    try:
        stream = client.chat.completions.create(
            model=cfg.llm.model, messages=messages,
            temperature=cfg.llm.temperature if temperature is None else temperature,
            max_tokens=cfg.llm.max_tokens, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece
    except Exception as exc:
        raise LLMUnavailable(_friendly_error(exc)) from exc
