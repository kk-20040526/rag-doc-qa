# -*- coding: utf-8 -*-
"""检查 02 项目的 .env 是否填好、百炼 Key 是否真能调通。

用法（在工程目录）：
    .\.venv\Scripts\python.exe scripts_check_key.py

它做三件事：
    1) 确认 .env 被读到（Key 不是占位文字）
    2) 打一次真实请求（1 个 token 级的小问题），看 Key/地址/模型名是否都对
    3) 出错时给具体原因，而不是一句 Exception
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

KEY = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")
BASE = os.getenv("QWEN_BASE_URL", "")

print("[1/3] 配置检查")
if not KEY:
    print("      没读到 Key：.env 里 DASHSCOPE_API_KEY 是空的")
    sys.exit(1)
if not KEY.startswith("sk-") or len(KEY) < 20:
    print("      Key 看起来不是有效值：%r（长度 %d）" % (KEY[:12], len(KEY)))
    print("      请把 .env 里的占位文字换成控制台复制的完整 Key")
    sys.exit(1)
print("      Key 已读到（长度 %d）" % len(KEY))
print("      model =", MODEL)
print("      base  =", BASE)
if not BASE:
    print("      QWEN_BASE_URL 为空，但代码会退回默认地址（不推荐）")

print("[2/3] 真实调用一次")
try:
    from openai import OpenAI
except ImportError:
    print("      缺少 openai 依赖：pip install openai")
    sys.exit(1)

client = OpenAI(api_key=KEY, base_url=BASE, timeout=30)
try:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "只回复两个字：就绪"}],
        temperature=0,
    )
    print("      返回：%r" % r.choices[0].message.content)
except Exception as exc:
    msg = str(exc)
    print("      失败：%s" % msg)
    if "401" in msg or "Incorrect API key" in msg or "invalid_api_key" in msg:
        print("      → Key 不对，或 Key 与 Base URL 不属于同一个业务空间")
    elif "404" in msg or "model" in msg.lower():
        print("      → 模型名 %s 可能在该空间不可用，试试 qwen-plus 或 qwen-turbo 全名" % MODEL)
    elif "Connection" in msg or "timed out" in msg.lower():
        print("      → 网络问题：确认能访问 %s" % BASE)
    sys.exit(1)

print("[3/3] 结论：Key 可用，02 项目可以生成答案了。")
