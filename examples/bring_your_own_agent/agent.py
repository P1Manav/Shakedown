"""Minimal OpenAI-compatible client; configure its endpoint with OPENAI_BASE_URL."""

import json
import os
from urllib.request import Request, urlopen


base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
request = Request(
    f"{base_url}/chat/completions",
    data=json.dumps({
        "model": os.environ.get("OPENAI_MODEL", "qwen2.5-coder:7b"),
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Say hello."},
        ],
        "temperature": 0,
    }).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=60) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
