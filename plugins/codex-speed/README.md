# Codex Speed

按需测量 Codex 输出速率并查询历史记录。需要 Codex CLI 和 Python 3.11+。

## 安装

```sh
codex plugin marketplace add tawanerguo-cn/codex-speed --ref main
codex plugin add codex-speed@tawanerguo-codex-tools
```

## 使用

安装后直接对 Codex 说“测一下 Codex 速率”或“查测速历史”。

生成速率按输出项时间估算，包含推理 token；整轮耗时包含模型等待。每次测速都会实际调用模型并消耗正常 token。

本插件部分代码来自 [TokenPulse](https://github.com/zzccchen/token-pulse)，遵循 MIT License；版权归 TokenPulse contributors。
