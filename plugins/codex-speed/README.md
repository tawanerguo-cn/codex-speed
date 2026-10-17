# Codex Speed

查询最近真实工作的生成速率。普通查询读取本地日志，不发起模型请求；明确要求跑测速时才发起一次新生成，并消耗正常模型 token。

## 安装

```sh
codex plugin marketplace add tawanerguo-cn/codex-speed --ref main
codex plugin add codex-speed@tawanerguo-codex-tools
```

## 使用

- “查一下速率”：读取最近真实工作的记录。
- “跑一次测速”：执行固定提示词的单轮测速。
- “查测速历史”：读取已保存记录。

没有可计算的近期记录时会说明原因，不会自动发起测速。需要 Codex CLI 和 Python 3.11+。

本插件部分代码来自 [TokenPulse](https://github.com/zzccchen/token-pulse)，遵循 MIT License。
