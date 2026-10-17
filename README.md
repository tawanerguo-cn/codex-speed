# Codex Speed

查最近真实工作的 Codex 生成速率；只有明确要求跑测速时，才额外生成内容。

## 安装

```sh
codex plugin marketplace add tawanerguo-cn/codex-speed --ref main
codex plugin add codex-speed@tawanerguo-codex-tools
```

## 使用

- “查一下速率”读取最近工作记录，不发起模型请求。
- “跑一次测速”运行固定测试提示词，会消耗正常模型 token。
- “查测速历史”查看已保存记录。

无可计算的近期样本时会直接说明，不会自动改跑测速。需要 Codex CLI 和 Python 3.11+。

本插件部分代码来自 [TokenPulse](https://github.com/zzccchen/token-pulse)，遵循 MIT License。
