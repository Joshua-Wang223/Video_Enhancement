---
name: qwen3-7-cc-switch-setup
description: Qwen 3.7 (阿里云百炼 DashScope) 通过 cc-switch 和翻译代理集成到 Claude Code/Codex
metadata: 
  node_type: memory
  type: project
  originSessionId: e240eb85-5b38-4477-93f1-bef9f2a97646
---

# Qwen 3.7 集成配置

## 背景

用户通过 cc-switch 为 Claude Code 和 Codex CLI 添加阿里云百炼 (DashScope) 的 Qwen 3.7 模型。

## API 信息

- **Endpoint**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **Model**: `qwen3.7-max` (Qwen 3.7 Max)
- **API Key**: 存储在 `.qwenv2key` 和 `.zshrc` 的 `OPENAI_API_KEY` 环境变量
- **协议**: DashScope 只支持 OpenAI 兼容 API（/v1/chat/completions）
  - Anthropic Messages API (/v1/messages) 不支持 → **404**
  - Anthropic 专用端点 (/anthropic/v1/messages) → **404**

## 翻译代理架构

由于 Claude Code 使用 Anthropic Messages API，而 DashScope 只支持 OpenAI API，因此需要一个翻译代理：

```
Claude Code → Anthropic API → 本地代理 (127.0.0.1:18999) → OpenAI API → DashScope Qwen
```

代理脚本: `/workspace/anthropic_openai_bridge.py`
- 监听 `127.0.0.1:18999`
- 翻译 Anthropic Messages API 请求 → OpenAI Chat Completions 请求
- 翻译响应格式返回
- 支持 streaming (SSE)

## cc-switch Providers

### Claude Code providers

| Provider ID | 名称 | Base URL | 状态 |
|-------------|------|----------|------|
| `deepseek-joshua` | DeepSeek (Joshua) | `https://api.deepseek.com/anthropic` | ✅ 当前默认 |
| `qwen-3.7` | Qwen 3.7 (百炼) | `http://127.0.0.1:18999` | ✅ 已添加 |
| `kimi` | Kimi | `https://api.moonshot.cn/anthropic` | ✅ 已添加 |

### 切换命令

```bash
# 切换到 Qwen 3.7 (需要先启动翻译代理)
cc-switch -a claude provider switch qwen-3.7

# 切换到 DeepSeek
cc-switch -a claude provider switch deepseek-joshua
```

## Codex CLI 配置

Codex OSS 模式文件: `~/.codex/config.toml`

```toml
model_provider = "Model_Studio"
model = "qwen3.7-max"

[model_providers.Model_Studio]
name = "Model_Studio"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
```

需要设置环境变量 `OPENAI_API_KEY` 后运行 Codex。

## 启动脚本

完整一键设置脚本: `/workspace/qwen_setup_complete.sh`
- 启动翻译代理
- 配置环境变量
- 验证连接

```bash
bash /workspace/qwen_setup_complete.sh
```

**Why:** DashScope 不支持 Anthropic API 格式，需要翻译代理才能给 Claude Code 使用。Codex CLI 可以直接通过 OSS mode 使用。

**How to apply:** 运行 `bash /workspace/qwen_setup_complete.sh` 一键启动，然后 `cc-switch -a claude provider switch qwen-3.7` 切换到 Qwen。
