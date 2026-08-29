---
name: curl-header-fetch-method
description: NVENC header file fetching — curl is reliable and preferred over WebFetch
metadata: 
  node_type: memory
  type: reference
  originSessionId: 7a4a34a4-f50c-4b16-bd8a-2b56af83decf
---

# 优先使用 curl 获取 NVENC 头文件

在诊断 NVENC ctypes struct 布局时，需要从 GitHub 获取原始头文件 `nvEncodeAPI.h` 确认准确偏移。

## 已验证的可靠方法

```bash
curl -L -o /tmp/nvEncodeAPI.h \
  "https://raw.githubusercontent.com/FFmpeg/nv-codec-headers/master/include/ffnvcodec/nvEncodeAPI.h"
```

**成功经验**：curl 直接下载在 Windows/Linux 上都已验证成功。之前 Fetch/WebFetch 工具多次失败（网络超时、URL 解析错误），但 curl 在同一环境始终可靠。

## 关键发现

- **`sdk/13.0` 分支不存在**：nv-codec-headers 仓库的分支只到 `sdk/12.2`，当前最新定义在 `master` 分支
- 正确的 GitHub raw URL 格式：`https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>`
- curl 的输出是纯文本，可直接用 Python 正则解析 C 结构体定义

## 不应使用的方法

| 方法 | 问题 |
|------|------|
| WebFetch 工具 | 多次因网络/代理问题失败 |
| 克隆整个仓库 | 不必要，只需要一个头文件 |
| 依赖自动解析脚本 | `verify_rcparams_offset.py` 有已知解析错误 |

**Why:** curl 是最直接、最可靠的单文件获取方式，不依赖工具链的代理配置或 GitHub API 速率限制。已验证在相同的 Windows 开发环境和 Linux GPU 服务器上都可正常工作。

**How to apply:** 当需要验证 NVENC struct 布局或枚举值时，首先用 curl 下载最新的 `nvEncodeAPI.h` 到 `/tmp/`，然后用 grep/Python 定位关键 typedef 和 struct 定义。不要依赖记忆或间接推断 — 直接看头文件原文。
