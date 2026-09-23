---
name: dev-deploy-environment
description: 开发在 Windows，生产运行在 Linux，代码必须跨平台兼容
metadata: 
  node_type: memory
  type: project
  originSessionId: 9ac38aa0-722c-43f8-aa1e-b6fad7621a9e
---

本项目的开发环境是 Windows（代码编辑、调试在 Windows），但实际运行/部署环境是 Linux（生产环境中执行视频处理任务）。

**Why:** 用户在 Windows 上开发和修改代码，但将这些代码部署到 Linux 服务器上运行。

**How to apply:** 编写代码时注意跨平台兼容（路径分隔符、换行符、shell 命令等），优先使用 POSIX 路径风格和可移植的 Python API（如 `pathlib.Path`）。

跨平台规则详见 [[CLAUDE.md]] 中的"跨平台开发与部署"章节。
