---
name: 开发与运行环境
description: 开发环境 Windows，运行环境 Linux
type: project
originSessionId: 549bd23c-b680-4b5e-b833-3852c89608f0
---
本项目的开发环境是 Windows（代码编辑、调试在 Windows），但实际运行/部署环境是 Linux（生产环境中执行视频处理任务）。

**Why:** 用户在 Windows 上开发和修改代码，但将这些代码部署到 Linux 服务器上运行。

**How to apply:** 编写代码时注意跨平台兼容（路径分隔符、换行符、shell 命令等），优先使用 POSIX 路径风格和可移植的 Python API（如 `pathlib.Path`）。
