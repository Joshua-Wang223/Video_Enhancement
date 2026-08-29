---
name: episodic-memory-mcp-fix
description: episodic-memory MCP server 连接修复方法
metadata: 
  node_type: memory
  type: reference
  originSessionId: 9c09c69d-308d-4cab-9c3c-0d2f40d649bb
---

# episodic-memory MCP Server 修复

## 根因

WPS 方式安装的 Claude Code（`claude.exe` 是自包含二进制）不含独立 `node.exe`，而 episodic-memory 插件需要 Node.js 运行时。

## 修复步骤

1. **安装 Node.js LTS**
```powershell
winget install OpenJS.NodeJS.LTS --silent --accept-source-agreements --accept-package-agreements
```

2. **安装插件依赖（跳过 Broken postinstall）**
```powershell
cd ${PLUGIN_ROOT}  # C:\Users\Administrator\.claude\plugins\cache\superpowers-marketplace\episodic-memory\1.4.0
npm install --ignore-scripts --no-audit --no-fund
npm rebuild better-sqlite3
```

关键点：
- `postinstall` 脚本用了 Unix shell 语法（`2>/dev/null || true`），Windows 不兼容 → 必须 `--ignore-scripts`
- `better-sqlite3` 是原生 C++ 模块，需要 `npm rebuild` 单独触发预编译二进制下载
- 第一次 `npm install` 可能因网络不稳定导致 `better-sqlite3` 下载失败（ECONNRESET），需要 `npm cache clean --force` 后重试
- `robocopy /MIR` 清空法可解决 Windows 长路径删除问题

## 验证

```powershell
node ./dist/mcp-server.js  # 应保持运行不退出
```

重启 Claude Code 后 `/doctor` 不再报 episodic-memory MCP 错误。
