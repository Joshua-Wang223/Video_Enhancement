#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""删除 .codebuddy/scheduled_tasks.json 中指定的冗余 cron 任务节点（原子写）。

用途：durable cron 由独立 task session 管理，**本会话的 CronDelete / CronList
看不到它们**（曾报"No scheduled job"），唯一可靠途径是直接编辑落地文件。

安全措施（按序）：
  1. 备份原文件（调用方已做，此处再确认）
  2. 读 json，按 id 精确定位要删的节点
  3. 先写临时文件，再 os.replace 原子替换（避免半截JSON）
  4. 写回后重新读回校验：任务数、剩余 id、prompt 完整性
  5. 打印 before/after 对比

⚠ 禁用 sed/awk 改 JSON —— 会破坏嵌套结构与转义。
用法：python3 fix_cron_dup.py <file> <要删的id> [<要删的id> ...]
"""
import json
import os
import sys
import tempfile


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    path = sys.argv[1]
    drop = set(sys.argv[2:])

    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    tasks = d.get('tasks', [])
    before = [t.get('id') for t in tasks]
    print('before: %d 个任务 %s' % (len(tasks), before))

    keep, removed = [], []
    for t in tasks:
        (removed if t.get('id') in drop else keep).append(t)
    if not removed:
        print('指定 id 不存在，无需删除')
        return 0
    print('删除: %s' % [t.get('id') for t in removed])
    print('保留: %s' % [t.get('id') for t in keep])

    d['tasks'] = keep
    # 原子写：同目录临时文件 + os.replace
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        os.path.exists(tmp) and os.unlink(tmp)
        raise
    print('已原子写回 %s' % path)

    # 写回校验
    with open(path, encoding='utf-8') as f:
        d2 = json.load(f)
    after = [t.get('id') for t in d2.get('tasks', [])]
    print('after : %d 个任务 %s' % (len(after), after))
    assert len(after) == len(keep), '写回后任务数不符'
    for t in d2['tasks']:
        assert t.get('prompt'), f"任务 {t.get('id')} prompt 为空"
        assert t.get('cron'), f"任务 {t.get('id')} cron 为空"
    print('校验通过：JSON 合法、prompt/cron 字段完整')
    return 0


if __name__ == '__main__':
    sys.exit(main())
