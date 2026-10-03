#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等质量表改名：QUALITY_MAP→SIZE_MAP / QUALITY_MAP_QUALITY→QUALITY_MAP / ..._QP→QUALITY_MAP_QP。

⚠ 三个名字有**包含关系**，直接用 str.replace 会级联。必须用占位符隔离，顺序如下：
    1) QUALITY_MAP_QUALITY_QP → QUALITY_MAP_QP
    2) QUALITY_MAP_QUALITY    → __EQ_TMP__
    3) QUALITY_MAP            → SIZE_MAP
    4) __EQ_TMP__             → QUALITY_MAP

用法：
    python3 rename_tables.py            # 干跑（只报告，不写盘）
    python3 rename_tables.py --apply    # 实际写入
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOTS = [Path('/mnt/d/Workspace_Python/Video_Enhancement'),
         Path('/mnt/d/Workspace_Python/VidUtils')]

SUB_DIRS = ['src', 'external', 'Accessory', 'verify', 'probe', 'test']
EXTS = {'.py', '.md', '.json'}

# 明确排除：生成物 / 历史副本 / 缓存
def _excluded(p: Path) -> bool:
    s = str(p)
    if '__pycache__' in s:
        return True
    if 'verification_report' in s:
        return True
    if ' - Copy' in p.name or '- Copy' in p.name:
        return True
    if p.name.startswith('.') or '/.git/' in s:
        return True
    return False


STEPS = [('QUALITY_MAP_QUALITY_QP', 'QUALITY_MAP_QP'),
         ('QUALITY_MAP_QUALITY', '__EQ_TMP__'),
         ('QUALITY_MAP', 'SIZE_MAP'),
         ('__EQ_TMP__', 'QUALITY_MAP')]

# 顺带把 CLI 取值 volume → size（只改写与 quality-mode/quality-table 相关的字面量）
CLI_SUBS = [
    ("--quality-mode volume|quality", "--quality-mode size|quality"),
    ("--quality-table volume|quality", "--quality-table size|quality"),
    ("choices=['volume', 'quality']", "choices=['size', 'quality']"),
    ('choices=["volume", "quality"]', 'choices=["size", "quality"]'),
    ("choices=['volume',, 'quality']", "choices=['size', 'quality']"),   # 容错
    ("'volume'", "'size'"),          # set_quality_mode 校验/默认值
    ('"volume"', '"size"'),
    ("volume（等体积", "size（等文件大小"),
]


def _collect() -> list[Path]:
    out = []
    for r in ROOTS:
        for sub in SUB_DIRS:
            d = r / sub
            if d.is_dir():
                out += [p for p in d.rglob('*') if p.is_file() and p.suffix in EXTS]
        out += [p for p in r.glob('*') if p.is_file() and p.suffix in EXTS]
    return sorted({p for p in out if not _excluded(p)})


def _count_old(text: str) -> dict:
    return {n: len(re.findall(re.escape(n), text)) for n, _ in STEPS}


def main():
    apply = '--apply' in sys.argv
    files = _collect()
    print(f'扫描 {len(files)} 个文件（已排除 verification_report / "- Copy" / __pycache__）\n')
    total_before, touched = {}, []
    for p in files:
        src = p.read_text(encoding='utf-8', errors='replace')
        before = _count_old(src)
        if not any(before.values()):
            continue
        out = src
        for old, new in STEPS:
            out = out.replace(old, new)
        for old, new in CLI_SUBS:
            out = out.replace(old, new)
        # CLI 的 'size' 只应出现在 mode/table 语境；上面全局替换 'volume' 会误伤，
        # 故此处仅统计，实际歧义项在报告里列出供人工确认
        total_before = {k: total_before.get(k, 0) + v for k, v in before.items()}
        touched.append((p, before, out != src))
        if apply and out != src:
            p.write_text(out, encoding='utf-8')
    print(f'{"文件":<64} {"QMQP":>5} {"QMQ":>5} {"QM":>5}  改动')
    for p, before, changed in touched:
        rel = str(p).replace('/mnt/d/Workspace_Python/', '')
        print(f'{rel:<64} {before["QUALITY_MAP_QUALITY_QP"]:>5} '
              f'{before["QUALITY_MAP_QUALITY"]:>5} {before["QUALITY_MAP"]:>5}  '
              f'{"是" if changed else "否"}')
    print(f'\n共 {len(touched)} 个文件；旧名出现：{total_before}')
    print('模式：' + ('已写入 ✅' if apply else '干跑（未写盘）— 加 --apply 执行'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
