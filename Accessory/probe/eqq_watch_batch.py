#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等质量标定 —— 批量任务看护（进度 / ETA / 异常诊断 / 按resume 语义重启）。

历史脚本 `eqq_watch.sh` / `watch_ji.sh` / `eqq_status.sh` 把素材清单硬编码在
shell 里，且「完成判据 = 65 点」写死。本脚本改为从批次根目录读实际数据算判据。

⚠ **重启前必须确认没有另一进程在跑同一 workdir** —— harness 的 `make_prep()`
  先unlink 再建、`_save_points()` 无锁覆盖，同 workdir 并行会互删 prep / 丢点。

用法
----
    # 每10 min 记录一次进度到日志（常驻）
    python3 eqq_watch_batch.py --outroot /tmp/eqq_run --interval 600

    # 单次快照 + ETA（不常驻）
    python3 eqq_watch_batch.py --outroot /tmp/eqq_run --once

    # 异常自动重启（先做同 workdir 占用检查）
    python3 eqq_watch_batch.py --outroot /tmp/eqq_run --interval 600 --restart-on-abnormal
"""
import argparse
import json
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
#: 满档素材的点数：5 锚点（key 去重，不是 5×6）+ 4 软编 ×10 + rav1e ×10 + rav1e@10 ×10
FULL_POINTS = 65
#: 各档相对软编的单点成本倍数（实测：rav1e native ~180s / @10 ~54s vs 软编 ~35s）
RAV_SCALE = {'librav1e': 5.2, 'librav1e@10': 1.6}
PLAN = ['libx264', 'libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1',
        'librav1e', 'librav1e@10']
LOG_RE = re.compile(r'\s+(\S+)\s+([\d.]+) ->.*\((\d+(?:\.\d+)?)s\)')


def scan(outroot):
    """扫批次根目录 → [{tag, total, by_tier, log_bytes, prep, ...}]。"""
    out = []
    for d in sorted(Path(outroot).iterdir()):
        pf = d / 'points.json'
        if not d.is_dir() or not pf.is_file():
            continue
        try:
            pts = json.loads(pf.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as e:
            out.append({'tag': d.name, 'broken': str(e)})
            continue
        # 批处理写 workdir/run.log；历史批次写 <outroot>/<tag>.log ⇒ 两处都找
        lg = next((p for p in (d / 'run.log', Path(outroot) / f'{d.name}.log',
                               Path(outroot) / f'{d.name}.points.json.log')
                   if p.is_file()), None)
        per = {}
        if lg is not None:
            for line in lg.read_text(errors='replace').splitlines():
                m = LOG_RE.match(line)
                if m:
                    per.setdefault(m.group(1), []).append(float(m.group(3)))
        out.append({
            'tag': d.name, 'broken': None, 'total': len(pts),
            'by_tier': dict(Counter(k.split('|')[1] for k in pts)),
            'log_bytes': lg.stat().st_size if lg else 0,
            'prep_exists': (d / 'prep.mp4').is_file(),
            'cost': {k: (statistics.mean(v) if v else None) for k, v in per.items()},
        })
    return out


def eta(info):
    """按 tier 分别用实测单点成本估算。剩余点数少 ≠ 剩余时间少（rav1e 单点最贵）。"""
    if info.get('broken') or not info['total']:
        return None, None
    done = Counter(info['by_tier'])
    soft = [v for k, v in info['cost'].items()
            if k not in RAV_SCALE and v]
    base = statistics.mean(soft) if soft else 35.0
    per_tier = {'libx264': max(0, 5 - done.get('libx264', 0)) * base * 0.25}
    for t in PLAN[1:]:
        unit = base * RAV_SCALE[t] if t in RAV_SCALE else base
        per_tier[t] = max(0, 10 - done.get(t, 0)) * unit
    worst = max(per_tier, key=per_tier.get)
    total = sum(per_tier.values())
    # 全档完成时无剩余 ⇒ 不报「最贵档位」（否则会误导成 libx264 最贵）
    return total, (worst if total > 0 else None)


def running(outroot, tag):
    """该 workdir 是否已有测量进程在跑（同 workdir 并行会互删 prep / 丢点）。"""
    r = subprocess.run(['pgrep', '-f', f'--out {Path(outroot) / tag}'],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def report(outroot, expect, log):
    infos = scan(outroot)
    if not infos:
        log(f'[{time.strftime("%F %T")}] {outroot} 下无 points.json')
        return infos
    log(f'[{time.strftime("%F %T")}] {outroot}  {len(infos)} 个 workdir')
    for i in infos:
        if i.get('broken'):
            log(f'  {i["tag"]:<40}⚠ {i["broken"]}')
            continue
        sec, worst = eta(i)
        eta_s = f'{sec / 60:>6.0f}min' if sec else '     —'
        log(f'  {i["tag"]:<40}{i["total"]:>4}/{expect}'
            f'  ETA {eta_s} (最贵 {worst or "—"})  log={i["log_bytes"]}B'
            f'{"  prep残留" if i["prep_exists"] else ""}')
    return infos


def main():
    ap = argparse.ArgumentParser(description='等质量标定批量看护')
    ap.add_argument('--outroot', required=True)
    ap.add_argument('--expect', type=int, default=FULL_POINTS, help=f'每素材满档点数（默认 {FULL_POINTS}）')
    ap.add_argument('--interval', type=int, default=600, help='轮询间隔秒（默认 600）')
    ap.add_argument('--once', action='store_true', help='只快照一次就退出')
    ap.add_argument('--log', default='', help='进度日志（默认 stdout）')
    ap.add_argument('--restart-on-abnormal', action='store_true',
                    help='异常时重启（会先检查同 workdir 是否已有进程）')
    ap.add_argument('--duration', type=float, default=6.0, help='--restart 时传给子进程的 --duration')
    args = ap.parse_args()

    lf = open(args.log, 'a', encoding='utf-8') if args.log else sys.stdout

    def log(msg):
        print(msg, file=lf, flush=True)

    stop = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__('flag', True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__('flag', True))

    while not stop['flag']:
        try:
            infos = report(args.outroot, args.expect, log)
            if args.restart_on_abnormal:
                for i in infos:
                    if i.get('broken') or i['total'] >= args.expect:
                        continue
                    tag = i['tag']
                    if running(args.outroot, tag):
                        log(f'  [{tag}] 进程在跑 ⇒ 不重启')
                        continue
                    log(f'  [{tag}] 进程不在且点数 {i["total"]}/{args.expect}'
                        f' ⇒ 按 resume 语义重启')
                    src = Path(args.outroot) / tag / 'src.txt'
                    dur_f = Path(args.outroot) / tag / 'duration.txt'
                    if not src.is_file():
                        log(f'  [{tag}] 缺 src.txt ⇒ 无法自动重启'
                            f'（用 --only 手工指定，或在该 workdir 写入 src.txt）')
                        continue
                    dur = (float(dur_f.read_text().strip()) if dur_f.is_file()
                           else args.duration)
                    subprocess.Popen(
                        [sys.executable, str(_HERE / 'eqq_calibrate_clip.py'),
                         '--src', src.read_text(encoding='utf-8').strip(),
                         '--out', str(Path(args.outroot) / tag),
                         '--duration', str(dur)],
                        stdout=(Path(args.outroot) / tag / 'run.log').open('a',
                                                                          encoding='utf-8'),
                        stderr=subprocess.STDOUT)
        except Exception as e:                      # 看护本身不能因单次异常退出
            log(f'[{time.strftime("%F %T")}] 看护异常：{e!r}')
        if args.once:
            return 0
        for _ in range(max(1, args.interval // 5)):
            if stop['flag']:
                return 0
            time.sleep(5)
    log('看护退出')
    return 0


if __name__ == '__main__':
    sys.exit(main())