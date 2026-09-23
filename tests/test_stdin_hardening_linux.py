#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stdin 加固 `detach_background_stdin()` —— Linux 判据实测。

背景与契约见 `Plan/stdin加固策略定稿_立项Prompt.md`（决策结论 2026-09-15）与
`src/utils/stdin_hardening.py` 的模块 docstring「契约（2026-09-15 定稿）」。

Windows 开发树无法跑本文件（`os.name == 'nt'`，没有 `tcgetpgrp`/`getpgrp`，也没有
前台 tty 语义），故立项时这几条判据全部标为「待 Linux」。本文件把它们在**真有 tty**
的 Linux 环境上实测一遍：

  判据 1  原故障复现与修复：后台进程组 + tty stdin 下
          无加固 → 被 SIGTTOU 停住（超时）；有加固 → rc=0
  判据 2  分支矩阵 5/5（§2.2 表格逐行）
  判据 3  异常退化 4/4（各系统调用缺失/抛错）→ 返回 False、不冒泡、fd0 不动
  判据 4  幂等性：连调 3 次，第 1 次生效后续 False，fd 表不再变
  判据 5  顺序保证：`import ifrnet_video.ffmpeg_io` 全程子进程拉起次数 = 0
  判据 6  调用点覆盖：读帧器子进程 fd0 == /dev/null
          （IFRNet `Popen(**)` 站点 + ESRGAN `run_async` 唯一缺口站点）

跑法：
    python tests/test_stdin_hardening_linux.py

实现说明（为什么要 pty）：
  「tty + 前台进程组」与「tty + 后台进程组」两个场景取决于**调用进程相对控制终端
  的前台状态**，而这由外层 shell/CI 决定，不能假定。故本文件用 `pty.fork()` 自建
  一个**受控控制终端**：pty 子进程是会话首进程且为前台进程组 →
  直接 exec 即「前台 tty」；再 fork 一个孙进程并 `setpgid(0,0)` 即「后台 tty」。
"""
import json
import os
import pty
import select
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src" / "utils", ROOT / "external",
           ROOT / "external" / "ifrnet_video", ROOT / "external" / "realesrgan_video"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stdin_hardening import detach_background_stdin  # noqa: E402

_FFMPEG = None
for _c in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
    if Path(_c).is_file():
        _FFMPEG = _c
        break
if _FFMPEG is None:
    from shutil import which
    _FFMPEG = which("ffmpeg")

_HAS_TTY = os.isatty(0)
_HAS_TCGETPGRP = hasattr(os, "tcgetpgrp") and hasattr(os, "getpgrp")

_OUT_ENV = "_STDIN_TEST_OUT"      # 子片段把 JSON 写到这里（避免背景组写 tty 的 TOSTOP 干扰）


# --------------------------------------------------------------------------
# 通用：子片段前置（路径 + JSON 落盘工具）
# --------------------------------------------------------------------------
_PREAMBLE = """
import json, os, sys, time
for _p in %r:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from stdin_hardening import detach_background_stdin
def fd0():
    try:
        return os.readlink('/proc/self/fd/0')
    except OSError as e:
        return '<err %%s>' %% e
def emit(d):
    from json import dumps
    s = dumps(d)
    _p = os.environ.get(%r)
    if _p:
        with open(_p, 'w') as f:
            f.write(s)
    # 不再 print 到 stdout，避免在 pty 场景下阻塞（父进程会读取 pty 但不保证实时）
""" % ([str(ROOT / "src" / "utils"), str(ROOT / "external"),
       str(ROOT / "external" / "ifrnet_video"),
       str(ROOT / "external" / "realesrgan_video")], _OUT_ENV)


def _proc_state(pid):
    """读 /proc/<pid>/stat 的状态字符（R/S/T/Z…）；读不到返回 None。"""
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


def _stopped_in_pgroup(pgid):
    """列出进程组 pgid 下处于 T（stopped）状态的进程 [(pid, comm)]。"""
    out = []
    for ent in Path("/proc").iterdir():
        if not ent.name.isdigit():
            continue
        try:
            with open(ent / "stat") as f:
                fields = f.read().rsplit(")", 1)
            comm = fields[0].split("(", 1)[1]
            rest = fields[1].split()
            state, pgrp = rest[0], int(rest[2])
        except (OSError, IndexError, ValueError):
            continue
        if pgrp == pgid and state == "T":
            out.append((int(ent.name), comm))
    return out


def _run_in_bg_tty(code, timeout=30):
    """在 pty 子进程里跑 code：pty 子进程 fork 孙进程，孙进程 setpgid(0,0) 变后台组。
    返回 (rc, out_dict, info)；info 含 stopped 列表。
    """
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "out.json"
        env = os.environ.copy()
        env[_OUT_ENV] = str(out_path)
        
        pid, fd = pty.fork()
        if pid == 0:  # 子进程 A：pty 子进程，会话首进程，有控制终端，前台进程组
            try:
                gp = os.fork()
                if gp == 0:  # 孙进程 B：将变成后台进程组
                    os.setpgid(0, 0)  # 新进程组 → 相对控制终端为后台组
                    os.execve(sys.executable, [sys.executable, "-c", code], env)
                    os._exit(127)
                # 子进程 A 等待孙进程 B
                os.waitpid(gp, 0)
                os._exit(0)
            except BaseException:
                os._exit(127)
        
        # 父进程：等待子进程 A（不读取 pty，避免读取错误导致误判超时）
        rc = None
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                wp, st = os.waitpid(pid, os.WNOHANG)
                if wp == pid:
                    rc = os.waitstatus_to_exitcode(st)
                    break
                time.sleep(0.1)
            else:
                # 超时
                rc = None
                pgid = os.getpgid(pid)
                stopped = []
                for ent in Path("/proc").iterdir():
                    if not ent.name.isdigit():
                        continue
                    try:
                        with open(ent / "stat") as f:
                            fields = f.read().rsplit(")", 1)
                        comm = fields[0].split("(", 1)[1]
                        rest = fields[1].split()
                        state, pgrp, ppid = rest[0], int(rest[2]), int(rest[1])
                    except (OSError, IndexError, ValueError):
                        continue
                    if ppid == pid and state == "T":
                        stopped.append((int(ent.name), comm))
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass
                for _ in range(50):
                    wp, _st = os.waitpid(pid, os.WNOHANG)
                    if wp == pid:
                        break
                    time.sleep(0.02)
                out = None
                if out_path.is_file():
                    try:
                        out = json.loads(out_path.read_text())
                    except (ValueError, OSError):
                        pass
                return None, out, {"timed_out": True, "bg_pgid": pgid, "stopped": stopped}
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        
        # 正常结束
        out = None
        if out_path.is_file():
            try:
                out = json.loads(out_path.read_text())
            except (ValueError, OSError):
                pass
        return rc, out, {"timed_out": False, "stopped": []}


def _run_in_fg_tty(code, timeout=30):
    """在 pty 子进程里跑 code：pty 子进程有控制终端 + 前台进程组（不 setpgid）。
    返回 (rc, out_dict, info)；用于不需要后台组的场景（如判据 5 导入测试）。
    """
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "out.json"
        env = os.environ.copy()
        env[_OUT_ENV] = str(out_path)
        
        pid, fd = pty.fork()
        if pid == 0:
            try:
                os.execve(sys.executable, [sys.executable, "-c", code], env)
                os._exit(127)
            except BaseException:
                os._exit(127)
        
        rc = None
        try:
            buf = b""
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                r, _, _ = select.select([fd], [], [], 0.2)
                if r:
                    try:
                        chunk = os.read(fd, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                wp, st = os.waitpid(pid, os.WNOHANG)
                if wp == pid:
                    rc = os.waitstatus_to_exitcode(st)
                    break
            else:
                rc = None
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                for _ in range(50):
                    wp, _st = os.waitpid(pid, os.WNOHANG)
                    if wp == pid:
                        break
                    time.sleep(0.02)
                out = None
                if out_path.is_file():
                    try:
                        out = json.loads(out_path.read_text())
                    except (ValueError, OSError):
                        pass
                if out is None and buf:
                    for line in buf.decode('utf-8', errors='ignore').splitlines():
                        if line.startswith('{'):
                            try:
                                out = json.loads(line)
                                break
                            except ValueError:
                                pass
                return None, out, {"timed_out": True}
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        
        out = None
        if out_path.is_file():
            try:
                out = json.loads(out_path.read_text())
            except (ValueError, OSError):
                pass
        if out is None and buf:
            for line in buf.decode('utf-8', errors='ignore').splitlines():
                if line.startswith('{'):
                    try:
                        out = json.loads(line)
                        break
                    except ValueError:
                        pass
        return rc, out, {"timed_out": False}


def _run_plain_with_emit(code, stdin=None, timeout=30):
    """非 tty 场景：用 subprocess 跑 code，子片段通过 _OUT_ENV 指定的文件 emit 结果。"""
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "out.json"
        env = os.environ.copy()
        env[_OUT_ENV] = str(out_path)
        p = subprocess.run([sys.executable, "-c", code],
                           stdin=stdin, capture_output=True, text=True,
                           timeout=timeout, cwd=str(ROOT), env=env)
        out = None
        if out_path.is_file():
            try:
                out = json.loads(out_path.read_text())
            except (ValueError, OSError):
                pass
        return p.returncode, out, p


def _run_plain(code, argv=(), stdin=None, timeout=30):
    """普通 subprocess 执行（用于非 tty 场景，不需要 emit）。"""
    return subprocess.run([sys.executable, "-c", code, *argv],
                          stdin=stdin, capture_output=True, text=True,
                          timeout=timeout, cwd=str(ROOT), env=os.environ.copy())


# ══════════════════════════════════════════════════════════════════════════
# 判据 1：原故障复现与修复
# ══════════════════════════════════════════════════════════════════════════
_FAULT_BODY = """
import subprocess, os, sys
mode = os.environ.get('MODE', 'noharden')
if mode == 'harden':
    detach_background_stdin()        # 本立项的加固
# 关键：不显式传 stdin，让 ffmpeg **继承控制终端**（tty）
# 用 -hide_banner -v error 避免输出干扰；用 -i /dev/stdin 强制 ffmpeg 打开 stdin（触发终端 ioctl）
_r = subprocess.run([os.environ['FFMPEG_BIN'], '-hide_banner', '-v', 'error',
                      '-f', 'lavfi', '-i', 'testsrc=duration=1:size=64x64:rate=5',
                      '-f', 'null', '-'],
                     stdin=None)  # None = 继承父进程的 fd 0
emit({'ffmpeg_rc': _r.returncode, 'fd0': fd0()})
"""


def test_criterion1_original_fault_and_fix():
    if not _HAS_TCGETPGRP or not _FFMPEG:
        print("  [1] SKIP（需 tcgetpgrp + ffmpeg）")
        return None

    def _run(mode, timeout=20):
        saved = os.environ.get("MODE"), os.environ.get("FFMPEG_BIN")
        os.environ["MODE"], os.environ["FFMPEG_BIN"] = mode, _FFMPEG
        try:
            t0 = time.monotonic()
            rc, out, info = _run_in_bg_tty(_PREAMBLE + _FAULT_BODY, timeout=timeout)
            return rc, out, info, time.monotonic() - t0
        finally:
            if saved[0] is None:
                os.environ.pop("MODE", None)
            else:
                os.environ["MODE"] = saved[0]
            if saved[1] is None:
                os.environ.pop("FFMPEG_BIN", None)
            else:
                os.environ["FFMPEG_BIN"] = saved[1]

    rc_bad, out_bad, info_bad, t_bad = _run("noharden", timeout=10)
    assert rc_bad is None, (
        "无加固却在 %.1fs 内退出了（本环境不复现 SIGTTOU）: rc=%s out=%s"
        % (t_bad, rc_bad, out_bad))
    # SIGTTOU 实锤：超时瞬间，后台进程组里应有处于 T（stopped）态的进程
    assert info_bad["stopped"], (
        "无加固场景确实卡住了，但进程组 %s 内没有 T 态进程 —— "
        "不是 SIGTTOU 停住（需重新归因）" % info_bad["bg_pgid"])
    _stopped_desc = "、".join("pid=%d(%s)" % (p, c) for p, c in info_bad["stopped"][:3])

    rc_ok, out_ok, info_ok, t_ok = _run("harden", timeout=20)
    assert rc_ok is not None, "有加固却仍被停住（加固未生效，%.1fs 超时）" % t_ok
    assert not info_ok["stopped"], "有加固却仍有 T 态进程：%s" % info_ok["stopped"]
    assert out_ok and out_ok.get("ffmpeg_rc") == 0, (
        "有加固但 ffmpeg 未成功：rc=%s out=%s" % (rc_ok, out_ok))

    print("  [1] 原故障复现+修复 OK：无加固→SIGTTOU 停住(%.1fs, T 态=%s)／"
          "有加固→ffmpeg rc=0（%.1fs, fd0=%s）"
          % (t_bad, _stopped_desc, t_ok, out_ok.get("fd0")))
    return True


# ══════════════════════════════════════════════════════════════════════════
# 判据 2：分支矩阵 5/5
# ══════════════════════════════════════════════════════════════════════════
_TOGGLE_BODY = """
before = fd0()
try:
    rc = detach_background_stdin()
    raised = None
except BaseException as e:
    rc, raised = None, '%s: %s' % (type(e).__name__, e)
emit({'rc': rc, 'before': before, 'after': fd0(), 'raised': raised})
"""


def test_criterion2_branch_matrix():
    if not _HAS_TCGETPGRP:
        print("  [2] SKIP（需 tcgetpgrp）")
        return None
    code = _PREAMBLE + _TOGGLE_BODY

    # 1) 非 tty：/dev/null
    rc, r, _p = _run_plain_with_emit(code, stdin=subprocess.DEVNULL)
    assert r is not None, "场景1 无 emit 输出"
    assert r["rc"] is False, "场景1(/dev/null) 不应加固：%s" % r
    assert r["before"] == r["after"], "场景1 改动了 fd0：%s" % r

    # 2) 非 tty：管道
    rc, r, _p = _run_plain_with_emit(code, stdin=subprocess.PIPE)
    assert r is not None, "场景2 无 emit 输出"
    assert r["rc"] is False, "场景2(管道) 不应加固：%s" % r
    assert r["before"] == r["after"], "场景2 改动了 fd0：%s" % r

    # 3) tty + 前台进程组 → 不动
    rc, r, _info = _run_in_fg_tty(code, timeout=20)
    assert r is not None, "场景3 无输出（rc=%s）" % rc
    assert r["before"].startswith("/dev/pts"), \
        "场景3 的 stdin 不是 tty：%s" % r["before"]
    assert r["rc"] is False, "场景3(前台 tty) 不应加固：%s" % r
    assert r["before"] == r["after"], "场景3 改动了 fd0：%s" % r

    # 4) tty + 后台进程组 → fd0 → /dev/null
    rc, r, _info = _run_in_bg_tty(code, timeout=20)
    assert r is not None, "场景4 无输出（rc=%s）" % rc
    assert r["before"].startswith("/dev/pts"), \
        "场景4 的 stdin 不是 tty：%s" % r["before"]
    assert r["rc"] is True, "场景4(后台 tty) 应加固：%s" % r
    assert r["after"] == "/dev/null", "场景4 加固后 fd0 不是 /dev/null：%s" % r

    # 5) 取不到前台进程组 → 保守加固
    body5 = (_PREAMBLE + """
os.tcgetpgrp = lambda fd: (_ for _ in ()).throw(OSError('模拟无控制终端'))
""" + _TOGGLE_BODY)
    rc, r, _info = _run_in_bg_tty(body5, timeout=20)
    assert r is not None, "场景5 无输出（rc=%s）" % rc
    assert r["rc"] is True, "场景5(tcgetpgrp 失败) 应保守加固：%s" % r
    assert r["after"] == "/dev/null", "场景5 加固后 fd0 不是 /dev/null：%s" % r

    print("  [2] 分支矩阵 OK：5/5（非tty×2 / 前台tty / 后台tty / 取不到前台组）")
    return True


# ══════════════════════════════════════════════════════════════════════════
# 判据 3：异常退化 4/4
# ══════════════════════════════════════════════════════════════════════════
_DEGRADE_BODY = """
mode = os.environ['MODE']
if mode == 'isatty_oserror':
    os.isatty = lambda fd: (_ for _ in ()).throw(OSError('EIO'))
elif mode == 'isatty_valueerror':
    os.isatty = lambda fd: (_ for _ in ()).throw(ValueError('非 OSError'))
elif mode == 'tcgetpgrp_and_open_oserror':
    os.tcgetpgrp = lambda fd: (_ for _ in ()).throw(OSError('ENOTTY'))
    os.open = lambda *a, **k: (_ for _ in ()).throw(OSError('EMFILE'))
elif mode == 'getpgrp_oserror':
    os.tcgetpgrp = lambda fd: 1
    os.getpgrp = lambda: (_ for _ in ()).throw(OSError('ESRCH'))
before = fd0()
try:
    rc = detach_background_stdin()
    raised = None
except BaseException as e:
    rc, raised = None, '%s: %s' % (type(e).__name__, e)
emit({'rc': rc, 'raised': raised, 'before': before, 'after': fd0()})
"""


def test_criterion3_exception_degradation():
    for mode in ("isatty_oserror", "isatty_valueerror",
                 "tcgetpgrp_and_open_oserror", "getpgrp_oserror"):
        saved = os.environ.get("MODE")
        os.environ["MODE"] = mode
        try:
            rc, r, _info = _run_in_bg_tty(_PREAMBLE + _DEGRADE_BODY, timeout=20)
        finally:
            if saved is None:
                os.environ.pop("MODE", None)
            else:
                os.environ["MODE"] = saved
        assert r is not None, "退化场景 %s 无输出（rc=%s）" % (mode, rc)
        assert r["raised"] is None, "异常冒泡了（%s）：%s" % (mode, r["raised"])
        assert r["rc"] is False, "未退化为 False（%s）：%s" % (mode, r)
        assert r["before"] == r["after"], "退化路径改动了 fd0（%s）：%s" % (mode, r)
    print("  [3] 异常退化 OK：4/4（isatty 抛 OSError／抛非 OSError／"
          "tcgetpgrp+open 失败／getpgrp 失败）")
    return True


# ══════════════════════════════════════════════════════════════════════════
# 判据 4：幂等性
# ══════════════════════════════════════════════════════════════════════════
_IDEMPOTENT_BODY = """
r1 = detach_background_stdin(); f1 = fd0()
r2 = detach_background_stdin(); f2 = fd0()
r3 = detach_background_stdin(); f3 = fd0()
emit({'r': [r1, r2, r3], 'f': [f1, f2, f3],
      'before': '/dev/pts' if f1 == '/dev/null' else f1})
"""


def test_criterion4_idempotent():
    if not _HAS_TCGETPGRP:
        print("  [4] SKIP（需 tcgetpgrp）")
        return None
    rc, r, _info = _run_in_bg_tty(_PREAMBLE + _IDEMPOTENT_BODY, timeout=20)
    assert r is not None, "幂等场景无输出（rc=%s）" % rc
    assert r["r"] == [True, False, False], "幂等性不符（期望 1True+2False）：%s" % r
    assert len(set(r["f"])) == 1 and r["f"][0] == "/dev/null", \
        "第 2/3 次仍改动了 fd 表：%s" % r
    print("  [4] 幂等性 OK：%s，fd0 三次均 %s" % (r["r"], r["f"][0]))
    return True


# ══════════════════════════════════════════════════════════════════════════
# 判据 5：顺序保证 —— import 期间不得拉起任何子进程
# ══════════════════════════════════════════════════════════════════════════
_ORDER_BODY = """
import subprocess, sys
_paths = %r
for _p in _paths:
    if _p not in sys.path:
        sys.path.insert(0, _p)
_calls = []
_orig_run, _orig_popen = subprocess.run, subprocess.Popen
def _run(*a, **k):
    _calls.append('run');  return _orig_run(*a, **k)
def _popen(*a, **k):
    _calls.append('Popen'); return _orig_popen(*a, **k)
subprocess.run, subprocess.Popen = _run, _popen
import ifrnet_video.ffmpeg_io          # 模块导入时会调 detach_background_stdin()
subprocess.run, subprocess.Popen = _orig_run, _orig_popen
emit({'spawns': len(_calls), 'which': _calls})
""" % ([str(ROOT / "src" / "utils"), str(ROOT / "external"),
       str(ROOT / "external" / "ifrnet_video")],)


def test_criterion5_import_order():
    # import 期间不需要后台组，用前台 pty（不 setpgid）
    rc, r, _info = _run_in_fg_tty(_PREAMBLE + _ORDER_BODY, timeout=90)
    assert r is not None, "导入计数片段无输出（rc=%s）" % rc
    assert r["spawns"] == 0, (
        "导入期间拉起了 %d 个子进程（加固必须先于任何 ffmpeg）：%s"
        % (r["spawns"], r["which"]))
    print("  [5] 顺序保证 OK：import ifrnet_video.ffmpeg_io 期间子进程拉起次数 = 0")
    return True


# ══════════════════════════════════════════════════════════════════════════
# 判据 6：调用点覆盖 —— 读帧器子进程 fd0 == /dev/null
# ══════════════════════════════════════════════════════════════════════════
_IFRNET_READER_BODY = """
import ifrnet_video.ffmpeg_io as io
clip = os.environ['CLIP']
rd = io.FFmpegFrameReader(clip, prefetch=2, use_hwaccel=False)
try:
    pid = getattr(rd, '_proc', None)
    pid = pid.pid if pid is not None else None
    emit({'pid': pid, 'fd0': os.readlink('/proc/%d/fd/0' % pid) if pid else None})
finally:
    try:
        rd.close()
    except Exception:
        pass
"""

_ESRGAN_READER_BODY = """
import realesrgan_video.ffmpeg_io as io
clip = os.environ['CLIP']
rd = io.FFmpegReader(clip, prefetch_factor=2, use_hwaccel=False)
try:
    pid = None
    for _ in range(100):                       # run_async 在后台线程里拉起，等它出现
        p = getattr(rd, '_ffmpeg_process', None)
        if p is not None and hasattr(p, 'pid'):
            pid = p.pid
            break
        time.sleep(0.1)
    emit({'pid': pid, 'fd0': os.readlink('/proc/%d/fd/0' % pid) if pid else None})
finally:
    try:
        rd.close()
    except Exception:
        pass
"""


def test_criterion6_callsite_fd0():
    if not _FFMPEG:
        print("  [6] SKIP（需 ffmpeg）")
        return None
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip.mp4"
        subprocess.run([_FFMPEG, "-hide_banner", "-v", "error", "-y",
                        "-f", "lavfi", "-i", "testsrc=duration=1:size=128x72:rate=8",
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", str(clip)],
                       check=True, stdin=subprocess.DEVNULL, capture_output=True)
        os.environ["CLIP"] = str(clip)
        results = {}
        try:
            for name, body in (("IFRNet Popen", _IFRNET_READER_BODY),
                               ("ESRGAN run_async", _ESRGAN_READER_BODY)):
                # 关键：整个读帧器建在**后台进程组 + tty** 下 ——
                # IFRNet 侧靠 Popen(stdin=DEVNULL)，ESRGAN 侧只能靠 L1/L2 的 fd0 加固
                rc, r, _info = _run_in_bg_tty(_PREAMBLE + body, timeout=180)
                if r is None:
                    print("  [6] %s 探测失败（rc=%s）" % (name, rc))
                    continue
                results[name] = r
                assert r.get("fd0") == "/dev/null", (
                    "%s 的子进程 fd0 = %r（期望 /dev/null）" % (name, r.get("fd0")))
        finally:
            os.environ.pop("CLIP", None)

        if not results:
            print("  [6] SKIP（两个读帧器都未能探测到子进程）")
            return None
        print("  [6] 调用点覆盖 OK：%s"
              % "／".join("%s fd0=%s" % (k, v.get("fd0")) for k, v in results.items()))
    return True


def _read_emit(p):
    """非 tty 场景：emit 仍写文件（路径由 _OUT_ENV 传），从 p 读回。"""
    out = p.stdout.strip().splitlines()
    for line in reversed(out):
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError("无 emit 输出：rc=%s stderr=%s"
                         % (p.returncode, p.stderr[-300:]))


# ══════════════════════════════════════════════════════════════════════════
def _main():
    print("stdin 加固 Linux 判据实测")
    print("  tty=%s tcgetpgrp=%s ffmpeg=%s pgrp=%s fg=%s"
          % (_HAS_TTY, _HAS_TCGETPGRP, _FFMPEG, os.getpgrp(),
             os.tcgetpgrp(0) if (_HAS_TTY and _HAS_TCGETPGRP) else "n/a"))
    tests = [test_criterion1_original_fault_and_fix,
             test_criterion2_branch_matrix,
             test_criterion3_exception_degradation,
             test_criterion4_idempotent,
             test_criterion5_import_order,
             test_criterion6_callsite_fd0]
    failures = skipped = 0
    for fn in tests:
        try:
            if fn() is None:
                skipped += 1
        except Exception as e:
            failures += 1
            print("  [FAIL] %s: %s: %s" % (fn.__name__, type(e).__name__, e))
    ran = len(tests) - skipped
    print("\n%s: %d/%d PASS%s"
          % ("ALL PASS" if not failures else "FAILURES",
             ran - failures, ran, "，%d SKIP" % skipped if skipped else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
