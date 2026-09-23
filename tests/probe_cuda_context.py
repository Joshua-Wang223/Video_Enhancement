#!/usr/bin/env python
"""probe_cuda_context.py —— CUDA context push/pop/setCurrent 语义探针（非侵入）。

本项目在此探针上确立的驱动实测语义（Tesla T4 / driver 580.65）：
  · cuCtxPushCurrent(p)：p 已是本线程 current → 恒返回 201 且**不入栈**
  · cuCtxPopCurrent   ：三种返回 ——
        (rc=0, *pctx=真指针) 真的解绑
        (rc=0, *pctx=NULL)   no-op：栈空但 current 仍在（context 被占用/refcount>1）
        rc=201               无 current，栈已空
  · cuCtxSetCurrent(p)：不进栈，无需配对 pop
  · 危险组合：另一线程持有同一 context current 时，本线程 pop 会**真的解绑**，
    而随后的 push 仍返回 201（context 非 floating）→ 本线程永久失去 current

模式：
  depth  : 测量当前线程 context 栈深度（破坏性，仅用于末尾探测）
  cycle  : 纯 ctypes 复刻 NVENCEncoder 创建/关闭的栈操作，逐周期打印 rc
  seq    : 逐步打印每次操作后的 rc / 弹出指针 / 当前 context
  thread : 复现「pop 真解绑 + push 201 → 丢失 current」的危险路径
  sim    : 完整复刻生产时序（每段：编码线程 setCurrent → 创建 → 关闭）
  exit   : 编码线程已退出时的对照组
  real   : 真实实例化 NVENCEncoder，记录全部 cuCtx* 的 rc 与指针

用法:
  python tests/probe_cuda_context.py depth --torch
  python tests/probe_cuda_context.py seq --torch
  python tests/probe_cuda_context.py sim --cycles 20
  python tests/probe_cuda_context.py real --cycles 5 --codec hevc --la 8
"""
import argparse
import ctypes
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "external"))

V = ctypes.c_void_p
U = ctypes.c_uint32

_lib = ctypes.CDLL("libcuda.so.1")
_lib.cuInit(0)
_lib.cuDevicePrimaryCtxRetain.restype = U
_lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(V), ctypes.c_int]
_lib.cuCtxSetCurrent.restype = U
_lib.cuCtxSetCurrent.argtypes = [V]
_lib.cuCtxPushCurrent.restype = U
_lib.cuCtxPushCurrent.argtypes = [V]
_lib.cuCtxPopCurrent.restype = U
_lib.cuCtxPopCurrent.argtypes = [ctypes.POINTER(V)]
_lib.cuCtxGetCurrent.restype = U
_lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(V)]


def hx(p):
    return "0x%x" % (p or 0)


def cur():
    c = V()
    rc = _lib.cuCtxGetCurrent(ctypes.byref(c))
    return (rc, c.value)


def retain():
    p = V()
    rc = _lib.cuDevicePrimaryCtxRetain(ctypes.byref(p), ctypes.c_int(0))
    return rc, (p.value if rc == 0 else None)


def push(p):
    return _lib.cuCtxPushCurrent(V(p))


def pop():
    o = V()
    rc = _lib.cuCtxPopCurrent(ctypes.byref(o))
    return rc, o.value


def init_torch():
    import torch
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    return torch


def measure_depth(tag):
    """破坏性：pop 直到失败，返回栈深度。"""
    n = 0
    vals = []
    while n < 64:
        rc, v = pop()
        if rc != 0:
            print(f"[{tag}] pop#{n} rc={rc}  → pop 失败（栈已空）", flush=True)
            break
        if v is None:
            print(f"[{tag}] pop#{n} rc=0 但 *pctx=NULL  → 栈已空（本次 pop 是 no-op）", flush=True)
            break
        vals.append(hx(v))
        n += 1
    print(f"[{tag}] 栈深度 = {n}  弹出顺序(自顶向底) = {vals}", flush=True)
    return n


def mode_depth(args):
    if args.torch:
        init_torch()
        print("torch cuda initialized", flush=True)
    rc, c = cur()
    print(f"getcur rc={rc} ctx={hx(c)}", flush=True)
    measure_depth("depth")


def mode_cycle(args):
    if args.torch:
        init_torch()
        print("torch cuda initialized", flush=True)

    rc, c0 = cur()
    print(f"[init] getcur rc={rc} ctx={hx(c0)}", flush=True)

    for i in range(args.cycles):
        # ── 编码器创建：_acquire_cuda_context() ──
        rc_s, saved = cur()
        rc_r, primary = retain()
        rc_push_create = push(primary)          # 生产代码此处忽略返回值
        rc_a, cur_after_create = cur()
        # ── 编码器关闭：close() 的 Restore saved context 块 ──
        rc_pop, popped = pop()                  # 无条件 pop
        rc_b, cur_after_pop = cur()
        rc_push_saved = push(saved) if saved is not None else -1
        rc_c, cur_after_restore = cur()
        warn = "  <<< 201 警告" if rc_push_saved == 201 else ""
        print(f"[cycle {i}] saved={hx(saved)} primary={hx(primary)} equal={saved == primary} | "
              f"create: retain={rc_r} push={rc_push_create} | "
              f"close: pop={rc_pop}(popped={hx(popped)}) push(saved)={rc_push_saved}{warn} | "
              f"cur: create后={hx(cur_after_create)} pop后={hx(cur_after_pop)} restore后={hx(cur_after_restore)}",
              flush=True)

    print("--- 压测结束，测量剩余栈深度 ---", flush=True)
    measure_depth("cycle-end")


# --------------------------------------------------------------------------- #
# real 模式：真实 NVENCEncoder + libcuda 调用日志
# --------------------------------------------------------------------------- #

class _FnProxy:
    """包装 ctypes 函数对象：透传 restype/argtypes，拦截调用并记录 rc。"""

    def __init__(self, real, name, logger):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_logger", logger)

    def __setattr__(self, k, v):
        setattr(object.__getattribute__(self, "_real"), k, v)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_real"), k)

    def __call__(self, *a, **k):
        rc = object.__getattribute__(self, "_real")(*a, **k)
        try:
            object.__getattribute__(self, "_logger")(
                object.__getattribute__(self, "_name"), a, rc)
        except Exception:
            pass
        return rc


class _CudaProxy:
    """只代理 libcuda 的 CDLL：记录所有 cuCtx* 调用的入参与返回码。"""

    CTX_FUNCS = ("cuCtxPushCurrent", "cuCtxPopCurrent", "cuCtxSetCurrent",
                 "cuCtxGetCurrent", "cuDevicePrimaryCtxRetain",
                 "cuDevicePrimaryCtxRelease")

    def __init__(self, real, logger):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_logger", logger)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name):
        cache = object.__getattribute__(self, "_cache")
        if name in cache:
            return cache[name]
        real = object.__getattribute__(self, "_real")
        try:
            fn = getattr(real, name)
        except AttributeError:
            raise
        if name.startswith("cuCtx") or name.startswith("cuDevicePrimaryCtx"):
            w = _FnProxy(fn, name, object.__getattribute__(self, "_logger"))
        else:
            w = fn
        cache[name] = w
        return w


def mode_seq(args):
    """逐步打印：每步操作后打印 rc / 弹出的指针 / 当前 context，用于精确判定栈语义。"""
    if args.torch:
        init_torch()
        print("torch cuda initialized", flush=True)

    def step(desc, fn):
        out = fn()
        rc, c = cur()
        print(f"  {desc:<34} -> {out}   | current={hx(c)} getcur_rc={rc}", flush=True)

    print("--- A: 直接 pop x3（不经 push）---", flush=True)
    step("getcur", lambda: f"rc={cur()[0]} ctx={hx(cur()[1])}")
    for i in range(3):
        step(f"pop#{i}", lambda: (lambda r: f"rc={r[0]} out={hx(r[1])}")(pop()))

    print("--- B: retain + push + pop x3 ---", flush=True)
    step("retain", lambda: (lambda r: f"rc={r[0]} ctx={hx(r[1])}")(retain()))
    _, p = retain()
    step(f"push({hx(p)})", lambda: f"rc={push(p)}")
    for i in range(3):
        step(f"pop#{i}", lambda: (lambda r: f"rc={r[0]} out={hx(r[1])}")(pop()))
    print("--- B 结束：恢复 current（setCurrent）---", flush=True)
    step(f"setCurrent({hx(p)})", lambda: f"rc={_lib.cuCtxSetCurrent(V(p))}")

    print("--- C: setCurrent 后再 push x2 + pop x3 ---", flush=True)
    step(f"setCurrent({hx(p)})", lambda: f"rc={_lib.cuCtxSetCurrent(V(p))}")
    for i in range(2):
        step(f"push#{i}", lambda: f"rc={push(p)}")
    for i in range(3):
        step(f"pop#{i}", lambda: (lambda r: f"rc={r[0]} out={hx(r[1])}")(pop()))
    step(f"setCurrent({hx(p)})", lambda: f"rc={_lib.cuCtxSetCurrent(V(p))}")


def mode_thread(args):
    """验证「另一线程持有同一 context current」时，主线程的 pop/push 行为。

    这是「pop 真的解绑了而 push 却失败 → 主线程丢失 current context」这一
    残余风险的唯一可能触发路径，必须实测排除。
    """
    import threading
    import time

    init_torch()
    print("torch cuda initialized", flush=True)

    _, p = retain()          # 编码器创建时的 retain（refcount +1）
    print(f"retain ctx={hx(p)}", flush=True)

    stop = threading.Event()

    def worker():
        # 模拟编码线程 _loop() 的 [FIX-ENC-CTX]
        rc = _lib.cuCtxSetCurrent(V(p))
        print(f"  [worker] setCurrent rc={rc}", flush=True)
        stop.wait(10)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(1.0)

    print("--- 主线程：模拟 close() 的 release + pop + push ---", flush=True)
    rc0, c0 = cur()
    print(f"  当前 current={hx(c0)} rc={rc0}", flush=True)
    print(f"  cuDevicePrimaryCtxRelease -> rc={_lib.cuDevicePrimaryCtxRelease(ctypes.c_int(0))}", flush=True)
    rc1, o1 = pop()
    rc2, c2 = cur()
    print(f"  pop -> rc={rc1} out={hx(o1)} ; pop 后 current={hx(c2)}", flush=True)
    rc3 = push(p)
    rc4, c4 = cur()
    print(f"  push(saved) -> rc={rc3} ; push 后 current={hx(c4)}", flush=True)

    stop.set()
    t.join()
    print("--- 结论判定 ---", flush=True)
    if rc1 == 0 and o1 is None and rc3 == 201:
        print("  pop 是 no-op（未解绑），push 因「已 current」返回 201 → 净变化 0，无害", flush=True)
    elif rc1 == 0 and o1 is not None and rc3 == 0:
        print("  pop 真的解绑，push 成功恢复 → 净变化 0，无害（无警告产生）", flush=True)
    elif rc1 == 0 and o1 is not None and rc3 != 0:
        print(f"  ⚠️ 危险：pop 解绑成功(out={hx(o1)})但 push 失败(rc={rc3}) → 主线程丢失 current context", flush=True)
    else:
        print(f"  其它组合：pop rc={rc1} out={hx(o1)} / push rc={rc3}", flush=True)


def mode_sim(args):
    """完整复刻生产时序：每段「编码线程 setCurrent → 编码器创建 → 编码 → 编码器关闭」。

    --alive : 编码线程在 close() 时仍然存活（未 join）
    默认     : 编码线程在 close() 前已 join（模拟段处理完成后再建下一段编码器）
    """
    import threading
    import time

    init_torch()
    print("torch cuda initialized", flush=True)
    _, p = retain()
    print(f"primary ctx={hx(p)}  (--alive={args.alive})", flush=True)

    stop = threading.Event()
    worker_started = threading.Event()

    def worker():
        _lib.cuCtxSetCurrent(V(p))
        worker_started.set()
        if args.alive:
            stop.wait(30)

    for seg in range(args.cycles):
        worker_started.clear()
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        worker_started.wait(5)
        if not args.alive:
            t.join()          # 段处理结束：编码线程已退出

        # ── 编码器创建 _acquire_cuda_context() ──
        rc_s, saved = cur()
        retain()
        rc_push_create = push(p)
        rc_a, cur_a = cur()

        # ── 编码器关闭 close() ──
        rc_rel = _lib.cuDevicePrimaryCtxRelease(ctypes.c_int(0))
        rc_pop, out = pop()
        rc_b, cur_b = cur()
        rc_push_saved = push(saved) if saved is not None else None
        rc_c, cur_c = cur()

        warn = "  <<< 201 警告" if rc_push_saved == 201 else ""
        lost = "  ⚠️ 主线程丢失 current context" if cur_c is None else ""
        print(f"[seg {seg}] saved={hx(saved)} | "
              f"create: push={rc_push_create} cur={hx(cur_a)} | "
              f"close: release={rc_rel} pop={rc_pop}(out={hx(out)}) cur_after_pop={hx(cur_b)} "
              f"push(saved)={rc_push_saved} cur={hx(cur_c)}{warn}{lost}", flush=True)
        if args.alive:
            stop.set()
            t.join()

        # 下一段前做一次 torch CUDA 运算，验证主线程 context 是否仍可用
        try:
            import torch
            _ = (torch.zeros(4, device="cuda") + 1).sum().item()
            print(f"         torch CUDA OK", flush=True)
        except Exception as _e:
            print(f"         torch CUDA 失败: {type(_e).__name__}: {_e}", flush=True)
            break

    print("--- 结束，测量剩余栈深度 ---", flush=True)
    measure_depth("sim-end")


def mode_exit(args):
    """决定性实验：编码线程「先 setCurrent 再退出（已被 join）」之后，
    主线程在 refcount==1 时执行 close() 的 pop + push 会怎样。

    对照：
      mode_thread（线程仍存活）→ pop 解绑成功 + push 201 → 主线程丢失 context（危险）
      mode_exit（线程已退出）  → ?
    """
    import threading

    init_torch()
    print("torch cuda initialized  (refcount = torch 的 1)", flush=True)
    _, p = retain()

    def worker():
        _lib.cuCtxSetCurrent(V(p))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join()
    print("worker 线程已 setCurrent(primary) 并退出（已 join）", flush=True)

    rc0, c0 = cur()
    print(f"  close 前 current={hx(c0)}", flush=True)
    print(f"  cuDevicePrimaryCtxRelease -> rc={_lib.cuDevicePrimaryCtxRelease(ctypes.c_int(0))}", flush=True)
    rc1, o1 = pop()
    rc2, c2 = cur()
    print(f"  pop -> rc={rc1} out={hx(o1)} ; pop 后 current={hx(c2)}", flush=True)
    rc3 = push(p)
    rc4, c4 = cur()
    print(f"  push(saved) -> rc={rc3} ; push 后 current={hx(c4)}", flush=True)
    print("--- 判定 ---", flush=True)
    if rc1 == 0 and o1 is None:
        print("  pop 是 no-op（refcount>1 或 context 仍被占用）→ push 201 无害，净变化 0", flush=True)
    elif rc1 == 0 and o1 is not None and rc3 == 0:
        print("  pop 解绑 + push 成功 → 无警告产生（与生产现象不符）", flush=True)
    elif rc1 == 0 and o1 is not None and rc3 == 201:
        print("  ⚠️ pop 解绑成功但 push 失败 → 主线程丢失 current context（危险路径）", flush=True)
    else:
        print(f"  其它：pop rc={rc1} out={hx(o1)} push rc={rc3}", flush=True)


def mode_real(args):
    import ifrnet_video.nvenc_sdk as nsdk  # noqa: E402

    log_fp = open(args.log, "w") if args.log else None

    def logger(name, a, rc):
        if name in _CudaProxy.CTX_FUNCS:
            extra = ""
            if a:
                a0 = a[0]
                # byref(c_void_p) 形式的出参：读回 *pctx
                if hasattr(a0, "_obj"):
                    try:
                        extra = " out=%s" % hx(a0._obj.value)
                    except Exception:
                        extra = " out=?"
                elif isinstance(a0, ctypes.c_void_p):
                    try:
                        extra = " arg0=%s" % hx(a0.value)
                    except Exception:
                        pass
                elif isinstance(a0, int):
                    extra = " arg0=%d" % a0
            line = f"    [libcuda] {name}{extra} -> rc={rc}"
            print(line, flush=True)
            if log_fp:
                log_fp.write(line + "\n")

    # 仅替换 nvenc_sdk 模块内的 ctypes.CDLL，避免影响其它模块
    shim = types.ModuleType("ctypes_shim")
    shim.__dict__.update(ctypes.__dict__)

    def patched_cdll(name, *a, **k):
        real = ctypes.CDLL(name, *a, **k)
        if name and "libcuda" in str(name):
            return _CudaProxy(real, logger)
        return real

    shim.CDLL = patched_cdll
    nsdk.ctypes = shim

    NVENCEncoder = nsdk.NVENCEncoder

    class ProbeEncoder(NVENCEncoder):
        def _acquire_cuda_context(self):
            r = super()._acquire_cuda_context()
            saved = getattr(self, "_saved_ctx", None)
            prim = getattr(self, "_primary_ctx", None)
            print(f"  >> create: _saved_ctx={hx(saved.value if saved else None)} "
                  f"_primary_ctx={hx(prim.value if prim else None)} "
                  f"equal={(saved is not None and prim is not None and saved.value == prim.value)}",
                  flush=True)
            if log_fp:
                log_fp.write(f"create saved={hx(saved.value if saved else None)} "
                             f"primary={hx(prim.value if prim else None)}\n")
            return r

        def close(self):
            saved = getattr(self, "_saved_ctx", None)
            prim = getattr(self, "_primary_ctx", None)
            rc, c = cur()
            print(f"  >> close : _saved_ctx={hx(saved.value if saved else None)} "
                  f"_primary_ctx={hx(prim.value if prim else None)} "
                  f"current={hx(c)}(rc={rc})", flush=True)
            if log_fp:
                log_fp.write(f"close  saved={hx(saved.value if saved else None)} "
                             f"primary={hx(prim.value if prim else None)} "
                             f"current={hx(c)}\n")
            return super().close()

    # 生产时序：torch 先于编码器初始化
    init_torch()
    print("torch cuda initialized", flush=True)
    rc, c0 = cur()
    print(f"[init] getcur rc={rc} ctx={hx(c0)}", flush=True)

    W, H = 640, 360
    primary_first = None
    for i in range(args.cycles):
        print(f"===== 编码器 #{i} =====", flush=True)
        enc = ProbeEncoder(width=W, height=H, fps=30.0, preset="veryslow", qp=21,
                           codec=args.codec, pipeline_depth=4,
                           rate_mode="vbr_hq", la_depth=args.la)
        if primary_first is None:
            primary_first = getattr(enc, "_primary_ctx", None)
        rc, c1 = cur()
        print(f"  -- 创建后 current={hx(c1)} (rc={rc})", flush=True)
        enc.close()
        rc, c2 = cur()
        print(f"  -- 关闭后 current={hx(c2)} (rc={rc})", flush=True)
        del enc

        # 每轮关闭后做一次真实 torch CUDA 运算，验证主进程 context 是否仍然有效
        try:
            import torch
            _ = (torch.zeros(4, device="cuda") + 1).sum().item()
            print(f"  -- 关闭后 torch CUDA 运算 OK", flush=True)
        except Exception as _e:
            print(f"  -- 关闭后 torch CUDA 运算失败: {_e}", flush=True)

    print("--- 压测结束，测量剩余栈深度（破坏性）---", flush=True)
    measure_depth("real-end")
    if log_fp:
        log_fp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["depth", "cycle", "seq", "thread", "sim", "exit", "real"])
    ap.add_argument("--torch", action="store_true")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--codec", default="hevc")
    ap.add_argument("--la", type=int, default=8)
    ap.add_argument("--log", default="")
    ap.add_argument("--alive", action="store_true")
    a = ap.parse_args()
    {"depth": mode_depth, "cycle": mode_cycle, "seq": mode_seq,
     "thread": mode_thread, "sim": mode_sim, "exit": mode_exit,
     "real": mode_real}[a.mode](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
