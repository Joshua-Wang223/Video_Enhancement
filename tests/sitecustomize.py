"""非侵入式 libcuda context 调用日志（通过 PYTHONPATH 注入，不修改任何生产代码）。

仅当环境变量 NVENC_CTXLOG 指向一个路径时才记录；未设置时本模块完全惰性，
不会包装任何 CDLL，对被测进程零影响（可安全长期留在 tests/ 下）。

记录范围：cuCtx* / cuDevicePrimaryCtx*，含线程名、返回码、出参值。

用法：
  NVENC_CTXLOG=/workspace/Video_Enhancement/temp/retest/ctx.log \
  PYTHONPATH=/workspace/Video_Enhancement/tests \
  python src/main_video_optimized.py ...
"""
import os
import threading
import time

_LOG_PATH = os.environ.get("NVENC_CTXLOG", "")
if _LOG_PATH:
    import ctypes

    _fp = open(_LOG_PATH, "a", buffering=1)
    _lock = threading.Lock()
    _t0 = time.time()

    _CTX_PREFIX = ("cuCtx", "cuDevicePrimaryCtx")

    def _fmt(v):
        if v is None:
            return "NULL"
        try:
            return "0x%x" % v
        except Exception:
            return repr(v)

    class _FnProxy:
        def __init__(self, real, name):
            object.__setattr__(self, "_real", real)
            object.__setattr__(self, "_name", name)

        def __setattr__(self, k, v):
            setattr(object.__getattribute__(self, "_real"), k, v)

        def __getattr__(self, k):
            return getattr(object.__getattribute__(self, "_real"), k)

        def __call__(self, *a, **k):
            rc = object.__getattribute__(self, "_real")(*a, **k)
            name = object.__getattribute__(self, "_name")
            if name.startswith(_CTX_PREFIX):
                parts = []
                for x in a:
                    if hasattr(x, "_obj"):          # byref(...) 出参
                        try:
                            parts.append("out=%s" % _fmt(x._obj.value))
                        except Exception:
                            parts.append("out=?")
                    elif isinstance(x, ctypes._Pointer):
                        try:
                            parts.append("ptr=%s" % _fmt(x.contents.value if hasattr(x.contents, "value") else None))
                        except Exception:
                            parts.append("ptr=?")
                    elif isinstance(x, ctypes.c_void_p):
                        parts.append("ctx=%s" % _fmt(x.value))
                    elif isinstance(x, int):
                        parts.append("%d" % x)
                cur = ctypes.c_void_p()
                try:
                    g = object.__getattribute__(self, "_real")
                    cu = ctypes.CDLL.__dict__ and None
                except Exception:
                    cu = None
                with _lock:
                    _fp.write("[%8.3f] tid=%-5d %-28s %-28s -> rc=%s | %s\n" % (
                        time.time() - _t0,
                        threading.get_ident() % 100000,
                        threading.current_thread().name,
                        name, rc, " ".join(parts)))
            return rc

    class _LibCudaProxy:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)
            object.__setattr__(self, "_cache", {})

        def __getattr__(self, name):
            cache = object.__getattribute__(self, "_cache")
            if name in cache:
                return cache[name]
            fn = getattr(object.__getattribute__(self, "_real"), name)  # 缺失符号照常抛 AttributeError
            if name.startswith(_CTX_PREFIX):
                w = _FnProxy(fn, name)
            else:
                w = fn
            cache[name] = w
            return w

    _orig_CDLL = ctypes.CDLL

    def _patched_CDLL(name, *a, **k):
        lib = _orig_CDLL(name, *a, **k)
        try:
            if name and "libcuda" in str(name):
                return _LibCudaProxy(lib)
        except Exception:
            pass
        return lib

    ctypes.CDLL = _patched_CDLL
