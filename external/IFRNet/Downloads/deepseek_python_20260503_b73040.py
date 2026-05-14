import warnings

def _sample(self):
    # ---------- 初始化 NVML（兼容新旧包） ----------
    pynvml_module = None
    pynvml_handle = None
    try:
        # 优先使用推荐的新包
        import nvidia_ml_py as nvml
        pynvml_module = nvml
    except ImportError:
        try:
            # 回退到旧的 pynvml（此时仍会触发警告，但至少运行）
            import pynvml as nvml
            pynvml_module = nvml
        except ImportError:
            pass

    if pynvml_module is not None:
        try:
            pynvml_module.nvmlInit()
            pynvml_handle = pynvml_module.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pynvml_handle = None

    # ---------- 采样循环 ----------
    while not self._stop_event.is_set():
        util = 0.0
        mem_used_gib = 0.0
        try:
            if pynvml_handle is not None:
                util = float(
                    pynvml_module.nvmlDeviceGetUtilizationRates(pynvml_handle).gpu
                )
                mem_info = pynvml_module.nvmlDeviceGetMemoryInfo(pynvml_handle)
                mem_used_gib = mem_info.used / (1024**3)
            elif torch.cuda.is_available():
                try:
                    util = float(torch.cuda.utilization(0))
                except Exception:
                    util = 0.0
                free, total = torch.cuda.mem_get_info(0)
                mem_used_gib = (total - free) / (1024**3)
        except Exception:
            pass
        self.util_samples.append(util)
        self.mem_samples.append(mem_used_gib)
        self._stop_event.wait(self.interval)

    # ---------- 清理 ----------
    if pynvml_handle is not None and pynvml_module is not None:
        try:
            pynvml_module.nvmlShutdown()
        except Exception:
            pass