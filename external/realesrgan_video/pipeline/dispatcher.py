#!/usr/bin/env python3
import threading, queue, time
from typing import Dict, Any, Optional, List
from realesrgan_video.utils.shm import SharedMemoryDoubleBuffer

class AsyncGFPGANDispatcher:
    def __init__(self, subprocess_obj, shm_buf: Optional[SharedMemoryDoubleBuffer] = None):
        self.subprocess, self.shm_buf = subprocess_obj, shm_buf
        self._results: Dict[int, Any] = {}
        self._lock = threading.Lock(); self._cv = threading.Condition(self._lock)
        self._validate_results: Dict[int, bool] = {}
        self._validate_lock = threading.Lock(); self._validate_cv = threading.Condition(self._validate_lock)
        self._running = True
        self._collector = threading.Thread(target=self._collect_loop, daemon=True, name='async_gfpgan_collector')
        self._collector.start()
    def _collect_loop(self):
        while self._running:
            try: res = self.subprocess.result_queue.get(timeout=1.0)
            except queue.Empty: continue
            if res is None: break
            if isinstance(res, tuple) and len(res) == 3 and res[0] == '__validate__':
                _, val_id, ok = res
                with self._validate_lock: self._validate_results[val_id] = ok; self._validate_cv.notify_all()
                continue
            if isinstance(res, tuple) and len(res) == 2:
                task_id, result = res
                with self._lock: self._results[task_id] = result; self._cv.notify_all()
    def wait_result(self, task_id: int, timeout: float = 120.0, slot: Optional[int] = None) -> Optional[list]:
        deadline = time.time() + timeout
        with self._lock:
            while task_id not in self._results:
                remaining = deadline - time.time()
                if remaining <= 0: return None
                self._cv.wait(timeout=min(1.0, remaining))
            if not self._running: return None
            raw = self._results.pop(task_id)
            if isinstance(raw, int) and self.shm_buf is not None and slot is not None:
                restored_list = self.shm_buf.read_output(slot, raw)
                return [r if r.any() else None for r in restored_list]
            return raw
    def wait_validate(self, val_id: int, timeout: float = 180.0) -> bool:
        deadline = time.time() + timeout
        with self._validate_lock:
            while val_id not in self._validate_results:
                remaining = deadline - time.time()
                if remaining <= 0: return False
                if self.subprocess.process is not None and not self.subprocess.process.is_alive(): return False
                self._validate_cv.wait(timeout=min(5.0, remaining))
            return self._validate_results.pop(val_id)
    def submit_validate(self, val_id: int):
        try: self.subprocess.task_queue.put(('__validate__', val_id), timeout=5.0)
        except queue.Full: pass
    def close(self):
        self._running = True
        with self._lock: self._cv.notify_all()
        with self._validate_lock: self._validate_cv.notify_all()
        if self._collector.is_alive(): self._collector.join(timeout=5.0)