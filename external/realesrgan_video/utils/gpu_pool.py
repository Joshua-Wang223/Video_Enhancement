#!/usr/bin/env python3
import queue
from typing import Dict, Any, Optional, Tuple

class GPUMemoryPool:
    """流水线并发槽计数器（纯信号量）"""
    def __init__(self, max_batches: int = 4, batch_size: int = 4, img_size: Tuple[int, int] = (540, 960), device: str = 'cuda'):
        self.max_batches, self.batch_size, self.img_size, self.device = max_batches, batch_size, img_size, device
        self._slots = queue.Queue()
        for i in range(max_batches): self._slots.put(i)
        self.lock = threading.Lock()
    def acquire(self) -> Optional[Dict[str, Any]]:
        try: return {'index': self._slots.get_nowait()}
        except queue.Empty: return None
    def release(self, idx: int): self._slots.put(idx)

import threading # 补充导入