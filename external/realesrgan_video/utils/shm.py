#!/usr/bin/env python3
import numpy as np
import queue
from typing import List, Optional
import multiprocessing.shared_memory as shm

class SharedMemoryDoubleBuffer:
    """双缓冲共享内存，用于主进程与 GFPGAN 子进程之间的零拷贝数据传输。
    FIX-SHM-IPC / FIX-SLOT-POOL 完整保留。
    """
    N_SLOTS = 2
    MAX_FACES = 64
    FACE_SHAPE = (512, 512, 3)
    def __init__(self):
        face_bytes = int(np.prod(self.FACE_SHAPE))
        slot_bytes = self.MAX_FACES * face_bytes
        self._input_shms = []
        self._output_shms = []
        for _ in range(self.N_SLOTS):
            self._input_shms.append(shm.SharedMemory(create=True, size=slot_bytes))
            self._output_shms.append(shm.SharedMemory(create=True, size=slot_bytes))
        self._slot_pool = queue.Queue(maxsize=self.N_SLOTS)
        for i in range(self.N_SLOTS): self._slot_pool.put(i)

    @property
    def input_names(self) -> List[str]: return [s.name for s in self._input_shms]
    @property
    def output_names(self) -> List[str]: return [s.name for s in self._output_shms]

    def acquire_slot(self, timeout: float = 30.0) -> int:
        try: return self._slot_pool.get(timeout=timeout)
        except queue.Empty: raise TimeoutError(f'无法在 {timeout}s 内获取空闲 slot')

    def try_acquire_slot(self) -> Optional[int]:
        try: return self._slot_pool.get_nowait()
        except queue.Empty: return None

    def release_slot(self, slot: int):
        if slot is not None: self._slot_pool.put(slot)

    def write_input(self, slot: int, crops: List[np.ndarray]) -> int:
        n = min(len(crops), self.MAX_FACES)
        buf = np.ndarray((self.MAX_FACES, *self.FACE_SHAPE), dtype=np.uint8, buffer=self._input_shms[slot].buf)
        for i in range(n):
            c = crops[i]
            if c.shape == self.FACE_SHAPE: buf[i] = c
            else:
                import cv2; buf[i] = cv2.resize(c, (self.FACE_SHAPE[1], self.FACE_SHAPE[0]))
        return n

    def read_output(self, slot: int, n: int) -> List[np.ndarray]:
        buf = np.ndarray((self.MAX_FACES, *self.FACE_SHAPE), dtype=np.uint8, buffer=self._output_shms[slot].buf)
        return [buf[i].copy() for i in range(n)]

    def close(self):
        for s_list in [self._input_shms, self._output_shms]:
            for s in s_list:
                try: s.close()
                except Exception: pass
                try: s.unlink()
                except Exception: pass