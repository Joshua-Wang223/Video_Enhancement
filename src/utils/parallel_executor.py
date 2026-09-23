#!/usr/bin/env python3
"""
统一并行执行引擎
参考模式：tests/analyze_video_pipeline_v3.py (run_tasks_parallel + 任务级信号量) +
         tests/verify_segment_bitstream_v5.py (run_verify_parallel + 结果顺序保留) +
         tests/benchmark_ifrnet_versions_v3.py (自动 workers + GPU 动态上限)
功能：ThreadPoolExecutor/ProcessPoolExecutor 统一封装、GPU 信号量闸门、
     进度回调、异常聚合、两阶段流水线支持、顺序保留
"""
import os
import pickle
import time
import contextlib
import threading
import traceback
import multiprocessing
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, List, Dict, Any, Optional, Tuple, TypeVar, Generic, cast
from dataclasses import dataclass

# 兼容两种导入方式：包内相对导入(src.utils.parallel_executor)，以及项目把
# src/utils 加入 sys.path 后的顶层模块导入(parallel_executor)。
# 仅用相对导入会在顶层导入场景下抛 "attempted relative import with no known
# parent package"，导致调用方整块功能静默失效。
if __package__:
    from .system_resources import (
        SystemResourceDetector,
        SystemResources,
        get_gpu_semaphore,
        compute_auto_workers,
    )
else:
    from system_resources import (
        SystemResourceDetector,
        SystemResources,
        get_gpu_semaphore,
        compute_auto_workers,
    )

T = TypeVar("T")

# workers 硬上限：避免大核机器上创建过多线程/进程导致调度抖动与内存放大
MAX_WORKERS_CAP = 32
# GPU 并发闸门硬上限（与 system_resources.get_gpu_semaphore 保持一致）
MAX_GPU_SLOTS_CAP = 8
# 合法执行模式，提前校验避免未知值静默退化成 thread
VALID_MODES = ("thread", "process")

# 任务描述: (可调用对象, 位置参数, 关键字参数, 任务名)
TaskSpec = Tuple[Callable[..., Any], Tuple[Any, ...], Dict[str, Any], str]


@dataclass
class TaskResult(Generic[T]):
    index: int
    success: bool
    result: Optional[T] = None
    error: Optional[str] = None
    elapsed: float = 0.0
    worker_id: int = 0


@dataclass
class ProgressInfo:
    completed: int
    total: int
    failed: int
    elapsed: float
    eta: float


# 进程模式下由 initializer 注入子进程的 GPU 闸门（信号量只能通过进程继承传递）
_WORKER_GATE: Optional[Any] = None


def _init_worker(gate: Optional[Any]) -> None:
    """ProcessPoolExecutor 初始化器：把 GPU 闸门以继承方式注入子进程。

    threading/multiprocessing 信号量都不能随 submit 参数走调用队列
    （会触发 "should only be shared between processes through inheritance"），
    只有通过 Process 构造参数（initializer/initargs）继承才合法。
    """
    global _WORKER_GATE
    _WORKER_GATE = gate


def _run_task(
    fn: Callable[..., T],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    gpu_gate: Optional[Any],
    name: str,
) -> "TaskResult[T]":
    """任务包装器（必须是模块级函数）。

    说明：
      · 定义在模块层是为了可被 pickle —— 原先使用嵌套闭包，process 模式下会直接
        抛 PicklingError/AttributeError，导致 parallel_mode="process" 完全不可用。
      · GPU 闸门在任务体外部获取，异常路径也会由 with 正常释放，不会泄漏信号量。
      · 任何异常都被聚合成 TaskResult 返回，绝不向上抛，避免单个任务拖垮整批任务。
      · worker_id 融合 pid 与线程 id，thread/process 两种模式都能区分真实执行者。
    """
    gate = gpu_gate
    if gate is None:
        gate = _WORKER_GATE if _WORKER_GATE is not None else contextlib.nullcontext()
    worker_id = hash((os.getpid(), threading.get_ident())) & 0x7FFFFFFF
    t0 = time.time()
    try:
        with gate:
            result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        return TaskResult(index=-1, success=True, result=result,
                          elapsed=elapsed, worker_id=worker_id)
    except Exception as exc:
        elapsed = time.time() - t0
        err_msg = f"[{name}] {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        return TaskResult(index=-1, success=False, error=err_msg,
                          elapsed=elapsed, worker_id=worker_id)


def _find_unpicklable(specs: List[TaskSpec]) -> Optional[str]:
    """预检任务是否可序列化，返回首个不兼容任务的描述；全部可序列化时返回 None。

    process 模式依赖 pickle 把任务送进子进程（闭包/lambda/局部函数不可序列化）。
    提前预检可以把晦涩的 PicklingError 转换成一次明确的降级告警。
    """
    for fn, args, kwargs, name in specs:
        try:
            pickle.dumps(fn)
            pickle.dumps(args)
            pickle.dumps(kwargs)
        except Exception as exc:
            return f"{name}: {type(exc).__name__}: {exc}"
    return None


class ParallelExecutor:
    def __init__(
        self,
        workers: Optional[int] = None,
        parallel_mode: str = "thread",
        task_ram_mb: int = 1024,
        reserve_ratio: float = 0.10,
        gpu_task: bool = False,
        gpu_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        resources: Optional[SystemResources] = None,
    ):
        # 入参校验：非法值提前失败，避免静默退化成错误配置
        if parallel_mode not in VALID_MODES:
            raise ValueError(
                f"parallel_mode 必须是 {VALID_MODES} 之一，当前值: {parallel_mode!r}")
        if task_ram_mb <= 0:
            raise ValueError(f"task_ram_mb 必须为正整数，当前值: {task_ram_mb}")
        if not 0.0 <= reserve_ratio < 1.0:
            raise ValueError(f"reserve_ratio 必须落在 [0.0, 1.0)，当前值: {reserve_ratio}")

        self.workers = workers
        self.parallel_mode = parallel_mode
        self.task_ram_mb = task_ram_mb
        self.reserve_ratio = reserve_ratio
        self.gpu_task = gpu_task
        self.gpu_workers = gpu_workers
        self.progress_callback = progress_callback
        self.resources = resources or SystemResourceDetector().detect()

        if self.workers is None:
            self.workers = compute_auto_workers(
                task_ram_mb=self.task_ram_mb,
                reserve_ratio=self.reserve_ratio,
                resources=self.resources,
                gpu_task=self.gpu_task,
            )
        self.workers = max(1, min(int(self.workers), MAX_WORKERS_CAP))
        # GPU 闸门容量（0 表示不启用）；信号量实例在运行时按最终执行模式惰性创建
        self._gpu_slots = self._resolve_gpu_slots()
        self._gpu_sem: Optional[Any] = None
        self._completed = 0
        self._failed = 0
        self._total = 0
        self._start_time = 0.0
        self._lock = threading.Lock()
        # 运行锁：禁止同一实例并发执行，否则进度计数器与 _total 会互相污染
        self._run_lock = threading.Lock()

    def submit_tasks(
        self,
        tasks: List[Callable[[], T]],
        task_names: Optional[List[str]] = None,
    ) -> List[TaskResult[T]]:
        specs: List[TaskSpec] = []
        for idx, task in enumerate(tasks):
            name = (task_names[idx] if task_names
                    and idx < len(task_names) else f"task_{idx}")
            specs.append((task, (), {}, name))
        return self._execute(specs)

    def map_tasks(
        self,
        fn: Callable[..., T],
        args_list: List[tuple],
        kwargs_list: Optional[List[dict]] = None,
    ) -> List[TaskResult[T]]:
        if kwargs_list is None:
            # 必须是独立字典，共享同一个 {} 会让任务间误传可变状态
            kwargs_list = [{} for _ in args_list]
        elif len(kwargs_list) != len(args_list):
            raise ValueError(
                f"kwargs_list 长度({len(kwargs_list)}) 与 args_list 长度"
                f"({len(args_list)}) 不一致")
        fn_name = getattr(fn, "__name__", "task")
        # 直接把 (fn, args, kwargs) 交给执行器，不再用 lambda 包一层：
        # lambda 无法被 pickle，是 process 模式失效的另一根因
        specs: List[TaskSpec] = [
            (fn, tuple(args), dict(kwargs), f"{fn_name}_{i}")
            for i, (args, kwargs) in enumerate(zip(args_list, kwargs_list))
        ]
        return self._execute(specs)

    def _execute(self, specs: List[TaskSpec]) -> List[TaskResult[T]]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError(
                "ParallelExecutor 实例正在执行任务，不支持并发调用 "
                "submit_tasks/map_tasks（进度状态会互相污染）")
        try:
            return self._execute_locked(specs)
        finally:
            self._run_lock.release()

    def _execute_locked(self, specs: List[TaskSpec]) -> List[TaskResult[T]]:
        total = len(specs)
        self._total = total
        self._completed = 0
        self._failed = 0
        self._start_time = time.time()
        if total == 0:
            return []

        mode = self.parallel_mode
        if mode == "process":
            bad = _find_unpicklable(specs)
            if bad is not None:
                # 闭包/lambda/局部函数无法进入子进程，与其整批报 PicklingError，
                # 不如明确告警后降级为线程模式，保证任务仍能跑完
                print(f"[WARN] process 模式要求任务可序列化，检测到({bad})，"
                      f"自动降级为 thread 模式")
                mode = "thread"

        # 任务数少于 workers 时不创建空转线程/进程，减少调度与内存开销
        effective_workers = max(1, min(self.workers, total))
        self._gpu_sem = self._build_gpu_gate(mode)
        gate = self._gpu_gate()
        gpu_desc = (f" | gpu_sem={self._gpu_slots}" if self._gpu_slots
                    else " | gpu_sem=off")
        print(
            f"并行执行引擎启动: {total} 任务 | workers={effective_workers}"
            f"/{self.workers} | mode={mode} | "
            f"gpu_task={self.gpu_task}{gpu_desc if self.gpu_task else ''}")

        results: List[Optional[TaskResult[T]]] = [None] * total
        if mode == "process":
            # 信号量通过 initializer 继承进子进程，不能作为 submit 参数走调用队列
            executor = ProcessPoolExecutor(max_workers=effective_workers,
                                           initializer=_init_worker,
                                           initargs=(self._gpu_sem,))
            submit_gate: Optional[Any] = None
        else:
            executor = ThreadPoolExecutor(max_workers=effective_workers)
            submit_gate = gate
        with executor:
            futures: Dict[Future, int] = {}
            for idx, (fn, args, kwargs, name) in enumerate(specs):
                fut = executor.submit(_run_task, fn, args, kwargs, submit_gate, name)
                futures[fut] = idx
            pending = set(futures)
            try:
                for fut in as_completed(futures):
                    pending.discard(fut)
                    idx = futures[fut]
                    results[idx] = self._collect(fut, idx)
            except Exception as exc:
                # worker 进程崩溃(BrokenProcessPool)等致命错误：剩余任务标记失败，
                # 保证返回长度与输入一致，而不是整批结果全部丢失
                fatal = f"{type(exc).__name__}: {exc}"
                print(f"[WARN] 并行执行中断，剩余 {len(pending)} 个任务标记为失败: {fatal}")
                for fut in pending:
                    fut.cancel()
                    results[futures[fut]] = TaskResult(
                        index=futures[fut], success=False, error=fatal)

        # 兜底补齐：任何未回填的位置都视为失败，维持「顺序保留 + 长度一致」契约
        for idx in range(total):
            if results[idx] is None:
                results[idx] = TaskResult(
                    index=idx, success=False, error="task_not_executed")

        total_elapsed = time.time() - self._start_time
        ok_count = sum(1 for r in results if r and r.success)
        failed = total - ok_count
        print(
            f"并行执行完成: {ok_count}/{total} 成功 | 失败: {failed} | "
            f"总耗时: {total_elapsed:.2f}s")
        return cast(List[TaskResult[T]], results)

    def _collect(self, fut: "Future", idx: int) -> TaskResult:
        """收集单个 future 结果：worker 侧异常在此兜底，不会中断整批任务。"""
        try:
            res = fut.result()  # type: ignore[assignment]
        except Exception as exc:
            res = TaskResult(
                index=idx,
                success=False,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        res.index = idx
        # 进度统计放在父线程：跨进程回调不可行，且避免用户回调在工作线程被并发触发
        self._update_progress(res.success)
        return res

    def _resolve_gpu_slots(self) -> int:
        """GPU 并发闸门容量，0 表示不启用闸门。

        仅当 gpu_task=True 且确实存在 GPU 时才限流 —— 原先无条件向
        get_gpu_semaphore 取信号量，纯 CPU 任务也会被 GPU 闸门误限流。
        """
        if not self.gpu_task or self.resources.gpu_count == 0:
            return 0
        if self.gpu_workers == 0:  # 显式关闭闸门
            return 0
        slots = self.gpu_workers
        if slots is None:
            sessions = sum(g.max_nvdec_sessions for g in self.resources.gpus)
            slots = sessions if sessions > 0 else 1
        return max(1, min(int(slots), MAX_GPU_SLOTS_CAP))

    def _build_gpu_gate(self, mode: str) -> Optional[Any]:
        """按最终执行模式构建 GPU 并发闸门。

        thread  模式：threading.BoundedSemaphore（进程内共享，开销最小）
        process 模式：multiprocessing.BoundedSemaphore —— threading 信号量内部持有
                      _thread.lock，无法跨进程序列化，必须换成多进程版本，
                      否则提交任务时就会抛 pickle 错误。
        """
        if self._gpu_slots <= 0:
            return None
        if mode == "process":
            try:
                return multiprocessing.BoundedSemaphore(self._gpu_slots)
            except Exception as exc:
                print(f"[WARN] 多进程 GPU 闸门创建失败，退化为不限流: {exc}")
                return None
        return get_gpu_semaphore(self.resources, self.gpu_workers)

    def _gpu_gate(self) -> Any:
        """可复用的闸门上下文管理器；未启用闸门时返回 nullcontext。"""
        return self._gpu_sem if self._gpu_sem is not None else contextlib.nullcontext()

    def _update_progress(self, success: bool):
        # 只在锁内更新计数并生成快照，用户回调放到锁外执行：
        # 回调若耗时、阻塞或重入，持锁调用会把所有 worker 串行化甚至造成死锁
        with self._lock:
            self._completed += 1
            if not success:
                self._failed += 1
            callback = self.progress_callback
            if callback is None:
                return
            elapsed = time.time() - self._start_time
            rate = self._completed / elapsed if elapsed > 0 else 0.0
            eta = (self._total - self._completed) / rate if rate > 0 else 0.0
            info = ProgressInfo(
                completed=self._completed,
                total=self._total,
                failed=self._failed,
                elapsed=elapsed,
                eta=eta,
            )
        try:
            callback(info)
        except Exception as exc:
            # 进度回调只是可观测性装饰，失败不应中断主流程
            print(f"[WARN] progress_callback 执行失败: {type(exc).__name__}: {exc}")
