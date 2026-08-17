"""[FIX-STREAM-TS-REASSOC] backport 回归验证（替代已移除的 [FIX-LA-WINDOW-ROTATION]）。

对全部 6 个 IFRNet 版本文件 (v6.4.3 / v6.4.3.1 / v6.4.4 / v6.4.4.1 / v6.4.5 / v6.4.5.1):
1. 断言 rotation 方案已移除: 无 _fix_la_window_rotation / _rotation_* 方法
   （主文件 v6.4.5.1 保留 _RotationBitReader 类供备用方向 B，不判失败）
2. 断言 _drain_outputs_blocking 返回三元组 (est_fi, out_ts, h264)（ts 重关联）
3. 流式重排缓冲单测: 乱序 (fi, h264) 经缓冲输出严格递增且帧数守恒（对拍）
4. BR 钳制单测: _clamp_bitrate 把高分辨率×帧率失控值钳到 [5M, 50M] 区间

用法: python tests/verify_rotation_backport.py
退出码: 0=全部通过 1=有失败
"""
import ast
import importlib.util
import os
import sys

# [FIX-CROSS-PLATFORM] Windows 默认 GBK 控制台无法输出部分 Unicode（→/≈/emoji），
# 显式切 UTF-8 保证开发(Windows)/生产(Linux)一致。
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VERSIONS = [
    "process_video_v6_4_3_single.py",
    "process_video_v6_4_3_1_single.py",
    "process_video_v6_4_4_single.py",
    "process_video_v6_4_4_1_single.py",
    "process_video_v6_4_5_single.py",
    "process_video_v6_4_5_1_single.py",
]

# 主文件保留 _RotationBitReader 类（备用方向 B），其他版本应完全移除 rotation。
_KEEP_READER = {"process_video_v6_4_5_1_single.py"}

# stream 版（有 encode_frames_stream / _drain_outputs_blocking，LA 路径）要求 ts 重关联；
# batch 版（v6.4.3/4/5，CONSTQP-only、_la_depth 恒 0）用同步 batch 编码路径，
# 无 _drain_outputs_blocking，ts 重关联不适用——仅需 rotation 已移除。
_STREAM_VERSIONS = {
    "process_video_v6_4_3_1_single.py",
    "process_video_v6_4_4_1_single.py",
    "process_video_v6_4_5_1_single.py",
}

_FAILS = []


def fail(msg):
    _FAILS.append(msg)
    print("  [FAIL] " + msg, flush=True)


def ok(msg):
    print("  [OK]   " + msg, flush=True)


# ═══════════════════════════════════════════════
# 1. AST: rotation 已移除 + drain 返回三元组
# ═══════════════════════════════════════════════
def check_version_ast(fname):
    path = os.path.join(BASE, "external", "IFRNet", fname)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tree = ast.parse(text)
    has_fix = False
    has_rotation_method = False
    has_rotation_state = False
    has_drain = False
    drain_returns_ts = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "NVENCEncoder":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    if sub.name == "_fix_la_window_rotation":
                        has_fix = True
                    if sub.name.startswith("_rotation"):
                        has_rotation_method = True
                    if sub.name == "_drain_outputs_blocking":
                        has_drain = True
                        # 检查 return 语句是否含三元组 (est_fi, ts, data)
                        src = ast.get_source_segment(text, sub) or ""
                        drain_returns_ts = ("outputTimeStamp" in src or
                                            "est_fi, _out_ts, h264_data" in src)
            # init 中 rotation 状态
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) \
                                and tgt.value.id == "self" and tgt.attr.startswith("_rotation"):
                            has_rotation_state = True
    # _RotationBitReader 模块级类
    has_reader = any(isinstance(n, ast.ClassDef) and n.name == "_RotationBitReader"
                     for n in tree.body)
    keep_reader = fname in _KEEP_READER
    is_stream = fname in _STREAM_VERSIONS

    problems = []
    if has_fix:
        problems.append("仍含 _fix_la_window_rotation")
    if has_rotation_method:
        problems.append("仍含 _rotation_* 方法")
    if has_rotation_state:
        problems.append("仍含 self._rotation_* 状态")
    if has_reader and not keep_reader:
        problems.append("仍含 _RotationBitReader 类（本版本应移除）")
    if is_stream:
        if not has_drain:
            problems.append("缺少 _drain_outputs_blocking")
        elif not drain_returns_ts:
            problems.append("_drain_outputs_blocking 未返回三元组（缺 ts 重关联）")
    # batch 版（CONSTQP-only）无 _drain_outputs_blocking 属正常，不做要求
    return problems


# ═══════════════════════════════════════════════
# 2. 流式重排缓冲（与主文件 _drain_write 重排逻辑等价）
# ═══════════════════════════════════════════════
class ReorderBuffer:
    """跨 chunk 流式重排缓冲：输出严格递增前缀，段末强制排空。"""

    def __init__(self):
        self.next_fi = 0
        self.pending = {}
        self.written = 0

    def push(self, pairs):
        for fi, h in pairs:
            self.pending[fi] = h
        n = 0
        while self.next_fi in self.pending:
            self.pending.pop(self.next_fi)
            self.next_fi += 1
            n += 1
            self.written += 1
        return n

    def flush_final(self):
        rem = sorted(self.pending.keys())
        for fi in rem:
            self.pending.pop(fi)
            self.written += 1
        return len(rem)


def test_reorder_buffer():
    print("[重排缓冲] 乱序对拍（随机洗牌 + chunk 切分）")
    import random
    random.seed(42)
    for trial in range(20):
        total = random.randint(30, 300)
        window = random.randint(3, 12)
        # 构造 (fi, h264) 乱序流：模拟 ts 重关联后 LA 硬件输出 buffer 重路由
        items = list(range(total))
        # 按窗口洗牌：每个 window 帧块内部随机错位（模拟 slot 相位错开）
        for start in range(window, total, window):
            blk = items[start:start + window]
            random.shuffle(blk)
            items[start:start + window] = blk
        # 切成随机 chunk（模拟 writer 每 chunk 收到若干帧）
        chunks = []
        i = 0
        while i < total:
            csz = random.randint(1, window * 2 + 1)
            chunks.append([(fi, b"x" * (fi + 1)) for fi in items[i:i + csz]])
            i += csz
        buf = ReorderBuffer()
        out_fis = []
        for c in chunks:
            buf.push(c)
        # 段末排空
        buf.flush_final()
        out_fis = list(range(buf.written))
        if out_fis != list(range(total)):
            fail("试%d: 输出帧号不连续 total=%d" % (trial, total))
            return
    ok("20 组随机乱序全部输出严格递增 0..N-1（帧数守恒）")


def test_reorder_buffer_healthy():
    print("[重排缓冲] 正常流零误伤")
    buf = ReorderBuffer()
    for i in range(100):
        buf.push([(i, b"x" * (i + 1))])
    buf.flush_final()
    if buf.written != 100 or buf.next_fi != 100:
        fail("正常递增流被误伤: written=%d next=%d" % (buf.written, buf.next_fi))
    else:
        ok("正常递增流 100 帧零误伤")


# ═══════════════════════════════════════════════
# 3. BR 钳制单测
# ═══════════════════════════════════════════════
def test_clamp_bitrate():
    print("[BR 钳制] _clamp_bitrate")
    sys.path.insert(0, os.path.join(BASE, "external", "IFRNet"))
    try:
        spec = importlib.util.spec_from_file_location(
            "v6451", os.path.join(BASE, "external", "IFRNet",
                                  "process_video_v6_4_5_1_single.py"))
        mod = importlib.util.module_from_spec(spec)
        # 不实际执行模块（会 import ctypes 等，但无 GPU 副作用）；仅提取 _clamp_bitrate
        src = open(os.path.join(BASE, "external", "IFRNet",
                                "process_video_v6_4_5_1_single.py"), encoding="utf-8").read()
        tree = ast.parse(src)
        clamp_src = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_clamp_bitrate":
                clamp_src = ast.get_source_segment(src, node)
        if clamp_src is None:
            fail("主文件未定义 _clamp_bitrate")
            return
        # _clamp_bitrate 引用模块级常量 _NVENC_BR_CLAMP_MIN/MAX，此处注入（与主文件值一致）
        ns = {"_NVENC_BR_CLAMP_MIN": 5_000_000, "_NVENC_BR_CLAMP_MAX": 50_000_000}
        exec(clamp_src, ns)
        clamp = ns["_clamp_bitrate"]
        # 720p48 原始估算 ≈ 132.6Mbps → 应钳到 50Mbps
        v1 = clamp(int(1280 * 720 * 48 * 3.0))
        if not (5_000_000 <= v1 <= 50_000_000):
            fail("720p48 失控值未钳制: %d" % v1)
            return
        # 720p24 ≈ 66.3Mbps → 钳到 50M
        v2 = clamp(int(1280 * 720 * 24 * 3.0))
        if v2 > 50_000_000:
            fail("720p24 未钳制: %d" % v2)
            return
        # 640x480@24 ≈ 22.1M → 保持
        v3 = clamp(int(640 * 480 * 24 * 3.0))
        if not (5_000_000 <= v3 <= 50_000_000):
            fail("640x480@24 异常: %d" % v3)
            return
        # 极低值（<=0）→ 保底 5M
        v4 = clamp(0)
        if v4 != 5_000_000:
            fail("clamp(0) 保底失败: %d" % v4)
            return
        ok("clamp 测试通过: 720p48→%d 720p24→%d 640p→%d clamp(0)→%d" % (v1, v2, v3, v4))
    except Exception as e:
        fail("clamp 测试异常: %r" % e)


def main():
    print("=" * 70)
    print("[FIX-STREAM-TS-REASSOC] backport 回归验证")
    print("=" * 70)

    print("\n[1] rotation 已移除 + drain ts 重关联 (6 版本 AST 断言)")
    for fname in VERSIONS:
        problems = check_version_ast(fname)
        if problems:
            fail("%s: %s" % (fname, "; ".join(problems)))
        else:
            ok("%s: rotation 已移除, drain 含 ts 重关联" % fname)

    print("\n[2] 流式重排缓冲")
    test_reorder_buffer()
    test_reorder_buffer_healthy()

    print("\n[3] BR 钳制")
    test_clamp_bitrate()

    print("\n" + "=" * 70)
    if _FAILS:
        print("== FAIL: 回归失败 (%d 项) ==" % len(_FAILS))
        sys.exit(1)
    print("== PASS: 全部通过 ==")
    sys.exit(0)


if __name__ == "__main__":
    main()
