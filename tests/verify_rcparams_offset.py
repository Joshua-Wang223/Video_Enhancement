#!/usr/bin/env python3
"""
从 nvEncodeAPI.h 头文件解析 NV_ENC_RC_PARAMS struct 布局，输出完整偏移表。

修复记录 (2026-06-09):
  v1 原始版本: NV_ENC_QP 被当作 4B enum（line 115-116: `if "NV_ENC" in type_str: size = 4`）
              → 累积偏移少 8B → 错误的 "union" 假说 → targetQuality@76(应为88)
  v2 修正版:   递归解析嵌套 struct 定义，NV_ENC_QP 正确识别为 12B sequential struct
              → targetQuality@88, averageBitRate@20, multiPass@100 (已 GPU 验证通过)

用法:
  python verify_rcparams_offset.py                          # 自动查找 nvEncodeAPI.h
  python verify_rcparams_offset.py /path/to/nvEncodeAPI.h   # 指定头文件
  python verify_rcparams_offset.py --fetch                  # 自动下载最新头文件后解析
"""

import re
import sys
import os
import subprocess

# ── 从 nvEncodeAPI.h master 已知的正确嵌套类型大小 ──
# 这些是无法从单文件自动推断的跨 struct 依赖
KNOWN_NESTED_TYPES = {
    "NV_ENC_QP":         12,   # {uint32_t qpInterP; uint32_t qpInterB; uint32_t qpIntra;}
    "NV_ENC_MULTI_PASS":  4,   # enum (uint32)
    "NV_ENC_PARAMS_RC_MODE": 4,  # enum (uint32)
    "NV_ENC_QP_MAP_MODE": 4,   # enum (uint32)
    "NV_ENC_BFRAME_STRATEGY": 4,  # enum (uint32)
}


def fetch_header():
    """Download nvEncodeAPI.h from GitHub master branch using curl."""
    url = "https://raw.githubusercontent.com/FFmpeg/nv-codec-headers/master/include/ffnvcodec/nvEncodeAPI.h"
    out_path = "/tmp/nvEncodeAPI.h"
    print(f"[DOWNLOAD] 下载: {url}")
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", out_path, url],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and os.path.getsize(out_path) > 10000:
            print(f"   [OK] 下载完成 ({os.path.getsize(out_path):,} bytes)")
            return out_path
        else:
            print(f"   [ERR] curl 失败: {result.stderr[:200]}")
            return None
    except Exception as e:
        print(f"   [ERR] 下载异常: {e}")
        return None


def find_header():
    """Find nvEncodeAPI.h — local paths first, then try download if --fetch."""
    paths = [
        "/tmp/nvEncodeAPI.h",
        "/tmp/nv-codec-headers/include/ffnvcodec/nvEncodeAPI.h",
        "/usr/include/ffnvcodec/nvEncodeAPI.h",
        "/usr/local/include/ffnvcodec/nvEncodeAPI.h",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    if len(sys.argv) > 1 and sys.argv[1] != "--fetch":
        p = sys.argv[1]
        if os.path.exists(p):
            return p
    if "--fetch" in sys.argv:
        return fetch_header()
    return None


def parse_all_structs(content):
    """Parse all typedef struct definitions from the header.
    Returns dict: struct_name -> dict of fields."""
    structs = {}

    # Pattern: typedef struct ... StructName { ... } StructName;
    # Handle both "typedef struct _NV_ENC_QP {" and "typedef struct {"
    pattern = r'typedef\s+struct\s+(?:_\w+)?\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\s*(\w+)\s*;'
    # Simpler: split by "typedef struct"
    # Better approach: find each typedef individually
    for match in re.finditer(
        r'typedef\s+struct\s+(?:\w+\s*)?\{((?:[^{}]|(?:\{[^{}]*\}))*)\}\s*(\w+)\s*;',
        content, re.DOTALL
    ):
        body = match.group(1)
        name = match.group(2)
        if name.startswith("_"):
            continue  # skip internal tag names
        # Check if it's a "real" struct (not just a forward declaration)
        if len(body) > 50:  # has actual fields
            structs[name] = body

    return structs


def calculate_type_size(type_str, known_structs, parsed_types):
    """Calculate byte size of a C type string.
    Uses known_structs for nested types, recursing if needed."""
    type_str = type_str.strip().replace("*", "").strip()

    # Pointers are 8 bytes on x86-64
    if type_str.endswith("*"):
        return 8

    # Known nested types
    if type_str in KNOWN_NESTED_TYPES:
        return KNOWN_NESTED_TYPES[type_str]

    # Look up in parsed structs
    if type_str in parsed_types:
        return parsed_types[type_str]

    # Standard types
    type_sizes = {
        "uint8_t": 1, "int8_t": 1, "uint8": 1, "int8": 1,
        "uint16_t": 2, "int16_t": 2, "uint16": 2, "int16": 2,
        "uint32_t": 4, "int32_t": 4, "uint32": 4, "int32": 4,
        "float": 4,
        "uint64_t": 8, "int64_t": 8, "uint64": 8, "int64": 8,
        "double": 8,
        "void": 8,  # void* → pointer
    }

    # NV_ENC_ prefixed enums are all uint32
    if type_str.startswith("NV_ENC_") and type_str not in KNOWN_NESTED_TYPES:
        return 4  # default: enum = uint32

    if type_str in type_sizes:
        return type_sizes[type_str]

    # Default for unknown: assume uint32 (4 bytes)
    print(f"    [WARN]  未知类型 '{type_str}', 默认 4 字节")
    return 4


def _strip_doxygen_comments(body):
    """Remove all Doxygen /**< ... */ and /* ... */ block comments from struct body.
    This is critical because Doxygen comments in nvEncodeAPI.h contain English
    sentences whose words get falsely parsed as C field names."""
    # Remove /**< ... */ (Doxygen post-field comments)
    body = re.sub(r'/\*\*<.*?\*/', '', body, flags=re.DOTALL)
    # Remove /* ... */ (regular block comments)
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)
    # Remove // line comments
    body = re.sub(r'//[^\n]*', '', body)
    return body


def parse_fields(struct_body, known_structs=None, parsed_types=None):
    """Parse struct body and calculate byte offsets.
    Returns (fields_list, total_size)."""
    if known_structs is None:
        known_structs = {}
    if parsed_types is None:
        parsed_types = {}

    # Pre-process: strip all Doxygen and block comments BEFORE parsing
    struct_body = _strip_doxygen_comments(struct_body)

    fields = []
    offset = 0
    _prev_was_bitfield = False  # track consecutive bitfields (share one uint32 container)

    for line in struct_body.split("\n"):
        line = line.strip()

        # Skip empty lines and standalone semicolons
        if not line:
            continue
        if line == ";":
            continue
        # Clean any remaining comment artifacts
        line = line.replace(";", "").strip()
        if not line:
            continue
        # Skip lines that don't look like C declarations (no type keyword pattern)
        looks_like_field = bool(re.match(r'^[\w_]+\s+\*?\w+', line))
        if not looks_like_field:
            _prev_was_bitfield = False
            continue

        # Handle nested struct/union bodies — skip these lines
        if line.startswith("{") or line.startswith("}"):
            _prev_was_bitfield = False
            continue
        if line.startswith("union") or line.startswith("struct"):
            _prev_was_bitfield = False
            continue

        # Handle bitfield: "type name : N"
        # Consecutive :N fields pack into ONE uint32 container — only the FIRST
        # allocates 4 bytes; subsequent ones share the same offset.
        bf_match = re.match(r'(\w[\w\s*]*?)\s+(\w+)\s*:\s*(\d+)', line)
        if bf_match:
            type_str = bf_match.group(1).strip()
            name = bf_match.group(2)
            bits = int(bf_match.group(3))
            fields.append((offset, name, f"bitfield:{bits}", ""))
            if not _prev_was_bitfield:
                offset += 4  # first bitfield allocates the uint32 container
            _prev_was_bitfield = True
            continue

        _prev_was_bitfield = False

        # Handle arrays: "type name[N]"
        arr_match = re.match(r'([\w\s*]+?)\s+(\w+)\s*\[(\d+)\]', line)
        if arr_match:
            type_str = arr_match.group(1).strip()
            name = arr_match.group(2)
            count = int(arr_match.group(3))
            elem_size = calculate_type_size(type_str, known_structs, parsed_types)
            size = elem_size * count
            fields.append((offset, name, f"{type_str}[{count}]", ""))
            offset += size
            continue

        # Regular field: "type name"
        # Support: "const type name", "type *name", "type* name"
        reg_match = re.match(r'(?:const\s+)?([\w\s*]+?)\s+(\*?\w+)', line)
        if reg_match:
            type_str = reg_match.group(1).strip()
            name = reg_match.group(2).strip()

            # Alignment: uint16 → 2-byte, uint64 → 8-byte, uint32 → 4-byte
            if "uint64" in type_str or "int64" in type_str or "double" in type_str:
                offset = (offset + 7) & ~7
            elif "uint16" in type_str or "int16" in type_str:
                offset = (offset + 1) & ~1

            size = calculate_type_size(type_str, known_structs, parsed_types)

            fields.append((offset, name, type_str, ""))
            offset += size

    return fields, offset


def resolve_nested_structs(content):
    """First pass: resolve sizes of all nested structs.
    Returns dict of {struct_name: total_size}. """
    struct_bodies = parse_all_structs(content)
    parsed = {}

    # Iteratively resolve — some structs depend on others
    max_iterations = 5
    for _ in range(max_iterations):
        made_progress = False
        for name, body in struct_bodies.items():
            if name in parsed:
                continue
            try:
                fields, total = parse_fields(body, known_structs=struct_bodies, parsed_types=parsed)
                parsed[name] = total
                made_progress = True
            except Exception:
                pass  # may depend on unresolved types
        if not made_progress:
            break

    # Ensure NV_ENC_QP is correctly known
    if "NV_ENC_QP" not in parsed:
        parsed["NV_ENC_QP"] = 12  # hardcode: {uint32 qpInterP; uint32 qpInterB; uint32 qpIntra;}

    return parsed


def main():
    header = find_header()
    if not header:
        print("[ERR] 找不到 nvEncodeAPI.h")
        print("   用法: python verify_rcparams_offset.py [path/to/nvEncodeAPI.h]")
        print("         python verify_rcparams_offset.py --fetch  (自动下载)")
        return 1

    print(f"[FILE] 头文件: {header}")

    with open(header, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # First pass: resolve nested types
    parsed_types = resolve_nested_structs(content)
    print(f"\n[SCAN] 已解析的嵌套类型:")
    for name, size in sorted(parsed_types.items()):
        marker = " [WARN] " if name == "NV_ENC_QP" and size != 12 else ""
        print(f"   {name:<35} → {size} 字节{marker}")

    if parsed_types.get("NV_ENC_QP", 0) != 12:
        print("\n   [ERR] NV_ENC_QP 大小 ≠ 12！强制修正为 12")
        parsed_types["NV_ENC_QP"] = 12

    # Second pass: parse NV_ENC_RC_PARAMS
    # Find the struct body directly
    rc_match = re.search(
        r'typedef\s+struct\s+(?:_NV_ENC_RC_PARAMS\s*)?\{((?:[^{}]|(?:\{[^{}]*\}))*)\}\s*NV_ENC_RC_PARAMS\s*;',
        content, re.DOTALL
    )
    if not rc_match:
        print("[ERR] 找不到 NV_ENC_RC_PARAMS 定义")
        return 1

    body = rc_match.group(1)
    fields, total_size = parse_fields(body, known_structs={}, parsed_types=parsed_types)

    print(f"\n[LAYOUT] NV_ENC_RC_PARAMS struct 布局 (SEQUENTIAL, total: {total_size} 字节)")
    print(f"{'Offset':>6}  {'rc_ptr':>7}  {'字段名':<35} {'类型':<25} 备注")
    print(f"{'------':>6}  {'------':>7}  {'------':<35} {'------':<25} {'----'}")

    key_fields = {}
    for offset, name, type_str, comment in fields:
        # Calculate rc_ptr index for uint32 fields
        is_u32 = "bitfield" in type_str or type_str in ("uint32_t", "uint32", "NV_ENC_PARAMS_RC_MODE",
                                                         "NV_ENC_MULTI_PASS", "NV_ENC_QP_MAP_MODE",
                                                         "NV_ENC_BFRAME_STRATEGY")
        rc_ptr_idx = f"rc_ptr[{offset//4}]" if is_u32 else ""

        # Mark key fields
        is_key = name in ("targetQuality", "targetQualityLSB", "rateControlMode",
                         "averageBitRate", "maxBitRate", "constQP",
                         "vbvBufferSize", "vbvInitialDelay", "lookaheadDepth",
                         "multiPass", "minQP", "maxQP", "initialRCQP",
                         "temporallayerIdxMask", "temporalLayerQP",
                         "lowDelayKeyFrameScale", "qpMapMode", "lookaheadLevel")

        marker = " ← KEY" if is_key else ""

        if is_u32:
            offset_str = f"{offset:>6}  {rc_ptr_idx:>7}  {name:<35} {type_str:<25}{marker}"
        else:
            offset_str = f"{offset:>6}  {'':>7}  {name:<35} {type_str:<25}{marker}"

        print(offset_str)

        if is_key:
            key_fields[name] = offset

    # ── Cross-reference with known correct values from code ──
    print(f"\n{'='*70}")
    print(f"[KEY] 关键字段: 头文件实际偏移 vs 代码使用值")
    print(f"{'='*70}")

    CODE_USAGE = {
        "rateControlMode":   ("rc_ptr[1] → byte 4",  4),
        "averageBitRate":    ("rc_ptr[5] → byte 20", 20),
        "maxBitRate":        ("rc_ptr[6] → byte 24", 24),
        "vbvBufferSize":     ("rc_ptr[7] → byte 28", 28),
        "vbvInitialDelay":   ("rc_ptr[8] → byte 32", 32),
        "targetQuality":     ("uint8 @ byte 88",      88),
        "targetQualityLSB":  ("uint8 @ byte 89",      89),
        "lookaheadDepth":    ("uint16 @ byte 90",     90),
        "multiPass":         ("rc_ptr[25] → byte 100",100),
        "qpMapMode":         ("rc_ptr[24] → byte 96", 96),
        "lookaheadLevel":    ("rc_ptr[28] → byte 112",112),
    }

    all_ok = True
    for name, (code_desc, code_offset) in CODE_USAGE.items():
        if name in key_fields:
            actual = key_fields[name]
            match = actual == code_offset
            status = "[OK]" if match else "[ERR] MISMATCH!"
            if not match:
                all_ok = False
                # Calculate delta to help debugging
                delta = actual - code_offset
                print(f"  {name:<25} header={actual:>4}  code={code_desc:<30} {status} (delta={delta:+d})")
            else:
                print(f"  {name:<25} header={actual:>4}  code={code_desc:<30} {status}")

    print()
    if all_ok:
        print("[OK] 所有关键字段偏移与头文件一致 — 代码正确！")
    else:
        print("[ERR] 存在偏移不匹配 — 需要修正代码中的 rc_ptr 索引")
        print()
        print("   常见修复方向:")
        print("   · NV_ENC_QP 是 12B struct，不是 4B — 确保解析器正确处理嵌套类型")
        print("   · targetQuality 是 uint8，不是 uint32 — 用 uint8 pointer 写入")
        print("   · multiPass 是独立字段@100，不在 bitfield@36 中")
        print("   · lookaheadDepth 是 uint16@90，不是 byte 78")

    # ── 输出可直接使用的代码模板 ──
    print(f"\n{'='*70}")
    print("[TEMPLATE] 正确的 rcParams Python 写入模板:")
    print(f"{'='*70}")
    print("""
    rc_ptr = cast(byref(preset_config, 8 + 40), ctypes.POINTER(c_uint32))
    rc_ptr[0] = _sdk13_ver(1)        # NV_ENC_RC_PARAMS_VER
    rc_ptr[1] = 32                    # NV_ENC_PARAMS_RC_VBR_HQ
    # ── averageBitRate @20, maxBitRate @24 ──
    rc_ptr[5] = _est_br               # averageBitRate
    rc_ptr[6] = _est_br * 2           # maxBitRate
    # ── bitfield @36: AQ + Temporal AQ ──
    rc_ptr[9] = rc_ptr[9] | (1 << 3) | (1 << 8)
    # ── targetQuality: uint8 at rcParams+88 ──
    _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))
    _tq8_ptr[0] = max(1, 51 - crf) & 0xFF
    # ── multiPass: SEPARATE field @100 ──
    rc_ptr[25] = 0                    # NV_ENC_MULTI_PASS_DISABLED (VBR_HQ/QVBR 不支持 two-pass)
    # ── lookaheadDepth: uint16 @90 ──
    _la_ptr = cast(byref(preset_config, 8 + 40 + 90), ctypes.POINTER(c_uint16))
    _la_ptr[0] = la_depth
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
