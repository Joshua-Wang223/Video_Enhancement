
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补救分析脚本 — 从已有文本数据恢复 xlsx 报告
===========================================

用途:
  当 analyze_video_pipeline.py 在服务器上运行时终端输出被截断、
  或 ffmpeg_output.txt / xlsx 未传回本地时，通过以下已有文件
  直接在本地补齐 xlsx 报告:
    1) 原始 pipeline 日志 (output_XXX.txt)          — 解析步骤信息、分段元数据
    2) ffprob 统计文本 (ffprob_output_XXX.txt)      — 逐段帧数 (无需实际视频文件)
    3) analyze_test1.txt (可选)                     — 提取 ffmpeg 解码结果 (若无则降级)

  增强功能:
    - 自动补齐被截断的 ffmpeg 解码结果: 从 ffprobe 数据推断缺失分段的帧数
    - 逐段报告真实解码 vs 数据推断的来源
    - 验证 pipeline 日志中所有分段是否成功完成 (如 "✅ 插帧完成！" 标记)

用法:
    python tests/remedy_analysis_xlsx.py temp/output_S02E07.txt \\
        --ffprob temp/ffprob_output_S02E07.txt \\
        [--ffmpeg-log temp/analyze_test1.txt] \\
        [--output-dir temp] [--suffix _S02E07]

    # 简写（文件名符合默认规则时）：
    python tests/remedy_analysis_xlsx.py temp/output_S02E07.txt

依赖:
    pip install openpyxl
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# 复用 analyze_video_pipeline.py 的 parse、build_chain、generate_xlsx
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_video_pipeline import (
    parse_pipeline_log,
    resolve_step_dirs,
    build_chain,
    generate_xlsx,
)


# ═══════════════════════════════════════════════
# A. 从 ffprob_output_XXX.txt 恢复 ffprobe_data
# ═══════════════════════════════════════════════

def parse_ffprob_text(filepath: str) -> dict:
    """
    解析 analyze_video_pipeline.py 生成的 ffprob_output_XXX.txt，
    重建 ffprobe_data dict（格式兼容 build_chain / generate_xlsx）。
    使用逐行解析，避免复杂 regex 跨行边界问题。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = {}
    current_order = None
    current_section = None  # 'in_packets' | 'in_frames' | 'out_packets' | 'out_frames'
    final_packets = None
    final_frames = None
    collecting_final = False

    for line in lines:
        # Step header
        m = re.match(r'── Step (\d+):', line)
        if m:
            current_order = int(m.group(1))
            data[current_order] = {'in_packets': [], 'in_frames': [],
                                   'out_packets': [], 'out_frames': []}
            current_section = None
            collecting_final = False
            continue

        # Section headers
        m = re.match(r'\s*\[(input|output)\]\s+nb_read_(packets|frames):', line)
        if m and current_order in data:
            direction = 'in' if m.group(1) == 'input' else 'out'
            current_section = f'{direction}_{m.group(2)}'
            continue

        # 最终输出视频
        if '最终输出视频' in line:
            collecting_final = True
            current_section = None
            continue

        # Data lines: segment_XXX.mp4: N  /  upscaled_segment_XXX.mp4: N / interpolated_XXX.mp4: N
        m = re.match(r'\s+\S+segment_\d{3}\.mp4:\s*(\d+)', line)
        if m and current_section and current_order in data:
            data[current_order][current_section].append(m.group(1))

        # 最终视频帧数
        m = re.match(r'\s*nb_read_packets:\s*(\d+)', line)
        if m and collecting_final:
            final_packets = m.group(1)
        m = re.match(r'\s*nb_read_frames\s*:\s*(\d+)', line)
        if m and collecting_final:
            final_frames = m.group(1)

    if final_packets and final_frames:
        data['final'] = {'packets': final_packets, 'frames': final_frames}

    return data


# ═══════════════════════════════════════════════
# B. 从 analyze_test1.txt（终端输出）恢复 ffmpeg_results
# ═══════════════════════════════════════════════

def parse_ffmpeg_from_analyze_log(logpath: str, info: dict) -> dict:
    """
    从 analyze_video_pipeline.py 的终端输出日志 restore ffmpeg_results。
    匹配形如:
        [ffmpeg] Step1 upscaled_segment_000.mp4: frame=9074, dup=None, drop=None, errors=0
        [ffmpeg] Step1 upscaled_segment_001.mp4: frame=8957, dup=None, drop=None, errors=0
    """
    with open(logpath, 'r', encoding='utf-8') as f:
        text = f.read()

    results = {}

    for step in info['steps']:
        order = step['order']
        n_seg = len(step['segments'])
        ff_frames = [None] * n_seg
        ff_errors = [[] for _ in range(n_seg)]
        ff_dd = [None] * n_seg

        # 匹配: [ffmpeg] Step{order} <filename>: frame=N, dup=X, drop=Y, errors=Z
        pattern = rf'\[ffmpeg\] Step{order}\s+\S+:\s*frame=(\d+),\s*dup=(\S+),\s*drop=(\S+),\s*errors=(\d+)'
        for m in re.finditer(pattern, text):
            # 按顺序填充
            seg_idx = len([x for x in ff_frames if x is not None])
            if seg_idx < n_seg:
                ff_frames[seg_idx] = int(m.group(1))
                dup_val = None if m.group(2) in ('None', '') else int(m.group(2))
                drop_val = None if m.group(3) in ('None', '') else int(m.group(3))
                ff_dd[seg_idx] = {'dup': dup_val, 'drop': drop_val}

        results[order] = (ff_frames, ff_errors, ff_dd)

    return results


# ═══════════════════════════════════════════════
# C. 填补缺失的 ffmpeg 解码结果
# ═══════════════════════════════════════════════

def _to_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def fill_missing_ffmpeg(ffmpeg_results: dict, info: dict, ffprobe_data: dict) -> dict:
    """
    对于 ffmpeg_results 中缺失（None）的分段，从 ffprobe output_frames
    推断帧数作为模拟 ffmpeg decode 值。同时验证 pipeline 日志中每段的
    完成状态。

    返回填充后的 ffmpeg_results，并在控制台打印来源报告。
    """
    filled = False
    for step in info['steps']:
        order = step['order']
        if order not in ffmpeg_results:
            ffmpeg_results[order] = ([None] * len(step['segments']),
                                     [[] for _ in step['segments']],
                                     [None] * len(step['segments']))

        ff_frames, ff_errors, ff_dd = ffmpeg_results[order]
        out_fr = ffprobe_data.get(order, {}).get('out_frames', [])

        for i in range(len(step['segments'])):
            if ff_frames[i] is not None:
                continue  # 已有真实解码结果

            # 从 ffprobe output_frames 推断
            simulated = _to_int(out_fr[i]) if i < len(out_fr) else None
            if simulated is not None:
                ff_frames[i] = simulated
                ff_dd[i] = {'dup': 0, 'drop': 0}   # 推断无丢帧
                print(f"   [FILL] Step{order} seg_{i:03d}: ffmpeg 缺失，从 ffprobe 推断 "
                      f"frame={simulated} (dup=0, drop=0, errors=0)")
                filled = True
            else:
                print(f"   [WARN] Step{order} seg_{i:03d}: ffmpeg 缺失且 ffprobe 也无数据，"
                      f"跳过")
        ffmpeg_results[order] = (ff_frames, ff_errors, ff_dd)

    if not filled:
        print("   [INFO] 所有分段 ffmpeg 数据均完整，无需填补")

    return ffmpeg_results


# ═══════════════════════════════════════════════
# D. pipeline 日志完整性校验
# ═══════════════════════════════════════════════

def verify_pipeline_completion(logpath: str, info: dict) -> list:
    """
    从原始 pipeline 日志 output_XXX.txt 中提取每段完成状态。

    匹配规则 (避免跨步骤叠加重叠):
      - IFRNet 插帧阶段:  用 "🎬.*✅ 插帧完成！" + 下行原始帧信息
      - ESRGAN 超分阶段:  用 "🎨.*✅ 处理完成:" (含 🎨 前缀避免与插帧阶段混淆)

    返回 [(order, seg_idx, status, frame_info)] 列表。
    """
    with open(logpath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # 分步逐段统计
    step_texts = {}
    # 按阶段标题切分文本块
    for step in info['steps']:
        order = step['order']
        stype = step['type']
        label = 'Real-ESRGAN' if stype == 'upscale' else 'IFRNet'
        m = re.search(f'阶段 {order}:.*?—.*?{label}', text)
        next_m = None
        for s2 in info['steps']:
            if s2['order'] > order:
                label2 = 'Real-ESRGAN' if s2['type'] == 'upscale' else 'IFRNet'
                next_m = re.search(f'阶段 {s2["order"]}:.*?—.*?{label2}', text)
                break
        start = m.end() if m else 0
        end = next_m.start() if next_m else len(text)
        step_texts[order] = text[start:end]

    results = []
    for step in info['steps']:
        order = step['order']
        stype = step['type']
        block = step_texts.get(order, text)

        if stype == 'interpolate':
            pattern = r'✅ 插帧完成！\s*\n\s*原始帧:\s*(\d+)\s*→\s*输出帧:\s*(\d+)'
            for m in re.finditer(pattern, block):
                src_frames = int(m.group(1))
                out_frames = int(m.group(2))
                seg_idx = len([r for r in results if r[0] == order])
                results.append((order, seg_idx, 'completed',
                                f'{src_frames}>{out_frames}'))
        elif stype == 'upscale':
            pattern = r'✅ 处理完成:\s*输出时长\s+[\d:.]+'
            for m in re.finditer(pattern, block):
                seg_idx = len([r for r in results if r[0] == order])
                results.append((order, seg_idx, 'completed', 'ok'))

    return results


# ═══════════════════════════════════════════════
# E. 主函数
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='补救分析脚本 — 从已有文本数据恢复 xlsx 报告（无需实际视频文件）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('log_file', help='原始 pipeline 日志 (output_XXX.txt)')
    parser.add_argument('--ffprob', default=None,
                        help='ffprob_output_XXX.txt (默认: 与 log_file 同目录，从 log 文件名推导)')
    parser.add_argument('--ffmpeg-log', default=None,
                        help='analyze_test1.txt 终端输出日志，含 [ffmpeg] 行 (可选)')
    parser.add_argument('--output-dir', default=None,
                        help='xlsx 输出目录 (默认: log_file 同目录)')
    parser.add_argument('--suffix', default=None,
                        help='输出文件后缀 (默认: 从 log 文件名推导)')
    parser.add_argument('--skip-ffmpeg', action='store_true',
                        help='完全跳过 ffmpeg 解码结果恢复')
    parser.add_argument('--no-fill', action='store_true',
                        help='不自动填补缺失的 ffmpeg 解码结果（仅使用真实解码数据）')
    args = parser.parse_args()

    # ── 路径推导 ──
    log_path = Path(args.log_file).resolve()
    if not log_path.exists():
        print(f"[ERROR] 日志文件不存在: {log_path}")
        sys.exit(1)

    log_dir = log_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else log_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.suffix
    if suffix is None:
        suffix = re.sub(r'^output', '', log_path.stem) or '_pipeline'

    # ffprob 文件路径
    if args.ffprob:
        ffprob_path = Path(args.ffprob)
    else:
        ffprob_path = log_dir / f"ffprob_output{suffix}.txt"
    if not ffprob_path.exists():
        print(f"[ERROR] ffprobe 统计文件不存在: {ffprob_path}")
        print("  请通过 --ffprob 参数指定，或确保文件名为 ffprob_output{suffix}.txt")
        sys.exit(1)

    # ── Step 1: 解析 pipeline 日志 ──
    print(f"[INFO] 解析 pipeline 日志: {log_path}")
    info = parse_pipeline_log(str(log_path))
    resolve_step_dirs(info, log_dir, None)
    if not info['steps']:
        print("[ERROR] 未能从日志中解析出任何处理步骤")
        sys.exit(1)

    print(f"   处理模式: {info.get('mode', 'N/A')} | 分段数: {info['num_segments']}")
    for step in info['steps']:
        print(f"   Step {step['order']}: {step['name']} [{step['type']}] factor={step['factor']} 段数={len(step['segments'])}")
    print(f"   最终输出: {info.get('output_video', 'N/A')}")

    # ── Step 2: 从 ffprob 文本恢复 ffprobe_data ──
    print(f"\n[FFPROBE] 恢复 ffprobe 数据: {ffprob_path}")
    ffprobe_data = parse_ffprob_text(str(ffprob_path))
    final_frames = ffprobe_data.get('final', {}).get('frames', 'N/A')
    print(f"   恢复 Step 数: {len([k for k in ffprobe_data if isinstance(k, int)])}")
    print(f"   最终视频帧数: {final_frames}")

    # ── Step 3: pipeline 日志完整性校验 ──
    print(f"\n[VERIFY] 校验 pipeline 分段完成状态...")
    completion = verify_pipeline_completion(str(log_path), info)
    expected_total = sum(len(s['segments']) for s in info['steps'])
    print(f"   检测到 {len(completion)}/{expected_total} 段完成标记")
    if len(completion) == expected_total:
        print(f"   [OK] 全部 {expected_total} 段处理成功完成")
    else:
        missing = expected_total - len(completion)
        print(f"   [WARN] {missing} 段完成标记未匹配（可能日志格式不兼容）")

    # ── Step 4: 从 terminal 输出恢复 ffmpeg_results ──
    ffmpeg_results = {}
    ffmpeg_source_info = {}  # 记录每段的来源: 'real' | 'filled' | 'missing'
    if not args.skip_ffmpeg:
        if args.ffmpeg_log:
            alog_path = Path(args.ffmpeg_log)
        else:
            candidates = [
                log_dir / f"analyze{suffix}.txt",
                log_dir / "analyze_test1.txt",
            ]
            alog_path = None
            for c in candidates:
                if c.exists():
                    alog_path = c
                    break

        if alog_path and alog_path.exists():
            print(f"\n[FFMPEG] 从终端输出恢复 ffmpeg 解码结果: {alog_path}")
            ffmpeg_results = parse_ffmpeg_from_analyze_log(str(alog_path), info)
            # 统计真实解码的段数
            real_total = sum(
                1 for v in ffmpeg_results.values()
                for x in v[0] if x is not None
            )
            expected_total_seg = sum(len(s['segments']) for s in info['steps'])
            print(f"   真实解码: {real_total}/{expected_total_seg} 段")

            # ── Step 4b: 填补缺失 ──
            if not args.no_fill and real_total < expected_total_seg:
                print(f"\n[FILL] 自动填补缺失的 ffmpeg 解码结果...")
                ffmpeg_results = fill_missing_ffmpeg(ffmpeg_results, info, ffprobe_data)
                filled_total = sum(
                    1 for v in ffmpeg_results.values()
                    for x in v[0] if x is not None
                )
                print(f"   填补后: {filled_total}/{expected_total_seg} 段")
                if filled_total < expected_total_seg:
                    print(f"   [WARN] 仍有 {expected_total_seg - filled_total} 段无法填补")

            # 记录来源信息用于摘要
            for step in info['steps']:
                order = step['order']
                if order not in ffmpeg_results:
                    continue
                ff_frames, _, _ = ffmpeg_results[order]
                for i in range(len(step['segments'])):
                    if i not in ffmpeg_source_info:
                        ffmpeg_source_info[f'{order}_{i}'] = \
                            'real' if ff_frames[i] is not None else 'missing'
            # 标记哪些是填补的 —— 需要从终端日志的实际行数反推
            # 一个近似方法: 如果 ffmpeg_results 中某段的 ff_frames[i]
            # 来自 ffprobe 推断 = filled; 如果来自终端日志 = real
            # 但因为 fill_missing_ffmpeg 不区分，我们从头构建
            if alog_path:
                real_ffmpeg = parse_ffmpeg_from_analyze_log(str(alog_path), info)
                for step in info['steps']:
                    order = step['order']
                    if order not in real_ffmpeg:
                        continue
                    real_frames, _, _ = real_ffmpeg[order]
                    for i in range(len(step['segments'])):
                        key = f'{order}_{i}'
                        if i < len(real_frames) and real_frames[i] is not None:
                            ffmpeg_source_info[key] = 'real'
                        elif ffmpeg_results.get(order, ([], [], []))[0][i] is not None:
                            ffmpeg_source_info[key] = 'filled'
                        else:
                            ffmpeg_source_info[key] = 'missing'
        else:
            print(f"\n[WARN] 未找到 ffmpeg 解码日志")
            print("   可通过 --ffmpeg-log 指定 analyze_test1.txt 路径")
            if not args.no_fill:
                print("[FILL] 无任何真实解码数据，全部从 ffprobe 推断...")
                ffmpeg_results = {}
                for step in info['steps']:
                    order = step['order']
                    n_seg = len(step['segments'])
                    out_fr = ffprobe_data.get(order, {}).get('out_frames', [])
                    ff_frames = []
                    ff_dd = []
                    for i in range(n_seg):
                        sim = _to_int(out_fr[i]) if i < len(out_fr) else None
                        ff_frames.append(sim)
                        ff_dd.append({'dup': 0, 'drop': 0} if sim is not None else None)
                        key = f'{order}_{i}'
                        ffmpeg_source_info[key] = 'filled' if sim is not None else 'missing'
                    ffmpeg_results[order] = (ff_frames, [[] for _ in range(n_seg)], ff_dd)
                print(f"   全部从 ffprobe 推断: {sum(1 for v in ffmpeg_results.values() for x in v[0] if x is not None)}/{sum(len(s['segments']) for s in info['steps'])} 段")
    else:
        for step in info['steps']:
            order = step['order']
            for i in range(len(step['segments'])):
                ffmpeg_source_info[f'{order}_{i}'] = 'skipped'

    # ── Step 5: 构建帧数链 ──
    print(f"\n[CHAIN] 构建帧数链...")
    chain_rows = build_chain(info, ffprobe_data, ffmpeg_results)
    print(f"   构建 {len(chain_rows)} 段帧数链")

    # ── Step 6: 生成 xlsx ──
    xlsx_path = output_dir / f"帧数差值统计汇总_output{suffix}.xlsx"
    print(f"\n[XLSX] 生成 xlsx: {xlsx_path}")
    try:
        generate_xlsx(info, ffprobe_data, ffmpeg_results, chain_rows, str(xlsx_path))
    except ImportError:
        print("[ERROR] openpyxl 未安装。请执行: pip install openpyxl")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] xlsx 生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Step 7: 简易摘要 ──
    print(f"\n{'=' * 60}")
    print(f"  [OK] 补救分析完成！")
    print(f"     xlsx: {xlsx_path}")
    print(f"")
    print(f"  ffmpeg 数据来源:")
    real_n = sum(1 for v in ffmpeg_source_info.values() if v == 'real')
    filled_n = sum(1 for v in ffmpeg_source_info.values() if v == 'filled')
    missing_n = sum(1 for v in ffmpeg_source_info.values() if v == 'missing')
    skipped_n = sum(1 for v in ffmpeg_source_info.values() if v == 'skipped')
    if real_n > 0:
        print(f"    - 真实解码 (from analyze_test1.txt): {real_n} 段")
    if filled_n > 0:
        print(f"    - 数据推断 (from ffprobe):           {filled_n} 段")
    if missing_n > 0:
        print(f"    - 缺失未填补:                          {missing_n} 段")
    if skipped_n > 0:
        print(f"    - 跳过:                               {skipped_n} 段")
    print(f"")

    # 最终输出校验
    final_probe = ffprobe_data.get('final', {})
    if final_probe.get('frames'):
        chain_sum = sum(
            r['chain'][-1]['actual'] or 0
            for r in chain_rows if r['chain']
        )
        actual_final = _to_int(final_probe['frames']) or 0
        diff = actual_final - chain_sum
        if diff == 0:
            print(f"  [OK] 最终输出校验: {actual_final} 帧 = {chain_sum} 帧 (分段总和)")
        else:
            print(f"  [WARN] 最终输出校验: ffprobe={actual_final}, 分段总和={chain_sum}, 差值={diff}")
            # 回退: 直接对比最后一个步骤的 expected 总和
            if info['steps']:
                last_step = info['steps'][-1]
                exp_sum = sum(
                    r['chain'][-1]['expected'] or 0
                    for r in chain_rows if r['chain']
                )
                diff2 = actual_final - exp_sum
                print(f"         期望总和={exp_sum}, vs ffprobe 差值={diff2}")
    print(f"")
    # 打印每段帧数链摘要 (带来源标记)
    print(f"  帧数链摘要 (FFMPEG来源: R=真实, F=推断, -=缺失):")
    for r in chain_rows:
        seg_idx = r['index']
        chain_info = []
        for e in r['chain']:
            step_order = e['step_order']
            key = f'{step_order}_{seg_idx}'
            source_mark = ffmpeg_source_info.get(key, '?')
            src = {'real': 'R', 'filled': 'F', 'missing': '-', 'skipped': 'S', '?': '?'}.get(source_mark, '?')
            t = 'UP' if e['step_type'] == 'upscale' else 'IF'
            chain_info.append(
                f"Step{e['step_order']}:{e['in_frames']}>{e['actual'] or '?'}"
                f"({t})[{src}]"
            )
        print(f"    seg_{seg_idx:03d}:  {' | '.join(chain_info)}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
