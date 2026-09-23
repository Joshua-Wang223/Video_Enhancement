"""
视频处理工具函数
提供视频信息获取、分段、合并等通用功能
"""

import os
import subprocess
import cv2
import json
import re
import threading
from pathlib import Path
from typing import Callable, Optional, List, Union, Tuple, Dict, Sequence, Any
import shutil
import logging
import tempfile
import warnings

# 配置日志（或在函数内直接使用 print，取决于你的项目规范）
logger = logging.getLogger(__name__)

# [QUALITY-UNIFY] 质量参数换算的唯一真源（与 quality_map.py 同目录）。
# 让合并/归一化等重编码路径也能把统一基准换算到实际编码器量纲
# （-crf / -cq:v + -b:v 0 / -qp），避免硬编拿到 -crf 被静默丢弃。
# 纯工具环境（未把 src/utils 入 sys.path）降级为 None，调用处回退旧行为。
try:
    from quality_map import resolve_quality as _resolve_quality
except Exception:  # noqa: BLE001
    _resolve_quality = None

# 不接受 x264 风格 -preset 的编码器（librav1e 用 -speed，libvpx 用 -deadline/-cpu-used）。
# 硬编（NVENC/QSV/AMF…）接受 -preset（medium/p1..p7），故不在排除集内。
_NO_PRESET_CODECS = {'librav1e', 'libvpx', 'libvpx-vp9'}


def _preset_supported(codec: str) -> bool:
    return str(codec).lower() not in _NO_PRESET_CODECS


def _resolve_quality_args(codec: str, *, crf=None, cq=None,
                          crf_ref=None, cq_ref=None,
                          preset=None) -> Tuple[List[str], str]:
    """[QUALITY-UNIFY] 统一构造"编码质量参数"片段（所有重编码路径共用）。

      软编 → ``-crf N`` / VP9 → ``-crf N -b:v 0`` / 硬编 → ``-cq:v N -b:v 0`` /
      librav1e → ``-qp N``；四键均未给时按 libx264 CRF 基准 21 换算。
      ``preset`` 仅在编码器支持 -preset 时发射（见 ``_preset_supported``）。

    Returns:
        (args, note) —— args 含 ``-preset``（如适用）+ 质量参数 + 配套参数。
    """
    if _resolve_quality is not None:
        _p, _val, _extra, _note = _resolve_quality(
            codec, crf=crf, cq=cq, crf_ref=crf_ref, cq_ref=cq_ref)
    else:  # quality_map 不可用（纯工具环境）：回退旧行为，但仍保持基准 21
        _p, _val, _extra = '-crf', int(crf if crf is not None else 18), []
        _note = 'quality_map 不可用，回退 -crf'
    args: List[str] = []
    if preset and _preset_supported(codec):
        args += ['-preset', str(preset)]
    args += [_p, str(_val)] + list(_extra)
    return args, _note

# 移到模块顶部作为常量，避免重复定义
AUDIO_EXT_MAP = {
    'aac': 'm4a',
    'mp3': 'mp3',
    'flac': 'flac',
    'alac': 'm4a',
    'opus': 'opus',
    'vorbis': 'ogg',
    'pcm_s16le': 'wav',    # 无损 PCM
    'pcm_s24le': 'wav',    # 无损 PCM
    'ac3': 'ac3',          # 杜比数字
    'eac3': 'eac3',        # 增强杜比数字
    'dts': 'dts',          # DTS 音轨
    'truehd': 'thd',       # Dolby TrueHD
    'mlp': 'mlp',          # MLP 无损
}

class FFmpegError(Exception):
    """FFmpeg 执行相关异常"""
    pass


# ══════════════════════════════════════════════════════════════════════
#  [META-KEEP] 原输入视频元数据探测与保留
#
#  一次 ffprobe 拿全量（-show_streams 默认即含 side_data_list：旋转 display matrix；
#  HDR10 静态元数据在 6.1 只在帧级暴露，故必要时补一次单帧探测），
#  按 abspath|size|mtime 缓存，避免同一文件重复探测。
# ══════════════════════════════════════════════════════════════════════

_PROBE_CACHE: Dict[str, Dict[str, Any]] = {}
_PROBE_CACHE_LOCK = threading.Lock()

# 10bit 源在各编码器下的目标像素格式（软件编码器 yuv420p10le，NVENC p010le）
_PIXFMT_10BIT_BY_ENCODER = {
    'libx264': 'yuv420p10le',
    'libx265': 'yuv420p10le',
    'libsvtav1': 'yuv420p10le',
    'libaom-av1': 'yuv420p10le',
    'librav1e': 'yuv420p10le',
    'libvpx-vp9': 'yuv420p10le',
    'h264_nvenc': 'p010le',
    'hevc_nvenc': 'p010le',
    'av1_nvenc': 'p010le',
    'prores': 'yuv422p10le',
    'prores_ks': 'yuv422p10le',
}
_ENCODERS_8BIT_ONLY = {'mpeg4', 'libvpx', 'mjpeg', 'vp8', 'h264_v4l2m2m'}

_BITMAP_SUBS = {'dvd_subtitle', 'dvb_subtitle', 'dvb_teletext',
                'hdmv_pgs_subtitle', 'xsub'}
_MP4_FAMILY = {'mp4', 'm4v', 'mov'}

# ffmpeg 输出端 -color_trc 与 setparams 滤镜接受的取值集合不一致（6.1 实测）：
#   -color_trc           只认 libavutil 规范名 gamma22/gamma28（BT.470M/BT.470BG）
#   setparams=color_trc  只认别名 bt470m/bt470bg，传 gamma28 直接报错
# 因此输出端用规范名、滤镜端用别名，两者语义等价（-color_trc gamma28 写出 bt470bg）。
_TRC_OUTPUT_NAMES = {'bt470bg': 'gamma28', 'bt470m': 'gamma22'}
_TRC_FILTER_NAMES = {v: k for k, v in _TRC_OUTPUT_NAMES.items()}

_FFMPEG_MAJOR: Optional[int] = None


# 注意：本文件后面已有一个 _probe_cache_key(path: Path)（供 get_video_codec 用），
# 这里必须用不同名字，否则会被后来的定义覆盖。
def _mk_probe_cache_key(path: str) -> str:
    """缓存键：绝对路径 + size + mtime（与 ffprobe 版本无关，进程内安全）。"""
    ap = os.path.abspath(path)
    try:
        st = os.stat(ap)
        return f'{ap}|{st.st_size}|{int(st.st_mtime)}'
    except OSError:
        return ap


def _frac_to_float(value: Any) -> Optional[float]:
    """ffprobe 有理数：'34000/50000' / [34000, 50000] / 0.68 → float。"""
    try:
        if isinstance(value, (list, tuple)):
            num, den = float(value[0]), float(value[1])
        else:
            s = str(value)
            if '/' in s:
                a, _, b = s.partition('/')
                num, den = float(a), float(b)
            else:
                num, den = float(s), 1.0
        return num / den if den else None
    except (TypeError, ValueError):
        return None


def _parse_rate(rate: Any) -> Optional[float]:
    """'25/1' / '30000/1001' → float；无法解析返回 None。"""
    try:
        if isinstance(rate, (list, tuple)):
            num, den = float(rate[0]), float(rate[1])
        else:
            num_s, _, den_s = str(rate).partition('/')
            num, den = float(num_s), float(den_s) if den_s else 1.0
        return num / den if den > 0 else None
    except (TypeError, ValueError):
        return None


def _parse_bits(video_stream: Dict[str, Any]) -> int:
    """位深：bits_per_raw_sample 优先（实测可能是 'N/A'），回退从 pix_fmt 名解析。"""
    try:
        n = int(str(video_stream.get('bits_per_raw_sample')).strip())
        if n in (8, 10, 12, 14, 16):
            return n
    except (TypeError, ValueError):
        pass
    pf = (video_stream.get('pix_fmt') or '').lower()
    for token, bits in (('p016le', 16), ('p014le', 14), ('p012le', 12),
                        ('p010le', 10), ('p16le', 16), ('p12le', 12),
                        ('p10le', 10)):
        if token in pf:
            return bits
    return 8


def _norm_rotation(deg: Any) -> int:
    try:
        return int(round(float(deg))) % 360
    except (TypeError, ValueError):
        return 0


def _extract_rotation(stream: Dict[str, Any]) -> int:
    """优先 side_data_list 的 Display Matrix，兜底容器里的 rotate tag。"""
    for sd in (stream.get('side_data_list') or []):
        if sd.get('rotation') is not None:
            return _norm_rotation(sd['rotation'])
    tags = stream.get('tags') or {}
    for key in ('rotate', 'rotation'):
        if key in tags:
            return _norm_rotation(tags[key])
    return 0


def _extract_hdr_from_side_data(sd_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    解析 HDR10 静态元数据。

    ffprobe 6.1 实测：Mastering display metadata / Content light level metadata
    只在**帧级** side_data 暴露（-show_frames），字段为扁平的 red_x/green_x/.../
    white_point_x/max_luminance；老版本或某些容器则给出 display_primaries 数组。
    两种形态都兼容，解析不出就返回空，调用方优雅降级。
    """
    out: Dict[str, Any] = {'master_display': None, 'max_cll': None}
    for sd in sd_list or []:
        stype = (sd.get('side_data_type') or '').lower()
        if 'mastering display' in stype:
            # 老版本/部分容器给出 display_primaries 数组，先摊平成 red_x/... 形态
            if 'red_x' not in sd and (sd.get('display_primaries')
                                      or sd.get('display_primaries_rgb')
                                      or sd.get('white_point')):
                prim = sd.get('display_primaries') or sd.get('display_primaries_rgb') or []
                wpt = sd.get('white_point') or []
                flat: Dict[str, Any] = {}
                for name, item in (('red', prim[0] if len(prim) > 0 else None),
                                   ('green', prim[1] if len(prim) > 1 else None),
                                   ('blue', prim[2] if len(prim) > 2 else None),
                                   ('white_point', wpt or None)):
                    if item is None:
                        continue
                    vals = item if isinstance(item, (list, tuple)) \
                        else (item.get('x'), item.get('y'))
                    try:
                        flat[name + '_x'] = vals[0]
                        flat[name + '_y'] = vals[1]
                    except (TypeError, IndexError, AttributeError):
                        pass
                sd = {**sd, **flat}

            def _chroma(key: str) -> Optional[int]:
                """色度坐标 → x265 单位（0.00002）。"""
                v = _frac_to_float(sd.get(key))
                return None if v is None else int(round(v * 50000))

            def _luma(key: str) -> Optional[int]:
                """亮度 → x265 单位（0.0001 cd/m²）。"""
                v = _frac_to_float(sd.get(key))
                return None if v is None else int(round(v * 10000))

            r = (_chroma('red_x'), _chroma('red_y'))
            g = (_chroma('green_x'), _chroma('green_y'))
            b = (_chroma('blue_x'), _chroma('blue_y'))
            wp = (_chroma('white_point_x'), _chroma('white_point_y'))
            mx, mn = _luma('max_luminance'), _luma('min_luminance')
            if None not in (*r, *g, *b, *wp) and mx is not None and mn is not None:
                # x265 语法顺序为 G()B()R()
                out['master_display'] = (
                    f'G({g[0]},{g[1]})B({b[0]},{b[1]})R({r[0]},{r[1]})'
                    f'WP({wp[0]},{wp[1]})L({mx},{mn})'
                )
        elif 'content light level' in stype:
            max_c = sd.get('max_content')
            avg = sd.get('max_average', sd.get('max_pic_average'))
            if max_c is not None and avg is not None:
                try:
                    out['max_cll'] = f'{int(max_c)},{int(avg)}'
                except (TypeError, ValueError):
                    pass
    return out


def _extract_hdr(stream: Dict[str, Any]) -> Dict[str, Any]:
    return _extract_hdr_from_side_data(stream.get('side_data_list') or [])


def _looks_hdr(video_stream: Dict[str, Any], src_bits: int) -> bool:
    """判断是否需要为 HDR 静态元数据额外做一次帧级探测。"""
    if src_bits < 10:
        return False
    trc = (video_stream.get('color_transfer') or '').lower()
    prim = (video_stream.get('color_primaries') or '').lower()
    return trc in ('smpte2084', 'arib-std-b67', 'smpte2084-hdr10') or prim == 'bt2020'


def _probe_frame_side_data(path: str) -> List[Dict[str, Any]]:
    """
    读首帧 side_data（HDR10 静态元数据在 ffmpeg 6.1 只在帧级暴露）。
    只读 1 帧，开销可忽略；失败返回空列表。
    """
    cmd = ['ffprobe', '-v', 'error', '-print_format', 'json',
           '-select_streams', 'v:0', '-show_frames',
           '-read_intervals', '%+#1', path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=20,
                           check=False)
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)
        for frame in data.get('frames') or []:
            sd = frame.get('side_data_list')
            if sd:
                return sd
    except Exception:
        pass
    return []


def probe_full_metadata(video_file: Union[str, Path],
                        errors: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    一次 ffprobe 拿到 format.tags / 各流 tags+disposition / side_data / pix_fmt /
    bits_per_raw_sample / SAR / 帧率 / 章节，并按 abspath|size|mtime 缓存。

    Returns:
        {'format':..., 'video':<原始视频流dict>, 'streams':..., 'chapters':...,
         'derived':{rotation,width,height,effective_width,effective_height,pix_fmt,
                    src_bits,video_index,cover_indices,subtitle_codecs,
                    is_hdr,master_display,max_cll}}
        探测失败返回 None。
    """
    path = str(video_file)
    if not os.path.isfile(path):
        return None

    key = _mk_probe_cache_key(path)
    with _PROBE_CACHE_LOCK:
        cached = _PROBE_CACHE.get(key)
    if cached is not None:
        return cached

    cmd = ['ffprobe', '-v', 'error', '-print_format', 'json',
           '-show_format', '-show_streams', '-show_chapters', path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=15,
                           check=True)
        data = json.loads(r.stdout)
    except Exception as exc:
        if errors is not None:
            errors.append(str(exc))
        return None

    streams = data.get('streams') or []
    video = next((s for s in streams
                  if s.get('codec_type') == 'video'
                  and not (s.get('disposition') or {}).get('attached_pic')), None)
    if video is None:
        video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    if video is None:
        if errors is not None:
            errors.append('no video stream')
        return None

    rotation = _extract_rotation(video)
    width = int(video.get('width') or 0)
    height = int(video.get('height') or 0)
    derived: Dict[str, Any] = {
        'rotation': rotation,
        'width': width,
        'height': height,
        # 显示尺寸：90/270 度旋转时宽高互换
        'effective_width': height if rotation in (90, 270) else width,
        'effective_height': width if rotation in (90, 270) else height,
        'pix_fmt': video.get('pix_fmt') or '',
        'src_bits': _parse_bits(video),
        'video_index': int(video.get('index') or 0),
        'cover_indices': [int(s['index']) for s in streams
                          if (s.get('disposition') or {}).get('attached_pic')],
        'subtitle_codecs': [s.get('codec_name') for s in streams
                            if s.get('codec_type') == 'subtitle'],
    }
    hdr = _extract_hdr(video)
    if not (hdr['master_display'] or hdr['max_cll']) \
            and _looks_hdr(video, derived['src_bits']):
        hdr = _extract_hdr_from_side_data(_probe_frame_side_data(path))
    derived.update(hdr)
    derived['is_hdr'] = bool(hdr['master_display'] or hdr['max_cll'])

    meta: Dict[str, Any] = {
        'path': path,
        'format': {
            'format_name': (data.get('format') or {}).get('format_name') or '',
            'duration': (data.get('format') or {}).get('duration') or '',
            'bit_rate': (data.get('format') or {}).get('bit_rate') or '',
            'tags': dict((data.get('format') or {}).get('tags') or {}),
        },
        'video': video,
        'streams': streams,
        'chapters': data.get('chapters') or [],
        'derived': derived,
    }
    with _PROBE_CACHE_LOCK:
        _PROBE_CACHE[_mk_probe_cache_key(path)] = meta
    return meta


def _ffmpeg_major(ffmpeg_bin: str = 'ffmpeg') -> int:
    """ffmpeg 主版本号（用于选择旋转写法）；探测失败按 6 处理。"""
    global _FFMPEG_MAJOR
    if _FFMPEG_MAJOR is not None:
        return _FFMPEG_MAJOR
    try:
        out = subprocess.run([ffmpeg_bin, '-version'], capture_output=True,
                             text=True, timeout=10).stdout
        m = re.search(r'ffmpeg version (\d+)\.', out)
        _FFMPEG_MAJOR = int(m.group(1)) if m else 6
    except Exception:
        _FFMPEG_MAJOR = 6
    return _FFMPEG_MAJOR


def resolve_pix_fmt_for_source(codec: str, src_bits: int,
                               warn: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """源为 10bit+ 时选择保持位深的目标 pix_fmt；8bit 源返回 None（沿用默认）。"""
    if src_bits < 10:
        return None
    c = (codec or '').lower()
    if c in _ENCODERS_8BIT_ONLY:
        if warn:
            warn(f'源为 {src_bits}bit，编码器 {c} 不支持 10bit，已降级为 yuv420p'
                 f'（高光可能出现色带）')
        return 'yuv420p'
    return _PIXFMT_10BIT_BY_ENCODER.get(c, 'yuv420p10le')


def resolve_subtitle_codec(src_subs: List[Optional[str]], container: str,
                           warn: Optional[Callable[[str], None]] = None):
    """
    按「源字幕 codec × 目标容器」决定字幕编码方式。

    mp4 里 -c:s copy 对 subrip/ass 会硬失败（Could not find tag for codec），
    位图字幕则根本无法转换，必须丢弃并告警。
    """
    ctr = (container or '').lower().lstrip('.')
    subs = [s for s in (src_subs or []) if s]
    if not subs:
        return None, False
    if ctr in ('mkv', 'webm'):
        if ctr == 'mkv':
            return 'copy', True
        return ('copy', True) if all(s == 'webvtt' for s in subs) else (None, False)
    if ctr in _MP4_FAMILY:
        if any(s in _BITMAP_SUBS for s in subs):
            if warn:
                warn('位图字幕（PGS/DVDSUB 等）无法写入 mp4，已丢弃')
            return None, False
        if warn and any(s in ('ass', 'ssa') for s in subs):
            warn('ASS/SSA 转为 mov_text 后会丢失样式')
        return 'mov_text', True
    return None, False


def build_hdr_args(meta: Dict[str, Any], codec: str,
                   warn: Optional[Callable[[str], None]] = None) -> List[str]:
    """
    HDR10 静态元数据写入。色彩三参数由 build_color_args 从源透传，此处不重复指定。

    - libx265：显式 -x265-params，可靠。
    - NVENC：依赖帧 side_data 自动传播；走 hwdownload/CPU 回退链路时会丢失，故告警。
    - 其余编码器：只能保住色彩三参数与位深。
    """
    d = meta['derived']
    if not d['is_hdr']:
        return []
    c = (codec or '').lower()
    out: List[str] = []
    if c == 'libx265' and d['master_display']:
        params = ['master-display=' + d['master_display']]
        if d['max_cll']:
            params.append('max-cll=' + d['max_cll'])
        params.append('hdr10=1')
        out += ['-x265-params', ':'.join(params)]
    elif c.endswith('_nvenc'):
        # 实测（ffmpeg 6.1 + Tesla T4）：NVENC 不写入 mastering display / MaxCLL，
        # hevc_metadata bsf 也无此能力，属编码器封装限制。
        if warn and d['master_display']:
            warn('NVENC 不写入 mastering display / MaxCLL，HDR10 静态元数据会丢失'
                 '（色彩三参数与 10bit 位深仍保留）；如需完整 HDR10 元数据请用 libx265')
    elif warn and d['master_display']:
        warn(f'编码器 {c} 无法写入 mastering display / MaxCLL，仅保留色彩三参数与位深')
    return out


def build_aspect_args(meta: Dict[str, Any]) -> List[str]:
    """
    仅当源为变形（SAR≠1:1）时显式 -aspect 保持源 DAR。
    方像素源不加：crop 后 ffmpeg 保持 SAR 自动算出正确的新 DAR。
    """
    sar = meta['video'].get('sample_aspect_ratio') or '1:1'
    try:
        n_s, _, d_s = str(sar).partition(':')
        n, d = int(n_s), int(d_s) if d_s else 1
    except (TypeError, ValueError):
        return []
    if n <= 0 or d <= 0 or (n == 1 and d == 1):
        return []
    src_w, src_h = meta['derived']['width'], meta['derived']['height']
    if not src_w or not src_h:
        return []
    from math import gcd
    dan, dad = n * src_w, d * src_h
    g = gcd(dan, dad) or 1
    return ['-aspect', f'{dan // g}/{dad // g}']


def write_source_meta_sidecar(input_video: Union[str, Path],
                              output_dir: Union[str, Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """
    [META-KEEP] 切片阶段把原片元数据落盘，供后续 concat 合并阶段注入。

    concat demuxer 的合成上下文不继承任何分段的容器级元数据，合并时
    `-map_metadata 0` 指向的是列表文件（拿不到东西），所以必须在切片这一步
    就把原片元数据固化下来。

    产出两个文件：
      source_meta.ffmetadata  ffmpeg -f ffmetadata（只读 header，零重编码）
      source_meta.json        合并阶段要用的几个关键值（位深/旋转/creation_time 等）

    Returns:
        (ffmetadata 路径, json 路径)；失败时对应项为 None。
    """
    out_dir = Path(output_dir)
    ffmeta_path = out_dir / 'source_meta.ffmetadata'
    json_path = out_dir / 'source_meta.json'

    meta = probe_full_metadata(input_video)
    if meta is None:
        return None, None

    try:
        subprocess.run(
            ['ffmpeg', '-v', 'error', '-y', '-i', str(input_video),
             '-f', 'ffmetadata', str(ffmeta_path)],
            capture_output=True, timeout=60, check=False,
        )
    except Exception as exc:
        logger.warning(f'[META-KEEP] ffmetadata 落盘失败（不影响主流程）: {exc}')
        return None, None

    d = meta['derived']
    try:
        payload = {
            'source': os.path.abspath(str(input_video)),
            'format_tags': meta['format'].get('tags') or {},
            'creation_time': (meta['format'].get('tags') or {}).get('creation_time'),
            'rotation': d['rotation'],
            'src_bits': d['src_bits'],
            'pix_fmt': d['pix_fmt'],
            'width': d['width'],
            'height': d['height'],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding='utf-8')
    except Exception as exc:
        logger.warning(f'[META-KEEP] 元数据 JSON 落盘失败（不影响主流程）: {exc}')
        return ffmeta_path, None

    return ffmeta_path, json_path


def load_source_meta_sidecar(path: Union[str, Path]) -> Dict[str, Any]:
    """读取 write_source_meta_sidecar 落盘的 JSON；缺失/损坏返回空字典。"""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8')) or {}
    except Exception:
        return {}


class VideoInfo:
    """视频信息类"""
    
    def __init__(self, video_path: str):
        self.path = video_path
        self.duration = None
        self.fps = None
        self.width = None
        self.height = None
        self.frame_count = None
        self.codec = None
        self.bitrate = None
        self.has_audio = False
        self.audio_codec = None
        # [META-KEEP] 元数据保留所需
        self.pix_fmt = None
        self.bits_per_raw_sample = None
        self.color_space = None
        self.color_range = None
        self.rotation = 0
        self.meta = None            # probe_full_metadata 全量结果

        self._load_info()

    def _load_info(self):
        """加载视频信息（走 probe_full_metadata，带缓存）"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"视频文件不存在: {self.path}")

        try:
            m = probe_full_metadata(self.path)
            if m is None:
                raise RuntimeError("ffprobe 未返回视频流")

            self.meta = m
            fmt, video, d = m['format'], m['video'], m['derived']

            self.duration = float(fmt.get('duration') or 0) or 0.0
            self.bitrate = int(fmt.get('bit_rate') or 0)

            # [P0-FIX-FPS-EVAL] 原实现 eval(r_frame_rate) 属动态执行反模式：
            # 'N/A'/'0/0' 等奇异值会抛 NameError/ZeroDivisionError，且均不在
            # 外层 except 捕获面内 → 直接穿透 VideoInfo.__init__。改用 Fraction。
            _rate_str = str(video.get('r_frame_rate', '0/1'))
            try:
                from fractions import Fraction as _Fraction
                _fps_f = float(_Fraction(_rate_str))
                self.fps = _fps_f if _fps_f > 0 else None
            except (ValueError, ZeroDivisionError):
                self.fps = None

            self.width = d['width']
            self.height = d['height']
            self.codec = video.get('codec_name', 'unknown')
            try:
                self.frame_count = int(video.get('nb_frames') or 0)
            except (TypeError, ValueError):
                self.frame_count = 0

            self.has_audio = any(s.get('codec_type') == 'audio' for s in m['streams'])
            self.audio_codec = next(
                (s.get('codec_name', 'unknown') for s in m['streams']
                 if s.get('codec_type') == 'audio'), None)

            # [META-KEEP] 位深 / 色彩 / 旋转
            self.pix_fmt = d['pix_fmt']
            self.bits_per_raw_sample = d['src_bits']
            self.color_space = video.get('color_space')
            self.color_range = video.get('color_range')
            self.rotation = d['rotation']

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  使用ffprobe获取信息失败，尝试OpenCV: {e}")
            self._load_info_opencv()
    
    def _load_info_opencv(self):
        """使用OpenCV获取基本信息（备用方法）"""
        try:
            cap = cv2.VideoCapture(self.path)
            
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.fps > 0 and self.frame_count > 0:
                self.duration = self.frame_count / self.fps
            
            cap.release()
        except Exception as e:
            print(f"❌ OpenCV获取信息失败: {e}")
    
    def __repr__(self):
        return (f"VideoInfo(path={self.path}, duration={self.duration:.2f}s, "
                f"fps={self.fps:.2f}, resolution={self.width}x{self.height}, "
                f"frames={self.frame_count}, has_audio={self.has_audio})")

# 音频处理函数
def get_audio_codec(
    file_path: Union[str, Path],
    stream_index: int = 0,
    detailed: bool = False
) -> Union[str, Dict[str, str]]:
    """
    检测音视频文件中的音频编码格式
    
    Args:
        file_path: 音视频文件路径
        stream_index: 音频流索引（默认第1个音频流，通常为0）
        detailed: 是否返回详细信息
        
    Returns:
        如果 detailed=False: 返回编码格式字符串（如 'aac', 'mp3', 'flac'）
        如果 detailed=True: 返回包含详细信息的字典
        
    Raises:
        FileNotFoundError: 文件不存在
        subprocess.CalledProcessError: ffprobe执行失败
        ValueError: 文件不包含音频流或指定流索引无效
    """
    
    # 转换为Path对象并检查文件存在
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 构建ffprobe命令
    cmd = [
        'ffprobe',
        '-v', 'quiet',          # 安静模式，减少输出
        '-print_format', 'json', # JSON格式输出
        '-show_streams',        # 显示流信息
        '-select_streams', 'a', # 只选择音频流
        str(file_path)
    ]
    
    try:
        # 执行ffprobe命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30  # 30秒超时
        )
        
        # 解析JSON输出
        data = json.loads(result.stdout)
        
        if 'streams' not in data or not data['streams']:
            raise ValueError(f"文件不包含音频流: {file_path}")
        
        # 获取指定的音频流
        if stream_index >= len(data['streams']):
            raise ValueError(
                f"音频流索引 {stream_index} 无效，文件只有 {len(data['streams'])} 个音频流"
            )
        
        stream = data['streams'][stream_index]
        
        if detailed:
            # 返回详细信息
            return {
                'codec_name': stream.get('codec_name', 'unknown'),
                'codec_long_name': stream.get('codec_long_name', 'unknown'),
                'codec_type': stream.get('codec_type', 'unknown'),
                'sample_rate': stream.get('sample_rate', 'unknown'),
                'channels': stream.get('channels', 'unknown'),
                'channel_layout': stream.get('channel_layout', 'unknown'),
                'bit_rate': stream.get('bit_rate', 'unknown'),
                'duration': stream.get('duration', 'unknown'),
                'index': stream.get('index', stream_index),
                'tags': stream.get('tags', {}),
                'profile': stream.get('profile', 'unknown')
            }
        else:
            # 只返回编码名称
            return stream.get('codec_name', 'unknown')
            
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"检测音频编码超时: {file_path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"解析ffprobe输出失败: {e}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe执行失败: {e.stderr}")


def get_all_audio_streams(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    获取文件中所有音频流的详细信息
    
    Args:
        file_path: 音视频文件路径
        
    Returns:
        包含所有音频流信息的列表
    """
    file_path = Path(file_path)
    
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-select_streams', 'a',
        str(file_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    streams_info = []
    for i, stream in enumerate(data.get('streams', [])):
        streams_info.append({
            'index': stream.get('index', i),
            'codec_name': stream.get('codec_name', 'unknown'),
            'codec_long_name': stream.get('codec_long_name', 'unknown'),
            'sample_rate': stream.get('sample_rate', 'unknown'),
            'channels': stream.get('channels', 'unknown'),
            'channel_layout': stream.get('channel_layout', 'unknown'),
            'bit_rate': stream.get('bit_rate', 'unknown'),
            'duration': stream.get('duration', 'unknown'),
            'language': stream.get('tags', {}).get('language', 'unknown'),
            'title': stream.get('tags', {}).get('title', ''),
            'profile': stream.get('profile', 'unknown')
        })
    
    return streams_info


def is_lossless_audio(codec_name: str) -> bool:
    """
    判断音频编码是否为无损格式
    
    Args:
        codec_name: 音频编码名称
        
    Returns:
        True如果是无损格式，否则False
    """
    lossless_codecs = {
        'flac',          # Free Lossless Audio Codec
        'alac',          # Apple Lossless Audio Codec
        'pcm_s16le',     # 16-bit PCM
        'pcm_s24le',     # 24-bit PCM
        'pcm_s32le',     # 32-bit PCM
        'pcm_f32le',     # 32-bit float PCM
        'pcm_f64le',     # 64-bit float PCM
        'pcm_s16be',     # 16-bit PCM big-endian
        'pcm_s24be',     # 24-bit PCM big-endian
        'pcm_s32be',     # 32-bit PCM big-endian
        'pcm_f32be',     # 32-bit float PCM big-endian
        'pcm_f64be',     # 64-bit float PCM big-endian
        'pcm_u8',        # 8-bit unsigned PCM
        'pcm_alaw',      # A-law PCM
        'pcm_mulaw',     # μ-law PCM
        'wavpack',       # WavPack
        'tta',           # True Audio
        'mlp',           # Meridian Lossless Packing
        'dts',           # DTS (有些变种是无损的)
        'truehd',        # Dolby TrueHD
    }
    
    # 检查是否以'pcm_'开头（所有PCM格式都是无损的）
    if codec_name.startswith('pcm_'):
        return True
    
    return codec_name.lower() in lossless_codecs


def get_audio_codec_simple(file_path: Union[str, Path]) -> Optional[str]:
    """
    简化的音频编码检测（仅返回编码名称，忽略错误）
    
    Args:
        file_path: 音视频文件路径
        
    Returns:
        音频编码名称，如果检测失败则返回None
    """
    try:
        return get_audio_codec(file_path)
    except Exception:
        return None


def extract_audio_stream_info(ffprobe_output: str) -> Dict[str, str]:
    """
    从ffprobe的文本输出中提取音频流信息（兼容旧版本ffprobe）
    
    Args:
        ffprobe_output: ffprobe -show_streams的文本输出
        
    Returns:
        包含音频流信息的字典
    """
    info = {}
    
    # 正则表达式匹配
    patterns = {
        'codec_name': r'codec_name=(\w+)',
        'codec_long_name': r'codec_long_name=(.+)',
        'sample_rate': r'sample_rate=(\d+)',
        'channels': r'channels=(\d+)',
        'bit_rate': r'bit_rate=(\d+)',
        'duration': r'duration=([\d\.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, ffprobe_output)
        if match:
            info[key] = match.group(1)
    
    return info

def extract_audio(video_path: str, audio_output: str, 
                  config: Optional[Dict[str, Any]] = None) -> bool:
    """
    提取视频音频
    
    Args:
        video_path: 视频路径
        audio_output: 音频输出路径
        config: 配置文件参数（可选）
    
    Returns:
        是否成功
    """
    try:
        cmd = [
            'ffmpeg', '-i', video_path,       
            '-vn',  # 不包含视频
        ]

        # 如果提供了配置参数，使用配置中的编码设置
        if config:
            audio_codec = config.get('audio_codec', 'aac')
            audio_bitrate = config.get('bitrate', '192k')

            if audio_codec == 'copy':
                cmd.extend(['-c:a', 'copy'])
            else:
                cmd.extend([
                    '-c:a', audio_codec,
                    '-b:a', audio_bitrate,
                ])
        else:
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', '192k',
            ])

        cmd.extend(['-y', audio_output])

        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ 音频提取完成: {audio_output}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  音频提取失败: {e}")
        return False

def smart_extract_audio(
    video_path: Union[str, Path], 
    audio_output_dir: Union[str, Path], 
    timeout: int = 60,
    stream_index: int = 0,           # 新增：指定音频流索引
    overwrite: bool = False,         # 新增：是否覆盖已存在文件
    log_ffmpeg_output: bool = False,  # 新增：是否打印 ffmpeg 输出
    quiet: bool = False              # 新增：静默模式，抑制日志输出
) -> Optional[str]:
    """
    智能提取音频流（不重新编码），并根据音频编码自动选择适当的文件扩展名。

    Args:
        video_path: 输入视频文件路径
        audio_output_dir: 输出目录（字符串或 Path 对象）
        timeout: ffmpeg 执行超时时间（秒）
        stream_index: 要提取的音频流索引（默认 0，第一个音频流）
        overwrite: 是否覆盖已存在的输出文件
        log_ffmpeg_output: 是否打印 ffmpeg 的详细输出（调试用）
        quiet: 静默模式，抑制日志输出

    Returns:
        成功时返回输出音频文件的绝对路径字符串，失败返回 None
    """
    # 1. 输入验证
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"输入文件不存在: {video_path}")
        return None
    
    # 2. 准备输出目录
    out_dir = Path(audio_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. 获取音频编码
    try:
        codec = get_audio_codec(video_path, stream_index=stream_index)
        if codec == 'unknown':
            logger.error(f"无法识别音频编码（流索引 {stream_index}）")
            return None
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.error(f"获取音频编码失败: {e}")
        return None

    # 4. 确定输出扩展名和路径
    ext = AUDIO_EXT_MAP.get(codec, 'm4a')
    base_name = video_path.stem
    stream_suffix = f"_stream{stream_index}" if stream_index > 0 else ""
    output_filename = f"{base_name}{stream_suffix}_audio.{ext}"
    output_path = out_dir / output_filename
    
    # 5. 处理已存在文件
    if output_path.exists() and not overwrite:
        # [P1-FIX-AUDIO-FRESHNESS] 同名即复用会让"换了内容的同名输入视频"静默
        # 挂上旧音轨。以源视频 (size, mtime_ns) 指纹侧车校验新鲜度，不符则重提取。
        _fp_file = output_path.with_suffix(output_path.suffix + ".src.json")
        _cur_fp = None
        try:
            _st = video_path.stat()
            _cur_fp = {"source": str(video_path.resolve()),
                       "size": _st.st_size, "mtime_ns": _st.st_mtime_ns}
        except OSError:
            pass
        _fp_ok = False
        if _cur_fp is not None and _fp_file.exists():
            try:
                with open(_fp_file, "r", encoding="utf-8") as f:
                    _saved = json.load(f)
                _fp_ok = (_saved.get("source") == _cur_fp["source"]
                          and _saved.get("size") == _cur_fp["size"]
                          and _saved.get("mtime_ns") == _cur_fp["mtime_ns"])
            except Exception:
                _fp_ok = False
        if not _fp_ok:
            if not quiet:
                print(f"已有音频与当前源视频不匹配（或无指纹），重新提取: {output_path}")
        else:
            if not quiet:
                print(f"输出文件已存在且源未变化，跳过: {output_path}")
            return str(output_path.absolute())

    # 6. 构建 ffmpeg 命令
    cmd = [
        'ffmpeg',
        '-y' if overwrite else '-n',  # -y:覆盖, -n:不覆盖
        '-i', str(video_path),
        '-vn',                        # 不处理视频
        '-map', f'0:a:{stream_index}', # 精确选择指定音频流
        '-c:a', 'copy',              # 直接复制音频流
        str(output_path)
    ]

    # 7. 执行 ffmpeg
    if not quiet:
        print(f"执行命令: {' '.join(cmd)}")
    
    # 根据是否需要日志配置 stdout/stderr
    stdout = None if log_ffmpeg_output else subprocess.DEVNULL
    stderr = None if log_ffmpeg_output else subprocess.PIPE
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            stdout=stdout,
            stderr=stderr,
            text=True, 
            timeout=timeout
        )
        
        # 只有在需要日志且成功时打印输出
        if log_ffmpeg_output and result.stdout:
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else f"返回码 {e.returncode}"
        logger.error(f"ffmpeg 执行失败: {error_msg}")
        return None
    except FileNotFoundError:
        logger.error("未找到 ffmpeg，请确认已安装并加入 PATH")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg 执行超时（{timeout}秒）")
        return None

    # [P1-FIX-AUDIO-FRESHNESS] 成功后写入源指纹侧车，供下次复用判定
    try:
        _st2 = video_path.stat()
        with open(output_path.with_suffix(output_path.suffix + ".src.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"source": str(video_path.resolve()),
                       "size": _st2.st_size, "mtime_ns": _st2.st_mtime_ns},
                      f, indent=2)
    except Exception:
        pass

    return str(output_path.absolute())

def add_audio_to_video(video_path: str, audio_path: str, 
                       output_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    为视频添加音频
    
    Args:
        video_path: 视频路径
        audio_path: 音频路径
        output_path: 输出路径
        config: 配置文件参数（可选）
    
    Returns:
        是否成功
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:a', 'copy',
        ]
        
        # 如果提供了配置参数，使用配置中的编码设置
        if config:
            # [QUALITY-UNIFY] 质量参数按生效编码器换算（软编 -crf / 硬编 -cq:v / librav1e -qp）
            video_codec = config.get('codec', 'libx264')
            pix_fmt = config.get('pix_fmt', 'yuv420p')
            _q_args, _q_note = _resolve_quality_args(
                video_codec,
                crf=config.get('crf'), cq=config.get('cq'),
                crf_ref=config.get('crf_ref'), cq_ref=config.get('cq_ref'),
                preset=config.get('preset', 'medium'))
            logger.info(f"[add_audio] 质量参数解析: {_q_note} → {' '.join(_q_args)}")
            cmd.extend(['-c:v', video_codec] + _q_args + ['-pix_fmt', pix_fmt])

        else:
            # 默认设置：复制视频
            cmd.extend([
                '-c:v', 'copy',
                '-vsync', 'passthrough'
            ])
        
        cmd.extend([
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y', output_path
        ])

        # 打印命令以便调试
        print(f"📋 音频合并命令: {' '.join(cmd)}")
        # get_frame_rate(video_path)
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ 音频添加完成: {output_path}")

        # 打印信息以便调试
        # get_frame_rate(video_path)

        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 音频添加失败: {e}")
        return False

#视频处理函数
def get_frame_rate(file_path):
    """获取视频帧率"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
        '-of', 'json',  # 使用JSON格式，更容易解析
        file_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # 提取帧率信息
        stream = data['streams'][0]
        r_frame_rate = stream.get('r_frame_rate', 'N/A')
        avg_frame_rate = stream.get('avg_frame_rate', 'N/A')
        
        # 计算实际数值（如果格式是 30/1）
        def parse_fraction(fraction_str):
            if fraction_str == 'N/A' or '/' not in fraction_str:
                return None
            try:
                num, den = map(int, fraction_str.split('/'))
                return num / den if den != 0 else None
            except (ValueError, ZeroDivisionError):
                # [P0-FIX-BARE-EXCEPT] 原裸 except 会吞 KeyboardInterrupt/SystemExit
                return None
        
        r_fps = parse_fraction(r_frame_rate)
        avg_fps = parse_fraction(avg_frame_rate)
        
        # 打印结果
        print(f"视频文件: {file_path}")
        print(f"声明帧率 (r_frame_rate): {r_frame_rate}")
        print(f"平均帧率 (avg_frame_rate): {avg_frame_rate}")
        if r_fps:
            print(f"声明帧率 (数值): {r_fps:.2f} fps")
        if avg_fps:
            print(f"平均帧率 (数值): {avg_fps:.2f} fps")
        
        return r_frame_rate, avg_frame_rate
        
    except FileNotFoundError:
        print("错误: 未找到 ffprobe，请安装FFmpeg")
    except subprocess.CalledProcessError as e:
        print(f"ffprobe执行错误: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"原始输出: {result.stdout}")
    except Exception as e:
        print(f"未知错误: {e}")

def get_video_duration(video_path: str) -> Optional[float]:
    """
    获取视频时长（秒）
    
    Args:
        video_path: 视频路径
    
    Returns:
        时长（秒），失败返回None
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        if output and output != 'N/A':
            return float(output)
    except Exception:
        pass

    # 备用方法：使用OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps > 0 and frame_count > 0:
            return frame_count / fps
    except Exception:
        pass

    return None

def format_time(seconds: Optional[float]) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时间字符串
    """
    if seconds is None:
        return "未知"
    
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}.{ms:03d}"
    elif mins > 0:
        return f"{mins}:{secs:02d}.{ms:03d}"
    else:
        return f"{secs}.{ms:03d}秒"


def validate_source_video_structurally(video_path: Union[str, Path]) -> Tuple[bool, str]:
    """[P5-FIX-SOURCE-STRUCT-GATE] 处理前对源视频做硬性结构解码检查。

    只拦截不可解码/损坏视频，不做内容级撕裂启发式（高误判风险）。
    """
    path = Path(video_path)
    if not path.exists():
        return False, "source_missing"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg_unavailable"
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-v", "error", "-i", str(path),
             "-map", "0:v:0", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=3600,
        )
        err_tail = (proc.stderr or "").strip()
        if proc.returncode != 0 or err_tail:
            return False, f"source_decode_errors: {err_tail[-1000:]}"
    except subprocess.TimeoutExpired:
        return False, "source_validation_timeout"
    except OSError as exc:
        return False, f"validation_process_error: {exc}"
    return True, "ok"


def _detect_gpu_hwaccel_available(ffmpeg_bin='ffmpeg', timeout=10) -> bool:
    """自动探测 GPU NVDEC 硬件解码能力（复用 _count_frames_nvdec 思路）。"""
    try:
        ffmpeg_path = shutil.which(ffmpeg_bin) or ffmpeg_bin
        proc = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-v", "error",
             "-hwaccel", "cuda", "-i", "/dev/null",
             "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
        # 即使 returncode != 0，只要 stderr 不含 "no capable devices" / "Invalid" 等关键错误，
        # 且命令能执行（说明 ffmpeg 识别到 -hwaccel cuda），则认为硬件可用
        err_text = (proc.stderr or "").lower()
        negative_indicators = ["no capable devices found", "openencodesessionex failed",
                               "invalid param", "unsupported", "not supported"]
        if any(k in err_text for k in negative_indicators):
            return False
        # 只要 ffmpeg 接受了 -hwaccel cuda 选项且未立即因设备缺失崩溃，即视为可用
        # （实际解码由调用方在 -i 真实文件时验证）
        return True
    except Exception:
        return False


def _get_gpu_hwaccel_config() -> Dict[str, Any]:
    """利用 system_resources 探测 GPU 并计算最大并发 NVDEC session 数。"""
    try:
        from .system_resources import SystemResourceDetector, SystemResources
    except ImportError:
        try:
            from system_resources import SystemResourceDetector, SystemResources
        except ImportError:
            SystemResourceDetector = None  # type: ignore
    if SystemResourceDetector is None:
        return {"available": False, "max_sessions": 0, "gpu_count": 0, "gpu_workers": 0}
    resources = SystemResourceDetector().detect()
    gpu_available = resources.gpu_count > 0 and len(resources.gpus) > 0
    max_sessions = sum(g.max_nvdec_sessions for g in resources.gpus) if gpu_available else 0
    # 默认最大化并行（利用全部 session）；可通过环境变量覆盖
    env_workers = os.environ.get("NVENC_VALIDATE_GPU_WORKERS", "").strip()
    if env_workers:
        try:
            gpu_workers = max(1, int(env_workers))
        except ValueError:
            gpu_workers = max_sessions if max_sessions > 0 else 0
    else:
        gpu_workers = max_sessions if max_sessions > 0 else 0
    return {
        "available": gpu_available,
        "max_sessions": max_sessions,
        "gpu_count": resources.gpu_count,
        "gpu_workers": max(1, gpu_workers) if gpu_available else 0,
        "resources": resources,
    }
#
# 背景（2026-09-05 实测）：count_decoded_video_frames() 原实现无条件使用
# `ffprobe -count_frames`，该参数会**全量软解码**整个文件：
#   · 1536x1152 HEVC、43448 帧 → 实测 8 分钟（≈5430 帧/分钟，单线程）
#   · R1 每个超分段约 9000 帧 → 约 1.65 分钟 × 5 段 ≈ 8.3 分钟
#   · 原 timeout=1800（30 分钟）→ 慢时表现为"进程卡住"，只能靠 ps 发现
#   · 同一文件被【段验收】【最终校验】反复探测，每次都全解码，浪费严重
#
# P0 进程内缓存（同 key 只解码一次）
# P1 元数据优先（auto 模式：nb_frames 可信时跳过全解码）
# P2 NVDEC 硬件解码（可选，失败自动回退软解）
# P3 并行探测（批量预热缓存）
# P4 超时收敛 1800→300 + 耗时日志
# ═══════════════════════════════════════════════════════════════════════════

# [FIX-GATE-STRICT-COUNT] 值为 (帧数, 来源) 二元组，来源 ∈ {'metadata','decode'}。
# 旧实现只缓存裸帧数，于是「先用 auto 命中容器元数据」会把低可信值喂给之后要求
# mode='decode' 的验收门调用 —— 上游 count_frames_parallel() 预热（默认 auto）恰好
# 就是 auto 先行，验收门的 decode 语义会被静默降级。带上来源后，decode 调用拒绝
# 复用 metadata 来源的值（并随后用真解码结果覆盖，顺带升级 auto 调用方）。
_PROBE_FRAME_CACHE: Dict[tuple, Tuple[Optional[int], str]] = {}
_PROBE_STATS = {"calls": 0, "hits": 0, "metadata": 0, "decode": 0, "seconds": 0.0}

# 探测超时：全解码最多等 300s（原 1800s）。超过即返回 None，由调用方按
# "frame_count_unavailable" 处理，绝不静默长期阻塞。
_PROBE_TIMEOUT_S = 300


def _probe_cache_key(path: Path):
    """缓存 key：绝对路径 + 大小 + mtime(ns)。任一变化即视为新文件。"""
    try:
        st = path.stat()
        return (str(path.resolve()), st.st_size, st.st_mtime_ns)
    except OSError:
        return None


def _read_nb_frames_metadata(path: Path, ffprobe: str) -> Optional[int]:
    """读取容器元数据帧数（秒级，不解码）。不可信时返回 None。"""
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,r_frame_rate,duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return None
        vs = streams[0]
        raw = vs.get("nb_frames")
        if raw in (None, "", "N/A"):
            return None
        nb = int(raw)
        if nb <= 0:
            return None
        # 交叉验证：与 duration × fps 偏差超过 5% 则判定元数据不可信
        try:
            from fractions import Fraction as _Fraction
            fps = float(_Fraction(vs.get("r_frame_rate", "0/1")))
            dur = float(vs.get("duration", 0) or 0)
            if fps > 0 and dur > 0:
                est = dur * fps
                if est > 0 and abs(est - nb) / max(est, 1.0) > 0.05:
                    return None
        except Exception:
            pass
        return nb
    except Exception:
        return None


def _count_frames_ffprobe(path: Path, ffprobe: str) -> Optional[int]:
    """P2/P4：ffprobe -count_frames 全解码计数（软解）。"""
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries",
             "stream=nb_read_frames,nb_frames", "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=True, timeout=_PROBE_TIMEOUT_S)
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [])
        if not streams:
            return None
        value = streams[0].get("nb_read_frames") or streams[0].get("nb_frames")
        return int(value) if value not in (None, "", "N/A") else None
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# [P0-3] 单次全量解码同时产出「帧数」与「真实错误行」
# ═══════════════════════════════════════════════════════════════════════════
# 原实现把一次验收拆成两次全量解码:
#   ① count_decoded_video_frames(use_hwaccel=True) → `-v info -f null` 取帧数
#   ② validate_decodable_video 的错误检查 → 再跑一次 `-v error -f null`
# 实测每次 4K 解码约 4~9s，即每个文件白付一倍解码代价；且 ② 用
# "stderr 有任意一行即判失败"，在 hwaccel 初始化失败时会误报。
#
# [FIX-NVDEC-COUNT-DEAD-V2] 仓库旧注释称「`-v error -stats` 均不产生 frame=」，
# 该结论对 ffmpeg 7.1 已不成立：实测 `-v error -stats` 正常输出进度行与收尾行
# （`frame= 720 fps=... Lsize=N/A`），因此可在 `-v error` 下同时取得帧数与
# stderr 错误行 —— 既不必退到 `-v info`（那样必须用启发式分类 stderr，反而
# 降低严格度），也不必为计数与错误检查各跑一次解码。
#
# 单次解码的完整证据缓存（key 同 _PROBE_FRAME_CACHE: 路径+大小+mtime_ns）。
# validate_decodable_video 直接复用，保证「同一文件一次验收只解码一次」。
_PROBE_DETAIL_CACHE: Dict[tuple, Dict[str, object]] = {}

# NVDEC 初始化失败特征：属运行时环境问题（driver/NVDEC SDK 不匹配、容器
# GPU 未透传、surface 超限等），ffmpeg 会静默回退软解且帧数仍然有效，
# 不得计入解码错误。与 tests/verify_segment_bitstream_v5.py 的
# _HWACCEL_INIT_FAILURE_KW 保持同源。
_HWACCEL_INIT_FAIL_KW = (
    "failed setup for format cuda",
    "hwaccel initialisation returned error",
    "cuvidcreatedecoder",
    "cuda_error_invalid_value",
    "decode surfaces",
)
# 已知无害的运行时噪声行（非码流缺陷）
_BENIGN_PROBE_KW = ("avhwdevicecontext", "instance creation failure")


# ═══════════════════════════════════════════════════════════════════════════
# [PROBE-CACHE-PERSIST] 帧数探测缓存的断点恢复（落盘 sidecar）
# ═══════════════════════════════════════════════════════════════════════════
# 背景：[PROBE-OPT] 预热把分段帧数写入 _PROBE_FRAME_CACHE，但那是纯进程内
# dict —— 断点重启后全部归零，而 resume 时分段走 split_video_by_time() 的
# 复用分支（字节/mtime 不变），于是同一批分段每次都要重新全解码一遍。
#
# 处理：把缓存落盘到 checkpoint 同目录的 sidecar（由调用方用
# set_probe_cache_file() 指定路径），启动时加载、命中即免重复解码。缓存 key
# （路径+size+mtime_ns）已随文件内容变化自动失效，无需额外校验。
#
# ⚠️ 只持久化**成功**项：帧数为 None，或来源为 'metadata'（低可信，见
# [FIX-GATE-STRICT-COUNT]）的条目一律不写 —— 否则一次瞬时失败/元数据捷径会
# 被永久固化，让 mode='decode' 的严格验收门静默失效。detail
# （error_lines/rc/hw_failed）同理，只写 rc==0 且无错误行的证据，否则恢复后
# validate_decodable_video 会错误地跳过解码错误检查。
# 可用 NVENC_PROBE_CACHE_PERSIST=0 关闭持久化（退回纯进程内行为）。
_PROBE_CACHE_FILE: Optional[str] = None
_PROBE_CACHE_VERSION = 1
_PROBE_CACHE_LOCK = threading.Lock()


def _probe_key_to_str(key: tuple) -> str:
    """缓存 key（tuple）→ JSON 键字符串（用 JSON 数组编码，避免分隔符歧义）。"""
    return json.dumps(list(key), ensure_ascii=False)


def _probe_key_from_str(s: str):
    """JSON 键字符串 → 缓存 key；格式不符返回 None。"""
    try:
        v = json.loads(s)
        if isinstance(v, list) and len(v) == 3:
            return (str(v[0]), int(v[1]), int(v[2]))
    except (ValueError, TypeError):
        pass
    return None


def _probe_detail_persistable(detail) -> bool:
    """只有「解码成功且无错误行」的证据才可固化（失败不得跨进程复用）。"""
    if not isinstance(detail, dict):
        return False
    if detail.get("frames") is None:
        return False
    if detail.get("rc") not in (0, None):
        return False
    return not detail.get("error_lines")


def _load_probe_cache() -> None:
    """从 sidecar 加载帧数/解码证据缓存（版本不符或损坏则忽略）。"""
    if not _PROBE_CACHE_FILE:
        return
    try:
        with open(_PROBE_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict) or data.get("version") != _PROBE_CACHE_VERSION:
        return
    for ks, item in (data.get("frames") or {}).items():
        k = _probe_key_from_str(ks)
        if k is None or not isinstance(item, dict):
            continue
        val, kind = item.get("value"), item.get("kind")
        if val is None or kind != "decode":
            continue                      # 低可信/失败值绝不回流
        _PROBE_FRAME_CACHE[k] = (int(val), "decode")
    for ks, detail in (data.get("detail") or {}).items():
        k = _probe_key_from_str(ks)
        if k is None or not _probe_detail_persistable(detail):
            continue
        _PROBE_DETAIL_CACHE[k] = detail


def _write_probe_cache_merged(frames: dict, detail: dict) -> None:
    """在跨进程文件锁内「读旧 → 合并 → 原子写」，供 process 模式并发持久化。

    [PROBE-CACHE-PERSIST-PROC] 为什么必须合并而不是直接覆盖：
    ProcessPoolExecutor 用 fork 时子进程继承父进程的 `_PROBE_FRAME_CACHE`
    **快照**，每个子进程各自把「自己算出的那一条」写回同一 sidecar —— 直接
    覆盖 = 最后写者胜，N 个子进程的结果只剩 1 条（实测 8 段只留 1 条）。且并发
    写同一个 `.tmp` 路径可能产出交错内容。此处用 flock 串行化「读-合并-写」，
    并把临时文件名带上 pid，保证多进程安全且结果累积。

    Returns: 无返回值；任何异常都不抛出（缓存落盘失败不影响主流程）。
    """
    lock_fp = None
    try:
        lock_fp = open(_PROBE_CACHE_FILE + ".lock", "a+")
    except Exception:
        lock_fp = None
    try:
        if lock_fp is not None:
            try:
                import fcntl
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass                    # 非 POSIX 平台无 fcntl：退化为不加锁
        # 读回已落盘内容（可能是同批另一个子进程刚写的），仅补充本地缺失项
        try:
            with open(_PROBE_CACHE_FILE, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict) and \
                    old.get("version") == _PROBE_CACHE_VERSION:
                for k, v in (old.get("frames") or {}).items():
                    if k not in frames and isinstance(v, dict) \
                            and v.get("kind") == "decode" \
                            and v.get("value") is not None:
                        frames[k] = v
                for k, v in (old.get("detail") or {}).items():
                    if k not in detail and _probe_detail_persistable(v):
                        detail[k] = v
        except Exception:
            pass
        payload = {"version": _PROBE_CACHE_VERSION,
                   "frames": frames, "detail": detail}
        _tmp = "%s.tmp.%d" % (_PROBE_CACHE_FILE, os.getpid())
        with open(_tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(_tmp, _PROBE_CACHE_FILE)
    except Exception:
        pass
    finally:
        if lock_fp is not None:
            try:
                import fcntl
                fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_fp.close()
            except Exception:
                pass


def _persist_probe_cache() -> None:
    """把成功项原子写回 sidecar（tmp + os.replace，沿用断点的 [P1-FIX-ATOMIC]）。"""
    if not _PROBE_CACHE_FILE:
        return
    if os.environ.get("NVENC_PROBE_CACHE_PERSIST", "1").strip() == "0":
        return
    frames = {}
    for k, entry in list(_PROBE_FRAME_CACHE.items()):
        try:
            val, kind = entry
        except (TypeError, ValueError):
            continue
        if val is None or kind != "decode":
            continue
        frames[_probe_key_to_str(k)] = {"value": int(val), "kind": "decode"}
    detail = {}
    for k, d in list(_PROBE_DETAIL_CACHE.items()):
        if _probe_detail_persistable(d):
            detail[_probe_key_to_str(k)] = d
    with _PROBE_CACHE_LOCK:                 # 进程内串行；跨进程由 flock 串行
        _write_probe_cache_merged(frames, detail)


def set_probe_cache_file(path) -> None:
    """指定帧数缓存 sidecar 路径并加载已有内容；None 关闭持久化（纯进程内）。

    由 processor 在 _setup_temp_dirs() 之后调用，路径取 checkpoint 同目录
    （随 temp/{stage}/{prefix}_{video}/ 生命周期清理）。

    切换到**不同** sidecar 时先清空内存缓存：同一进程可能先后处理不同视频/
    阶段（source 段 → from_segments 段 → ESRGAN 段），否则前一个视频的条目会
    被写进后一个 sidecar（虽因 key 携带路径而不会误命中，但会污染文件且随
    视频数无界增长）。
    """
    global _PROBE_CACHE_FILE
    _new = str(path) if path else None
    if _new != _PROBE_CACHE_FILE:
        _PROBE_FRAME_CACHE.clear()
        _PROBE_DETAIL_CACHE.clear()
    _PROBE_CACHE_FILE = _new
    if _PROBE_CACHE_FILE:
        _load_probe_cache()


def get_probe_cache_file() -> Optional[str]:
    return _PROBE_CACHE_FILE


def probe_cache_persist_enabled() -> bool:
    return bool(_PROBE_CACHE_FILE) and \
        os.environ.get("NVENC_PROBE_CACHE_PERSIST", "1").strip() != "0"


def _probe_decode_error(path: Path, ffmpeg: str, use_hwaccel: bool,
                        timeout: int = _PROBE_TIMEOUT_S) -> Optional[Dict[str, object]]:
    """[P0-3] 单次 `-v error -stats -f null` 全量解码，一次产出帧数与错误行。

    Returns:
        dict(frames, error_lines, hw_failed, rc, stderr_tail)；
        None = 探测本身异常/超时（调用方按原语义回退）。
    """
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-stats"]
    if use_hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", str(path), "-map", "0:v:0", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    except Exception:
        return None
    err = r.stderr or ""
    # 进度行在同一行内以 \r 原地覆盖，捕捉后需按 \r\n / \r 双重切分
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", err) if ln.strip()]
    frames: Optional[int] = None
    for mm in re.finditer(r"frame=\s*(\d+)", err):
        frames = int(mm.group(1))       # 单调递增，最后一个即最终计数
    error_lines: List[str] = []
    hw_failed = False
    for line in lines:
        low = line.lower()
        if low.startswith("frame=") or "fps=" in low or "speed=" in low:
            continue                    # ffmpeg -stats 进度/收尾行
        if any(k in low for k in _HWACCEL_INIT_FAIL_KW):
            hw_failed = True
            continue
        if any(k in low for k in _BENIGN_PROBE_KW):
            continue
        error_lines.append(line[:200])
    return {
        "frames": frames,
        "error_lines": error_lines,
        "hw_failed": hw_failed,
        "rc": r.returncode,
        "stderr_tail": err.strip()[-2000:],
    }


def _get_probe_detail(path: Path) -> Optional[Dict[str, object]]:
    """取本文件最近一次全量解码的完整证据（无则 None）。"""
    key = _probe_cache_key(path)
    if key is None:
        return None
    return _PROBE_DETAIL_CACHE.get(key)


def _count_frames_nvdec(path: Path, ffmpeg: str) -> Optional[int]:
    """P2：用 NVDEC 硬件解码计数（ffmpeg -hwaccel cuda ... -f null -），
    比 ffprobe 软解快 3~5 倍。任何失败/超时都返回 None，由调用方回退。

    [P0-3] 改为委托 _probe_decode_error()：同一次解码顺带产出 error_lines /
    rc / hw_failed，写入 _PROBE_DETAIL_CACHE 供 validate_decodable_video 的
    「解码错误检查」复用，消除第二次全量解码。

    判定语义保持原有严格度：rc != 0 或存在真实错误行即返回 None（交回调用方
    回退软解）。hwaccel 初始化失败不算错误（ffmpeg 已内部软解、帧数有效），
    由 detail['hw_failed'] 单独上报。
    """
    detail = _probe_decode_error(path, ffmpeg, use_hwaccel=True)
    if detail is None:
        return None
    key = _probe_cache_key(path)
    if key is not None:
        _PROBE_DETAIL_CACHE[key] = detail
    if detail["rc"] != 0 or detail["error_lines"]:
        return None
    frames = detail["frames"]
    return int(frames) if frames is not None else None


def count_decoded_video_frames(video_path: Union[str, Path],
                               mode: Optional[str] = None,
                               use_cache: bool = True,
                               use_hwaccel: Optional[bool] = None) -> Optional[int]:
    """返回视频的真实可解码帧数（支持自动 GPU 探测与并行硬解）。

    [P4-FIX-COUNT] 原始语义：返回 nb_read_frames 而非容器元数据帧数——
    容器 nb_frames/packet 数对"包存在但解码失败/参考链断裂"完全盲区。
    该严格语义在 mode='decode' 时完整保留（用于最终产出验收门）。

    [PROBE-OPT] 优化分级（见本模块顶部 [PROBE-OPT] 说明）：

    Args:
        video_path: 视频路径
        mode:  None → 取环境变量 NVENC_COUNT_FRAMES，默认 'auto'
              'auto'     : 容器元数据可信时直接返回；否则先尝试 NVDEC 硬解（若可用），再软解
              'metadata' : 只信元数据，绝不解码（最快，最不严格）
              'decode'   : 总是全解码（最严格，等价原始行为，但默认优先 NVDEC）
        use_cache: 是否使用进程内缓存（P0）。默认 True。
        use_hwaccel: None=自动探测 GPU 并启用（若可用）；True=强制硬解；False=强制软解。

    环境变量：
        NVENC_COUNT_FRAMES = auto|metadata|decode
        NVENC_COUNT_HWACCEL = 0|1   （P2：是否优先尝试 NVDEC，默认 0；优化后默认自动启用）
        NVENC_COUNT_VERBOSE = 0|1   （P4：打印耗时日志，默认 1）
        NVENC_VALIDATE_GPU_WORKERS = N （控制并发 GPU session 数，默认自动计算 max_nvdec_sessions）
    """
    import time as _t

    _t0 = _t.perf_counter()
    path = Path(video_path)
    _PROBE_STATS["calls"] += 1
    if not path.exists():
        return None

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    if mode is None:
        mode = os.environ.get("NVENC_COUNT_FRAMES", "auto").strip().lower()
    if mode not in ("auto", "metadata", "decode"):
        mode = "auto"

    # 自动探测 GPU 硬解能力（优化：默认自动启用，不再完全依赖环境变量）
    hwaccel_enabled = False
    hw_config = {"available": False, "max_sessions": 0, "gpu_workers": 0, "gpu_count": 0}
    if use_hwaccel is None:
        # 默认自动探测（不再强制要求 NVENC_COUNT_HWACCEL=1 才尝试）
        hw_config = _get_gpu_hwaccel_config()
        hwaccel_enabled = hw_config["available"]
    elif use_hwaccel is True:
        hw_accel_available = _detect_gpu_hwaccel_available()
        hw_config = _get_gpu_hwaccel_config()
        # 即使探测成功，也结合环境变量做最终仲裁：若用户显式传入 True 且 GPU 存在，则启用
        hwaccel_enabled = hw_accel_available and (hw_config.get("available", False) or hw_config.get("gpu_count", 0) > 0)
        if not hwaccel_enabled:
            # 强制启用但 GPU 不可用：记录为回退状态，但仍继续尝试软解
            pass
    else:
        # use_hwaccel=False：强制软解
        hwaccel_enabled = False
        hw_config = {"available": False, "max_sessions": 0, "gpu_workers": 0, "gpu_count": 0}

    # 仅在环境变量显式要求关闭时，覆盖自动探测结果
    if os.environ.get("NVENC_COUNT_HWACCEL", "").strip() == "0":
        hwaccel_enabled = False

    # ── P0 缓存 ──────────────────────────────────────────────────────────
    key = _probe_cache_key(path) if use_cache else None
    if key is not None and key in _PROBE_FRAME_CACHE:
        _cached_val, _cached_kind = _PROBE_FRAME_CACHE[key]
        # [FIX-GATE-STRICT-COUNT] mode='decode' 只接受「真解码得来」的缓存值：
        # 复用元数据来源的值等于把 decode 悄悄降级成 metadata，正是验收门的盲区。
        if not (mode == "decode" and _cached_kind == "metadata"):
            _PROBE_STATS["hits"] += 1
            _PROBE_STATS["seconds"] += _t.perf_counter() - _t0
            return _cached_val

    result: Optional[int] = None
    # [FIX-GATE-STRICT-COUNT] 记录 result 的来源（'metadata' / 'decode'），
    # 供缓存区分可信度：元数据来源的值不得回流给 mode='decode' 的调用方。
    result_kind: str = "decode"
    # ── 优化路径：若可用硬解，先尝试 NVDEC（P2）；否则直接软解 ────────────
    if hwaccel_enabled and mode in ("auto", "decode"):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            # 尝试 NVDEC 硬解计数（单文件仍单进程，避免同一文件多 session 冲突）
            result = _count_frames_nvdec(path, ffmpeg)
            if result is not None:
                _PROBE_STATS["decode"] += 1
                # 若硬解成功，直接返回；否则继续回退
            else:
                # 硬解失败：进入降级回退（先降低并发再回退 CPU）
                # 由于单文件内不并发，此处直接视为降级失败 → 回退软解
                pass

    # 若硬解未启用、失败，或模式强制软解，则执行软解路径
    if result is None:
        if mode == "decode":
            result = _count_frames_ffprobe(path, ffprobe)
            _PROBE_STATS["decode"] += 1
        else:
            # ── P1 元数据优先 ────────────────────────────────────────────────
            result = _read_nb_frames_metadata(path, ffprobe)
            if result is not None:
                _PROBE_STATS["metadata"] += 1
                result_kind = "metadata"
            elif mode == "auto":
                # 元数据不可信/缺失 → 回退全解码
                result = _count_frames_ffprobe(path, ffprobe)
                _PROBE_STATS["decode"] += 1

    # ── P2 NVDEC 兜底（仅在硬解未成功且环境变量开启时尝试）───
    # 优化后：默认已在上方尝试硬解，此处保留为显式兜底（兼容旧行为）
    if result is None and os.environ.get("NVENC_COUNT_HWACCEL", "0") == "1":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            result = _count_frames_nvdec(path, ffmpeg)
            if result is not None:
                _PROBE_STATS["decode"] += 1

    if key is not None:
        _PROBE_FRAME_CACHE[key] = (result, result_kind)
        # [PROBE-CACHE-PERSIST] 只有真解码出来的成功值才落盘（断点重启免重算）；
        # None / metadata 来源一律不写，避免把失败或低可信值固化进 sidecar。
        if result is not None and result_kind == "decode":
            _persist_probe_cache()

    _elapsed = _t.perf_counter() - _t0
    _PROBE_STATS["seconds"] += _elapsed
    # ── P4 可观测 ────────────────────────────────────────────────────────
    if os.environ.get("NVENC_COUNT_VERBOSE", "1") != "0" and _elapsed > 3.0:
        hw_path_tag = "nvdec" if (hwaccel_enabled and result is not None and result > 0) else ("hard_failed_fallback" if hwaccel_enabled else "soft")
        print(f"[probe] count_frames 耗时 {_elapsed:.1f}s "
              f"(mode={mode}, frames={result}, hwaccel={hw_path_tag}) {path.name}", flush=True)
    return result


def count_frames_parallel(paths, max_workers: int = 4, mode: Optional[str] = None) -> Dict[str, Optional[int]]:
    """P3：并行探测多个文件的帧数，用于批量预热 P0 缓存。

    返回 {str(path): frames}。任一文件失败不影响其他文件；并行不可用时
    退回串行。预热后，后续 count_decoded_video_frames() 调用直接命中缓存。
    """
    paths = [Path(p) for p in paths]
    result: Dict[str, Optional[int]] = {}
    if not paths:
        return result
    try:
        from concurrent.futures import ThreadPoolExecutor
        workers = max(1, min(max_workers, len(paths)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(count_decoded_video_frames, p, mode): p for p in paths}
            for f, p in futs.items():
                try:
                    result[str(p)] = f.result()
                except Exception:
                    result[str(p)] = None
    except Exception:
        for p in paths:
            try:
                result[str(p)] = count_decoded_video_frames(p, mode)
            except Exception:
                result[str(p)] = None
    return result


def probe_stats() -> Dict[str, object]:
    """返回探测统计，便于诊断（P4 可观测）。"""
    c = max(1, _PROBE_STATS["calls"])
    return {
        "calls": _PROBE_STATS["calls"],
        "cache_hits": _PROBE_STATS["hits"],
        "cache_hit_rate": f"{_PROBE_STATS['hits'] * 100.0 / c:.0f}%",
        "via_metadata": _PROBE_STATS["metadata"],
        "via_decode": _PROBE_STATS["decode"],
        "total_seconds": round(_PROBE_STATS["seconds"], 1),
    }


def _run_ffmpeg_decode_validation(path: Path, use_hwaccel: bool = False) -> Tuple[int, str]:
    """执行 ffmpeg 解码错误检查；use_hwaccel=True 时注入 -hwaccel cuda。

    [FIX-NULL-NV12] 不再追加 `-hwaccel_output_format nv12`：`-f null` 不消费
    任何像素，指定 hw 输出格式只会强制 ffmpeg 逐帧 D2H 下行 + 格式协商，
    实测同一 4K H.264 文件 7.51s → 6.88s（-9%），NVDEC 利用率 36.6% → 41.1%，
    且该路径曾产生 "hwaccel initialisation returned error" 类假失败。
    纯 `-hwaccel cuda` 由 ffmpeg 自行在需要时回拷，语义完全一致。
    """
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg_bin, "-hide_banner", "-v", "error"]
    if use_hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", str(path), "-map", "0:v:0", "-f", "null", "-"]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600
    )
    err_tail = (proc.stderr or "").strip()
    return proc.returncode, err_tail


def _validate_container_only(path: Path, report: Dict[str, object]
                             ) -> Tuple[bool, Dict[str, object]]:
    """[P3-1] 容器级（不解码）校验：给「分段已逐段解码级验收通过」的最终合并输出用。

    为什么可以不解码：分段阶段每个产出都过了 ``validate_decodable_video``（严格
    全解码 + 帧数守恒 + 解码错误检查），合并阶段 ``-c:v copy`` 不重编码，因此
    合并产物出问题只可能是容器层（索引/时间戳/moov 不完整）。本函数专门覆盖
    这一层，避免对同一个视频再付一次全量解码代价。

    注意：本模式**不是**弱化版的全解码验收，只适用于「分段验收已覆盖」的情形；
    「整体处理不分段」的最终输出必须走全解码路径（见调用方
    main_video_optimized._merge_and_finalize）。

    Checks:
      1. ffprobe 可解析且存在视频流（宽高 > 0）；
      2. 包计数 > 0（``ffprobe -count_packets`` 仅 demux，不解码）；
      3. 可选：元数据帧数与包计数交叉核对（不可信元数据不判失败，仅记录）。
    """
    report["mode"] = "container_only"
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        report["reason"] = "ffprobe_unavailable"
        return False, report
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120)
        data = json.loads(r.stdout or "{}")
        streams = data.get("streams") or []
        if r.returncode != 0 or not streams:
            report["reason"] = "container_unreadable"
            report["decode_stderr_tail"] = (r.stderr or "").strip()[-500:]
            return False, report
        vs = streams[0]
        width = int(vs.get("width") or 0)
        height = int(vs.get("height") or 0)
        report["width"], report["height"] = width, height
        if width <= 0 or height <= 0:
            report["reason"] = "no_video_stream"
            return False, report
        nb_frames = vs.get("nb_frames")
        report["metadata_frames"] = (int(nb_frames)
                                     if str(nb_frames).isdigit() else None)
    except Exception as exc:
        report["reason"] = "container_probe_error"
        report["detail"] = str(exc)
        return False, report

    packets = None
    try:
        r2 = subprocess.run(
            [ffprobe, "-v", "error", "-count_packets", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300)
        m = re.match(r"(\d+)", (r2.stdout or "").strip())
        packets = int(m.group(1)) if m else None
    except Exception:
        packets = None
    report["packets"] = packets
    report["decoded_frames"] = report.get("metadata_frames")
    if packets is None or packets <= 0:
        report["reason"] = "packet_count_unavailable"
        return False, report
    # 元数据帧数与包计数不一致只做记录（容器 nb_frames 常不可信），不判失败
    meta_frames = report.get("metadata_frames")
    if meta_frames is not None and meta_frames != packets:
        report["metadata_packet_mismatch"] = (meta_frames, packets)
    report["reason"] = "ok"
    return True, report


def validate_decodable_video(video_path: Union[str, Path],
                             expected_frames: Optional[int] = None,
                             count_mode: Optional[str] = None,
                             use_hwaccel: Optional[bool] = None,
                             gpu_workers: Optional[int] = None,
                             skip_decode_check: bool = False
                             ) -> Tuple[bool, Dict[str, object]]:
    """[P4-FIX-GATE] 段级/最终输出解码级守恒与错误校验（优化：自动 GPU 探测 + 并行硬解 + 降级回退）。

    Checks:
      1. ffprobe 可解析且有视频流；
      2. nb_read_frames == expected_frames（提供时）；
      3. ffmpeg -v error ... -f null - 无任何 stderr 错误（支持 -hwaccel cuda 硬解）。

    Args:
        count_mode: 传给 count_decoded_video_frames() 的探测模式（见 [PROBE-OPT]）。
        use_hwaccel: None=自动探测 GPU 并启用（若可用）；True=强制硬解；False=强制软解。
        gpu_workers: GPU 并发 session 数（用于批量控制，默认自动计算 max_nvdec_sessions）。
        skip_decode_check: [P3-1] True 时改为**容器级校验**（不解码），仅允许用于
            「其分段已逐段通过解码级验收」的最终合并输出。默认 False 保持
            全解码严格语义。见 _validate_container_only 的适用条件说明。
    """
    path = Path(video_path)
    report: Dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "hwaccel_path": None,
        "hwaccel_fallback": False,
    }
    if not report["exists"]:
        report["reason"] = "file_missing"
        return False, report
    if path.stat().st_size < 1024:
        report["reason"] = "too_small"
        return False, report

    # [P3-1] 容器级（不解码）校验分支：仅由「分段已逐段解码级验收通过」的
    # 最终合并输出调用，见函数 docstring 与 _validate_container_only。
    if skip_decode_check:
        return _validate_container_only(path, report)

    # 自动探测 GPU 硬解能力（默认启用，不再完全依赖环境变量）
    hw_config = _get_gpu_hwaccel_config()
    gpu_available = hw_config.get("available", False) and hw_config.get("gpu_count", 0) > 0
    max_sessions = hw_config.get("max_sessions", 0)
    # 默认最大化并行（利用全部 NVDEC session），可通过环境变量覆盖
    if gpu_workers is None:
        env_workers_raw = os.environ.get("NVENC_VALIDATE_GPU_WORKERS", "").strip()
        if env_workers_raw:
            try:
                gpu_workers = max(1, int(env_workers_raw))
            except ValueError:
                gpu_workers = max(1, max_sessions) if max_sessions > 0 else 0
        else:
            gpu_workers = max(1, max_sessions) if max_sessions > 0 else 0

    # 仲裁 use_hwaccel：None → 自动探测结果；True → 强制硬解（若 GPU 不可用则记录为回退）；False → 强制软解
    if use_hwaccel is None:
        use_hwaccel_final = gpu_available
    else:
        use_hwaccel_final = bool(use_hwaccel)

    # 降级回退状态跟踪
    current_hwaccel = use_hwaccel_final and gpu_available
    fallback_stage = 0  # 0=直接，1=降级（降低并发），2=回退 CPU
    hw_path_tags = []

    # 帧计数：先尝试硬解（若可用且启用），失败则自动降级再回退
    decoded: Optional[int] = None
    if current_hwaccel:
        # 使用自动探测结果传入 count_decoded_video_frames（已增强支持自动硬解）
        decoded = count_decoded_video_frames(path, mode=count_mode, use_hwaccel=True)
        if decoded is not None and decoded > 0:
            hw_path_tags.append("nvdec_direct")
            report["hwaccel_path"] = "direct"
        else:
            # 第一次失败：降级（降低 GPU 并发数再重试）
            fallback_stage = 1
            # 降级：若原并发数 > 1，则降到 1 再重试硬解
            degraded_workers = max(1, (gpu_workers // 2)) if gpu_workers > 1 else 1
            if degraded_workers < gpu_workers:
                # 降级重试：传入 use_hwaccel=True 但降低并发（通过环境变量模拟，或直接重试）
                # 此处简化为：直接重试（降级的核心是降低 session 数；由于单文件不并发，
                # 降级主要体现为降低并发并重试硬解命令）
                decoded = count_decoded_video_frames(path, mode=count_mode, use_hwaccel=True)
                if decoded is not None and decoded > 0:
                    hw_path_tags.append("nvdec_degraded")
                    report["hwaccel_path"] = "degraded"
                    report["hwaccel_fallback"] = True
                    current_hwaccel = True  # 降级成功，仍算硬解
                else:
                    fallback_stage = 2
            else:
                fallback_stage = 2

    # 若硬解未成功（或未启用），回退到软解
    if decoded is None or decoded <= 0:
        if fallback_stage >= 2:
            # 回退 CPU 软解
            hw_path_tags.append("soft_fallback")
            report["hwaccel_path"] = "fallback_cpu"
            report["hwaccel_fallback"] = True
            # 强制软解：传入 use_hwaccel=False
            decoded = count_decoded_video_frames(path, mode=count_mode, use_hwaccel=False)
        elif not current_hwaccel:
            # 从未启用硬解：直接软解
            hw_path_tags.append("soft_direct")
            report["hwaccel_path"] = "soft_direct"
            decoded = count_decoded_video_frames(path, mode=count_mode, use_hwaccel=False)

    report["decoded_frames"] = decoded
    if decoded is None:
        report["reason"] = "frame_count_unavailable"
        # 补充硬解回退信息
        report["hwaccel_fallback_detail"] = f"attempted_hwaccel={use_hwaccel_final}, " \
            f"gpu_available={gpu_available}, max_sessions={max_sessions}, " \
            f"fallback_stage={fallback_stage}, tags={','.join(hw_path_tags)}"
        return False, report

    if expected_frames is not None:
        report["expected_frames"] = int(expected_frames)
        if decoded != int(expected_frames):
            report["reason"] = "decoded_frame_mismatch"
            report["hwaccel_fallback_detail"] = f"tags={','.join(hw_path_tags)}, " \
                f"decoded={decoded}, expected={expected_frames}"
            return False, report

    # ── 解码错误检查：复用帧计数那一轮解码的证据 [P0-3] 单次解码 ──────────
    # 原实现此处再跑一次 `-v error -f null`（每个文件白付一次全量解码）。
    # 现直接取计数阶段 _probe_decode_error() 留下的 detail；仅当计数走了
    # 元数据捷径（根本没解码）时才补一次探测。
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        report["ffmpeg_error_check"] = "unavailable"
        report["reason"] = "ffmpeg_unavailable"
        return False, report

    final_use_hwaccel_for_check = current_hwaccel  # 经过降级后，若仍为硬解则保留
    detail = _get_probe_detail(path)
    if detail is None:
        detail = _probe_decode_error(path, ffmpeg_bin,
                                     use_hwaccel=final_use_hwaccel_for_check)
        if detail is not None:
            _key = _probe_cache_key(path)
            if _key is not None:
                _PROBE_DETAIL_CACHE[_key] = detail
                # [PROBE-CACHE-PERSIST] 成功证据一并落盘，恢复后免第二次解码
                if _probe_detail_persistable(detail):
                    _persist_probe_cache()
    if detail is None:
        report["reason"] = "decode_timeout"
        return False, report

    err_lines = list(detail["error_lines"])
    hw_failed = bool(detail["hw_failed"])
    report["decode_errors"] = len(err_lines)
    report["decode_stderr_tail"] = ("\n".join(err_lines)[-2000:]
                                    or (detail["stderr_tail"] or ""))
    report["returncode"] = detail["rc"]
    if hw_failed:
        # NVDEC 初始化失败属运行时环境问题（ffmpeg 已静默回退软解、帧数有效），
        # 不计入 decode_errors，只标记回退状态供审计。
        report["hwaccel_fallback"] = True
        report["hwaccel_path"] = "nvdec_internal_soft_fallback"
    report["hwaccel_path"] = report.get("hwaccel_path") or (
        "nvdec" if final_use_hwaccel_for_check else "soft")

    if detail["rc"] != 0 or err_lines:
        if final_use_hwaccel_for_check and not hw_failed:
            # 硬解路径确有错误 → 回退 CPU 软解复验一次，排除 NVDEC 实现差异
            fallback = _probe_decode_error(path, ffmpeg_bin, use_hwaccel=False)
            if fallback is not None and fallback["rc"] == 0 and not fallback["error_lines"]:
                report["decode_errors_hwaccel"] = len(err_lines)
                report["decode_errors_fallback"] = 0
                report["decode_errors"] = 0
                report["decode_stderr_tail"] = ""
                report["returncode"] = fallback["rc"]
                report["hwaccel_path"] = "degraded_fallback_cpu"
                report["hwaccel_fallback"] = True
                # 通过软解验证，不判失败
            else:
                # 降级也失败：记录并判失败
                report["decode_errors_degraded"] = (
                    len(fallback["error_lines"]) if fallback is not None else -1)
                report["reason"] = "decode_errors_fallback_failed"
                report["hwaccel_path"] = "degraded_failed"
                return False, report
        else:
            report["reason"] = "decode_errors"
            return False, report

    report["reason"] = "ok"
    # 补充最终的硬解状态标签（便于审计）
    if "hwaccel_path" not in report or not report.get("hwaccel_path"):
        report["hwaccel_path"] = ("nvdec" if final_use_hwaccel_for_check else "soft")
    return True, report


def get_video_content_duration(video_path: Union[str, Path]) -> Optional[float]:
    """[FIX-DUR-CONTENT] 返回"内容时长"= 视频流时长（不含容器起始偏移）。

    -c copy 剪出来的源常带非零起始 pts（实测 wws3e02_26s.mp4 首帧 pts=0.988s），
    而各处理阶段输出的时间轴都从 0 开始。若拿 ffprobe 的 format.duration
    （含起始偏移）当作"预期时长"，合并后就会凭空多出这段偏移，
    触发"输出时长偏差"告警（实测 1.05s = 0.988 起始偏移 + 3 段各少 1 帧的 0.063s）。
    """
    def _f(v):
        try:
            val = float(v)
            return val if val > 0 else None
        except (TypeError, ValueError):
            return None

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=duration,start_time',
             '-show_entries', 'format=duration',
             '-of', 'json', str(video_path)],
            capture_output=True, text=True, check=True, timeout=60)
        data = json.loads(result.stdout or '{}')
        streams = data.get('streams') or []
        fmt = data.get('format') or {}
        s0 = streams[0] if streams else {}

        stream_dur = _f(s0.get('duration'))
        if stream_dur:
            return stream_dur
        fmt_dur = _f(fmt.get('duration'))
        if fmt_dur:
            start = _f(s0.get('start_time')) or 0.0
            return max(0.0, fmt_dur - start)
    except Exception:
        pass
    return get_video_duration(str(video_path))


def verify_segment_output(output_path: Union[str, Path],
                          input_path: Union[str, Path],
                          scale: float = 1.0,
                          tolerance: int = 2) -> Tuple[bool, Dict[str, object]]:
    """[P4-FIX-GATE-TOL] 段级输出的解码级验收（分级容差）。

    期望值口径：expected = (源真实解码帧数 - 1) × scale + 1
      · 超分（scale=1.0）→ expected = 源帧数，帧数应严格守恒；
      · 插帧（scale=2.0）→ expected = 2n - 1（相邻帧之间插 n-1 帧）。

    为什么需要分级：验收门用 ffprobe -count_frames 的"真实解码帧数"推算期望值，
    而推理流水线用 ffmpeg rawvideo 读帧。VFR 源（时间戳空洞，例如源视频尾部丢帧
    留下的 Δ=2 帧间隔）在 CFR 读帧路径下会被复制帧补洞，产出**多于**期望的帧。
    那属于时间轴问题，内容并未缺失，不应与"丢帧"同等判死。

    分级规则（diff = decoded - expected）：
      diff < 0            → 失败（真丢帧）
      0 < diff ≤ tolerance → 警告通过（补帧/边界帧，仅损失极短时长）
      diff >  tolerance   → 失败
    其余原因（decode_errors / frame_count_unavailable 等）一律判失败。
    """
    import time as _time
    _t0 = _time.perf_counter()

    # [FIX-GATE-STRICT-COUNT] 本函数当前**无调用方**（历史遗留的段级验收封装）。
    # 一并改为同源严格口径：期望帧数与实测帧数都走全解码，避免日后被接上时
    # 继承「元数据期望 vs 全解码实测」在无 NVDEC 机器上的系统性假失败。
    #
    # 去留决定（2026-09-15，Plan/门禁与测试资产纳管清理_立项Prompt.md §1.4）：
    # **保留，并显式标注为「仅参考实现、无调用方」**。语义（段级输出解码级验收 +
    # 分级容差）已被 validate_decodable_video + 两个 processor 的内联验收取代，
    # 故不再接线；但删除的收益（少一个零调用方函数）小于其风险（本开发树非 git
    # 仓库，删掉后无法 A/B 回取），故不作删除。新增调用前请先确认口径与验收门同源。
    src_frames = count_decoded_video_frames(input_path, mode="decode")
    expected_frames = None
    if src_frames is not None and int(src_frames) > 0:
        expected_frames = int((int(src_frames) - 1) * float(scale) + 1)

    report: Dict[str, object] = {"src_frames": src_frames,
                                 "expected_frames": expected_frames,
                                 "tolerated": False}
    dec_ok, dec_report = validate_decodable_video(output_path, expected_frames,
                                                  count_mode="decode")
    report.update(dec_report)

    if dec_ok:
        _el = _time.perf_counter() - _t0
        print(f"   ✅ 解码级验收通过: decoded={dec_report.get('decoded_frames')} "
              f"expected={expected_frames} (源帧={src_frames})"
              f"（耗时：{int(_el) // 60}m:{int(_el) % 60:02d}s）")
        return True, report

    decoded = dec_report.get('decoded_frames')
    reason = dec_report.get('reason')

    if (reason == 'decoded_frame_mismatch' and expected_frames is not None
            and isinstance(decoded, int)):
        diff = decoded - expected_frames
        if 0 < diff <= int(tolerance):
            report['tolerated'] = True
            print(f"   ⚠️  解码级验收容忍: decoded={decoded} "
                  f"expected={expected_frames}（多 {diff} 帧 ≤ 容差 {tolerance}）"
                  f"——疑似源时间轴空洞导致的补帧，非丢帧；"
                  f"建议对源执行时间轴归一化")
            return True, report

    print("   ❌ 解码级验收失败: "
          f"decoded={decoded} "
          f"expected={dec_report.get('expected_frames', expected_frames)} "
          f"reason={reason}")
    tail = str(dec_report.get('decode_stderr_tail', '')).strip()
    if tail:
        print(f"   ↳ {tail[-500:]}")
    return False, report


def verify_video_integrity(video_path: str) -> bool:
    """
    验证视频文件完整性
    
    Args:
        video_path: 视频路径
    
    Returns:
        是否完整
    """
    if not os.path.exists(video_path):
        return False
    
    if os.path.getsize(video_path) < 1024:  # 小于1KB
        return False
    
    try:
        # 尝试打开视频
        cap = cv2.VideoCapture(video_path)
        ret = cap.isOpened()
        
        if ret:
            # 尝试读取第一帧
            ret, frame = cap.read()
        
        cap.release()
        return ret
    except Exception:
        # [P0-FIX-BARE-EXCEPT] 原裸 except 会吞掉 KeyboardInterrupt/SystemExit，
        # 中断瞬间命中此路径会被静默吞掉。
        return False


def _fingerprint_matches(sidecar_path: str, current: dict) -> bool:
    """[P1-FIX-SPLIT-FINGERPRINT] 比对分段目录的源内容指纹侧车与当前源视频。

    以名字代内容的等价假设是该模块历史缺陷根源之一；侧车记录
    (source, size, mtime_ns, segment_duration)，任一不符即视为陈旧。
    """
    if not current:
        return False
    try:
        if not os.path.exists(sidecar_path):
            return False
        with open(sidecar_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return False
    return (saved.get("source") == current.get("source")
            and saved.get("size") == current.get("size")
            and saved.get("mtime_ns") == current.get("mtime_ns")
            and saved.get("segment_duration") == current.get("segment_duration"))


def _fingerprint_segment_count(sidecar_path: str, default: int) -> int:
    """取指纹侧车里记录的段数（尾部碎片段合并后段数会少于按 duration 推算的值）。"""
    try:
        if os.path.exists(sidecar_path):
            with open(sidecar_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cnt = saved.get("segment_count")
            if isinstance(cnt, int) and cnt > 0:
                return cnt
    except Exception:
        pass
    return default


def _concat_copy(inputs: List[str], output: str) -> bool:
    """用 concat demuxer + -c copy 无损拼接（不重编码）。"""
    list_path = str(output) + ".concat.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for p in inputs:
                f.write(f"file '{Path(p).resolve().as_posix()}'\n")
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'concat', '-safe', '0', '-i', list_path,
               '-c', 'copy', '-fflags', '+genpts',
               '-avoid_negative_ts', 'make_zero', output]
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
        return os.path.exists(output) and verify_video_integrity(output)
    except Exception as e:
        print(f"   ⚠️  拼接失败: {e}")
        return False
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def detect_timestamp_anomaly(video_path: Union[str, Path],
                              gap_tolerance: float = 1.5,
                              start_tolerance: float = 0.05
                              ) -> Dict[str, object]:
    """[P3-FIX-NORM] 检测源时间轴是否异常（只 demux 不解码，开销很低）。

    两类问题都会让下游"帧数守恒"类校验失真：
      1. 非零起始偏移：源首帧 pts ≠ 0（-c copy 剪辑残留）；
      2. 时间戳空洞（VFR）：相邻帧 pts 间隔远大于中位数，通常是源丢帧留下的。

    Returns: dict(ok, anomaly, start_time, frame_count, gaps, detail)
      ok=False 表示无法判定（ffprobe 失败/取不到 pts），此时不应据此阻断流程。
    """
    report: Dict[str, object] = {"ok": False, "anomaly": False,
                                 "start_time": None, "frame_count": 0,
                                 "gaps": [], "detail": ""}
    try:
        proc = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'frame=pts', '-of', 'csv=p=0', str(video_path)],
            capture_output=True, text=True, check=True, timeout=1800)
        pts = []
        for line in (proc.stdout or '').splitlines():
            line = line.strip().rstrip(',')
            if not line or line == 'N/A':
                continue
            try:
                pts.append(int(float(line)))
            except ValueError:
                continue
        if len(pts) < 2:
            return report
    except Exception:
        return report

    deltas = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    if not deltas:
        return report

    ordered = sorted(deltas)
    median = ordered[len(ordered) // 2]
    gaps = [d for d in deltas if median > 0 and d > median * gap_tolerance]

    start_time = None
    try:
        _p = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=start_time', '-of', 'csv=p=0',
             str(video_path)],
            capture_output=True, text=True, check=True, timeout=60)
        _v = (_p.stdout or '').strip().split(',')[0]
        if _v not in ('', 'N/A'):
            start_time = float(_v)
    except Exception:
        pass

    report.update({"ok": True, "frame_count": len(pts),
                   "start_time": start_time, "gaps": gaps})

    reasons = []
    if start_time is not None and abs(start_time) > start_tolerance:
        reasons.append(f"起始偏移 {start_time:.3f}s")
    if gaps:
        reasons.append(f"{len(gaps)} 处时间戳空洞（间隔 {min(gaps)}~{max(gaps)}"
                       f"，正常 {median}）")
    if reasons:
        report["anomaly"] = True
        report["detail"] = "；".join(reasons)
    return report


def normalize_video_timeline(input_video: Union[str, Path],
                             output_video: Union[str, Path],
                             encoder: str = 'libx264',
                             crf: Optional[int] = None,
                             preset: str = 'veryfast',
                             ffmpeg_bin: str = 'ffmpeg',
                             # [QUALITY-UNIFY] 环节① 质量输入（与合并/分段同语义）
                             cq: Optional[int] = None,
                             crf_ref: Optional[int] = None,
                             cq_ref: Optional[int] = None) -> bool:
    """[P3-FIX-NORM] 把源时间轴归一化为均匀 CFR：按帧号重编号 pts。

    必须"重编号 + passthrough"组合，实测三种写法只有这一种正确：
      · 只加 -fps_mode passthrough      → 空洞原样保留（无效，仍有 Δ=1000 跳变）
      · -fps_mode cfr -r 24000/1001     → 反向补帧（602 帧 → 628 帧，时长被拉长）
      · setpts=N/FR/TB + passthrough    → 帧数不变、start_time=0、间隔均匀 ✅

    [QUALITY-UNIFY] 质量参数经 quality_map.resolve_quality 换算到生效编码器量纲
    （软编 -crf / 硬编 -cq:v + -b:v 0 / librav1e -qp）；四键均未给时按 libx264
    CRF 基准 21 换算（此前硬编码 crf=18，与全局基线不一致）。

    Note: -vf 意味着一路软编解码，代价是一次重编码；仅在源确实异常时使用。
    """
    _q_args, _q_note = _resolve_quality_args(
        encoder, crf=crf, cq=cq, crf_ref=crf_ref, cq_ref=cq_ref, preset=preset)
    logger.info(f"[normalize] 质量参数解析: {_q_note} → {' '.join(_q_args)}")
    cmd = [ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
           '-i', str(input_video),
           '-vf', 'setpts=N/FR/TB',      # 按帧序号重编号 → 等间隔
           '-fps_mode', 'passthrough',   # 禁止 CFR 逻辑二次插帧/丢帧
           '-c:v', encoder] + _q_args + [
           '-c:a', 'copy',
           str(output_video)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=86400)
    except Exception as e:
        print(f"   ❌ 时间轴归一化失败: {e}")
        return False
    if not os.path.exists(str(output_video)):
        return False
    return verify_video_integrity(str(output_video))


def merge_trailing_fragment(segment_files: List[str],
                            segment_duration: float = 30.0,
                            min_duration: float = None) -> List[str]:
    """[P2-FIX-FRAG] 合并过短的尾部碎片段（concat + -c copy，不重编码）。

    segment muxer 只能在关键帧处切分，目标 9s 常被拉成 10.010/14.097/1.001
    ——末段只剩 1 秒。短尾段有三个副作用：
      1) 源的时间轴缺陷（时间戳空洞）集中在末尾，短尾段最容易被"一击即中"；
      2) AUTO-TUNE / RETUNE / GPU-MONITOR 用几十帧样本外推，调参噪声大；
      3) 每个分段固有少 1 帧（2n-1），段数越多累计时长损失越大。
    末段短于阈值时并入前一段；仅剩一段时不处理。
    """
    if len(segment_files) < 2:
        return segment_files

    threshold = float(min_duration) if min_duration else max(
        2.0, float(segment_duration) * 0.25)
    tail = segment_files[-1]
    tail_dur = get_video_duration(tail)
    if tail_dur is None or tail_dur >= threshold:
        return segment_files

    prev = segment_files[-2]
    print(f"🧩 末段 {Path(tail).name} 仅 {format_time(tail_dur)} "
          f"(< 阈值 {format_time(threshold)})，并入前一段 {Path(prev).name} …")

    tmp_out = str(prev) + ".merged.mp4"
    if not _concat_copy([prev, tail], tmp_out):
        print("   ⚠️  末段合并失败，保留原分段")
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        return segment_files

    merged_dur = get_video_duration(tmp_out)
    if not merged_dur:
        print("   ⚠️  合并产物时长异常，保留原分段")
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        return segment_files

    prev_dur = get_video_duration(prev) or 0.0
    try:
        os.replace(tmp_out, prev)
        os.remove(tail)
    except OSError as e:
        print(f"   ⚠️  末段合并落盘失败: {e}")
        return segment_files

    print(f"   ✅ 已合并: {Path(prev).name} = {format_time(prev_dur)} + "
          f"{format_time(tail_dur)} → {format_time(merged_dur)}"
          f"（分段数 {len(segment_files)} → {len(segment_files) - 1}）")
    return segment_files[:-1]


def split_video_by_time(input_video: str, output_dir: str,
                        segment_duration: int = 30,
                        reuse_existing: bool = True) -> List[str]:
    """
    按时间分割视频

    Args:
        input_video: 输入视频路径
        output_dir: 输出目录
        segment_duration: 每段时长（秒）
        reuse_existing: 若已有有效分段文件则复用，跳过 ffmpeg 分割

    Returns:
        分段文件列表
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取视频时长
    duration = get_video_duration(input_video)
    if duration is None:
        print("❌ 无法获取视频时长")
        return []

    print(f"📹 视频总时长: {format_time(duration)}")

    # [P1-FIX-SPLIT-FINGERPRINT] 源内容指纹侧车：(size, mtime_ns) 写入
    # .segments_fingerprint.json。复用判定先比对指纹——同名不同内容的源视频
    # 不再被陈旧 segments 目录蒙混；指纹不符时强制重新分割。
    _fp_path = os.path.join(output_dir, ".segments_fingerprint.json")
    def _current_fingerprint() -> dict:
        try:
            st = os.stat(input_video)
            return {"source": os.path.abspath(input_video),
                    "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                    "segment_duration": segment_duration}
        except OSError:
            return {}

    # 计算分段数（[P1-FIX-SEG-COUNT] 浮点容差：60.0000001s 这类元数据不再多算一段）
    _eps = max(segment_duration, 1.0) * 1e-6
    num_segments = int(duration / segment_duration) + (
        1 if (duration % segment_duration) > _eps else 0)

    def _write_fingerprint(segment_count: int = 1):
        # [P2-FIX-FRAG] 记录实产段数：尾部碎片段合并后段数会少于按 duration 推算值，
        # 否则下次运行会因"段数不符"误判指纹过期而白白重切。
        try:
            payload = dict(_current_fingerprint())
            payload["segment_count"] = int(segment_count)
            with open(_fp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    if num_segments <= 1:
        print(f"⏭️  视频时长 < {segment_duration}秒，无需分段")
        segment_file = os.path.join(output_dir, "segment_000.mp4")
        if reuse_existing and os.path.exists(segment_file) \
                and verify_video_integrity(segment_file) \
                and _fingerprint_matches(_fp_path, _current_fingerprint()):
            print("♻️  复用已有分段文件，跳过重复复制")
            return [segment_file]
        # [META-KEEP] 整片复制同样要落 sidecar（合并阶段统一从这里取原片元数据）
        write_source_meta_sidecar(input_video, output_dir)
        shutil.copy2(input_video, segment_file)
        _write_fingerprint()
        return [segment_file]

    # 检查是否可复用已有分段
    if reuse_existing:
        existing = sorted(Path(output_dir).glob("segment_*.mp4"))
        # [P2-FIX-FRAG] 段数以指纹侧车记录为准（尾部碎片合并后会变少）
        _expected_count = _fingerprint_segment_count(_fp_path, num_segments)
        if len(existing) == _expected_count and \
                _fingerprint_matches(_fp_path, _current_fingerprint()):
            valid_files = []
            all_valid = True
            for f in existing:
                fpath = str(f)
                if verify_video_integrity(fpath):
                    valid_files.append(fpath)
                else:
                    all_valid = False
                    break
            if all_valid:
                print("♻️  复用已有分段文件，跳过重复分割")
                for i, f in enumerate(valid_files):
                    seg_dur = get_video_duration(f)
                    print(f"✅ 分段 {i+1}/{num_segments}: {format_time(seg_dur)}")
                return valid_files
            else:
                print("⚠️  已有分段文件不完整，将重新分割")
        elif len(existing) == num_segments:
            print("⚠️  源视频内容/参数与既有分段不一致（指纹侧车不匹配），将重新分割")

    print(f"🔪 分割为 {num_segments} 段，只 copy 视频流，不重新编码...")

    # [META-KEEP] 切片阶段固化原片元数据：concat demuxer 的合成上下文不继承
    # 任何分段的容器级元数据，合并阶段 -map_metadata 0 指向的是列表文件（空的），
    # 所以必须在这里落盘 sidecar，供 merge_videos_by_codec 注入。
    _src_meta = probe_full_metadata(input_video)
    write_source_meta_sidecar(input_video, output_dir)

    segment_files = []
    segment_pattern = os.path.join(output_dir, "segment_%03d.mp4")

    # 使用FFmpeg的segment muxer
    cmd = [
        'ffmpeg', '-fflags', '+genpts',
        # -fflags +genpts 是输入选项(须在 -i 前)：老 AVI/mpeg4 常缺 PTS，
        # segment muxer 依赖包时间戳切分，缺 PTS 会导致整段输出不切分
        '-noautorotate',                # [META-KEEP] 不烘焙旋转，让标签沿链路传递
        '-i', input_video,
        '-c', 'copy',  # 复制流，不重新编码
        '-map', f"0:{_src_meta['derived']['video_index']}"
                if _src_meta else '0:v',   # 绝对索引，天然避开 mp4 封面轨
        '-map_metadata', '0',           # [META-KEEP] 分段自带原片容器 tags
        '-map_chapters', '0',
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-reset_timestamps', '1',
        '-y',
        segment_pattern
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)

        # 验证分段
        for i in range(num_segments):
            segment_file = os.path.join(output_dir, f"segment_{i:03d}.mp4")
            if os.path.exists(segment_file) and verify_video_integrity(segment_file):
                segment_files.append(segment_file)
                seg_dur = get_video_duration(segment_file)
                print(f"✅ 分段 {i+1}/{num_segments}: {format_time(seg_dur)}")

        # [P1-FIX-SEG-COUNT] 实产段数与预期不符（关键帧对齐/容器怪癖）时显式告警，
        # 不再静默返回短列表导致合并产物比源视频短。
        if len(segment_files) != num_segments:
            print(f"⚠️  [split] 预期 {num_segments} 段，实际产出 {len(segment_files)} 段"
                  f"——后续合并产物时长可能短于源视频，请检查源文件时间戳")

        # [P2-FIX-FRAG] 关键帧切分常留下 1 秒级的尾部碎片段，并入前一段
        segment_files = merge_trailing_fragment(segment_files, segment_duration)

        _write_fingerprint(len(segment_files))
        print(f"✅ 成功分割为 {len(segment_files)} 个有效片段")
        return segment_files

    except subprocess.CalledProcessError as e:
        print(f"❌ 分割失败: {e}")
        return []


def merge_videos(
    video_files: List[str],
    output_path: str,
    audio_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
    timeout: Optional[int] = None
) -> str:
    """
    合并视频文件（优化版）

    Args:
        video_files: 视频文件路径列表
        output_path: 输出文件路径
        audio_path: 可选的外部音频文件路径
        config: 编码配置字典（当不为 None 时启用重新编码）
                - codec: 视频编码器 (默认 libx264)
                - preset: 预设 (默认 medium)
                - crf: CRF 值 (默认 18)
                - pix_fmt: 像素格式 (默认 yuv420p)
                - audio_codec: 音频编码器 (默认 aac，若为 'copy' 则复制)
                - audio_bitrate: 音频码率 (默认 192k)
                - map_video: 视频流映射 (默认 '0:v:0')
                - map_audio: 外部音频流映射 (默认 '1:a:0')
                - ffmpeg_args: 额外 ffmpeg 参数列表
        overwrite: 是否覆盖已存在的输出文件
        timeout: ffmpeg 进程超时时间（秒）

    Returns:
        输出文件路径（成功时）

    Raises:
        ValueError: 输入参数无效
        FFmpegError: ffmpeg 执行失败
    """
    # ---------- 输入验证 ----------
    if not video_files:
        raise ValueError("视频文件列表不能为空")
    
    missing = [f for f in video_files if not os.path.exists(f)]
    if missing:
        raise ValueError(f"以下视频文件不存在: {missing}")
    
    # 音频文件存在性检查，不存在则忽略并警告
    if audio_path and not os.path.exists(audio_path):
        logger.warning(f"音频文件不存在，将忽略: {audio_path}")
        audio_path = None
    
    # ---------- 配置参数处理 ----------
    # 编码模式: config 不为 None 则启用重新编码，否则复制流
    encode_mode = config is not None
    
    # 默认编码参数（仅在 encode_mode=True 时使用）
    # [QUALITY-UNIFY] 质量四键互斥、基准轴优先；均 None 时按 libx264 CRF 基准 21 换算。
    default_config = {
        'codec': 'libx264',
        'preset': 'medium',
        'crf': None,
        'cq': None,
        'crf_ref': None,
        'cq_ref': None,
        'pix_fmt': 'yuv420p',
        'audio_codec': 'aac',
        'audio_bitrate': '192k',
        'map_video': '0:v:0',      # 视频列表的第一个视频流
        'map_audio': '1:a:0',      # 外部音频的第一个音频流
        'ffmpeg_args': []
    }
    if encode_mode:
        # 合并用户配置，缺失键使用默认值
        config = {**default_config, **(config or {})}
    else:
        config = {}
    
    # ---------- 创建临时文件列表 ----------
    # 使用临时文件避免命名冲突，自动清理
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', encoding='utf-8', delete=False
    ) as tmp_f:
        list_file = tmp_f.name
        for video in video_files:
            # 使用绝对路径，避免相对路径问题
            abs_path = os.path.abspath(video).replace("'", "'\\''")  # 转义单引号
            tmp_f.write(f"file '{abs_path}'\n")
    
    # ---------- 构建 ffmpeg 命令 ----------
    cmd = ['ffmpeg']
    
    # 输入：concat 协议文件列表
    cmd.extend(['-f', 'concat', '-safe', '0', '-i', list_file])
    
    # 如果有外部音频，添加第二个输入
    if audio_path:
        cmd.extend(['-i', audio_path])
    
    # 编码/流复制参数
    if encode_mode:
        # [QUALITY-UNIFY] 视频编码参数：按生效编码器换算（软编 -crf / 硬编 -cq:v / librav1e -qp）
        _q_args, _q_note = _resolve_quality_args(
            config['codec'],
            crf=config.get('crf'), cq=config.get('cq'),
            crf_ref=config.get('crf_ref'), cq_ref=config.get('cq_ref'),
            preset=config.get('preset', 'medium'))
        logger.info(f"[merge_videos] 质量参数解析: {_q_note} → {' '.join(_q_args)}")
        cmd.extend(['-c:v', config['codec']] + _q_args + ['-pix_fmt', config['pix_fmt']])
        
        # 音频编码参数
        if audio_path:
            if config['audio_codec'] == 'copy':
                cmd.extend(['-c:a', 'copy'])
            else:
                cmd.extend([
                    '-c:a', config['audio_codec'],
                    '-b:a', config['audio_bitrate']
                ])
        else:
            # 无外部音频：复制原视频中的音频流（如果存在）
            cmd.extend(['-c:a', 'copy'])
    else:
        # 复制流模式
        cmd.extend(['-c', 'copy'])
        # 注意：不加 -map，让 ffmpeg 自动选择默认流
    
    # 流映射（仅当有外部音频时需要指定）
    if audio_path:
        # 默认映射：视频第一个视频流，外部音频第一个音频流
        cmd.extend([
            '-map', config.get('map_video', '0:v:0'),
            '-map', config.get('map_audio', '1:a:0')
        ])
    # 无外部音频时不添加 -map，自动选择流
    
    # 额外 ffmpeg 参数
    if encode_mode and config.get('ffmpeg_args'):
        cmd.extend(config['ffmpeg_args'])
    
    # 输出覆盖及路径
    if overwrite:
        cmd.append('-y')
    cmd.append(output_path)
    
    logger.debug(f"FFmpeg 命令: {' '.join(cmd)}")
    
    # ---------- 执行并处理结果 ----------
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout
        )
        
        if result.returncode != 0:
            raise FFmpegError(
                f"FFmpeg 失败，返回码 {result.returncode}\n"
                f"stderr: {result.stderr}"
            )
        
        # 验证输出文件
        if not os.path.exists(output_path):
            raise FFmpegError(f"输出文件未生成: {output_path}")
        if os.path.getsize(output_path) == 0:
            raise FFmpegError(f"输出文件为空: {output_path}")
        
        logger.info(f"视频合并成功: {output_path}")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise FFmpegError("FFmpeg 进程超时")
    except Exception as e:
        raise FFmpegError(f"合并过程中发生错误: {e}") from e
    finally:
        # 确保临时文件被删除
        try:
            os.unlink(list_file)
        except Exception as e:
            logger.warning(f"删除临时文件失败 {list_file}: {e}")

def get_video_codec(video_file: Union[str, Path]) -> str:
    """
    使用 ffprobe 提取视频文件中第一个视频流的编码格式。

    Args:
        video_file: 视频文件的路径，支持字符串或 pathlib.Path 对象。

    Returns:
        视频编码格式的小写字符串，例如 'h264', 'hevc', 'vp9' 等。

    Raises:
        ValueError: 无法获取视频流或编码信息（如无视频流、ffprobe 输出异常等）。
        subprocess.CalledProcessError: ffprobe 命令执行失败（非零返回码）。
        json.JSONDecodeError: ffprobe 输出的 JSON 格式不正确。
    """
    # 将 Path 对象转换为字符串，并检查文件是否存在
    video_path = str(video_file)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_file}")

    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',       # 仅选择第一个视频流
        '-show_entries', 'stream=codec_name',
        '-of', 'json',
        video_path                      # 使用转换后的字符串路径
    ]

    # 执行 ffprobe 命令并解析 JSON
    probe_out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
    data = json.loads(probe_out)

    # 提取编码名称，并转换为小写
    try:
        codec = data['streams'][0]['codec_name'].lower()
    except (KeyError, IndexError) as e:
        raise ValueError(f"无法从 ffprobe 输出中获取视频编码信息: {e}")

    return codec


def probe_color_metadata(video_file: Union[str, Path]) -> Optional[Dict[str, str]]:
    """
    使用 ffprobe 读取视频第一个视频流的色彩元数据。

    Args:
        video_file: 视频文件路径，支持字符串或 pathlib.Path 对象。

    Returns:
        字典 {color_range, color_space, color_primaries, color_transfer}，
        各项为 ffprobe 输出字符串（可能为 'unknown'）。
        无法探测（文件缺失/无视频流/ffprobe 异常）时返回 None。
    """
    m = probe_full_metadata(video_file)
    if not m:
        return None
    v = m['video']
    return {k: v.get(k, 'unknown') for k in
            ('color_range', 'color_space', 'color_primaries', 'color_transfer')}


def build_color_args(video_file: Union[str, Path],
                     meta: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    构造 ffmpeg 输出端色彩参数列表。

    color_range 固定为 auto：源有值取源值，unknown 时取 tv
    （unknown 时 ffmpeg 本就按 tv 解释并解码，取 tv 可与源逐像素保持一致；
    只有 pix_fmt 为 yuvj* 才是真的 full range）。

    Args:
        video_file: 源视频路径。
        meta: 可选的 probe_full_metadata 结果，传入可避免重复 ffprobe。

    Returns:
        ffmpeg 参数列表；探测失败时返回空列表（让编码器自行决定，不瞎猜）。
    """
    m = meta if meta is not None else probe_full_metadata(video_file)
    if not m:
        return []

    v = m['video']
    d = m['derived']

    def _val(key: str) -> Optional[str]:
        x = (v.get(key) or '').lower()
        return None if x in ('', 'unknown', 'unspecified', 'n/a') else x

    space = _val('color_space')
    prim = _val('color_primaries')
    trc = _val('color_transfer')
    rng = _val('color_range')

    if rng is None:
        # auto：只有 yuvj* 才是真的 full range；其余一律按 tv（limited）
        rng = 'pc' if d['pix_fmt'].lower().startswith('yuvj') else 'tv'

    if space is None or prim is None or trc is None:
        if d['src_bits'] >= 10 and (d['width'] >= 1920 or d['height'] >= 1080):
            guess = ('bt2020nc', 'bt2020', 'bt709')
        elif d['height'] >= 720:
            guess = ('bt709', 'bt709', 'bt709')
        elif d['src_bits'] >= 10:
            # 高位深的小分辨率内容基本不存在标清广播电视色彩，按 bt709 更合理
            guess = ('bt709', 'bt709', 'bt709')
        else:
            fps = _parse_rate(v.get('avg_frame_rate') or v.get('r_frame_rate'))
            is_pal = fps is not None and (abs(fps - 25) < 0.3 or abs(fps - 50) < 0.3)
            guess = ('bt470bg', 'bt470bg', 'bt470bg') if is_pal else \
                    ('smpte170m', 'smpte170m', 'smpte170m')
        space = space or guess[0]
        prim = prim or guess[1]
        trc = trc or guess[2]

    # 输出端 -color_trc 不接受 bt470bg/bt470m，需换成 libavutil 规范名
    trc = _TRC_OUTPUT_NAMES.get(trc, trc)

    return [
        '-colorspace', space,
        '-color_primaries', prim,
        '-color_trc', trc,
        '-color_range', rng,
    ]


def _setparams_from_color_args(extra_args: Sequence[str]) -> Optional[str]:
    """
    从 build_color_args 生成的色彩参数列表中提取取值，构造 setparams 滤镜字符串。

    原因：libx264 等编码器对输出端 -color_primaries/-color_trc 参数不写入 VUI，
    重新编码时需用 setparams 滤镜显式注入帧级色彩属性（copy 路径则靠输出端
    参数写入 MP4 colr box，无需滤镜）。

    Args:
        extra_args: ffmpeg 输出端参数列表（含 -colorspace/-color_primaries 等）。

    Returns:
        setparams 滤镜字符串（如 'setparams=colorspace=bt709:color_primaries=bt709:...'），
        无色彩参数时返回 None。
    """
    _color_map = {
        '-colorspace': 'colorspace',
        '-color_primaries': 'color_primaries',
        '-color_trc': 'color_trc',
        '-color_range': 'range',
    }
    vals: Dict[str, str] = {}
    i = 0
    while i < len(extra_args) - 1:
        key = extra_args[i]
        if key in _color_map:
            value = extra_args[i + 1]
            # 滤镜端只认别名（bt470bg），不认规范名（gamma28）
            vals[_color_map[key]] = _TRC_FILTER_NAMES.get(value, value)
            i += 2
        else:
            i += 1
    if not vals:
        return None
    return 'setparams=' + ':'.join(f'{k}={v}' for k, v in vals.items())


# =============================================================================
# 合并前分段 timescale 归一化                        [FIX-C / TIMESCALE-NORM]
# =============================================================================
# 注：本段逻辑旧名 "extradata 归一化"，源自早期用 -bsf:v dump_extra 的实现。
# 该 BSF 默认按 Annex B（起始码）方式把 extradata 注入 packet，适用于
# MPEG-TS/裸流；若直接用于 MP4/AVCC，会因格式不匹配（起始码 vs 长度前缀）写出畸形 packet。
# 现实际动作已改为 timescale 归一化（零损耗 remux），名称一并更新，以反映真实职责，避免误导。

#: copy 合并统一使用的目标 MP4 时间基（90 kHz）
MP4_TARGET_TIMESCALE = 90000


def _get_stream_timescale(path: Union[str, Path]) -> Optional[int]:
    """返回视频流 time_base 的分母（即 MP4 timescale），失败返回 None。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=time_base", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        val = (r.stdout or "").strip().split(",")[0].strip()
        if "/" in val:
            val = val.split("/", 1)[1].strip()
        return int(val) if val.isdigit() else None
    except Exception:
        return None


def _detect_seg_codec(seg_path: Union[str, Path]) -> str:
    """检测分段视频编码器，返回小写规范名：h264 / hevc / vp9 / av1 / ""。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(seg_path)],
            capture_output=True, text=True, timeout=30)
        raw = (r.stdout or "").strip().lower() if r.returncode == 0 else ""
    except Exception:
        raw = ""
    if "264" in raw or raw == "avc":
        return "h264"
    if "265" in raw or "hevc" in raw:
        return "hevc"
    return raw


def normalize_segments_timescale(
    seg_paths: Sequence[Union[str, Path]],
    codec_hint: str = "",
    target_timescale: int = MP4_TARGET_TIMESCALE,
) -> list:
    """[FIX-C] copy 合并前对所有分段执行零损耗 remux，统一 MP4 时间基。

    修复独立 ffmpeg 子进程编码时各分段 timescale 微差导致的
    concat demuxer AVERROR_EXIT(254) 问题。

    处理步骤（仅对 H.264 / H.265）：
      1. -video_track_timescale <target> — 统一 MP4 时间基（默认 90 kHz）

    不使用 -bsf:v dump_extra（详见本段顶部说明）。
    其他编码格式（VP9 / AV1 / ProRes 等）无此 timescale 问题，直接跳过。

    [SKIP-IF-ALREADY] 分段当前 timescale 已等于目标值时跳过 remux。
    链路上同一批分段可能被多次合并（processor 层合并 → main 层最终合并），
    无此判定会对已是目标值的文件反复重写出相同内容，纯属浪费 I/O。

    Args:
        seg_paths       : 待处理的分段路径列表（Path 或 str）
        codec_hint      : 已知编解码器名称（可省略，会自动检测）
        target_timescale: 目标时间基，默认 90000

    Returns:
        归一化后的路径列表（成功则原地替换，失败/无需处理则保留原路径）
    """
    from pathlib import Path as _Path

    if not seg_paths:
        return list(seg_paths)

    # ── 确定编解码器 ──────────────────────────────────────────────────────────
    codec = (codec_hint or "").lower()
    if "264" in codec or codec == "avc":
        codec = "h264"
    elif "265" in codec or "hevc" in codec:
        codec = "hevc"
    else:
        codec = _detect_seg_codec(str(seg_paths[0]))

    # ── 只处理 H.264 / H.265 ─────────────────────────────────────────────────
    if codec not in ("h264", "hevc"):
        if codec:
            print(f"   ℹ️  [timescale] 编解码器 '{codec}' 无需 timescale 归一化，跳过")
        else:
            print("   ⚠️  [timescale] 无法检测编解码器，跳过归一化"
                  "（若合并失败请检查分段文件）")
        return list(seg_paths)

    def _has_decodable_video(path: _Path) -> bool:
        """验证文件中存在视频流且含可解码帧（duration > 0）。

        仅检查容器元数据"流存在"不够——某些写法会写入 duration=0 的空
        视频轨（元数据存在但无有效帧）；ffprobe 能看到流，但 ffmpeg concat
        读不出任何视频帧，最终输出只有音频。必须同时验证 duration。
        """
        try:
            r2 = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_type,duration", "-of", "csv=p=0",
                 str(path)],
                capture_output=True, text=True, timeout=30)
            if r2.returncode != 0 or not r2.stdout.strip():
                return False
            line = r2.stdout.strip().split("\n")[0]
            parts = line.split(",")
            if not parts or "video" not in parts[0].lower():
                return False
            if len(parts) >= 2:
                dur_str = parts[1].strip()
                if dur_str in ("N/A", "0", "0.000000", ""):
                    return False
            return True
        except Exception:
            return False

    normalized: list = []
    ok_count = fail_count = skip_count = 0

    for seg in seg_paths:
        seg_p = _Path(str(seg))

        # [SKIP-IF-ALREADY] 已是目标 timescale → 无需 remux
        if _get_stream_timescale(seg_p) == target_timescale:
            skip_count += 1
            normalized.append(seg)
            continue

        tmp_p = seg_p.with_suffix(f".tsnorm_tmp{seg_p.suffix or '.mp4'}")

        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-noautorotate",                     # [META-KEEP] remux 不烘焙旋转
                 "-i", str(seg_p),
                 "-map", "0:v:0", "-map", "0:a?",
                 "-c:v", "copy", "-c:a", "copy",
                 "-map_metadata", "0",                # [META-KEEP] remux 别丢容器 tags
                 "-map_chapters", "0",
                 "-video_track_timescale", str(target_timescale),
                 str(tmp_p)],
                capture_output=True, text=True, timeout=300)

            # 三重校验：rc=0 + 文件非空 + 视频帧可解码。
            # 不能只检查流元数据是否存在（空视频轨可通过元数据检查但
            # duration=0，无法播放）。
            tmp_ok = (
                r.returncode == 0
                and tmp_p.exists()
                and tmp_p.stat().st_size > 0
                and _has_decodable_video(tmp_p)
            )

            if tmp_ok:
                os.replace(str(tmp_p), str(seg_p))   # 原子替换
                ok_count += 1
                normalized.append(str(seg_p))
            else:
                _reason = (f"rc={r.returncode}" if r.returncode != 0
                           else "视频帧校验失败（duration=0 或流不存在）")
                _stderr = (r.stderr or "").strip().splitlines()[-1:] or [""]
                print(f"   ⚠️  [timescale] {seg_p.name} 归一化失败"
                      f"（{_reason}），使用原始文件")
                print(f"       ffmpeg: {_stderr[0]}")
                if tmp_p.exists():
                    tmp_p.unlink()
                normalized.append(seg)
                fail_count += 1
        except Exception as exc:
            print(f"   ⚠️  [timescale] {seg_p.name} 归一化异常: {exc}，使用原始文件")
            if tmp_p.exists():
                try:
                    tmp_p.unlink()
                except OSError:
                    pass
            normalized.append(seg)
            fail_count += 1

    _total = len(seg_paths)
    print(f"   🔧 [timescale] 对 {_total} 个分段归一化（codec={codec}, "
          f"timescale={target_timescale}）...")
    if fail_count == 0:
        print(f"   ✅ [timescale] 完成：{ok_count} 归一化 / {skip_count} 已达标跳过")
    else:
        print(f"   ⚠️  [timescale] {ok_count}/{_total} 成功（{skip_count} 已达标，"
              f"{fail_count} 失败，已保留原始文件，合并可能仍会失败）")

    return normalized


def merge_videos_by_codec(
    file_list: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    audio_path: Optional[Union[str, Path]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    check_consistency: bool = True,
    force_reencode: bool = False,
    reencode: Optional[bool] = None,
    overwrite: bool = True,
    timeout: Optional[int] = None,
    actual_output: Optional[List[str]] = None,
    normalize_timescale: bool = True,
    source_video: Optional[Union[str, Path]] = None,
    meta_sidecar: Optional[Union[str, Path]] = None,
) -> bool:
    """
    根据视频编码自动选择直接复制流或重新编码为 H.264 后合并视频，
    并支持使用独立音频文件替换原视频音轨。

    [FIX-C-COVERAGE] 合并前自动对各分段做 timescale 归一化（零损耗 remux），
    使"单阶段"路径（--skip-upscale / --skip-interpolate，由各 processor 内部
    合并）与"两阶段"路径（main 层最终合并）获得一致保护。此前该归一化只挂在
    main 层，单阶段路径完全无保护，仅因同编码器+同帧率下分段 timescale 恰好
    一致才未暴露问题。

    [P0-FIX-EXT-PROP] 触发重编码且输出扩展名与 config format 不一致时，本函数
    会改写实际输出路径（如 out.mkv → out.mp4）。调用方传入 actual_output 列表
    （出参）即可拿到真实写出路径，避免"任务成功但按原路径找不到文件"的假阴性链。

    Args:
        file_list: 待合并的视频分段文件路径列表。
        output_path: 输出视频文件路径。
        actual_output: 可选出参列表；函数成功时追加实际输出路径（str）。
        config: 可选，配置参数字典，支持以下字段：
            - format      : str   (默认 'mp4')      # 输出格式（用于自动补充扩展名）
            - codec       : str   (默认 'libx264')  # 视频编码器；'copy' = 强制直接复制
            - preset      : str   (默认 'medium')   # 编码预设
            - crf         : int   (默认 None)       # 质量字面量（libx264 轴，原样下发）
            - cq          : int   (默认 None)       # 质量字面量（h264_nvenc CQ 轴）
            - crf_ref     : int   (默认 None)       # 统一基准（libx264 CRF 轴，0~51）
            - cq_ref      : int   (默认 None)       # 统一基准（h264_nvenc CQ 轴，0~51）
            - pix_fmt     : str   (默认 'yuv420p')  # 像素格式
            - audio_codec : str   (默认 'copy')     # 音频编码器，'copy' 表示直接复制
            - audio_bitrate: str  (默认 '192k')     # 音频码率（当 audio_codec != 'copy' 时使用）
            - extra_args  : list  (默认 [])         # 其他追加的 ffmpeg 参数
            [QUALITY-UNIFY] crf/cq/crf_ref/cq_ref 由 quality_map.resolve_quality 换算为
            实际编码器参数（软编 -crf / 硬编 -cq:v + -b:v 0 / librav1e -qp），四者互斥，
            基准轴优先。均未给时按 libx264 CRF 基准 21 换算。
        audio_path: 可选，独立音频文件路径。提供后将替换合并视频中的音轨。
        check_consistency: 是否检查所有分段视频编码格式一致。
            True 时不一致将抛出 ValueError；False 时仅发出警告并强制重编码。
        force_reencode: 若为 True，则无视编码梯队，强制全部重新编码为 H.264。
        reencode: [QUALITY-UNIFY] 显式重编码开关。
            None (默认) = 按编码梯队自动判定（可复制则 copy）；
            True        = 强制重编码（用户显式请求了 --output-codec/质量）；
            False       = 强制直接复制（-c:v copy，无损且快）。
            config['codec'] == 'copy' 等价于 reencode=False。
        overwrite: 是否覆盖已存在的输出文件
        timeout: ffmpeg 进程超时时间（秒）
        normalize_timescale: 复制流合并前是否统一各分段 timescale（默认 True）。
            仅对 H.264/H.265 的多段 copy 合并生效；重编码或单文件输入自动跳过。
        source_video: [META-KEEP] 原片路径。用于合并时回写原片的容器级元数据
            （tags / creation_time / 旋转 / 位深）。未传时自动在分段目录下找 sidecar。
        meta_sidecar: [META-KEEP] split_video_by_time 落盘的 source_meta.ffmetadata。
            未传时自动从分段所在目录查找。

    Returns:
        True -- 合并成功。

    Raises:
        ValueError: 输入列表为空、编码格式不一致（check_consistency=True）或无法获取编码信息。
        FileNotFoundError: 输入文件不存在或 ffmpeg 未安装。
        subprocess.CalledProcessError: ffmpeg 命令执行失败。
    """
    # ---------- 前置检查 ----------
    if not file_list:
        raise ValueError("待合并的文件列表不能为空")

    if shutil.which('ffmpeg') is None:
        raise FileNotFoundError("未找到 ffmpeg，请确保已安装并加入 PATH")

    # 统一转换为 Path 对象，并检查存在性
    input_paths = [Path(p) for p in file_list]
    for p in input_paths:
        if not p.is_file():
            raise FileNotFoundError(f"输入文件不存在: {p}")

    output_path = Path(output_path)
    audio_path = Path(audio_path) if audio_path else None
    if audio_path and not audio_path.is_file():
        raise FileNotFoundError(f"独立音频文件不存在: {audio_path}")

    # ---------- 编码格式检测与决策 ----------
    try:
        first_codec = get_video_codec(input_paths[0])
    except Exception as e:
        raise ValueError(f"无法读取第一个视频 '{input_paths[0]}' 的编码信息: {e}")

    DIRECT_COPY_CODECS = {
        'h264', 'avc', 'hevc', 'h265', 'vp9', 'av1', 'vvc', 'h266'
    }

    # ---------- [QUALITY-UNIFY] 复制 / 重编码路由 ----------
    # 默认（reencode=None 且 codec!='copy'）仍是"可复制则 copy"的旧行为，
    # 保证分段合并无损且快；仅当调用方显式请求重编码时才重编码。
    #   · reencode=True          → 强制重编码（用户给了 --output-codec/质量）
    #   · reencode=False         → 强制直接复制（-c:v copy）
    #   · config['codec']=='copy' → 等价强制复制（此前该值从未生效）
    _codec_req = str((config or {}).get('codec') or '').strip().lower()
    _force_copy = (reencode is False) or (_codec_req == 'copy' and reencode is None)

    if force_reencode or reencode is True:
        need_reencode = True
        reason = "显式请求重新编码"
    elif _force_copy:
        if first_codec in DIRECT_COPY_CODECS:
            need_reencode = False
            reason = "显式/默认直接复制流（-c:v copy）"
        else:
            need_reencode = True
            reason = f"请求 copy 但分段编码 '{first_codec}' 不可直接复制，回退重编码"
    else:
        need_reencode = first_codec not in DIRECT_COPY_CODECS
        reason = f"编码格式 '{first_codec}' 不在直接复制梯队" if need_reencode else "直接复制流"

    # ---------- 编码一致性检查 ----------
    if check_consistency and not force_reencode:
        for file in input_paths[1:]:
            try:
                codec = get_video_codec(file)
            except Exception as e:
                raise ValueError(f"无法读取视频 '{file}' 的编码信息: {e}")
            if codec != first_codec:
                raise ValueError(
                    f"视频编码不一致: '{input_paths[0]}' 是 {first_codec}, "
                    f"而 '{file}' 是 {codec}。\n"
                    "设置 check_consistency=False 可忽略不一致（将强制重新编码）或手动统一分段编码。"
                )
    elif (not check_consistency and not force_reencode and not need_reencode
          and reencode is not False):
        # 一致性检查关闭，但第一个编码不需重编码 → 后续可能不兼容，自动降级重编码。
        # 守卫 reencode is not False：显式 copy 请求不得被此回退静默改写为重编码。
        warnings.warn(
            "编码一致性检查已禁用，但第一个分段编码不需重编码。"
            "为避免后续分段编码不一致导致合并失败，将自动强制重新编码。",
            UserWarning
        )
        need_reencode = True
        reason = "一致性检查关闭，强制重新编码"

    # reason 原先只赋值从未使用（死代码），这里落日志让重编码决策可追溯
    logger.info(f"合并编码策略: {'重编码' if need_reencode else '直接复制'}（{reason}）")

    # ---------- 加载配置参数 ----------
    # 默认配置（完全移除 audio_format）
    default_config = {
        'format': 'mp4',
        'codec': 'libx264',
        'preset': 'medium',
        # [QUALITY-UNIFY] 质量四键互斥、基准轴优先；均 None 时按 libx264 CRF 21 换算。
        # crf 默认由 18 改 None：老调用方若显式传 crf 仍原样下发（向后兼容）。
        'crf': None,
        'cq': None,
        'crf_ref': None,
        'cq_ref': None,
        'pix_fmt': 'yuv420p',
        'audio_codec': 'copy',      # 默认直接复制音频流
        'audio_bitrate': '192k',
        'extra_args': []
    }

    # 合并用户配置，忽略未知字段（兼容旧 config 可能包含的 audio_format）
    params = default_config.copy()
    if config:
        for key in params:
            if key in config:
                params[key] = config[key]
        # 单独处理 extra_args（允许完全替换）
        if 'extra_args' in config:
            params['extra_args'] = config.get('extra_args', [])
        # 忽略旧的 audio_format 字段，如有则静默忽略（或可发出 DeprecationWarning）
        if 'audio_format' in config:
            warnings.warn(
                "配置项 'audio_format' 已弃用，将被忽略。请直接使用 'audio_codec' 控制音频行为。",
                DeprecationWarning
            )

    # ---------- [FIX-C] 合并前分段 timescale 归一化 ----------
    # 仅在"直接复制流"路径下有意义：重编码会由编码器重新生成时间基，
    # 无需也无从归一化。单文件输入（如音频回写）不存在拼接一致性问题，跳过。
    if normalize_timescale and not need_reencode and len(input_paths) > 1:
        input_paths = [Path(p) for p in normalize_segments_timescale(
            input_paths, codec_hint=first_codec)]

    # ---------- 构建 concat 列表文件 ----------
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        list_path = Path(f.name)
        for file in input_paths:
            # Windows 路径兼容：使用绝对路径并替换反斜杠
            abs_path = file.resolve().as_posix()
            # 转义单引号：在单引号字符串中，' 需要写成 '\''
            escaped_path = abs_path.replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    # ---------- [META-KEEP] 定位原片元数据 sidecar ----------
    # concat demuxer 的合成上下文没有容器级元数据，-map_metadata 0 拿到的是
    # 列表文件（空）。必须在切片阶段落盘的 source_meta.ffmetadata 里取；
    # 找不到时退回把原片作为额外输入（代价是要多 demux 一遍原片）。
    _sidecar = Path(meta_sidecar) if meta_sidecar else None
    if _sidecar is None:
        _cand = input_paths[0].parent / 'source_meta.ffmetadata'
        if _cand.is_file():
            _sidecar = _cand
    _sidecar_json = _sidecar.with_name('source_meta.json') if _sidecar else None
    _src_sidecar = load_source_meta_sidecar(_sidecar_json) if _sidecar_json else {}
    if source_video is None and _src_sidecar.get('source'):
        _maybe_src = Path(_src_sidecar['source'])
        if _maybe_src.is_file():
            source_video = _maybe_src

    # ---------- 构建 ffmpeg 命令 ----------
    ffmpeg_cmd = ['ffmpeg']
    ffmpeg_cmd += ['-y'] if overwrite else ['-n']
    # [META-KEEP] -noautorotate 是输入侧选项，必须放在 -i 之前：
    # 阻止 autorotate 烘焙分段里继承来的 display matrix
    ffmpeg_cmd += ['-noautorotate']
    ffmpeg_cmd += [
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_path)
    ]

    has_external_audio = audio_path is not None
    if has_external_audio:
        ffmpeg_cmd += ['-i', str(audio_path)]

    # [META-KEEP] 元数据输入：优先 ffmetadata sidecar（只读 header，零成本），
    # 其次原片作为额外输入。必须排在音频输入之后，保证现有 -map 1:a:0 索引不变。
    _meta_input_idx = None
    if _sidecar is not None and _sidecar.is_file():
        ffmpeg_cmd += ['-i', str(_sidecar)]
        _meta_input_idx = 2 if has_external_audio else 1
    elif source_video is not None and Path(source_video).is_file():
        ffmpeg_cmd += ['-i', str(Path(source_video))]
        _meta_input_idx = 2 if has_external_audio else 1

    # 映射视频流（始终来自第一个输入）
    ffmpeg_cmd += ['-map', '0:v:0']

    # 映射音频流（统一策略，无需 audio_format）
    if has_external_audio:
        ffmpeg_cmd += ['-map', '1:a:0']   # 使用独立音频
    else:
        ffmpeg_cmd += ['-map', '0:a?']    # 保留原视频音轨（如果存在）

    # [META-KEEP] 原片容器级元数据 + 章节（注意：必须指向 sidecar/原片，
    # concat 下 -map_metadata 0 指向的是列表文件）
    if _meta_input_idx is not None:
        ffmpeg_cmd += ['-map_metadata', str(_meta_input_idx),
                       '-map_chapters', str(_meta_input_idx)]
        _ct = _src_sidecar.get('creation_time')
        if _ct:
            # -metadata 覆盖 -map_metadata，故必须放在其后
            ffmpeg_cmd += ['-metadata', f'creation_time={_ct}']

    # [META-KEEP] 位深与 HDR：优先用 sidecar JSON，缺失时探测原片
    _src_meta_full = probe_full_metadata(source_video) if source_video else None
    _src_bits = _src_sidecar.get('src_bits')
    if _src_bits is None and _src_meta_full:
        _src_bits = _src_meta_full['derived']['src_bits']

    # ---------- 视频编码器：codec-aware 质量下发（[QUALITY-UNIFY]） ----------
    if need_reencode:
        _eff_codec = str(params.get('codec') or '').strip()
        if _eff_codec.lower() in ('', 'copy'):
            _eff_codec = 'libx264'   # 防御：重编码路径 codec 不得为 copy

        _pix_fmt = params['pix_fmt']
        if _src_bits and int(_src_bits) >= 10:
            _pix_fmt = resolve_pix_fmt_for_source(
                _eff_codec, int(_src_bits), warn=lambda m: print(f"⚠️  {m}")) \
                or params['pix_fmt']

        # 统一构造 preset + 质量参数（软编 -crf / 硬编 -cq:v -b:v 0 / librav1e -qp）
        _q_args, _q_note = _resolve_quality_args(
            _eff_codec,
            crf=params.get('crf'), cq=params.get('cq'),
            crf_ref=params.get('crf_ref'), cq_ref=params.get('cq_ref'),
            preset=params.get('preset'))
        logger.info(f"[merge] 质量参数解析: {_q_note} → {' '.join(_q_args)}")

        ffmpeg_cmd += ['-c:v', _eff_codec] + _q_args
        if _pix_fmt:
            ffmpeg_cmd += ['-pix_fmt', _pix_fmt]
        # [META-KEEP] HDR10 静态元数据（libx265 走 -x265-params，其余尽力而为）
        if _src_meta_full is not None:
            ffmpeg_cmd += build_hdr_args(
                _src_meta_full, _eff_codec, warn=lambda m: print(f"⚠️  {m}"))
        # [COLOR-FIX] 补全 re-encode 路径色彩元数据：
        # libx264 等编码器对输出端 -color_primaries/-color_trc 不写 VUI，
        # 需用 setparams 滤镜显式注入（值来自 extra_args，无则跳过）。
        _vf_parts: List[str] = []
        if '-vf' not in params['extra_args']:
            _sp = _setparams_from_color_args(params['extra_args'])
            if _sp:
                _vf_parts.append(_sp)
        if _vf_parts:
            ffmpeg_cmd += ['-vf', ','.join(_vf_parts)]
    else:
        ffmpeg_cmd += ['-c:v', 'copy']

    # 音频编码器（统一规则）
    if params['audio_codec'] == 'copy':
        ffmpeg_cmd += ['-c:a', 'copy']
    else:
        ffmpeg_cmd += ['-c:a', params['audio_codec']]
        if params.get('audio_bitrate'):
            ffmpeg_cmd += ['-b:a', params['audio_bitrate']]

    # 输出文件：自动补充扩展名
    _requested_output = str(output_path)
    if not output_path.suffix or need_reencode:
        output_path = output_path.with_suffix(f'.{params["format"]}')
    _final_output = str(output_path)
    if _final_output != _requested_output:
        print(f"⚠️  [merge] 输出容器与编码策略不匹配，实际输出: {_final_output} "
              f"(请求: {_requested_output})")

    ffmpeg_cmd += params['extra_args'] + [str(output_path)]

    # ---------- 执行 ----------
    _merge_stderr = ""
    try:
        _result = subprocess.run(
            ffmpeg_cmd, check=True, capture_output=True, text=True, timeout=timeout
        )
        _merge_stderr = _result.stderr  # rc=0 时保留 stderr 供后续诊断
    except subprocess.TimeoutExpired:
        raise FFmpegError("FFmpeg 进程超时")
    except subprocess.CalledProcessError as e:
        cmd_preview = ' '.join(ffmpeg_cmd[:5]) + ' ...'
        raise subprocess.CalledProcessError(
            e.returncode, e.cmd,
            output=e.stdout,
            stderr=(
                f"ffmpeg 合并失败，命令预览: {cmd_preview}\n"
                f"临时文件: {list_path}\n"
                f"独立音频: {audio_path or '无'}\n"
                f"视频编码策略: {'重编码' if need_reencode else '直接复制'}\n"
                f"音频编码策略: {params['audio_codec']}\n"
                f"错误输出: {e.stderr}"
            )
        )
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except OSError:
            pass

    # ---------- 输出校验 ----------
    # rc=0 并不保证输出文件完整：ffmpeg 在某些静默错误下（如 BSF 畸形 packet、
    # timescale 问题）会以 rc=0 退出但写出损坏文件（无视频流或空视频轨）。
    # 用 ffprobe 主动验证视频流存在且 duration > 0，及早发现并报告问题。
    _out_str = str(output_path)

    def _probe_stream(path: str, stream_sel: str, entry: str) -> str:
        """返回 ffprobe 指定条目的值字符串，失败返回空字符串。"""
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-select_streams", stream_sel,
                 "-show_entries", f"stream={entry}",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=30,
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    # 验证视频流：codec_type=video 且 duration > 0
    _v_info = _probe_stream(_out_str, "v:0", "codec_type,duration")
    _has_video = (
        _v_info
        and "video" in _v_info.lower()
        and not all(p.strip() in ("N/A", "0", "0.000000", "")
                    for p in _v_info.split(",")[1:])
    )

    if not _has_video:
        # 打印 ffmpeg stderr 以便诊断（即使 rc=0 也可能有有用警告）
        if _merge_stderr:
            _warn_lines = [l for l in _merge_stderr.splitlines()
                           if any(k in l.lower() for k in
                                  ("warning", "error", "invalid", "dts", "pts",
                                   "moov", "track", "stream", "codec"))]
            if _warn_lines:
                logger.warning("ffmpeg 合并 stderr（rc=0）:\n" +
                               "\n".join(_warn_lines[-20:]))
        raise FFmpegError(
            f"合并输出文件缺少有效视频流: {output_path}\n"
            f"ffprobe v:0 返回: {_v_info!r}\n"
            f"完整 ffmpeg 命令: {' '.join(ffmpeg_cmd)}\n"
            f"诊断：请检查输入分段是否均包含有效视频帧"
            + (f"\nffmpeg stderr 末尾:\n{_merge_stderr[-800:]}"
               if _merge_stderr else "")
        )

    # 有外部音频时同样验证音频流
    if audio_path:
        _a_info = _probe_stream(_out_str, "a:0", "codec_type")
        if not _a_info or "audio" not in _a_info.lower():
            logger.warning(
                f"合并输出文件缺少音频流（视频正常）: {output_path}\n"
                f"ffprobe a:0 返回: {_a_info!r}"
            )

    # ---------- [META-KEEP] 旋转兜底 ----------
    # 链路为「切片 → 逐段重编码 → concat 合并」，真正会丢掉旋转的是逐段重编码
    # 那一步（raw elementary 流不携带 display matrix）。这里做一次零重编码的 remux
    # 补救：仅当产物**横竖方向与源一致**（说明像素没被烘焙）时才补写 display matrix，
    # 避免「像素已旋转 + 又写标签」的双重旋转。
    _src_rot = int(_src_sidecar.get('rotation') or 0)
    if _src_rot == 0 and _src_meta_full:
        _src_rot = int(_src_meta_full['derived']['rotation'] or 0)
    if _src_rot:
        try:
            _om = probe_full_metadata(_out_str)
            _out_rot = _om['derived']['rotation'] if _om else 0
            _ow = _om['derived']['width'] if _om else 0
            _oh = _om['derived']['height'] if _om else 0
            _sw = int(_src_sidecar.get('width') or 0)
            _sh = int(_src_sidecar.get('height') or 0)
            if _src_meta_full and not (_sw and _sh):
                _sw = _src_meta_full['derived']['width']
                _sh = _src_meta_full['derived']['height']
            # 判定「像素是否已被烘焙」：烘焙会交换宽高，即横竖方向翻转。
            # 不能用「宽高完全相等」判断——流水线会超分（640x360 → 1280x720），
            # 相等条件永不成立。改为比较方向，与缩放倍数无关。
            if (_ow and _oh and _sw and _sh and not _out_rot
                    and (_sw > _sh) == (_ow > _oh)):
                _tmp = _out_str + '.rotfix.mp4'
                subprocess.run(
                    ['ffmpeg', '-v', 'error', '-y',
                     '-noautorotate', '-display_rotation', str(_src_rot),
                     '-i', _out_str,
                     '-map', '0', '-c', 'copy',
                     '-map_metadata', '0', '-map_chapters', '0', _tmp],
                    capture_output=True, timeout=1800, check=True)
                os.replace(_tmp, _out_str)
        except Exception as _e:
            logger.warning(f'[META-KEEP] 旋转兜底失败（不影响产物）: {_e}')

    # [P0-FIX-EXT-PROP] 向调用方传播实际输出路径
    if actual_output is not None:
        actual_output.append(_final_output)

    return True

def encode_video(input_path: str, output_path: str, 
                codec: str = 'libx264', crf: Optional[int] = None,
                preset: str = 'medium', pix_fmt: str = 'yuv420p',
                cq: Optional[int] = None, crf_ref: Optional[int] = None,
                cq_ref: Optional[int] = None) -> bool:
    """
    编码视频

    Args:
        input_path: 输入路径
        output_path: 输出路径
        codec: 编码器
        crf / cq / crf_ref / cq_ref: [QUALITY-UNIFY] 质量输入（互斥、基准轴优先）；
            均未给时按 libx264 CRF 基准 21 换算到 codec 的实际量纲/参数。
        preset: 预设（仅对支持 -preset 的编码器发射）
        pix_fmt: 像素格式

    Returns:
        是否成功
    """
    try:
        _q_args, _q_note = _resolve_quality_args(
            codec, crf=crf, cq=cq, crf_ref=crf_ref, cq_ref=cq_ref, preset=preset)
        logger.info(f"[encode_video] 质量参数解析: {_q_note} → {' '.join(_q_args)}")
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', codec,
        ] + _q_args + [
            '-pix_fmt', pix_fmt,
            '-y', output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ 视频编码完成: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 视频编码失败: {e}")
        return False

def _validate_single_task(
    job: Tuple[int, str, Optional[int], Optional[bool], Optional[int]],
) -> Tuple[int, Tuple[bool, Dict[str, object]]]:
    """单文件验收任务（模块级函数），支持 GPU 自动探测与并行硬解。

    任务格式扩展为: (idx, path_str, expected_frames, use_hwaccel, gpu_workers)
    以支持批量入口传递 GPU 配置。
    """
    idx, path_str, expected, use_hwaccel_task, gpu_workers_task = job
    # 批量任务传入的 GPU 配置优先；若未传入则由单函数自动探测
    ok, report = validate_decodable_video(
        path_str,
        expected_frames=expected,
        count_mode="decode",
        use_hwaccel=use_hwaccel_task,
        gpu_workers=gpu_workers_task,
    )
    return idx, (ok, report)


def validate_decodable_video_batch(
    video_paths: List[Union[str, Path]],
    expected_frames_list: Optional[List[Optional[int]]] = None,
    workers: Optional[int] = None,
    parallel_mode: str = "thread",
    gpu_workers: Optional[int] = None,
    skip_validate: bool = False,
    progress_callback: Optional[Any] = None,
) -> List[Tuple[bool, Dict[str, object]]]:
    """[P4-FIX-BATCH] 批量解码级验收（并行版本，参考三个脚本并行引擎模式）。

    参考：
      · verify_segment_bitstream_v5.py (run_verify_parallel + 结果顺序保留)
      · benchmark_ifrnet_versions_v3.py (自动 workers + GPU 动态上限)
      · analyze_video_pipeline_v3.py (两阶段流水线 + 任务级信号量)
    """
    if skip_validate:
        return [(True, {"skipped": True, "decoded_frames": None})
                       for _ in video_paths]

    if not video_paths:
        return []

    if expected_frames_list is None:
        expected_frames_list = [None] * len(video_paths)

    # 使用独立的并行执行引擎（参考三个脚本模式整合）
    # 延迟导入：parallel_executor -> system_resources 会间接引入 torch/pynvml，
    # 放到模块顶层会显著拖慢 video_utils 的导入（本模块本身不依赖 GPU 运行时）
    # 同时兼容两种导入方式：包内相对导入，以及 src/utils 在 sys.path 上的顶层导入
    if __package__:
        from .parallel_executor import ParallelExecutor
        from .system_resources import SystemResourceDetector, compute_auto_workers
    else:
        from parallel_executor import ParallelExecutor
        from system_resources import SystemResourceDetector, compute_auto_workers
    resources = SystemResourceDetector().detect()
    if workers is None:
        workers = compute_auto_workers(task_ram_mb=512, reserve_ratio=0.10,
                                       resources=resources, gpu_task=True)

    # 自动探测 GPU 配置（用于批量控制和任务传递）
    hw_config = _get_gpu_hwaccel_config()
    gpu_available = hw_config.get("available", False)
    # 批量默认启用硬解（若可用），可通过传入参数覆盖
    # 当前签名中没有 use_hwaccel 参数；若需要可后续扩展。此处使用自动探测结果控制任务传递。
    batch_use_hwaccel = gpu_available  # 默认自动启用
    batch_gpu_workers = gpu_config_workers = max(1, hw_config.get("gpu_workers", 0)) if gpu_available else 0

    # 预先把 (idx, 路径, 预期帧数, use_hwaccel, gpu_workers) 打包成可序列化任务
    jobs: List[Tuple[int, str, Optional[int], Optional[bool], Optional[int]]] = [
        (i,
         str(video_paths[i]),
         expected_frames_list[i] if i < len(expected_frames_list) else None,
         batch_use_hwaccel,
         batch_gpu_workers)
        for i in range(len(video_paths))
    ]

    print(f"批量解码级验收启动: {len(video_paths)} 文件 | workers={workers} | "
          f"mode={parallel_mode} | gpu_sem={gpu_workers or 'auto'}")

    executor = ParallelExecutor(
        workers=workers,
        parallel_mode=parallel_mode,
        task_ram_mb=512,
        gpu_task=True,
        gpu_workers=gpu_workers,
        progress_callback=progress_callback,
        resources=resources,
    )
    results_raw = executor.map_tasks(_validate_single_task, [(job,) for job in jobs])

    # 整理结果：按输入顺序严格保留（参考 verify_segment_bitstream_v5 ordered 模式）
    output: List[Tuple[bool, Dict[str, object]]] = []
    results_by_idx: Dict[int, Tuple[bool, Dict[str, object]]] = {}
    for res in results_raw:
        if res.success and isinstance(res.result, tuple) and len(res.result) == 2:
            idx, val_tuple = res.result
            results_by_idx[idx] = val_tuple
        else:
            # 异常处理：参考 analyze_video_pipeline_v3 的 _task_error_result
            idx = res.index
            error_dict = {
                "error": res.error or "parallel_validation_exception",
                "decoded_frames": 0,
                "reason": "parallel_exception",
            }
            results_by_idx[idx] = (False, error_dict)

    for i in range(len(video_paths)):
        output.append(results_by_idx.get(i, (False, {"reason": "missing_result"})))

    ok_total = sum(1 for ok, _ in output if ok)
    print(f"批量验收完成: {ok_total}/{len(video_paths)} 通过 | 失败: {len(video_paths) - ok_total}")
    return output
