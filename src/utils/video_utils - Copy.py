"""
视频处理工具函数
提供视频信息获取、分段、合并等通用功能
"""

import os
import subprocess
import cv2
import json
import re
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Sequence, Any
import shutil
import logging
import tempfile
import warnings

# 配置日志（或在函数内直接使用 print，取决于你的项目规范）
logger = logging.getLogger(__name__)

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
        
        self._load_info()
    
    def _load_info(self):
        """加载视频信息"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"视频文件不存在: {self.path}")
        
        # 使用ffprobe获取详细信息
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_format', '-show_streams', self.path],
                capture_output=True,
                text=True,
                check=True
            )
            
            data = json.loads(result.stdout)
            
            # 解析格式信息
            if 'format' in data:
                self.duration = float(data['format'].get('duration', 0))
                self.bitrate = int(data['format'].get('bit_rate', 0))
            
            # 解析流信息
            for stream in data.get('streams', []):
                if stream['codec_type'] == 'video':
                    self.fps = eval(stream.get('r_frame_rate', '0/1'))
                    self.width = int(stream.get('width', 0))
                    self.height = int(stream.get('height', 0))
                    self.codec = stream.get('codec_name', 'unknown')
                    if 'nb_frames' in stream:
                        self.frame_count = int(stream['nb_frames'])
                elif stream['codec_type'] == 'audio':
                    self.has_audio = True
                    self.audio_codec = stream.get('codec_name', 'unknown')
        
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
    log_ffmpeg_output: bool = False  # 新增：是否打印 ffmpeg 输出
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
        logger.info(f"输出文件已存在，跳过: {output_path}")
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
    logger.info(f"执行命令: {' '.join(cmd)}")
    
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
            # 设置视频编码参数
            video_codec = config.get('codec', 'libx264')
            preset = config.get('preset', 'medium')
            crf = config.get('crf', 18)
            pix_fmt = config.get('pix_fmt', 'yuv420p')
            
            cmd.extend([
                '-c:v', video_codec,
                '-preset', preset,
                '-crf', str(crf),
                '-pix_fmt', pix_fmt
            ])

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
            except:
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
    except:
        pass
    
    # 备用方法：使用OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps > 0 and frame_count > 0:
            return frame_count / fps
    except:
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
    except:
        return False


def split_video_by_time(input_video: str, output_dir: str, 
                        segment_duration: int = 30) -> List[str]:
    """
    按时间分割视频
    
    Args:
        input_video: 输入视频路径
        output_dir: 输出目录
        segment_duration: 每段时长（秒）
    
    Returns:
        分段文件列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频时长
    duration = get_video_duration(input_video)
    if duration is None:
        print(f"❌ 无法获取视频时长")
        return []
    
    print(f"📹 视频总时长: {format_time(duration)}")
    
    # 计算分段数
    num_segments = int(duration / segment_duration) + (1 if duration % segment_duration > 0 else 0)
    
    if num_segments <= 1:
        print(f"⏭️  视频时长 < {segment_duration}秒，无需分段")
        # 复制整个视频
        segment_file = os.path.join(output_dir, "segment_000.mp4")
        shutil.copy2(input_video, segment_file)
        return [segment_file]
    
    print(f"🔪 分割为 {num_segments} 段...")
    
    segment_files = []
    segment_pattern = os.path.join(output_dir, "segment_%03d.mp4")
    
    # 使用FFmpeg的segment muxer
    cmd = [
        'ffmpeg', '-i', input_video,
        '-c', 'copy',  # 复制流，不重新编码
        '-map', '0:v',  # 只映射视频流
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
        
        print(f"✅ 成功分割为 {len(segment_files)} 个有效片段")
        return segment_files
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 分割失败: {e}")
        return []


# def merge_videos(video_files: List[str], output_path: str, 
#                  audio_path: Optional[str] = None,
#                  config: Optional[Dict[str, Any]] = None) -> bool:
#     """
#     合并视频文件
    
#     Args:
#         video_files: 视频文件列表
#         output_path: 输出路径
#         audio_path: 音频文件路径（可选）
#         config: 输出编码配置字典（如 codec, crf, preset 等参数）
    
#     Returns:
#         是否成功
#     """
#     if not video_files:
#         print("❌ 没有视频文件需要合并")
#         return False
    
#     print(f"🔗 合并 {len(video_files)} 个视频片段...")
    
#     # 创建文件列表
#     list_file = output_path + ".list.txt"
#     with open(list_file, 'w') as f:
#         for video in video_files:
#             f.write(f"file '{os.path.abspath(video)}'\n")
    
#     try:
#         # 构建基础命令
#         cmd = [
#             'ffmpeg',
#             '-f', 'concat',
#             '-safe', '0',
#             '-i', list_file,
#         ]
        
#         # 如果提供了输出编码配置
#         if config:  # 直接检查 config 是否不为空
#             # 使用配置中的编码参数
#             video_codec = config.get('codec', 'libx264')
#             preset = config.get('preset', 'medium')
#             crf = config.get('crf', 18)
#             pix_fmt = config.get('pix_fmt', 'yuv420p')
            
#             # 添加视频编码参数
#             cmd.extend([
#                 '-c:v', video_codec,
#                 '-preset', preset,
#                 '-crf', str(crf),
#                 '-pix_fmt', pix_fmt
#             ])
            
#             # 如果有音频，设置音频编码参数
#             if audio_path and os.path.exists(audio_path):
#                 cmd.extend(['-i', audio_path])
#                 audio_codec = config.get('audio_codec', 'aac')

#                 if audio_codec == 'copy':
#                     cmd.extend(['-c:a', 'copy'])
#                 else:
#                     audio_bitrate = config.get('audio_bitrate', '192k')
#                     cmd.extend([
#                         '-c:a', audio_codec,
#                         '-b:a', audio_bitrate,
#                     ])

#                 cmd.extend([
#                     '-map', '0:v:0',
#                     '-map', '1:a:0'
#                 ])
#             else:
#                 # 如果没有音频文件，只映射视频
#                 cmd.extend(['-map', '0:v:0'])
#                 # 如果输入视频有音频，也复制音频流
#                 cmd.extend(['-c:a', 'copy'])
                
#         else:
#             # 如果没有配置参数，使用默认行为（复制流）
#             cmd.extend(['-c', 'copy'])
#             cmd.extend(['-vsync', 'passthrough'])
            
#             # 如果有音频，添加音频
#             if audio_path and os.path.exists(audio_path):
#                 cmd.extend(['-i', audio_path])
#                 cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
#                 cmd.extend(['-map', '0:v', '-map', '1:a'])
#             else:
#                 # 如果没有音频文件，只映射视频
#                 cmd.extend(['-map', '0:v:0'])
#                 # 如果输入视频有音频，也复制音频流
#                 cmd.extend(['-c:a', 'copy'])
        
#         # 添加输出路径
#         cmd.extend(['-y', output_path])
        
#         # 打印命令以便调试
#         print(f"📋 合并命令: {' '.join(cmd)}")
        
#         # 执行合并
#         result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
#         if result.returncode != 0:
#             print(f"❌ 合并失败，错误信息:")
#             print(result.stderr)
#             # if os.path.exists(list_file):
#             #     os.remove(list_file)
#             return False
        
#         # 删除临时文件列表
#         os.remove(list_file)
        
#         # 验证输出文件
#         if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
#             print(f"✅ 视频合并完成: {output_path}")
            
#             # 输出文件信息
#             try:
#                 info = VideoInfo(output_path)
#                 print(f"📊 输出文件信息: {info}")
#             except:
#                 print("⚠️  无法获取输出文件详细信息")
            
#             return True
#         else:
#             print(f"❌ 合并完成但输出文件无效")
#             return False
        
#     except Exception as e:
#         print(f"❌ 合并过程中发生异常: {e}")
#         # if os.path.exists(list_file):
#         #     os.remove(list_file)
#         return False
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
    default_config = {
        'codec': 'libx264',
        'preset': 'medium',
        'crf': 18,
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
        # 视频编码参数
        cmd.extend([
            '-c:v', config['codec'],
            '-preset', config['preset'],
            '-crf', str(config['crf']),
            '-pix_fmt', config['pix_fmt']
        ])
        
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

def merge_videos_by_codec(
    file_list: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    audio_path: Optional[Union[str, Path]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    check_consistency: bool = True,
    force_reencode: bool = False,
    overwrite: bool = True,
    timeout: Optional[int] = None
) -> bool:
    """
    根据视频编码自动选择直接复制流或重新编码为 H.264 后合并视频，
    并支持使用独立音频文件替换原视频音轨。

    Args:
        file_list: 待合并的视频分段文件路径列表。
        output_path: 输出视频文件路径。
        config: 可选，配置参数字典，支持以下字段：
            - format      : str  (默认 'mp4')       # 输出格式（用于自动补充扩展名）
            - codec       : str  (默认 'libx264')   # 视频编码器（重编码时使用）
            - preset      : str  (默认 'medium')    # 编码预设
            - crf         : int  (默认 18)          # 视频质量
            - pix_fmt     : str  (默认 'yuv420p')   # 像素格式
            - audio_codec : str  (默认 'copy')      # 音频编码器，'copy' 表示直接复制
            - audio_bitrate: str (默认 '192k')      # 音频码率（当 audio_codec != 'copy' 时使用）
            - extra_args  : list (默认 [])          # 其他追加的 ffmpeg 参数
        audio_path: 可选，独立音频文件路径。提供后将替换合并视频中的音轨。
        check_consistency: 是否检查所有分段视频编码格式一致。
            True 时不一致将抛出 ValueError；False 时仅发出警告并强制重编码。
        force_reencode: 若为 True，则无视编码梯队，强制全部重新编码为 H.264。
        overwrite: 是否覆盖已存在的输出文件
        timeout: ffmpeg 进程超时时间（秒）

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

    if force_reencode:
        need_reencode = True
        reason = "强制重新编码"
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
    elif not check_consistency and not force_reencode and not need_reencode:
        # 一致性检查关闭，但第一个编码不需重编码 → 后续可能不兼容，自动降级重编码
        warnings.warn(
            "编码一致性检查已禁用，但第一个分段编码不需重编码。"
            "为避免后续分段编码不一致导致合并失败，将自动强制重新编码。",
            UserWarning
        )
        need_reencode = True
        reason = "一致性检查关闭，强制重新编码"

    # ---------- 加载配置参数 ----------
    # 默认配置（完全移除 audio_format）
    default_config = {
        'format': 'mp4',
        'codec': 'libx264',
        'preset': 'medium',
        'crf': 18,
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

    crf_str = str(params['crf'])

    # ---------- 构建 concat 列表文件 ----------
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        list_path = Path(f.name)
        for file in input_paths:
            # Windows 路径兼容：使用绝对路径并替换反斜杠
            abs_path = file.resolve().as_posix()
            f.write(f"file '{abs_path}'\n")

    # ---------- 构建 ffmpeg 命令 ----------
    # ffmpeg_cmd = [
    #     'ffmpeg',
    #     '-y',
    #     '-f', 'concat',
    #     '-safe', '0',
    #     '-i', str(list_path)
    # ]
    ffmpeg_cmd = ['ffmpeg']
    ffmpeg_cmd += ['-y'] if overwrite else ['-n']
    ffmpeg_cmd += [
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_path)
    ]

    has_external_audio = audio_path is not None
    if has_external_audio:
        ffmpeg_cmd += ['-i', str(audio_path)]

    # 映射视频流（始终来自第一个输入）
    ffmpeg_cmd += ['-map', '0:v:0']

    # 映射音频流（统一策略，无需 audio_format）
    if has_external_audio:
        ffmpeg_cmd += ['-map', '1:a:0']   # 使用独立音频
    else:
        ffmpeg_cmd += ['-map', '0:a?']    # 保留原视频音轨（如果存在）

    # 视频编码器
    if need_reencode:
        ffmpeg_cmd += [
            '-c:v', params['codec'],
            '-preset', params['preset'],
            '-crf', crf_str,
            '-pix_fmt', params['pix_fmt']
        ]
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
    #debug print
    print(f"first_codec: {first_codec}; need_reencode: {need_reencode}; format: .{params['format']}; output_path.suffix: {output_path.suffix}")
    if not output_path.suffix or need_reencode:
        output_path = output_path.with_suffix(f'.{params["format"]}')

    ffmpeg_cmd += params['extra_args'] + [str(output_path)]

    # ---------- 执行 ----------
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, timeout=timeout)
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

    return True

def encode_video(input_path: str, output_path: str, 
                codec: str = 'libx264', crf: int = 18,
                preset: str = 'medium', pix_fmt: str = 'yuv420p') -> bool:
    """
    编码视频
    
    Args:
        input_path: 输入路径
        output_path: 输出路径
        codec: 编码器
        crf: 质量参数(0-51)
        preset: 预设
        pix_fmt: 像素格式
    
    Returns:
        是否成功
    """
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', codec,
            '-crf', str(crf),
            '-preset', preset,
            '-pix_fmt', pix_fmt,
            '-y', output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ 视频编码完成: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 视频编码失败: {e}")
        return False