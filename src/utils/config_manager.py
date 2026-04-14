"""
配置管理模块
负责加载、验证和管理所有配置参数
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class Config:
    """配置管理类"""
    
    # 默认配置 —— 与 config/default_config.json 保持完全一致
    # （JSON 优先；此处作为纯代码环境的兜底，字段齐全则避免 KeyError）
    DEFAULT_CONFIG = {
        "processing": {
            "mode":                 "interpolate_then_upscale",
            "interpolation_factor": 2,
            "upscale_factor":       2,
            "segment_duration":     30,
            "auto_fix_corrupted":   False,
            "auto_cleanup_temp":    True,
            "batch_mode":           False,
        },
        "paths": {
            "base_dir":    "",
            "input_video": "",
            "input_dir":   "",
            "output_dir":  "",
            "temp_dir":    "",
            "log_dir":     "",
            "trt_cache_dir": "",  # 留空由 _setup_paths() 自动派生为 base_dir/.trt_cache
        },
        "models": {
            "ifrnet": {
                "model_name":     "IFRNet_S_Vimeo90K",  # 模型名称，processor 自动拼路径
                "model_path":     "",                    # 显式路径（优先级 > model_name）
                "use_gpu":        True,
                # 批处理
                "batch_size":     4,
                "max_batch_size": 8,
                # v5 推理优化
                "use_fp16":       True,
                "use_compile":    True,
                "use_cuda_graph": True,
                "use_tensorrt":   False,
                # v5 硬件解/编码
                "use_hwaccel":    True,
                "codec":          "libx264",
                "crf":            18,
                "keep_audio":     True,
                "ffmpeg_bin":     "ffmpeg",
                # 性能报告
                "report_json":    None,
            },
            "realesrgan": {
                "model_name":       "realesr-general-x4v3",
                "model_path":       "",
                "use_gpu":          True,
                # 基础推理
                "denoise_strength": 0.5,
                "tile_size":        0,
                "tile_pad":         10,
                "pre_pad":          0,
                # fp32 已移除；使用 use_fp16 控制精度（与 ifrnet 保持一致）
                "use_fp16":         True,
                "face_enhance":     False,
                # v6 推理优化（默认值与底层 main.py argparse 对齐）
                "batch_size":       12,
                "prefetch_factor":  48,
                "use_compile":      True,
                "use_cuda_graph":   True,
                "use_tensorrt":     False,
                "gfpgan_trt":       False,
                # v6 face_enhance 精细控制
                "gfpgan_model":      "1.4",
                "gfpgan_weight":     0.7,
                "gfpgan_batch_size": 4,
                "face_det_threshold": 0.7,
                "adaptive_batch":    True,
                # v5 硬件解/编码
                "use_hwaccel":    True,
                "video_codec":    "libx264",   # 底层 argparse 使用的字段名
                "codec":          "libx264",   # 兼容旧配置，处理器会映射到 video_codec
                "x264_preset":    "medium",
                "crf":            23,
                "ffmpeg_bin":     "ffmpeg",
                # v6 预览与报告
                "preview":         False,
                "preview_interval": 30,
                "report_json":    None,
            },
        },
        "output": {
            "format":        "mp4",
            "codec":         "libx264",
            "preset":        "medium",
            "crf":           18,
            "pix_fmt":       "yuv420p",
            "audio_format":  "smart",
            "audio_codec":   "copy",
            "audio_bitrate": "192k",
            # extract_audio() 内部读取 config.get('bitrate')，与 audio_bitrate 保持一致
            "bitrate":       "192k",
        },
        "temp_files": {
            "segment_prefix":    "segment_",
            "processed_prefix":  "processed_",
            "temp_video_suffix": "_temp.mp4",
        },
        "logging": {
            "level":          "INFO",
            "log_to_file":    True,
            "log_to_console": True,
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self.config_path = config_path
        
        if config_path:
            self.load_from_file(config_path)
        
        self._validate_config()
        self._setup_paths()
    
    def load_from_file(self, config_path: str):
        """从文件加载配置"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        
        # 递归合并配置
        self._merge_config(self.config, user_config)
        
        print(f"✅ 已加载配置文件: {config_path}")
    
    def _merge_config(self, base: Dict, override: Dict):
        """递归合并配置字典；以 '//' 开头的注释键自动跳过"""
        for key, value in override.items():
            if str(key).startswith("//"):
                continue
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _validate_config(self):
        """验证配置的有效性"""
        # 验证处理模式
        valid_modes = ["interpolate_then_upscale", "upscale_then_interpolate"]
        if self.config["processing"]["mode"] not in valid_modes:
            raise ValueError(f"无效的处理模式: {self.config['processing']['mode']}")
        
        # 验证插帧倍数
        if self.config["processing"]["interpolation_factor"] not in [2, 4, 8, 16]:
            raise ValueError(f"无效的插帧倍数: {self.config['processing']['interpolation_factor']}")
        
        # 验证超分倍数
        if self.config["processing"]["upscale_factor"] not in [2, 4]:
            raise ValueError(f"无效的超分倍数: {self.config['processing']['upscale_factor']}")
        
        # 验证分段时长
        if self.config["processing"]["segment_duration"] <= 0:
            raise ValueError("分段时长必须大于0")
    
    def _setup_paths(self):
        """设置和创建必要的路径"""
        paths = self.config["paths"]

        # base_dir：空时优先从 config 文件位置向上推算项目根，其次用 cwd
        # 推算逻辑：config 文件通常在 <project_root>/config/default_config.json
        if not paths["base_dir"]:
            if self.config_path:
                paths["base_dir"] = str(Path(self.config_path).parent.parent.resolve())
            else:
                paths["base_dir"] = str(Path.cwd())

        base_dir = Path(paths["base_dir"])

        # 输出 / 临时 / 日志目录：空时从 base_dir 派生
        if not paths["output_dir"]:
            paths["output_dir"] = str(base_dir / "output")
        if not paths["temp_dir"]:
            paths["temp_dir"] = str(base_dir / "temp")
        if not paths["log_dir"]:
            paths["log_dir"] = str(base_dir / "logs")

        for key in ["output_dir", "temp_dir", "log_dir"]:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

        # TRT Engine 缓存目录：空时自动派生为 base_dir/.trt_cache
        if not paths.get("trt_cache_dir", ""):
            paths["trt_cache_dir"] = str(base_dir / ".trt_cache")
        # 仅创建父目录；.trt_cache 本身由各处理器在首次构建 Engine 时按需创建
        Path(paths["trt_cache_dir"]).parent.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 模型路径自动派生：model_path 为空时按项目约定目录自动填充
        # ------------------------------------------------------------------
        # 注意：使用 "not value" 判断时，空字符串会被误判为有效值
        # 所以这里用显式检查：确保 model_path 是有效字符串
        # ------------------------------------------------------------------
        ifrnet_cfg = self.config["models"]["ifrnet"]
        esrgan_cfg = self.config["models"]["realesrgan"]

        # 检查 IFRNet 模型路径 - 必须是有效非空字符串
        ifrnet_model_path = ifrnet_cfg.get("model_path")
        if not ifrnet_model_path or not isinstance(ifrnet_model_path, str) or ifrnet_model_path.strip() == "":
            # model_path 是必须字段：IFRNetVideoProcessor 直接调用 torch.load(model_path)
            # 空值无前置校验，会在模型加载阶段抛出 FileNotFoundError
            ifrnet_cfg["model_path"] = str(
                base_dir / "models_IFRNet" / "checkpoints" / "IFRNet_S_Vimeo90K.pth"
            )
            print(f"   ℹ️  IFRNet 模型路径已自动派生: {ifrnet_cfg['model_path']}")

        # 检查 RealESRGAN 模型路径 - 必须是有效非空字符串
        esrgan_model_path = esrgan_cfg.get("model_path")
        if not esrgan_model_path or not isinstance(esrgan_model_path, str) or esrgan_model_path.strip() == "":
            # v6 底层脚本通过 model_name 自动拼路径，model_path 不实际生效
            # 此处仅填充供日志打印及未来可能的校验逻辑使用
            esrgan_cfg["model_path"] = str(
                base_dir / "models_RealESRGAN"
                / (esrgan_cfg.get("model_name", "realesr-general-x4v3") + ".pth")
            )
            print(f"   ℹ️  RealESRGAN 模型路径已自动派生: {esrgan_cfg['model_path']}")
    
    def get(self, *keys, default=None):
        """
        获取配置值
        
        Args:
            *keys: 配置键路径，如 get("processing", "mode")
            default: 默认值
        
        Returns:
            配置值
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys, value):
        """
        设置配置值
        
        Args:
            *keys: 配置键路径
            value: 要设置的值
        """
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径，如果为None则保存到原路径
        """
        if output_path is None:
            if self.config_path is None:
                raise ValueError("未指定保存路径")
            output_path = self.config_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 配置已保存到: {output_path}")
    
    def get_temp_dir(self, subdir: str = "") -> Path:
        """
        获取临时目录路径
        
        Args:
            subdir: 子目录名称
        
        Returns:
            临时目录路径
        """
        temp_dir = Path(self.config["paths"]["temp_dir"])
        if subdir:
            temp_dir = temp_dir / subdir
            temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def get_input_videos(self) -> list:
        """
        获取要处理的视频列表
        
        Returns:
            视频文件路径列表
        """
        videos = []
        
        if self.config["processing"]["batch_mode"]:
            # 批量模式：从目录读取
            input_dir = Path(self.config["paths"]["input_dir"])
            if not input_dir.exists():
                raise FileNotFoundError(f"输入目录不存在: {input_dir}")
            
            # 支持的视频格式
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            
            for ext in video_extensions:
                videos.extend(input_dir.glob(f"*{ext}"))
                videos.extend(input_dir.glob(f"*{ext.upper()}"))
            
            videos = [str(v) for v in videos]
        else:
            # 单文件模式
            input_video = self.config["paths"]["input_video"]
            if not input_video:
                raise ValueError("未指定输入视频")
            
            input_path = Path(input_video)
            if not input_path.exists():
                raise FileNotFoundError(f"输入视频不存在: {input_video}")
            
            videos = [input_video]
        
        return videos
    
    def get_output_path(self, input_path: str, suffix: str = "_processed") -> str:
        """
        生成输出文件路径
        
        Args:
            input_path: 输入文件路径
            suffix: 文件名后缀
        
        Returns:
            输出文件路径
        """
        input_path = Path(input_path)
        output_dir = Path(self.config["paths"]["output_dir"])
        
        # 生成输出文件名
        output_name = f"{input_path.stem}{suffix}{input_path.suffix}"
        output_path = output_dir / output_name
        
        return str(output_path)

    def get_section(self, section_key: str, default: Any = None) -> Any:
        """
        获取配置段落
        
        Args:
            section_key: 段落键名
            default: 默认值
        
        Returns:
            配置段落字典
        """
        return self.config.get(section_key, default)
    
    def __repr__(self):
        """字符串表示"""
        return json.dumps(self.config, indent=2, ensure_ascii=False)


def create_default_config(output_path: str):
    """
    创建默认配置文件
    
    Args:
        output_path: 输出路径
    """
    config = Config()
    config.save(output_path)
    print(f"✅ 默认配置文件已创建: {output_path}")