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
    
    # 默认配置
    DEFAULT_CONFIG = {
        "processing": {
            "mode": "interpolate_then_upscale",  # 处理顺序: interpolate_then_upscale 或 upscale_then_interpolate
            "interpolation_factor": 2,  # 插帧倍数 (2, 4, 8等)
            "upscale_factor": 2,  # 超分倍数 (2, 4等)
            "segment_duration": 30,  # 分段时长(秒)
            "auto_fix_corrupted": True,  # 是否自动修复损坏的视频
            "auto_cleanup_temp": False,  # 是否自动清理临时文件
            "batch_mode": False,  # 是否批量处理
        },
        "paths": {
            "base_dir": "",
            "input_video": "",  # 单个视频路径
            "input_dir": "",  # 批量处理时的输入目录
            "output_dir": "",
            "temp_dir": "",
            "log_dir": "",
        },
        "models": {
            "ifrnet": {
                "model_path": "",
                "model_size": "L",  # L (Large) 或 S (Small)
                "use_gpu": True,
                "batch_size": 1,
            },
            "realesrgan": {
                "model_name": "RealESRGAN_x2plus",
                "model_path": "",
                "denoise_strength": 0.5,  # 0-1
                "tile_size": 0,  # 0为自动, 或指定如400, 512等
                "tile_pad": 10,
                "pre_pad": 0,
                "use_gpu": True,
                "fp32": False,  # 使用FP32精度(更慢但更精确)
            }
        },
        "output": {
            "format": "mp4",
            "codec": "libx265",
            "preset": "medium",  # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
            "crf": 18,  # 0-51, 越小质量越好
            "pix_fmt": "yuv420p",
            "audio_codec": "aac",
            "audio_bitrate": "192k",
            "copy_audio": True,  # 是否复制原音频
        },
        "temp_files": {
            "segment_prefix": "segment_",
            "processed_prefix": "processed_",
            "temp_video_suffix": "_temp.mp4",
        },
        "logging": {
            "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
            "log_to_file": True,
            "log_to_console": True,
        }
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
        """递归合并配置字典"""
        for key, value in override.items():
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
        
        # 如果没有设置base_dir，使用当前目录
        if not paths["base_dir"]:
            paths["base_dir"] = str(Path.cwd())
        
        base_dir = Path(paths["base_dir"])
        
        # 设置默认路径
        if not paths["output_dir"]:
            paths["output_dir"] = str(base_dir / "output")
        if not paths["temp_dir"]:
            paths["temp_dir"] = str(base_dir / "temp")
        if not paths["log_dir"]:
            paths["log_dir"] = str(base_dir / "logs")
        
        # 创建必要的目录
        for key in ["output_dir", "temp_dir", "log_dir"]:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)
    
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