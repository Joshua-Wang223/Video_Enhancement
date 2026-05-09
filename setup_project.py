#!/usr/bin/env python3
"""
创建项目目录结构（output / temp / logs / models_* / .trt_cache），可选克隆外部仓库与下载权重。
默认真源配置见仓库内 config/default_config.json（本脚本默认不覆盖已有配置文件）。
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

class ProjectSetup:
    """项目设置管理器"""
    
    def __init__(self, base_dir=None):
        _repo_root = Path(__file__).resolve().parent
        self.base_dir = Path(base_dir).resolve() if base_dir else _repo_root
        self.config = {
            "base_dir": str(self.base_dir),
            "directories": {
                "output": "output",
                "temp": "temp",
                "logs": "logs",
                "models_IFRNet_checkpoints": "models_IFRNet/checkpoints",
                "models_RealESRGAN": "models_RealESRGAN",
                "models_GFPGAN": "models_GFPGAN",
                "trt_cache": ".trt_cache",
                "config": "config",
                "external": "external",
            },
            "repositories": {
                "IFRNet": {
                    "url": "https://github.com/ltkong218/IFRNet.git",
                    "path": "external/IFRNet"
                },
                "Real-ESRGAN": {
                    "url": "https://github.com/xinntao/Real-ESRGAN.git",
                    "path": "external/Real-ESRGAN"
                }
            },
            "models_to_download": {
                "IFRNet": [
                    {
                        "name": "IFRNet_S_Vimeo90K.pth",
                        "url": "https://github.com/ltkong218/IFRNet/releases/download/v1.0/IFRNet_S.pth",
                        "dir": "models_IFRNet/checkpoints"
                    },
                    {
                        "name": "IFRNet_L_Vimeo90K.pth",
                        "url": "https://github.com/ltkong218/IFRNet/releases/download/v1.0/IFRNet_L.pth",
                        "dir": "models_IFRNet/checkpoints"
                    }
                ],
                "RealESRGAN": [
                    {
                        "name": "realesr-general-x4v3.pth",
                        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                        "dir": "models_RealESRGAN"
                    },
                    {
                        "name": "RealESRGAN_x4plus.pth",
                        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                        "dir": "models_RealESRGAN"
                    }
                ]
            }
        }
    
    def create_directories(self):
        """创建项目目录结构"""
        print("=" * 60)
        print("📁 创建项目目录结构...")
        print("=" * 60)
        
        # 创建基础目录
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 基础目录: {self.base_dir}")
        
        # 创建所有子目录
        for name, path in self.config["directories"].items():
            full_path = self.base_dir / path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ {name}: {full_path}")
        
        print("\n✅ 目录结构创建完成！\n")
    
    def check_system_dependencies(self):
        """检查系统依赖"""
        print("=" * 60)
        print("🔍 检查系统依赖...")
        print("=" * 60)
        
        required_commands = ["ffmpeg", "ffprobe", "git"]
        missing = []
        
        for cmd in required_commands:
            try:
                if cmd in ["ffmpeg", "ffprobe"]:
                    # 检查 ffmpeg/ffprobe
                    result = subprocess.run([cmd, "-version"], 
                                          capture_output=True, 
                                          text=True,
                                          timeout=5)
                else:
                    # 检查 git
                    result = subprocess.run([cmd, "--version"], 
                                          capture_output=True, 
                                          text=True,
                                          timeout=5)
                
                if result.returncode == 0:
                    print(f"✅ {cmd}: 已安装")
                else:
                    missing.append(cmd)
                    print(f"❌ {cmd}: 未找到")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                missing.append(cmd)
                print(f"❌ {cmd}: 未找到")
        
        if missing:
            print(f"\n⚠️  缺少依赖: {', '.join(missing)}")
            print("请先安装这些依赖:")
            
            if any(cmd in missing for cmd in ["ffmpeg", "ffprobe"]):
                print("Ubuntu/Debian: sudo apt-get install ffmpeg")
                print("macOS: brew install ffmpeg")
            
            if "git" in missing:
                print("Ubuntu/Debian: sudo apt-get install git")
                print("macOS: brew install git")
            
            return False
        
        print("\n✅ 所有系统依赖已满足！\n")
        return True
    
    def install_python_dependencies(self):
        """提示安装 Python 依赖（使用仓库根目录 requirements.txt，不覆盖现有文件）"""
        print("=" * 60)
        print("📦 Python 依赖说明")
        print("=" * 60)

        req_file = self.base_dir / "requirements.txt"
        if req_file.is_file():
            print(f"✅ 使用项目依赖清单: {req_file}")
        else:
            print(f"⚠️  未找到 {req_file}，请从仓库拷贝 requirements.txt 后再安装。")

        print("\n请先按 README.md 安装匹配的 PyTorch（CUDA wheel），再执行:")
        print(f"  pip install -r {req_file}")
        print()
    
    def clone_repositories(self):
        """克隆开源项目"""
        print("=" * 60)
        print("📥 克隆开源项目...")
        print("=" * 60)
        
        for name, info in self.config["repositories"].items():
            repo_path = self.base_dir / info["path"]
            
            if repo_path.exists():
                print(f"⏭️  {name}: 已存在，跳过克隆")
                continue
            
            print(f"📥 克隆 {name}...")
            try:
                subprocess.run([
                    "git", "clone", "--depth", "1",
                    info["url"], str(repo_path)
                ], check=True, capture_output=True)
                print(f"✅ {name}: 克隆成功")
            except subprocess.CalledProcessError as e:
                print(f"❌ {name}: 克隆失败")
                print(f"   错误: {e.stderr.decode() if e.stderr else str(e)}")
        
        print("\n✅ 代码仓库克隆完成！\n")
    
    def download_models(self):
        """下载预训练模型"""
        print("=" * 60)
        print("📥 下载预训练模型...")
        print("=" * 60)
        print("⚠️  模型文件较大，需要一些时间...")
        print("💡 您也可以手动下载模型文件到对应目录\n")
        
        for category, models in self.config["models_to_download"].items():
            print(f"\n{category} 模型:")
            for model_info in models:
                model_dir = self.base_dir / model_info["dir"]
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / model_info["name"]
                
                if model_path.exists():
                    print(f"⏭️  {model_info['name']}: 已存在，跳过下载")
                    continue
                
                print(f"📥 下载 {model_info['name']}...")
                print(f"   URL: {model_info['url']}")
                print(f"   保存到: {model_path}")
                
                # 使用wget或curl下载
                try:
                    # 尝试使用wget
                    subprocess.run([
                        "wget", "-O", str(model_path),
                        model_info["url"]
                    ], check=True, capture_output=True)
                    print(f"✅ 下载完成")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    try:
                        # 尝试使用curl
                        subprocess.run([
                            "curl", "-L", "-o", str(model_path),
                            model_info["url"]
                        ], check=True, capture_output=True)
                        print(f"✅ 下载完成")
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        print(f"⚠️  自动下载失败，请手动下载")
                        print(f"   从: {model_info['url']}")
                        print(f"   到: {model_path}")
        
        print("\n✅ 模型下载流程完成！\n")
    
    def create_config_file(self):
        """若不存在则写入最小占位配置；否则保留仓库内的 default_config.json（真源）。"""
        print("=" * 60)
        print("⚙️  配置文件")
        print("=" * 60)

        config_path = self.base_dir / "config" / "default_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.is_file():
            print(f"⏭️  已存在，跳过写入（请直接编辑）: {config_path}\n")
            return config_path

        placeholder = {
            "//": "请替换为完整配置或从仓库拷贝 config/default_config.json",
            "processing": {"mode": "interpolate_then_upscale"},
            "paths": {"base_dir": ""},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(placeholder, f, indent=4, ensure_ascii=False)

        print(f"✅ 已创建占位配置: {config_path}")
        print("   建议用仓库中的 config/default_config.json 覆盖此文件。\n")
        return config_path
    
    def save_setup_info(self):
        """保存设置信息"""
        info_path = self.base_dir / "setup_info.json"
        
        setup_info = {
            "setup_date": datetime.now().isoformat(timespec="seconds"),
            "python_version": sys.version,
            "base_dir": str(self.base_dir),
            "config": self.config
        }
        
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(setup_info, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 设置信息已保存: {info_path}")
    
    def run_setup(self):
        """运行完整设置流程"""
        print("\n" + "=" * 60)
        print("🚀 视频处理项目初始化")
        print("=" * 60)
        print(f"📍 项目目录: {self.base_dir}\n")
        
        # 1. 创建目录结构
        self.create_directories()
        
        # 2. 检查系统依赖
        if not self.check_system_dependencies():
            print("⚠️  请先安装缺失的系统依赖后重新运行")
            return False
        
        # 3. 安装Python依赖
        self.install_python_dependencies()
        
        # 4. 克隆开源项目
        self.clone_repositories()
        
        # 5. 下载模型（可选）
        download = input("\n是否下载预训练模型? (y/n，默认n): ").strip().lower()
        if download == 'y':
            self.download_models()
        else:
            print("⏭️  跳过模型下载，请稍后手动下载")
        
        # 6. 创建配置文件
        self.create_config_file()
        
        # 7. 保存设置信息
        self.save_setup_info()
        
        print("\n" + "=" * 60)
        print("✅ 项目初始化完成！")
        print("=" * 60)
        print(f"📁 项目目录: {self.base_dir}")
        print(f"⚙️  配置文件: {self.base_dir / 'config/default_config.json'}")
        print(f"📝 要求文件: {self.base_dir / 'requirements.txt'}")
        print("\n下一步:")
        print("1. 按 README 安装 PyTorch 后: pip install -r requirements.txt")
        print("2. 将模型放入 models_IFRNet/checkpoints、models_RealESRGAN（及可选 models_GFPGAN）")
        print("3. 编辑 config/default_config.json（默认真源）")
        print("4. 运行: python src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output/out.mp4")
        print("   或: python run.py -c config/default_config.json -i input.mp4 -o output/out.mp4")
        print("=" * 60 + "\n")
        
        return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="视频处理项目初始化")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="项目根目录（默认: 本脚本所在目录，即仓库根）"
    )
    
    args = parser.parse_args()
    
    setup = ProjectSetup(base_dir=args.base_dir)
    success = setup.run_setup()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
