"""
视频修复工具
用于修复损坏的视频文件
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple, Dict


class VideoFixer:
    """视频修复器"""
    
    def __init__(self):
        pass
    
    def check_video_integrity(self, video_path: str) -> bool:
        """
        检查视频文件完整性
        
        Args:
            video_path: 视频路径
        
        Returns:
            是否完整
        """
        try:
            # 尝试用ffprobe分析
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_format', '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode != 0:
                return False
            
            # 尝试读取几帧
            cmd = [
                'ffmpeg', '-v', 'error',
                '-i', video_path,
                '-t', '1',  # 只读1秒
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
            
        except Exception as e:
            print(f"⚠️  检查视频完整性失败: {e}")
            return False
    
    def fix_video(self, input_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        修复损坏的视频
        
        Args:
            input_path: 输入视频路径
            output_path: 输出路径（可选）
        
        Returns:
            修复后的文件路径，失败返回None
        """
        if not output_path:
            output_path = str(Path(input_path).with_suffix('.fixed.mp4'))
        
        print(f"🔧 尝试修复视频: {input_path}")
        
        # 方法1: 重新编码
        if self._fix_by_reencoding(input_path, output_path):
            return output_path
        
        # 方法2: 复制流
        if self._fix_by_stream_copy(input_path, output_path):
            return output_path
        
        # 方法3: 忽略错误
        if self._fix_ignore_errors(input_path, output_path):
            return output_path
        
        print(f"❌ 无法修复视频")
        return None
    
    def _fix_by_reencoding(self, input_path: str, output_path: str) -> bool:
        """通过重新编码修复"""
        try:
            print(f"   尝试方法1: 重新编码...")
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_path):
                print(f"   ✅ 重新编码成功")
                return True
                
        except Exception as e:
            print(f"   ❌ 重新编码失败: {e}")
        
        return False
    
    def _fix_by_stream_copy(self, input_path: str, output_path: str) -> bool:
        """通过流复制修复"""
        try:
            print(f"   尝试方法2: 流复制...")
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c', 'copy',
                '-bsf:a', 'aac_adtstoasc',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=180)
            
            if result.returncode == 0 and os.path.exists(output_path):
                print(f"   ✅ 流复制成功")
                return True
                
        except Exception as e:
            print(f"   ❌ 流复制失败: {e}")
        
        return False
    
    def _fix_ignore_errors(self, input_path: str, output_path: str) -> bool:
        """忽略错误尝试修复"""
        try:
            print(f"   尝试方法3: 忽略错误...")
            cmd = [
                'ffmpeg', '-y',
                '-err_detect', 'ignore_err',
                '-i', input_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                print(f"   ✅ 忽略错误修复成功")
                return True
                
        except Exception as e:
            print(f"   ❌ 忽略错误修复失败: {e}")
        
        return False
    
    def auto_fix_if_needed(self, video_path: str) -> str:
        """
        如果视频损坏则自动修复
        
        Args:
            video_path: 视频路径
        
        Returns:
            视频路径（原始或修复后）
        """
        if self.check_video_integrity(video_path):
            print(f"✅ 视频完整: {video_path}")
            return video_path
        
        print(f"⚠️  检测到视频可能损坏: {video_path}")
        
        fixed_path = self.fix_video(video_path)
        
        if fixed_path and self.check_video_integrity(fixed_path):
            print(f"✅ 视频已修复: {fixed_path}")
            return fixed_path
        else:
            print(f"⚠️  无法修复视频，使用原始文件")
            return video_path


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="视频修复工具")
    parser.add_argument("input", help="输入视频文件")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("--check-only", action="store_true", 
                       help="仅检查完整性，不修复")
    
    args = parser.parse_args()
    
    fixer = VideoFixer()
    
    if args.check_only:
        is_ok = fixer.check_video_integrity(args.input)
        print(f"视频完整性: {'✅ 正常' if is_ok else '❌ 损坏'}")
        return 0 if is_ok else 1
    else:
        fixed = fixer.auto_fix_if_needed(args.input)
        if args.output and fixed != args.input:
            import shutil
            shutil.move(fixed, args.output)
            print(f"✅ 修复后的视频已保存到: {args.output}")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
