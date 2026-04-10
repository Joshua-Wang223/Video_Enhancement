#!/usr/bin/env python3
"""
性能对比测试脚本
对比原始v6.2版本与优化v6.3版本的性能差异
"""

import os
import sys
import time
import subprocess
import json
from os import path as osp

# 添加项目路径
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

# 测试配置
TEST_VIDEO = "/workspace/input_videos/test1.mp4"
OUTPUT_DIR = "/workspace/output/performance_test"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_original_v6_2():
    """运行原始v6.2版本"""
    print("=== 运行原始v6.2版本 ===")
    
    output_path = osp.join(OUTPUT_DIR, "v6_2_result.mp4")
    
    # 构建命令
    cmd = [
        sys.executable, 
        "inference_realesrgan_video_v6_2_single.py",
        "-i", TEST_VIDEO,
        "-o", output_path,
        "--batch-size", "4",
        "--prefetch-factor", "16"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=osp.dirname(__file__))
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 解析输出获取FPS信息
        fps = parse_fps_from_output(result.stdout)
        
        return {
            "version": "v6.2_original",
            "success": result.returncode == 0,
            "execution_time": execution_time,
            "fps": fps,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except Exception as e:
        return {
            "version": "v6.2_original",
            "success": False,
            "error": str(e),
            "execution_time": 0
        }

def run_optimized_v6_3():
    """运行优化v6.3版本"""
    print("=== 运行优化v6.3版本 ===")
    
    output_path = osp.join(OUTPUT_DIR, "v6_3_result.mp4")
    
    # 构建命令
    cmd = [
        sys.executable, 
        "inference_realesrgan_video_v6_3_optimized.py",
        "-i", TEST_VIDEO,
        "-o", output_path,
        "--batch-size", "4",
        "--prefetch-factor", "16"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=osp.dirname(__file__))
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 解析输出获取FPS信息
        fps = parse_fps_from_output(result.stdout)
        
        return {
            "version": "v6.3_optimized",
            "success": result.returncode == 0,
            "execution_time": execution_time,
            "fps": fps,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except Exception as e:
        return {
            "version": "v6.3_optimized",
            "success": False,
            "error": str(e),
            "execution_time": 0
        }

def parse_fps_from_output(output):
    """从输出中解析FPS信息"""
    lines = output.split('\n')
    for line in lines:
        if 'fps=' in line:
            # 查找类似 "fps=6.48" 的模式
            import re
            fps_match = re.search(r'fps=([\d.]+)', line)
            if fps_match:
                return float(fps_match.group(1))
    return 0.0

def compare_performance(results):
    """比较性能结果"""
    print("\n=== 性能对比结果 ===")
    
    v6_2_result = None
    v6_3_result = None
    
    for result in results:
        if result["version"] == "v6.2_original":
            v6_2_result = result
        elif result["version"] == "v6.3_optimized":
            v6_3_result = result
    
    if not v6_2_result or not v6_3_result:
        print("❌ 无法获取完整的性能数据")
        return
    
    if not v6_2_result["success"] or not v6_3_result["success"]:
        print("❌ 有测试版本运行失败")
        return
    
    # 计算性能提升
    time_improvement = (v6_2_result["execution_time"] - v6_3_result["execution_time"]) / v6_2_result["execution_time"] * 100
    fps_improvement = (v6_3_result["fps"] - v6_2_result["fps"]) / v6_2_result["fps"] * 100
    
    print(f"原始v6.2版本:")
    print(f"  - 执行时间: {v6_2_result['execution_time']:.2f}秒")
    print(f"  - 平均FPS: {v6_2_result['fps']:.2f}")
    
    print(f"优化v6.3版本:")
    print(f"  - 执行时间: {v6_3_result['execution_time']:.2f}秒")
    print(f"  - 平均FPS: {v6_3_result['fps']:.2f}")
    
    print(f"\n性能提升:")
    print(f"  - 时间减少: {time_improvement:.1f}%")
    print(f"  - FPS提升: {fps_improvement:.1f}%")
    
    # 保存详细结果
    result_file = osp.join(OUTPUT_DIR, "performance_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {result_file}")

def main():
    """主测试函数"""
    print("开始性能对比测试...")
    print(f"测试视频: {TEST_VIDEO}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 运行两个版本的测试
    results = []
    
    # 先运行优化版本（因为原始版本可能还在运行）
    results.append(run_optimized_v6_3())
    
    # 再运行原始版本
    results.append(run_original_v6_2())
    
    # 比较性能
    compare_performance(results)

if __name__ == "__main__":
    main()