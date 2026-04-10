#!/usr/bin/env python3
"""
参数调优脚本
针对优化架构进行参数调优，找到最优配置
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
OUTPUT_DIR = "/workspace/output/parameter_tuning"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 参数调优配置
PARAMETER_CONFIGS = [
    # 基础配置（当前最优）
    {
        "name": "base_optimal",
        "batch_size": 6,
        "prefetch_factor": 48,
        "memory_pool_batches": 4,
        "queue_depths": [48, 32, 16, 16]  # F, D, S, G
    },
    # 高内存配置（适合大显存）
    {
        "name": "high_memory",
        "batch_size": 8,
        "prefetch_factor": 64,
        "memory_pool_batches": 6,
        "queue_depths": [64, 48, 24, 24]
    },
    # 平衡配置（通用性最佳）
    {
        "name": "balanced",
        "batch_size": 4,
        "prefetch_factor": 32,
        "memory_pool_batches": 3,
        "queue_depths": [32, 24, 12, 12]
    },
    # 低内存配置（适合小显存）
    {
        "name": "low_memory",
        "batch_size": 2,
        "prefetch_factor": 16,
        "memory_pool_batches": 2,
        "queue_depths": [16, 12, 8, 8]
    },
    # 高吞吐量配置（适合高速存储）
    {
        "name": "high_throughput",
        "batch_size": 8,
        "prefetch_factor": 96,
        "memory_pool_batches": 8,
        "queue_depths": [96, 64, 32, 32]
    }
]

def run_optimized_with_config(config):
    """使用特定配置运行优化版本"""
    print(f"=== 测试配置: {config['name']} ===")
    
    output_path = osp.join(OUTPUT_DIR, f"{config['name']}_result.mp4")
    
    # 构建命令
    cmd = [
        sys.executable, 
        "inference_realesrgan_video_v6_3_optimized.py",
        "-i", TEST_VIDEO,
        "-o", output_path,
        "--batch-size", str(config["batch_size"]),
        "--prefetch-factor", str(config["prefetch_factor"])
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=osp.dirname(__file__))
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 解析输出获取性能指标
        performance_data = parse_performance_metrics(result.stdout)
        
        return {
            "config_name": config["name"],
            "parameters": config,
            "success": result.returncode == 0,
            "execution_time": execution_time,
            "performance": performance_data,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except Exception as e:
        return {
            "config_name": config["name"],
            "parameters": config,
            "success": False,
            "error": str(e),
            "execution_time": 0,
            "performance": {}
        }

def parse_performance_metrics(output):
    """从输出中解析性能指标"""
    metrics = {
        "fps": 0.0,
        "avg_processing_time": 0.0,
        "min_processing_time": 0.0,
        "max_processing_time": 0.0,
        "queue_utilization": {}
    }
    
    lines = output.split('\n')
    
    # 解析FPS
    for line in lines:
        if 'fps=' in line:
            import re
            fps_match = re.search(r'fps=([\d.]+)', line)
            if fps_match:
                metrics["fps"] = float(fps_match.group(1))
                break
    
    # 解析处理时间
    for line in lines:
        if '平均处理时间:' in line:
            import re
            time_match = re.search(r'平均处理时间:\s*([\d.]+)ms', line)
            if time_match:
                metrics["avg_processing_time"] = float(time_match.group(1))
        elif '最快处理时间:' in line:
            import re
            time_match = re.search(r'最快处理时间:\s*([\d.]+)ms', line)
            if time_match:
                metrics["min_processing_time"] = float(time_match.group(1))
        elif '最慢处理时间:' in line:
            import re
            time_match = re.search(r'最慢处理时间:\s*([\d.]+)ms', line)
            if time_match:
                metrics["max_processing_time"] = float(time_match.group(1))
    
    # 解析队列利用率
    for line in lines:
        if 'queue_sizes=' in line:
            import re
            queue_match = re.search(r'queue_sizes=([^,]+)', line)
            if queue_match:
                queue_str = queue_match.group(1)
                # 解析类似 "F:12/D:8/S:4/G:2" 的格式
                queue_parts = queue_str.split('/')
                for part in queue_parts:
                    if ':' in part:
                        key, value = part.split(':')
                        metrics["queue_utilization"][key.strip()] = int(value.strip())
    
    return metrics

def analyze_performance_results(results):
    """分析性能结果"""
    print("\n=== 参数调优分析结果 ===")
    
    successful_results = [r for r in results if r["success"]]
    
    if not successful_results:
        print("❌ 所有配置测试均失败")
        return
    
    # 按FPS排序
    successful_results.sort(key=lambda x: x["performance"]["fps"], reverse=True)
    
    print("\n配置性能排名（按FPS降序）:")
    print("-" * 80)
    
    for i, result in enumerate(successful_results, 1):
        perf = result["performance"]
        params = result["parameters"]
        
        print(f"{i}. {result['config_name']}:")
        print(f"   参数: batch_size={params['batch_size']}, prefetch={params['prefetch_factor']}")
        print(f"   性能: FPS={perf['fps']:.2f}, 时间={result['execution_time']:.1f}s")
        print(f"   处理时间: 平均={perf['avg_processing_time']:.1f}ms, 最小={perf['min_processing_time']:.1f}ms, 最大={perf['max_processing_time']:.1f}ms")
        
        if perf["queue_utilization"]:
            print(f"   队列利用率: {perf['queue_utilization']}")
        
        print()
    
    # 找出最优配置
    best_config = successful_results[0]
    print(f"✅ 最优配置: {best_config['config_name']}")
    print(f"   - FPS: {best_config['performance']['fps']:.2f}")
    print(f"   - 执行时间: {best_config['execution_time']:.1f}秒")
    print(f"   - Batch Size: {best_config['parameters']['batch_size']}")
    print(f"   - Prefetch Factor: {best_config['parameters']['prefetch_factor']}")
    
    # 保存详细分析结果
    analysis_file = osp.join(OUTPUT_DIR, "parameter_analysis.json")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump({
            "best_config": best_config,
            "all_results": results,
            "ranking": [r["config_name"] for r in successful_results]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细分析结果已保存到: {analysis_file}")
    
    return best_config

def generate_recommendations(best_config):
    """生成优化建议"""
    print("\n=== 优化建议 ===")
    
    params = best_config["parameters"]
    perf = best_config["performance"]
    
    print("基于测试结果，推荐以下优化策略:")
    
    # 根据最优配置给出建议
    if params["batch_size"] >= 6:
        print("✅ 当前Batch Size设置合理，适合充分利用GPU并行能力")
    else:
        print("⚠️ 考虑增加Batch Size以提升GPU利用率")
    
    if params["prefetch_factor"] >= 32:
        print("✅ 预取深度设置充分，有效减少I/O等待时间")
    else:
        print("⚠️ 考虑增加预取深度以进一步减少I/O瓶颈")
    
    if perf["queue_utilization"]:
        queue_info = perf["queue_utilization"]
        print(f"📊 队列利用率分析:")
        for queue_name, usage in queue_info.items():
            if usage > 0:
                print(f"   - {queue_name}队列: 活跃使用中（利用率: {usage}）")
            else:
                print(f"   - {queue_name}队列: 空闲（可能成为瓶颈）")
    
    # 根据处理时间给出建议
    if perf["max_processing_time"] / perf["min_processing_time"] > 3:
        print("⚠️ 处理时间波动较大，建议检查是否有不稳定的处理阶段")
    else:
        print("✅ 处理时间稳定，流水线运行良好")
    
    print(f"\n🎯 推荐命令行参数:")
    print(f"   --batch-size {params['batch_size']} --prefetch-factor {params['prefetch_factor']}")

def main():
    """主调优函数"""
    print("开始参数调优测试...")
    print(f"测试视频: {TEST_VIDEO}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"测试配置数量: {len(PARAMETER_CONFIGS)}")
    
    results = []
    
    # 依次测试每个配置
    for config in PARAMETER_CONFIGS:
        result = run_optimized_with_config(config)
        results.append(result)
        
        # 短暂休息，避免系统过载
        time.sleep(2)
    
    # 分析结果
    best_config = analyze_performance_results(results)
    
    if best_config:
        generate_recommendations(best_config)
    
    print("\n🎉 参数调优完成！")

if __name__ == "__main__":
    main()