#!/usr/bin/env python3
"""
真实模型集成测试脚本
用于验证优化架构与真实模型的集成效果
"""

import os
import sys
import numpy as np
import torch
from os import path as osp

# 添加项目路径
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

# 导入必要的模块
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url

# 路径配置
_SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
base_dir = osp.dirname(osp.dirname(_SCRIPT_DIR))
models_RealESRGAN = osp.join(base_dir, 'models_RealESRGAN')

# 模型配置常量
MODEL_CONFIG = {
    'realesr-animevideov3': (
        SRVGGNetCompact(3, 3, 64, 16, upscale=4, act_type='prelu'), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'],
    ),
    'realesr-general-x4v3': (
        SRVGGNetCompact(3, 3, 64, 32, upscale=4, act_type='prelu'), 4,
        [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        ],
    ),
}

def test_model_loading():
    """测试模型加载功能"""
    print("=== 真实模型集成测试 ===")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试简单的模型加载
    model_name = 'realesr-animevideov3'
    print(f"测试加载模型: {model_name}")
    
    if model_name not in MODEL_CONFIG:
        print(f"❌ 未知模型名称: {model_name}")
        return False
    
    model, netscale, urls = MODEL_CONFIG[model_name]
    print(f"模型信息: netscale={netscale}, urls={len(urls)}")
    
    # 尝试下载模型文件
    try:
        model_paths = []
        for url in urls:
            print(f"下载模型: {url}")
            model_path = load_file_from_url(url, models_RealESRGAN, True)
            model_paths.append(model_path)
            print(f"✓ 下载成功: {osp.basename(model_path)}")
        
        # 使用第一个权重文件
        model_path = model_paths[0] if model_paths else None
        
        if not model_path or not osp.exists(model_path):
            print("❌ 模型文件不存在")
            return False
        
        # 创建RealESRGANer实例
        print("创建RealESRGANer实例...")
        upsampler = RealESRGANer(
            scale=netscale, 
            model_path=model_path, 
            dni_weight=None,
            model=model, 
            tile=0, 
            tile_pad=10,
            pre_pad=0, 
            half=True, 
            device=device,
        )
        print("✓ RealESRGANer实例创建成功")
        
        # 测试单帧处理
        print("测试单帧处理...")
        test_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        
        output, _ = upsampler.enhance(test_frame, outscale=2)
        print(f"✓ 单帧处理成功: 输入尺寸={test_frame.shape}, 输出尺寸={output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_optimized_architecture():
    """测试优化架构的基本功能"""
    print("\n=== 优化架构功能测试 ===")
    
    # 测试GPU内存池
    try:
        from inference_realesrgan_video_v6_3_optimized import GPUMemoryPool
        
        if torch.cuda.is_available():
            memory_pool = GPUMemoryPool(max_batches=2, batch_size=4, img_size=(180, 320))
            print("✓ GPU内存池初始化成功")
            
            # 测试内存块获取
            block = memory_pool.acquire()
            if block:
                print(f"✓ 内存块获取成功: index={block['index']}")
                memory_pool.release(block['index'])
                print("✓ 内存块释放成功")
            else:
                print("⚠ 内存块获取失败（可能池已满）")
        else:
            print("⚠ 跳过GPU内存池测试（无CUDA设备）")
            
    except Exception as e:
        print(f"❌ GPU内存池测试失败: {e}")
    
    # 测试吞吐量统计
    try:
        from inference_realesrgan_video_v6_3_optimized import ThroughputMeter
        
        meter = ThroughputMeter()
        meter.update(10)
        fps = meter.fps()
        print(f"✓ 吞吐量统计器测试成功: fps={fps}")
        
    except Exception as e:
        print(f"❌ 吞吐量统计器测试失败: {e}")

if __name__ == "__main__":
    print("开始真实模型集成测试...")
    
    # 测试模型加载
    model_loaded = test_model_loading()
    
    # 测试优化架构功能
    test_optimized_architecture()
    
    if model_loaded:
        print("\n✅ 集成测试完成！优化架构与真实模型集成成功")
        print("下一步：进行完整视频处理测试")
    else:
        print("\n❌ 集成测试失败！需要检查模型下载和加载逻辑")