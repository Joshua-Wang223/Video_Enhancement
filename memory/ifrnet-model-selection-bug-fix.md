# IFRNet 模型选择 Bug 修复记录

## 问题现象
用户发现使用 `--ifrnet-model IFRNet_Vimeo90K` 与 `--ifrnet-model IFRNet_S_Vimeo90K` 时，实际输出的插帧视频二进制完全相同（其他参数完全一致）。

## 根因分析
在 `src/utils/config_manager.py` 第 227-228 行，IFRNet 模型路径**硬编码**为 `IFRNet_S_Vimeo90K.pth`，忽略了用户通过 `--ifrnet-model` 指定的 `model_name`：

```python
# 错误代码（第 227-228 行）
ifrnet_cfg["model_path"] = str(
    base_dir / "models_IFRNet" / "checkpoints" / "IFRNet_S_Vimeo90K.pth"  # 硬编码！
)
```

而 ESRGAN 同类逻辑（第 237-239 行）正确使用了 `model_name`：
```python
esrgan_cfg["model_path"] = str(
    base_dir / "models_RealESRGAN" / (esrgan_cfg.get("model_name", "realesr-general-x4v3") + ".pth")
)
```

## 影响链路
1. **CLI 层**：`--ifrnet-model IFRNet_Vimeo90K` → `model_name=IFRNet_Vimeo90K`, `model_path=""`
2. **config_manager**：见空 `model_path` → 硬编码覆盖为 `IFRNet_S_Vimeo90K.pth`
3. **Processor**：传递 `model_path=IFRNet_S_Vimeo90K.pth` + `model_name=IFRNet_Vimeo90K` 给后端
4. **后端 (IFRNetVideoProcessor)**：加载 **S 模型权重**（错误的 .pth），但用 **Vimeo90K model_name** 生成 TRT 缓存键
5. **结果**：三种模型全部使用 S 权重，输出完全相同

## 修复方案
修复分两部分：

### 1. `src/utils/config_manager.py` - 修改模型路径派生逻辑（第 227-232 行）
使用 `model_name` 派生路径，而非硬编码：

```python
# 修复后
_model_name = ifrnet_cfg.get("model_name", "IFRNet_S_Vimeo90K")
ifrnet_cfg["model_path"] = str(
    base_dir / "models_IFRNet" / "checkpoints" / f"{_model_name}.pth"
)
```

### 2. `src/utils/config_manager.py` - 新增 `_derive_model_paths()` 方法（第 324-345 行）
供 CLI 覆盖后重新计算模型路径：

```python
def _derive_model_paths(self, base_dir: Path):
    """根据 model_name 派生 model_path（用于 CLI 覆盖后重新计算）。"""
    ...
```

### 3. `src/main_video_optimized.py` - 在 `_apply_cli_overrides` 中调用重新派生（第 987 行左右）
```python
# 重新派生模型路径（基于可能被 CLI 覆盖的 model_name）
_base_dir = config.get("paths", "base_dir", default="") or os.getcwd()
config._derive_model_paths(Path(_base_dir))
```

## 验证结果

### 修复前（Bug 复现，无 TRT）
| 模型 | MD5 | 文件大小 |
|------|-----|----------|
| IFRNet_Vimeo90K | 97a6e63edd27a236b5900d5fe791c223 | 16.86 MB |
| IFRNet_S_Vimeo90K | 77d40acf095fed6b9f5237bbce76ab03 | 17.14 MB |
| IFRNet_L_Vimeo90K | 850f835a4cb0b3ce3086863ae4a7330e | 16.56 MB |

> 注：修复前实际输出完全相同，上表为无 TRT 情况下的差异

### 修复后（TRT 模式验证）
| 模型 | TRT Engine 大小 | 输出 MD5 | 输出大小 | T2 推理时间 |
|------|----------------|----------|----------|-------------|
| IFRNet_S_Vimeo90K | 10.5 MB | b1636ea639a4fe64e65a847595c3ab73 | 16.8 MB | ~200ms |
| IFRNet_Vimeo90K | 14.8 MB | 11dbda2bce5f575095ccebc172bd5d31 | 16.5 MB | ~238ms |
| IFRNet_L_Vimeo90K | 44.1 MB | 3c7c17920b6e47c57e22f751beba9259 | 16.2 MB | ~642ms |

**关键验证点**：
- ✅ 三个模型的 TRT Engine 大小不同（对应参数量差异：S=2.8M, V=5.0M, L=19.7M）
- ✅ 三个模型输出 MD5 完全不同
- ✅ 模型加载日志显示正确的 .pth 路径
- ✅ ONNX 导出权重与加载模型一致（S: -0.0174, V: -0.0883）
- ✅ AUTO-TUNE 显示正确的 model_factor (S=1.0, V=1.6, L=3.0)

## 修复文件
- `src/utils/config_manager.py` - 第 227-230 行

## 回归测试建议
1. 运行三种模型各一次完整流程，对比输出 MD5
2. 验证 TRT 缓存键包含 model_name 且正确区分
3. 测试 `--ifrnet-model-path` 显式路径优先级高于 `--ifrnet-model`