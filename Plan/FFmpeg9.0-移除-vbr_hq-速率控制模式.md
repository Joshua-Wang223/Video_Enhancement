# FFmpeg 9.0 移除 `vbr_hq` 速率控制模式

## 一句话结论

`vbr_hq` 的移除是 **NVIDIA NVENC SDK 升级导致的必然结果**，并非 FFmpeg 自身的决定。虽然选项名称消失，但通过 `-rc vbr -tune hq -multipass fullres` 的组合仍可实现高质量可变码率编码。

---

## 一、事件背景

| 项目 | 内容 |
|---|---|
| 发生版本 | FFmpeg **9.0**（代号 "Lei"） |
| 变更内容 | 正式移除 `vbr_hq` 速率控制模式 |
| 根本原因 | 所依赖的 **NVIDIA NVENC 视频编码 SDK** 发生根本性变化 |

**因果链**：NVIDIA 在新版 SDK 中废弃并移除了 `VBR_HQ` 等一批旧速率控制模式 → FFmpeg 9.0 相应移除已废弃的 NVENC 选项 → 同时**停止支持 11.1 版本之前的 SDK**。

---

## 二、具体表现

在 FFmpeg 9.0+ 中继续使用 `-rc:v vbr_hq`，会报错：

```
Specified rc mode is deprecated.
Use -rc constqp/cbr/vbr, -tune and -multipass instead.
```

---

## 三、替代方案

官方迁移建议：用 `-rc vbr` 配合 `-tune hq` 和 `-multipass` 实现同等或更优的画质控制。

```bash
ffmpeg -i input.mp4 -c:v h264_nvenc -rc vbr -tune hq -cq 23 -b:v 0 output.mp4
```

- `-cq`：控制质量，**数值越低质量越高**
- `-b:v 0`：让编码器完全依据 `-cq` 值决定码率

---

## 四、移除原因：从“黑盒预设”到“模块化”

旧版 `VBR_HQ` 是一个**一体化的“高级”预设**，内部固定启用若干高质量特性（如多遍编码）。NVIDIA 从 **Video Codec SDK 10.0** 起重构了这一设计：

1. **解耦与细化** — 把过去捆绑的功能拆开，使 `-rc`（速率控制模式）、`-tune`（编码调优）、`-multipass`（多遍编码）可独立控制。
2. **提升灵活性** — 开发者可按需精细调配编码设置，而非只能在几个固定“套餐”中选择。

---

## 五、核心区别：新组合 vs. 旧模式

| 对比维度 | `-rc vbr_hq`（旧） | `-rc vbr -tune hq`（新） |
|---|---|---|
| **控制粒度** | 黑盒预设，固定启用高质量分析与多遍编码，无法单独调整 | 模块化组合：`-tune hq` 调整成本函数权重以偏向高质量，多遍编码由 `-multipass` 独立控制 |
| **参数构成** | 隐含“多遍编码”等特性 | `-tune hq` **不含**多遍编码，需手动添加 `-multipass fullres`（或 `qres`）才能贴近旧效果 |
| **API 架构** | 旧的、已废弃的 API 架构 | 全新推荐的 API 架构，未来持续支持 |

---

## 六、总结

NVIDIA 移除 `vbr_hq` 是为了推动编码器 API 向**更现代化、更灵活**的方向演进。新方案虽然需要手动配置更多选项，但也换来了**更精细的控制能力**。

> **迁移速记**：`-rc vbr_hq` → `-rc vbr -tune hq -multipass fullres`（或 `qres`）

---

## 七、本仓实测与落地（2026-10-04，Tesla T4 / 驱动 580.65.06 / FFmpeg 9.0.2 / NVENCAPI 13.0）

### 7.1 补充事实：`qvbr` 同样被移除

FFmpeg 9.0 的 `-rc` 取值只剩 `constqp / vbr / cbr`（`ffmpeg -h encoder=h264_nvenc` 实测）；
`vbr_hq` 与 **`qvbr`** 都已消失。故旧代码里 `qvbr → 'vbr_hq'` 的映射一并作废，
本仓统一迁移为**裸 `vbr`**（见 7.4 二次校正）。

### 7.2 关键纠正：CLI 被移除 ≠ SDK/驱动被移除

- 头文件层面：`nv-codec-headers` **13.0/13.1** 已删除 `NV_ENC_PARAMS_RC_VBR_HQ`（`NV_ENC_PARAMS_RC_MODE` 仅剩 CONSTQP=0/VBR=1/CBR=2）。
- **但驱动运行时仍接受 `rc_ptr[1]=32`**：T4 实测 `NVENCEncoder(rate_mode="vbr_hq")` 初始化成功（apiVersion=0xd0=13.0），且行为验证证明 mode 32 **真实生效而非静默钳制**（同 30 帧噪声 `vbr_hq`=1,076,383B / `constqp`=2,159,969B / `qvbr`=1,018,148B，三者互异）。
- 结论：**头文件删除是开源 ffnvcodec 的裁剪，不等于驱动移除**；驱动对 32 做了向后兼容。

### 7.3 落地裁定：方案 A（仅迁移 CLI token）

| 层 | 处理 |
|---|---|
| **FFmpeg CLI（`ffmpeg_io.py` 的 `_rc_v_map`/`_NVENC_RC_MAP`、harness、探针）** | 迁移到**裸 `vbr`**（7.4 二次校正；`-tune/-multipass` 改显式 opt-in） |
| **SDK ctypes（`nvenc_sdk.py` 的 `rc_ptr[1]=32`）** | **不动**（驱动仍接受） |
| **内部 `rate_mode` 名 / config JSON / 缓存 key** | **不动**（Plan A 边界） |

画质实测（真实素材 640×360，cq=23，旧基线用备份 FFmpeg 6.1.1）：h264 ΔVMAF −0.048 / hevc −0.130（≤0.3 PASS）。

### 7.4 二次校正：CQ 默认改回**裸命令**（2026-10-04，据 VU A/B）

7.3 落地时给 CLI 追加了 `-tune hq -multipass fullres`。VU 侧 A/B（`Plan/ffmpeg_nvenc_knowledge.md` §5.1）
证明：`-tune hq` 是 ffmpeg **默认值**（写了等于没写），固定 CQ 下 `-multipass` **不升 VMAF**
（fullres ΔVMAF −0.006~−0.108 / qres −0.067~−0.335）。故：
- CLI 默认改回**裸 `-rc:v vbr -cq:v N -b:v 0 -preset p4`**；
- `-tune` / `-multipass` 改为显式 opt-in（`--nvenc-tune-* / --nvenc-multipass-*`），
  `cbr` 或目标码率时自动 `-multipass fullres`；唯一真源 `src/utils/nvenc_tuning.py`；
- 仅 ffmpeg_io / CLI 层；SDK ctypes 仍不动。
详见 `Plan/T4_NVENC_vbr_hq移除_验证专项.md` §11。

### 7.4 候选方案 B（未来驱动/SDK 不再接受 32 时）

① `nvenc_sdk._build_encoder_config` 新增 `vbr` 分支（`rc_ptr[1]=1` NV_ENC_PARAMS_RC_VBR + `targetQuality@rcParams+88` + avgBitrate 天花板）；② LA 门控改 `in ('vbr_hq','vbr','qvbr')`；③ 全链路内部名 `vbr_hq`→`vbr`（config/main/`_NVENC_LEVEL1_RATE_MODE`/缓存 key）；④ 跨仓同步。

⚠ 陷阱：`nvenc_sdk` 的 `else` 会把未知 rate_mode 静默当 CONSTQP，LA 门控又排除 `vbr` ⇒ 只改内部名不改分支会**静默降级**（实测 `--rate-mode-ifrnet vbr` 的 Ready 行 = `HEVC CONSTQP ... la=8`，la 仅回显）。

> 详细验证过程与 V1–V15 清单见 `Plan/T4_NVENC_vbr_hq移除_验证专项.md`。
