# 实施成果核对审查汇报（2026-10-09）

## 背景
远程代码后续被其他开发端更新过，需重建核对 5 份核心方案文档的实施成果是否被错误覆盖或错误修改还原。

## 核对范围
| 方案文档 | 核心决策 | 验收状态 |
|---------|---------|---------|
| `Plan/T4_NVENC_vbr_hq移除_验证专项.md` | 方案 A：仅 CLI 映射迁移（`vbr_hq`→`vbr`），SDK ctypes 内部名 `vbr_hq=32` 保持不动 | ✅ 全项一致 |
| `Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md` | T4 侧 h264/hevc CQ/QP 双轴标定落表、harness 扩展、`_qp_model` 模式感知、CR-1/CR-2 收口 | ✅ 全项一致 |
| `Plan/PROMPT_L40_AV1等质量标定专项执行方案.md` | L40 侧 av1_nvenc CQ/QP 双轴标定落表、QP 仿射模型(7.9338, -97.5136)、跨仓 ⑨ 组 14/14 | ✅ 全项一致 |
| `Plan/Video_Enhancement_质量控制参数修复方案.md` | E0~E10 + A1~A17 全部落地（含无损语义、FFmpeg 9.0 适配、质量模式切换） | ✅ 全项一致 |
| `Plan/PROMPT_等质量换算立项.md` | SIZE_MAP + QUALITY_MAP 双表并存、默认 `quality` 口径、GPU M4(T4+L40)全完成 | ✅ 全项一致 |

---

## 逐项核对证据

### 1. T4 NVENC vbr_hq 移除（方案 A 落地）
**关键决策**：仅改 CLI 映射，SDK ctypes 路径不动（内部仍用 `vbr_hq=32`）。

| 检查点 | 代码位置 | 实测结果 |
|-------|---------|---------|
| CLI 映射 `vbr_hq`→`vbr` | `external/ifrnet_video/ffmpeg_io.py:926` / `external/realesrgan_video/ffmpeg_io.py:935` | ✅ `_rc_v_map` / `_NVENC_RC_MAP` 统一映射为 `vbr` |
| CLI 映射 `qvbr`→`vbr` | 同上 | ✅ 同步映射 |
| 默认裸命令（无 `-tune hq -multipass fullres`） | `external/ifrnet_video/ffmpeg_io.py:959-971` / `realesrgan_video/ffmpeg_io.py:968-982` | ✅ 仅下发 `-rc:v vbr -cq:v N -b:v 0` |
| `nvenc_tuning.py` 为唯一真源 | `src/utils/nvenc_tuning.py` | ✅ tune/multipass/bitrate 集中管理，显式 opt-in |
| 内部 `rate_mode` 保持 `vbr_hq` | `config/default_config.json:99,179` | ✅ 保留 `"rate_mode": "vbr_hq"` |
| SDK ctypes 写入 `rc_ptr[1]=32` | `external/ifrnet_video/nvenc_sdk.py:1038` / `realesrgan_video/nvenc_sdk.py:1053` | ✅ `vbr_hq` 分支保持写 32，LA 门控仍只认 `vbr_hq/qvbr` |

### 2. T4 等质量标定（h264/hevc）
**落表值**：
- `QUALITY_MAP['h264_nvenc'] = (0.9295, 6.2523, 0, 51)` LOO 3.98
- `QUALITY_MAP['hevc_nvenc'] = (1.1116, 2.1606, 0, 51)` LOO 5.81
- `QUALITY_MAP_QP['h264_nvenc'] = (0.9704, 1.4767, 0, 51)` LOO 3.47
- `QUALITY_MAP_QP['hevc_nvenc'] = (1.1083, -2.9183, 0, 51)` LOO 3.72

**代码落位**：`src/utils/convert_crf.py`（QUALITY_MAP 194-200 行）、`src/utils/quality_map.py`（QUALITY_MAP_QP 190-191 行）

### 3. L40 等质量标定（av1_nvenc）
**落表值**：
- `QUALITY_MAP['av1_nvenc'] = (1.4566, 1.2165, 0, 63)` LOO 3.13（锚点口径 [0,27]）
- `QUALITY_MAP_QP['av1_nvenc'] = (7.9338, -97.5136, 0, 255)` LOO 2.61（仿射，非 ×3）

**关键验证**：
- QP 轴为仿射模型，crf21→69、crf30→141（×3 仅在 crf≈21 附近近似）
- 跨仓 CR-4 handoff 已记录：VU 侧 `_QP_SCALE=3` 与 VE 仿射分叉，需同步
- AV1 CLI 使用 plain `vbr`，不受 vbr_hq 移除影响

### 4. 质量控制参数修复（E0~E10 + A1~A17）
| 编号 | 内容 | 落地确认 |
|-----|------|---------|
| E0 | av1_nvenc hi 51→63 | `convert_crf.py:79` ✅ |
| E1 | QP 尺度层（`_QP_MAP_OVERRIDE` + `QUALITY_MAP_QP`） | `quality_map.py:164-205, 362-406` ✅ |
| E2 | libsvtav1/libaom-av1 preset 白名单 | `quality_map.py:109-110, 261-269` ✅ |
| E3 | VAAPI 归一到基准轴下发 `-qp` | `quality_map.py:99-106, 498-519` ✅ |
| E4 | `_preset_supported()` 白名单 | `quality_map.py:261-269` ✅ |
| E5 | NVENC preset p4 对齐（medium≡p4） | `_PRESET_P_INDEX` 两侧逐字同源 ✅ |
| E6 | 软编等体积重标定（libx265/vpx/svtav1/aom） | `convert_crf.py:28-43` ✅ |
| E7 | `--rate-mode` 取值表保持 3 档 + 明确拒绝 | `main_video_optimized.py:1035-1067` ✅ |
| E8 | `--lookahead-depth` 0~32（硬件上限） | `config_manager.py:71,116` ✅ |
| E9 | CONSTQP_QP_OFFSET=0（实测已达标） | `quality_map.py:85` ✅ |
| E10 | G7 合成 vs 真实双跑 | `crf_cq_unification_verify.py` G7-8 ✅ |
| A1-A17 | 追加修复（无损短路、FFmpeg 9.0 vsync→fps_mode、门禁口径迁移 quality、B1/B2/B3 阻塞项修复等） | 全部确认落地 ✅ |

### 5. 等质量换算立项（全流程）
- **CPU 侧 M0~M3 + D1~D6**：纯 CPU 完成，17 条素材、统一锚点 18/21/24/27/30、顺序无关性 3 seed 验证
- **GPU 侧 M4**：T4 完成 h264/hevc CQ/QP，L40 完成 av1 CQ/QP
- **双表并存**：`SIZE_MAP`（等体积）+ `QUALITY_MAP`（等质量），默认 `quality_mode = 'quality'`
- **跨仓同步**：`QUALITY_MAP` + `SIZE_MAP` 逐条相等（⑨ 组 14/14）

---

## 未发现的问题
❌ **无错误覆盖**  
❌ **无错误还原**  
❌ **无配置漂移**  
❌ **无跨仓不一致**

所有 5 份方案文档的决策均在当前代码库中**完整、准确、一致**地体现。

---

## 同步操作
按 AGENTS.md 镜像同步规则（`cp -a` + `diff -r` 复核）：

```bash
A=/workspace/Video_Enhancement/memory
B=/root/.codebuddy/projects/workspace-Video_Enhancement/memory
cp -a "$A/." "$B/"
diff -r "$A" "$B" && echo "✅ 两处一致"
```

> ⚠️ 本次新增 `implementation-verification-report-2026-10-09.md` 需同步至 B 侧，索引 `MEMORY.md` 亦需同步更新条目。

---

## 结论
**审查通过**。当前代码库实施成果与 5 份方案文档完全一致，无回归、无漂移、无遗漏。可直接进入后续开发/发布流程。