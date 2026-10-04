# vbr_hq → vbr 迁移验证证据（2026-10-04）

FFmpeg 9.0 移除 NVENC 的 `-rc:v vbr_hq`（及 `qvbr`）后，本项目按 **方案 A** 落地
（只迁移 CLI token，`nvenc_sdk` 的 ctypes 路径 `rc_ptr[1]=32` 与内部 `rate_mode` 名不动）。
本目录保存**该次落地的可复现证据与数据**，供后续方案待办（T4 标定主线 / L40 / 跨仓同步）继续利用。

- 验证专项：`Plan/T4_NVENC_vbr_hq移除_验证专项.md`（§0/§1.6/§10）
- 落地 commit：`4a1123a`
- 环境：Tesla T4 / 驱动 580.65.06 / CUDA 13.0 / FFmpeg 9.0.2 / NVENCAPI 13.0

## 目录结构

```
smoke/
  seg_src_5s.mp4             # E2E 冒烟输入源（640x360@30, 5s, h264；由 lavfi testsrc2 生成）
  seg_hevc_la8_vbrhq.mp4     # E2E 冒烟输出（hevc_nvenc + LA=8, rate_mode=vbr_hq, 299 帧）
quality_compare/             # §4 画质对比（锚点 cq=23，6s=183 帧）
  ref.mp4                    #   参考：真实素材前 6s 无损流拷贝
  old_h264_nvenc.mp4         #   旧：FFmpeg 6.1.1 `-rc:v vbr_hq -cq:v 23 -b:v 0 -bf 0`
  new_h264_nvenc.mp4         #   新：FFmpeg 9.0.2 `-rc:v vbr -tune hq -multipass fullres -cq:v 23 -b:v 0 -bf 0`
  old_hevc_nvenc.mp4 / new_hevc_nvenc.mp4
  la8.mp4 / la0.mp4          #   §5 LA 联动：hevc `-rc-lookahead 8` vs `0`（输出不同 ⇒ LA 生效）
writer_e2e/
  planA_ifrnet.mp4           # 真实 FFmpegWriter(rc_mode='vbr_hq') 直出（30 帧，验证 CLI 路径已迁移）
reports/
  crfcq_verify_no_gpu_quick.{md,json}   # `crf_cq_unification_verify --no-gpu --quick` 运行报告
```

## 关键数据（供引用，勿重算口径混淆）

质量对比（判据 ΔVMAF ≤ 0.3 → PASS）：

| codec | 旧 vbr_hq bytes | 新 vbr+th+mp bytes | ΔPSNR | 旧 VMAF | 新 VMAF | **ΔVMAF** |
|---|---|---|---|---|---|---|
| h264_nvenc | 1,539,271 | 1,467,218 (−4.7%) | −0.090 | 97.679 | 97.631 | **−0.048** |
| hevc_nvenc | 1,562,355 | 1,538,863 (−1.5%) | −0.077 | 97.513 | 97.383 | **−0.130** |

其余口径（详见 `reports/` 与专项 §1.5.5/§1.6）：
`crf_cq_unification_verify --no-gpu --quick` = **PASS 94 / FAIL 0 / SKIP 11**；
`plan_implementation_gate` 静态 **50/48/0/2** + 行为 **46/46** = 合并 **96/94/0/2**。

## 复现命令

```bash
cd /workspace/Video_Enhancement

# ① 冒烟输入（与 smoke/seg_src_5s.mp4 逐字节同源）
ffmpeg -hide_banner -y -f lavfi -i testsrc2=size=640x360:rate=30:duration=5 \
  -c:v h264 -pix_fmt yuv420p smoke_src.mp4 < /dev/null

# ② E2E 冒烟（hevc + LA=8；判据 原始帧=150 → 输出=299、解码级守恒）
python3 run.py -i smoke_src.mp4 -o seg_hevc_la8_vbrhq.mp4 \
  --skip-upscale --codec-ifrnet hevc_nvenc \
  --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 8 < /dev/null

# ③ 新路径直接编码（FFmpeg 9.0）
ffmpeg -hide_banner -y -i <源> -t 6 -an -c:v h264_nvenc \
  -rc:v vbr -tune hq -multipass fullres -cq:v 23 -b:v 0 -bf 0 new_h264.mp4 < /dev/null

# ④ 画质（口径陷阱见 memory/ffmpeg-metric-measurement-traps：必须显式 [0:v][1:v]，勿加 -v error）
ffmpeg -hide_banner -i new_h264.mp4 -i ref.mp4 -lavfi "[0:v][1:v]psnr" -f null - < /dev/null
ffmpeg -hide_banner -i new_h264.mp4 -i ref.mp4 -lavfi "[0:v][1:v]libvmaf=n_threads=4" -f null - < /dev/null
```

## 复现注意事项（重要）

- **旧产物 `old_*.mp4` 不可用 FFmpeg 9.0 重造**（`vbr_hq` 已被 CLI 拒绝，rc=234）。本目录保存的旧产物
  由**本机备份的 FFmpeg 6.1.1** 生成：`/var/backups/ffmpeg-v9/20261001-015049/usr_bin/ffmpeg`
  （机器本地路径，可能被清理 ⇒ 这是把旧产物入库的**主要理由**）。
- **存在 FFmpeg 版本混淆**（6.1.1 vs 9.0.2）：ΔVMAF 仅作量级参考，不是纯粹的"同版本新旧选项"对照。
- `ref.mp4` 源自仓库内 `input_vidiow/real_captured/test_video_640_360_real.mp4`（该目录被 `.gitignore`
  忽略、不入库），故此处保留其前 6s 无损流拷贝作为**参考基准**，保证证据自包含。
- `smoke/seg_src_5s.mp4` 由 lavfi 合成，可任意重建；保留只为免去重建步骤、并锁定字节一致。

## 文件指纹（md5）

```
529d0fd694b69eadb5361d4e6faccebe  old_h264_nvenc.mp4
f69f529cd1427870ba71173a77f7f15a  old_hevc_nvenc.mp4
a97dff62bd2c10637ce03801a3bd5ba1  new_h264_nvenc.mp4
ceda02b3991ec96d1cd882cd49e40eae  new_hevc_nvenc.mp4
3de3a033479dd7bcfc6a3f3e208a6d7d  ref.mp4
5821cbdb261f6bea75f3cc926e91468f  la8.mp4
04c2cd66384df24ddd4c1830a7239bc1  la0.mp4
9cc9e08c7b5b2f40289a210a56c3ac2e  seg_src_5s.mp4
14b5dff0502a5440ecfa5a6fda6fdabf  seg_hevc_la8_vbrhq.mp4
b25eed522792e6dd87708ddf43803c56  planA_ifrnet.mp4
```

## 未入库的会话产物

- 行为子集日志、commit message 草稿等一次性文本：无长期价值，未保留。
- `input_vidiow/` 原始真实素材、`/benchmark_output/` 等：被 `.gitignore` 忽略，且体积大，未纳入。
