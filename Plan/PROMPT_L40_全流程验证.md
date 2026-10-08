# L40 侧全流程验证立项

## 背景
CPU 侧 7/7 测试已通过，T4 侧 h264/hevc 已验收，需在 L40 (Ada, sm89) 上完成 AV1 NVENC 全链路验收。

## 环境要求
- L40 / Ada (sm89) / 驱动 ≥535 / CUDA 12.x
- FFmpeg 9.0+ with av1_nvenc, h264_nvenc, hevc_nvenc
- PyTorch CUDA 可用
- 长视频素材：≥330s 真实素材（如 `01 the race to mystery island fixed.avi`）
- 模型权重就位

## 验收范围（L40 独有 / 含 T4 复核）

| 测试项 | 脚本 | 关键判据 |
|--------|------|----------|
| plan_gate (完整) | `plan_implementation_gate.py` | 100/0/1 PASS（含 BEH-G9） |
| crf_cq --gpu | `crf_cq_unification_verify.py` | G7-6 av1_nvenc -cq PASS / G7/G8 FAIL=0 |
| av1_vp9_matrix (av1_nvenc) | `av1_vp9_quality_matrix.py` | AC1 QP=70 落带内 / AC2 -cq:v 32 PASS / AC4 B组 PASS |
| av1_pipeline_smoke (S1~S8) | `av1_pipeline_smoke.py` | 15/16 PASS：帧守恒/解码级/编码器确认/内存泄漏斜率≤50MB/min |
| nvenc_rc_diagnose | `nvenc_rc_mode_diagnose.py` | AV1 仅接受 constqp/vbr/cbr 确认 |
| eqq_pool_fit (含 gpu_l40) | `eqq_pool_fit_table.py` | av1 CQ LOO≤5.9 / av1 QP LOO≤7.5 / 顺序无关性 |

## 执行命令
```bash
cd /workspace/Video_Enhancement

# 0. 环境体检
nvidia-smi -L
python3 -c "import torch;print(torch.cuda.get_device_name(0),torch.cuda.is_available())"
for C in h264_nvenc hevc_nvenc av1_nvenc; do
  ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 -c:v $C -f null - < /dev/null; echo "$C rc=$?"
done
# 期望: h264=0, hevc=0, av1=0

# 1. 完整验证（含 AV1 长视频冒烟）
python3 Accessory/verify/comprehensive_verify.py --env l40 \
    -i input_videos/word_world_2.mp4 -o output.mp4 \
    --source input_videos/word_world_2.mp4 \
    --bitrate-source input_videos/new4_raw.mp4 \
    --long-source "/mnt/d/Workspace_Python/input_videos/01 the race to mystery island fixed.avi" \
    --smoke --smoke-mode interpolate_then_upscale \
    --segment-duration 30 --mem-interval 5 --mem-dump-dir /tmp/s8_mem

# 2. AV1 质量矩阵（仅 av1_nvenc）
python3 Accessory/probe/av1_vp9_quality_matrix.py --quality-mode quality \
    --src input_videos/word_world_2.mp4 --only av1_nvenc \
    --report verification_report/av1_vp9_matrix_L40_$(date +%F).md < /dev/null

# 3. GPU 画质判据
python3 Accessory/verify/crf_cq_unification_verify.py --gpu \
    --source input_videos/word_world_2.mp4 \
    --bitrate-source input_videos/new4_raw.mp4 \
    --report verification_report/crfcq_gpu_L40_$(date +%F).md < /dev/null

# 4. 落表器含 GPU 侧
python3 Accessory/probe/eqq_pool_fit_table.py --sides 6s,10s,legacy10s,gpu_l40 < /dev/null

# 5. 跨仓真源一致
cd /workspace/VidUtils && python3 verify/verify_quality_mapping.py < /dev/null
# ⑨ 组 14/14 必须一致
```

## 门禁基线（2026-10-06 L40 实测）
- `plan_implementation_gate`: 100 PASS / 0 FAIL / 1 SKIP
- `crf_cq --gpu`: 113 PASS / 0 FAIL / 3 WARN
- `av1_vp9_quality_matrix`: AC1 -qp 70 落带内 (1.07× / −1.13 dB) / AC2 +0.16 dB / AC4 1.14× / −0.29 dB
- `av1_pipeline_smoke`: S1~S8 退出码 0，constqp S3 计数差异非功能性，S8 斜率通过
- 跨仓 ⑨ 组: 14/14 一致

## 产出物
- `verification_report/verification_report_YYYYMMDD_HHMMSS.json/.md`
- `verification_report/crfcq_gpu_L40_YYYYMMDD.md`
- `verification_report/av1_vp9_matrix_L40_YYYYMMDD.md/.json`
- `verification_report/av1_smoke_L40_YYYYMMDD.md` + `/tmp/s8_mem/*.mem.tsv`
- 落表：`QUALITY_MAP['av1_nvenc']=(1.4566,1.2165,0,63)` `QUALITY_MAP_QP['av1_nvenc']=(7.9338,-97.5136,0,255)`