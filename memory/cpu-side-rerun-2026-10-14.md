# CPU 侧待补跑清单 10 项全量执行记录（2026-10-14）

**执行环境**：WSL 容器 `/mnt/d/Workspace_Python/Video_Enhancement`（非方案默认的 `/workspace/...` 布局），
ffmpeg 9.0.2，Python 3.14.4，8 核 / 10.6GB 可用 RAM，**无 GPU**（`torch.cuda.is_available()=False`、`nvidia-smi` 不存在）。
素材池：`../input_videos`（=`/mnt/d/Workspace_Python/input_videos`）；跨仓：`/mnt/d/Workspace_Python/VidUtils`。
动机：执行 `Plan/PROMPT_L40_全流程验证.md` §「CPU 侧待补跑清单」全部 10 项（该节 2026-10-14 判定仅 L40-1/2/3 真需 L40）。

## 逐项结果（10/10 完成，报告均落 `verification_report/`）

| # | 项 | 结果 | 实测数字 | 报告 |
|---|----|------|---------|------|
| 1 | `plan_gate` | ✅ | **101 项 / 96 PASS / 0 FAIL / 2 WARN / 3 SKIP**（exit 0） | `verification_report_20261014_132032.md` |
| 2 | `verify_equal_quality` | ✅ | **5/5 达标**（主门禁 ΔVMAF）：libx265 −0.448 / libvpx-vp9 +0.234 / libsvtav1 +0.012 / libaom-av1 −0.029 / librav1e −0.189；6 项参考指标超界（PSNR/PSNR-HVS，不判红）；exit 0 | `verify_equal_quality_cpu_2026-10-14.log` |
| 3 | `crf_cq_cpu`（`--quick --no-gpu`） | ✅ | **97 PASS / 0 FAIL / 0 WARN / 18 SKIP**；SKIP = G7/G8 共 12 项（未启用 GPU 组，预期）+ G6 共 6 项（缺 `basicsr`，见下） | `CRF_CQ统一验证报告_20261014_131745.md` |
| 4 | `av1_vp9_matrix_cpu` | ✅ | **3 PASS / 0 WARN / 0 FAIL**：libsvtav1 `-crf 29` 0.84×/−1.33 dB、libaom-av1 `-crf 26` 0.75×/−0.76 dB、libvpx-vp9 `-crf 26` 1.04×/−1.04 dB；B 组 AC1 SKIP（av1_nvenc 需 L40，预期） | `av1_vp9_matrix_cpu_2026-10-14.md/.json` |
| 5 | `segment_bitstream_verify` | ✅ | 软编 hevc 720×576/250 帧：**frames=250=packets**、单 IDR、frame_num 单调、无 pts_anomaly，707ms，rc=0 | `plan_gate` 之外单跑，无独立报告文件（stdout） |
| 6 | `eqq_pool_fit_selftest` | ✅ | 6 档 LOO 全达标：libx265 3.04 / libvpx-vp9 2.83 / libsvtav1 3.59 / libaom-av1 4.19 / librav1e@10 4.52 / librav1e 4.47（≤5.9/≤7.5）；**3 seed 顺序无关性逐位一致** | stdout |
| 7 | `calibrate_eq_selftest` | ✅ | **39 项全 PASS** + 自测通过 | stdout |
| 8 | `nvenc_tuning_verify` | ✅ | TUNE=(hq,ll,ull,lossless,uhq)、MULTIPASS=(disabled,qres,fullres)、is_bitrate('vbr'/'vbr_hq')=False/False | stdout |
| 9 | `nvenc_rc_diagnose` | ✅ | V11 结论复现：rc mode 枚举值表（CONSTQP 0 / VBR 1 / CBR 2 / VBR_MINQP 4 / CBR_LD_HQ 8 / CBR_HQ 16 / **VBR_HQ 32** / QVBR 64），targetQuality offset rc_ptr[29]→116B | stdout |
| 10 | 跨仓 ⑨ 组 | ✅ | **14/14 一致，rc=0**——2026-10-09 实测的 ⑦ `--threads` FAIL（VE `'3'`/VU `'2'`）**当前不复现** | stdout |

## 环境缺口与处置（重要）

1. **`ffmpeg-python` 漏装（已修）**：`requirements.txt:55` 声明 `ffmpeg-python>=0.2.0`，本容器未装
   ⇒ 首轮 plan_gate `BEH-H1` WARN（278 个 subprocess 调用点中 1 个 run_async 未校验）、`BEH-H3` SKIP、
   crf_cq G6 组 6 项 SKIP（原因均写 `ModuleNotFoundError: No module named 'ffmpeg'`）。
   处置：`pip install --break-system-packages ffmpeg-python`（0.2.0，纯 Python 零风险）。
   复跑：plan_gate 96 PASS（BEH-H1 恢复 PASS），G6 组 SKIP 原因变为 `basicsr`。
2. **`basicsr`/ESRGAN 后端链缺失（不修，记录在案）**：G6 组 6 项与 BEH-H3 的 SKIP 根因
   `No module named 'basicsr'`。该链依赖 torchvision 等重型包，而本容器 torch 2.9.1 是 **Debian 仓库版本**
   （`2.9.1+debian`），强装 torchvision 有 ABI 版本错配、破坏现有 torch 的风险（AGENTS.md 明确
   「先手动安装与 CUDA 匹配的 PyTorch，再 pip install -r requirements.txt」）。
   ⇒ 6 项 SKIP 属**环境依赖缺口而非逻辑失败**；若后续要消缺，应在装好匹配 torchvision 的环境复跑。
3. **plan_gate 的 2 WARN / 3 SKIP 全部为环境因素**：R5（torch 已装但 CUDA 不可用）、R7
   （h264_nvenc 探测 `Cannot load libcuda.so.1`）、R8（NVML 提示，主入口运行时会自动
   setdefault）、RT-0（未提供 --output，属调用方式）、BEH-H3（basicsr）。
   与 2026-10-10 的「97 PASS / 2 WARN / 2 SKIP」相比仅 BEH-H3 不同（那轮 basicsr 在场）。

## 与历史数字的差异说明

- crf_cq CPU 历史 103 PASS / 12 SKIP → 本轮 97 PASS / 18 SKIP：差 6 项即 G6 组（缺 basicsr）。
  G6 组测试名含 h264_nvenc/av1_nvenc，在无完整依赖的容器里只能 SKIP；补装依赖后可回到 103/12。
- verify_equal_quality 历史 537s → 本轮约 8 分钟（容器编码慢）。
- av1_vp9_matrix CPU 历史 582s → 本轮约 10 分钟。
- ⑨ 组：2026-10-09 实跑 13 PASS + 1 FAIL（⑦ `--threads`），本轮 **14/14** ⇒ VU 侧 `--threads` 钳位
  疑已在其他会话修复，或两侧环境收敛；**方案 L40-4/QP 轴门控相关的 `--threads` 立项目录可关闭**。

## L40 独占项状态（本容器跑不了）

GPU 不在场 ⇒ L40-1（Clifford 长视频冒烟）、L40-2（G7-6 av1 `-cq`）、L40-3（AC1 `-qp 70` 落带）、
L40-4（gpu_l40 双轴 LOO）**仍未执行**。恢复命令见 `Plan/PROMPT_L40_全流程验证.md` §执行命令。
L40-1 前置条件（源结构门禁 rc=0、sm89 engine 缓存）已于 2026-10-09 核验，仍需 GPU 在场时重跑。

## 镜像同步说明

本容器**无 B 侧镜像**（`/root/.codebuddy/projects/workspace-Video_Enhancement/memory` 与
`/workspace/` 均不存在；实际布局是 `/mnt/d/...`）。本条仅落 A 侧 canonical，待回到标准 Linux 生产
布局后按 AGENTS.md 规则 `cp -a` 同步。
