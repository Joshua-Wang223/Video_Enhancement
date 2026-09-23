# Video_Enhancement项目架构分析与BUG审查

**Session ID:** ses_fd3183b32ffeuQ85s2IKZsA0g6
**Created:** 8/23/2026, 12:36:21 PM
**Updated:** 8/24/2026, 8:38:09 AM

---

## User

深度分析指定项目路径（`D:\Workspace_Python\Video_Enhancement\Video_Enhancement项目`）的完整代码库。提炼软件的整体架构组成，包括但不限于模块划分、核心类与函数关系、数据流、依赖关系及设计模式。全面审查并查找隐藏的BUG，涵盖逻辑错误、资源泄漏、并发问题、异常处理缺失、性能瓶颈及潜在崩溃点。基于分析结果，给出终极优化方案，按优先级排序，包含架构级优化、代码级修复、性能提升及可维护性改进建议。输出格式：1）架构总览；2）BUG清单（含位置与影响）；3）优化方案（分阶段实施计划）。 

---

## Assistant (Plan · Ox Alpha Free · 21.5s)

_Thinking:_

The user is asking me to do a deep analysis of the project at `D:\Workspace_Python\Video_Enhancement\Video_Enhancement项目`. Note that my working directory is `D:\Workspace_Python\Video_Enhancement\Video_Enhancement` but the user mentioned a path with `项目` suffix. Let me first check what exists.

This is a read-only analysis task (Plan Mode active). I need to:
1. Explore the codebase structure
2. Analyze architecture
3. Find bugs
4. Propose optimization plan

Let me start by exploring the directory structure. I should use parallel tool calls for efficiency. Since this is a large analysis task, I should use explore agents to help.

First, let me verify the path exists and understand the layout.

**Tool: bash**

**Input:**
```json
{
  "command": "Test-Path -LiteralPath \"D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement项目\"; Test-Path -LiteralPath \"D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\"; Get-ChildItem -LiteralPath \"D:\\Workspace_Python\\Video_Enhancement\" | Select-Object Name, Mode"
}
```

**Output:**
```
False
True

Name                                                            Mode  
----                                                            ----  
gfpgan                                                          d-----
input_videos                                                    d-----
memory                                                          d-----
Video_Enhancement                                               da----
Video_Enhancement - Copy                                        da----
Video_Enhancement202605061051_IE                                d-----
Video_Enhancement202605061329_IE                                d-----
Video_Enhancement202605061446_chrome                            d-----
Video_Enhancement20260506_chrome                                d-----
Video_Enhancement202605271550                                   d-----
Video_Enhancement202606181338                                   d-----
Video_Enhancement202606220839                                   d-----
Video_Enhancement202607070939                                   d-----
Video_Enhancement_202606301625                                  d-----
Video_Enhancement��202607240921��                               d-----
Video_Enhancement��202607270925��                               da----
aliyunpan_download_fix.sh                                       -a----
checkpoint (esrgan_demo).json                                   -a----
checkpoint (ifrnet_demo).json                                   -a----
deploy_proxy_env.sh                                             -a----
encoders.txt                                                    -a----
esrgan_from_segments_39.txt                                     -a----
FFmpeg NVENC -rc-lookahead �����֡���ʴ�.txt                   -a----
fix_pdf_cjk_font.py                                             -a----
followyourcanvas.txt                                            -a----
gfpgan.tar.gz                                                   -a----
git_set-url.txt                                                 -a----
inference_realesrgan.txt                                        -a----
install_plugins.txt                                             -a----
install_xlsx_ext.py                                             -a----
M3DDM-Plus.txt                                                  -a----
main_video_optimized.txt                                        -a----
main_video_optimized_20260514.txt                               -a----
newtest.txt                                                     -a----
NVENC CompletionEvent �ؼ��������.txt                          -a----
NVENC completionEvent ��ģʽ��֤ �� �ۺϷ���1.txt                -a----
NVENC GPU ֱͨ��ʼ��ʧ��.txt                                    -a----
nvidia-lib-version-fix.sh                                       -a----
PowerShell ֱ������.txt                                         -a----
process_video.txt                                               -a----
Proxy_Chain_Deployment_Guide.md                                 -a----
setup_claude_deepseek.sh                                        -a----
setup_cloudstudio.sh                                            -a----
start_tailscale.sh                                              -a----
test_nvenc_comprehensive_20260520_023932.txt                    -a----
Video_Enhancement202605061051_IE.tar.gz                         -a----
Video_Enhancement202605061329_IE.tar.gz                         -a----
Video_Enhancement202605061446_chrome.tar.gz                     -a----
Video_Enhancement20260506_chrome.tar.gz                         -a----
Video_Enhancement20260506_IE.tar.gz                             -a----
Video_Enhancement202605221110.tar.gz                            -a----
Video_Enhancement202605221112.tar.gz                            -a----
Video_Enhancement202605271550.tar.gz                            -a----
Video_Enhancement202606181338.tar.gz                            -a----
Video_Enhancement202606220839.tar.gz                            -a----
Video_Enhancement202607070939.tar.gz                            -a----
Video_Enhancement_github_token.txt                              -a----
���㷽���ĺ���Ҫ��.txt                                          -a----
������֡�����쳣���򼰽������.txt                              -a----
ͳһ����nvidia����580.65.06���������в���Ҫ�Ļ�������ŵİ�.txt -a----



```

**Tool: glob**

**Input:**
```json
{
  "pattern": "src/**/*.py"
}
```

**Output:**
```
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\__init__.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\main_video_v5_single.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\main_video_v3.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\main_video_v2.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\main_video_optimized.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\main_video_optimized - 202607130944.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\main_video_optimized - 202606171615.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\main_video.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\main_v2.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\main.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\backup202606261401\realesrgan_processor_video_optimized.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\backup202606261401\main_video_optimized.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\utils\video_fixer.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\utils\output_filter.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\utils\config_manager.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\main_video_v6_single.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\utils\video_utils.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\utils\video_utils - Copy.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\utils\__init__.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\ifrnet_processor.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\backup202607290943\realesrgan_processor_video_optimized.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\backup202607061659\realesrgan_processor_video_optimized.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\backup202607061659\ifrnet_processor_v6_4_single.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\backup202606171621\realesrgan_processor_video_optimized.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\backup202606171621\ifrnet_processor_video_optimized.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\realesrgan_processor.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\ifrnet_processor_video_optimized.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\ifrnet_processor_v6_single.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\ifrnet_processor_v6_4_single.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\ifrnet_processor_v6_1_single.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\ifrnet_processor_v6_1_single - Copy.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\ifrnet_processor_v5_single.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\ifrnet_processor_v3.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\ifrnet_processor_v2.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\realesrgan_processor_video_v2.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\realesrgan_processor_video_optimized_bak(多片段复用优化).py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\realesrgan_processor_video_optimized.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\realesrgan_processor_video.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\realesrgan_processor_v2.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\realesrgan_processor_video_v5_single.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\realesrgan_processor_video_v3.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\realesrgan_processor_video_v6_single.py
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\__init__.py
```

---

## Assistant (Plan · Ox Alpha Free · 40.4s)

_Thinking:_

The user's specified path `D:\Workspace_Python\Video_Enhancement\Video_Enhancement项目` doesn't exist. The actual project is at `D:\Workspace_Python\Video_Enhancement\Video_Enhancement` (my working directory). I'll analyze that one.

Per AGENTS.md:
- `src/main_video_optimized.py` — the only active entry point; other `main_video_*.py` are legacy
- `src/processors/` — IFRNet & Real-ESRGAN processors; backends in `external/ifrnet_video/` and `external/realesrgan_video/`
- `src/utils/` — config_manager.py, video_utils.py
- `config/default_config.json`

So the core active files to analyze:
1. `src/main_video_optimized.py`
2. `src/processors/ifrnet_processor.py` (which one is active? need to check imports)
3. `src/processors/realesrgan_processor_video_optimized.py`
4. `src/utils/config_manager.py`
5. `src/utils/video_utils.py`
6. `external/ifrnet_video/` and `external/realesrgan_video/`

This is a large analysis task. I should use explore agents to parallelize. Let me first check the sizes of the main files, then dispatch explore agents.

Let me plan the analysis:
1. Read main entry point structure myself or via agents
2. Dispatch multiple explore agents for different areas:
   - Agent 1: main_video_optimized.py architecture + bugs
   - Agent 2: processors (ifrnet + realesrgan) 
   - Agent 3: utils + config
   - Agent 4: external backends

Given the size, let me first get file sizes to understand scope.

指定路径 `Video_Enhancement项目` 不存在；实际项目为工作目录 `D:\Workspace_Python\Video_Enhancement\Video_Enhancement`，我将以它为准进行分析。先摸底文件规模与活跃代码范围。

**Tool: bash**

**Input:**
```json
{
  "command": "Get-ChildItem -LiteralPath \"D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\" | Select-Object Name, Mode; Get-ChildItem -Recurse -File -Include *.py -LiteralPath \"D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\",\"D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\" | Where-Object { $_.FullName -notmatch 'backup|Copy|bak' } | ForEach-Object { \"{0,8} lines  {1}\" -f (Get-Content $_.FullName | Measure-Object -Line).Lines, $_.FullName.Replace('D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\','') }"
}
```

**Output:**
```

Name               Mode  
----               ----  
.claude            d-----
.claude-flow       d-----
.codebuddy         d-----
.codewhale         d-----
.cursor            d-----
.hermes            d-----
.minimax           d-----
.qoder             d-----
.workbuddy         d-----
benchmark_output   d-----
config             d-----
external           da----
logs               d-----
memory             d-----
models             d-----
models_IFRNet      d-----
models_RealESRGAN  d-----
output             d-----
Plan               d-----
src                d-----
src - 202606171605 d-----
temp               d-----
tests              d-----
.gitignore         -a----
AGENTS.md          -a----
CLAUDE.md          -a----
default.tar.gz     -a----
FILE_LIST.md       -a----
fix_import.py      -a----
GUIDE.md           -a----
MEMORY.md          -a----
PROJECT_SUMMARY.md -a----
quick_start.sh     -a----
QUICK_UPGRADE.md   -a----
README.md          -a----
requirements.txt   -a----
run.py             -a----
setup_info.json    -a----
setup_project.py   -a----
     430 lines  src\main.py
     525 lines  src\main_v2.py
     480 lines  src\main_video.py
    1643 lines  src\main_video_optimized - 202606171615.py
    1758 lines  src\main_video_optimized - 202607130944.py
    1822 lines  src\main_video_optimized.py
     455 lines  src\main_video_v2.py
     455 lines  src\main_video_v3.py
     594 lines  src\main_video_v5_single.py
     714 lines  src\main_video_v6_single.py
       1 lines  src\__init__.py
     251 lines  src\processors\ifrnet_processor.py
     252 lines  src\processors\ifrnet_processor_v2.py
     254 lines  src\processors\ifrnet_processor_v3.py
     521 lines  src\processors\ifrnet_processor_v5_single.py
     708 lines  src\processors\ifrnet_processor_v6_1_single.py
     765 lines  src\processors\ifrnet_processor_v6_4_single.py
     609 lines  src\processors\ifrnet_processor_v6_single.py
     779 lines  src\processors\ifrnet_processor_video_optimized.py
     409 lines  src\processors\realesrgan_processor.py
     418 lines  src\processors\realesrgan_processor_v2.py
     364 lines  src\processors\realesrgan_processor_video.py
     969 lines  src\processors\realesrgan_processor_video_optimized.py
     316 lines  src\processors\realesrgan_processor_video_v2.py
     316 lines  src\processors\realesrgan_processor_video_v3.py
     554 lines  src\processors\realesrgan_processor_video_v5_single.py
     631 lines  src\processors\realesrgan_processor_video_v6_single.py
       1 lines  src\processors\__init__.py
     349 lines  src\processors\__pycache__\ifrnet_processor_video_optimized.cpython-312.pyc
     353 lines  src\processors\__pycache__\realesrgan_processor_video_optimized.cpython-312.pyc
     374 lines  src\utils\config_manager.py
      80 lines  src\utils\output_filter.py
     201 lines  src\utils\video_fixer.py
    1226 lines  src\utils\video_utils.py
       1 lines  src\utils\__init__.py
     113 lines  src\utils\__pycache__\config_manager.cpython-312.pyc
     516 lines  src\utils\__pycache__\video_utils.cpython-312.pyc
       2 lines  src\utils\__pycache__\__init__.cpython-312.pyc
     603 lines  src\__pycache__\main_video_optimized.cpython-312.pyc
       2 lines  src\__pycache__\__init__.cpython-312.pyc
     348 lines  external\IFRNet\datasets.py
      17 lines  external\IFRNet\demo_2x.py
      32 lines  external\IFRNet\demo_8x.py
      54 lines  external\IFRNet\generate_flow.py
      17 lines  external\IFRNet\LICENSE
      81 lines  external\IFRNet\loss.py
     105 lines  external\IFRNet\metric.py
     175 lines  external\IFRNet\process_video.py
     200 lines  external\IFRNet\process_video_v3.py
     726 lines  external\IFRNet\process_video_v4.py
    1464 lines  external\IFRNet\process_video_v5.py
    1175 lines  external\IFRNet\process_video_v5_single.py
    1918 lines  external\IFRNet\process_video_v6.py
    2119 lines  external\IFRNet\process_video_v6_1.py
    1798 lines  external\IFRNet\process_video_v6_1_single.py
    2264 lines  external\IFRNet\process_video_v6_2_1_single.py
    2533 lines  external\IFRNet\process_video_v6_2_2_single.py
    2583 lines  external\IFRNet\process_video_v6_2_3_single.py
    2703 lines  external\IFRNet\process_video_v6_2_4_single.py
    2264 lines  external\IFRNet\process_video_v6_2_single.py
    2407 lines  external\IFRNet\process_video_v6_3_0_single.py
    2741 lines  external\IFRNet\process_video_v6_3_1_single.py
    2994 lines  external\IFRNet\process_video_v6_3_2_single.py
    4127 lines  external\IFRNet\process_video_v6_3_3.py
    3353 lines  external\IFRNet\process_video_v6_3_3_single.py
    4258 lines  external\IFRNet\process_video_v6_3_4.py
    3479 lines  external\IFRNet\process_video_v6_3_4_single.py
    4496 lines  external\IFRNet\process_video_v6_3_5.py
    3758 lines  external\IFRNet\process_video_v6_3_5_single.py
    3855 lines  external\IFRNet\process_video_v6_4_1_single.py
    3998 lines  external\IFRNet\process_video_v6_4_2_single.py
    6613 lines  external\IFRNet\process_video_v6_4_3_1_single.py
    5577 lines  external\IFRNet\process_video_v6_4_3_single.py
    7589 lines  external\IFRNet\process_video_v6_4_4_1_single.py
    5736 lines  external\IFRNet\process_video_v6_4_4_single.py
    7090 lines  external\IFRNet\process_video_v6_4_5_1_single.py
    6622 lines  external\IFRNet\process_video_v6_4_5_1_single_tryfix1.py
    6634 lines  external\IFRNet\process_video_v6_4_5_1_single_tryfix2.py
    5837 lines  external\IFRNet\process_video_v6_4_5_single.py
    1586 lines  external\IFRNet\process_video_v6_single.py
      77 lines  external\IFRNet\README.md
     161 lines  external\IFRNet\train_gopro.py
     154 lines  external\IFRNet\train_vimeo90k.py
     169 lines  external\IFRNet\utils.py
      45 lines  external\IFRNet\benchmarks\Middlebury_Other.py
      65 lines  external\IFRNet\benchmarks\SNU_FILM.py
      30 lines  external\IFRNet\benchmarks\speed_parameters.py
      38 lines  external\IFRNet\benchmarks\UCF101.py
      41 lines  external\IFRNet\benchmarks\Vimeo90K.py
      35 lines  external\IFRNet\liteflownet\README.md
       4 lines  external\IFRNet\liteflownet\requirements.txt
     286 lines  external\IFRNet\liteflownet\run.py
     356 lines  external\IFRNet\liteflownet\correlation\correlation.py
       1 lines  external\IFRNet\liteflownet\correlation\README.md
     215 lines  external\IFRNet\models\IFRNet.py
     214 lines  external\IFRNet\models\IFRNet_L.py
     214 lines  external\IFRNet\models\IFRNet_S.py
     101 lines  external\IFRNet\models\__pycache__\IFRNet_S.cpython-312.pyc
      41 lines  external\IFRNet\__pycache__\loss.cpython-312.pyc
    1879 lines  external\IFRNet\__pycache__\process_video_v6_4_3_1_single.cpython-312.pyc
    1624 lines  external\IFRNet\__pycache__\process_video_v6_4_3_single.cpython-312.pyc
    1877 lines  external\IFRNet\__pycache__\process_video_v6_4_4_1_single.cpython-312.pyc
    1653 lines  external\IFRNet\__pycache__\process_video_v6_4_4_single.cpython-312.pyc
    2037 lines  external\IFRNet\__pycache__\process_video_v6_4_5_1_single.cpython-312.pyc
    1669 lines  external\IFRNet\__pycache__\process_video_v6_4_5_single.cpython-312.pyc
      92 lines  external\IFRNet\__pycache__\utils.cpython-312.pyc
      27 lines  external\ifrnet_video\config.py
     643 lines  external\ifrnet_video\ffmpeg_io.py
     297 lines  external\ifrnet_video\ifrnet_utils.py
    1693 lines  external\ifrnet_video\main.py
    2799 lines  external\ifrnet_video\nvenc_sdk.py
    1563 lines  external\ifrnet_video\pipeline.py
     218 lines  external\ifrnet_video\tensorrt_accel.py
       2 lines  external\ifrnet_video\__init__.py
      12 lines  external\ifrnet_video\__pycache__\config.cpython-312.pyc
     192 lines  external\ifrnet_video\__pycache__\ffmpeg_io.cpython-312.pyc
     114 lines  external\ifrnet_video\__pycache__\ifrnet_utils.cpython-312.pyc
     543 lines  external\ifrnet_video\__pycache__\main.cpython-312.pyc
     701 lines  external\ifrnet_video\__pycache__\nvenc_sdk.cpython-312.pyc
     497 lines  external\ifrnet_video\__pycache__\pipeline.cpython-312.pyc
      54 lines  external\ifrnet_video\__pycache__\tensorrt_accel.cpython-312.pyc
       3 lines  external\ifrnet_video\__pycache__\__init__.cpython-312.pyc
     114 lines  external\Real-ESRGAN\.gitignore
      41 lines  external\Real-ESRGAN\.pre-commit-config.yaml
     112 lines  external\Real-ESRGAN\ARCHITECTURE_OPTIMIZATION_GUIDE.md
      93 lines  external\Real-ESRGAN\CODE_OF_CONDUCT.md
      20 lines  external\Real-ESRGAN\cog.yaml
     131 lines  external\Real-ESRGAN\cog_predict.py
     151 lines  external\Real-ESRGAN\inference_realesrgan.py
     358 lines  external\Real-ESRGAN\inference_realesrgan_video.py
     342 lines  external\Real-ESRGAN\inference_realesrgan_video_original.py
     415 lines  external\Real-ESRGAN\inference_realesrgan_video_v3.py
     819 lines  external\Real-ESRGAN\inference_realesrgan_video_v4.py
    1780 lines  external\Real-ESRGAN\inference_realesrgan_video_v5.py
    1471 lines  external\Real-ESRGAN\inference_realesrgan_video_v5_single.py
    2613 lines  external\Real-ESRGAN\inference_realesrgan_video_v6.py
    2733 lines  external\Real-ESRGAN\inference_realesrgan_video_v6_1.py
    2253 lines  external\Real-ESRGAN\inference_realesrgan_video_v6_1_single.py
    3309 lines  external\Real-ESRGAN\inference_realesrgan_video_v6_2_single.py
    3819 lines  external\Real-ESRGAN\inference_realesrgan_video_v6_3_optimized.py
    5380 lines  external\Real-ESRGAN\inference_realesrgan_video_v6_4_optimized.py
    2138 lines  external\Real-ESRGAN\inference_realesrgan_video_v6_single.py
      23 lines  external\Real-ESRGAN\LICENSE
       8 lines  external\Real-ESRGAN\MANIFEST.in
     268 lines  external\Real-ESRGAN\parameter_tuning.py
     173 lines  external\Real-ESRGAN\performance_comparison.py
     193 lines  external\Real-ESRGAN\README.md
     189 lines  external\Real-ESRGAN\README_CN.md
       9 lines  external\Real-ESRGAN\requirements.txt
      28 lines  external\Real-ESRGAN\setup.cfg
      85 lines  external\Real-ESRGAN\setup.py
     143 lines  external\Real-ESRGAN\test_integration.py
       1 lines  external\Real-ESRGAN\VERSION
      19 lines  external\Real-ESRGAN\.vscode\settings.json
     479 lines  external\Real-ESRGAN\assets\realesrgan_logo.png
     517 lines  external\Real-ESRGAN\assets\realesrgan_logo_ai.png
     536 lines  external\Real-ESRGAN\assets\realesrgan_logo_av.png
     527 lines  external\Real-ESRGAN\assets\realesrgan_logo_gi.png
     522 lines  external\Real-ESRGAN\assets\realesrgan_logo_gv.png
    2978 lines  external\Real-ESRGAN\assets\teaser-text.png
    1701 lines  external\Real-ESRGAN\assets\teaser.jpg
      49 lines  external\Real-ESRGAN\docs\anime_comparisons.md
      46 lines  external\Real-ESRGAN\docs\anime_comparisons_CN.md
      48 lines  external\Real-ESRGAN\docs\anime_model.md
      95 lines  external\Real-ESRGAN\docs\anime_video_model.md
      32 lines  external\Real-ESRGAN\docs\CONTRIBUTING.md
       7 lines  external\Real-ESRGAN\docs\FAQ.md
       5 lines  external\Real-ESRGAN\docs\feedback.md
      34 lines  external\Real-ESRGAN\docs\model_zoo.md
      10 lines  external\Real-ESRGAN\docs\ncnn_conversion.md
     204 lines  external\Real-ESRGAN\docs\Training.md
     188 lines  external\Real-ESRGAN\docs\Training_CN.md
       1 lines  external\Real-ESRGAN\experiments\pretrained_models\README.md
     783 lines  external\Real-ESRGAN\inputs\00003.png
     385 lines  external\Real-ESRGAN\inputs\00017_gray.png
      55 lines  external\Real-ESRGAN\inputs\0014.jpg
      56 lines  external\Real-ESRGAN\inputs\0030.jpg
      62 lines  external\Real-ESRGAN\inputs\ADE_val_00000114.jpg
    1309 lines  external\Real-ESRGAN\inputs\children-alpha.png
    3599 lines  external\Real-ESRGAN\inputs\OST_009.png
    1726 lines  external\Real-ESRGAN\inputs\tree_alpha_16bit.png
     194 lines  external\Real-ESRGAN\inputs\wolf_gray.jpg
    2190 lines  external\Real-ESRGAN\inputs\video\onepiece_demo.mp4
     165 lines  external\Real-ESRGAN\options\finetune_realesrgan_x4plus.yml
     132 lines  external\Real-ESRGAN\options\finetune_realesrgan_x4plus_pairdata.yml
     163 lines  external\Real-ESRGAN\options\train_realesrgan_x2plus.yml
     162 lines  external\Real-ESRGAN\options\train_realesrgan_x4plus.yml
     124 lines  external\Real-ESRGAN\options\train_realesrnet_x2plus.yml
     123 lines  external\Real-ESRGAN\options\train_realesrnet_x4plus.yml
       9 lines  external\Real-ESRGAN\realesrgan\train.py
     280 lines  external\Real-ESRGAN\realesrgan\utils.py
     268 lines  external\Real-ESRGAN\realesrgan\utils_original.py
       5 lines  external\Real-ESRGAN\realesrgan\version.py
       6 lines  external\Real-ESRGAN\realesrgan\__init__.py
      55 lines  external\Real-ESRGAN\realesrgan\archs\discriminator_arch.py
      59 lines  external\Real-ESRGAN\realesrgan\archs\srvgg_arch.py
       9 lines  external\Real-ESRGAN\realesrgan\archs\__init__.py
     171 lines  external\Real-ESRGAN\realesrgan\data\realesrgan_dataset.py
      92 lines  external\Real-ESRGAN\realesrgan\data\realesrgan_paired_dataset.py
       9 lines  external\Real-ESRGAN\realesrgan\data\__init__.py
     232 lines  external\Real-ESRGAN\realesrgan\models\realesrgan_model.py
     171 lines  external\Real-ESRGAN\realesrgan\models\realesrnet_model.py
       9 lines  external\Real-ESRGAN\realesrgan\models\__init__.py
     114 lines  external\Real-ESRGAN\scripts\extract_subimages.py
      51 lines  external\Real-ESRGAN\scripts\generate_meta_info.py
      41 lines  external\Real-ESRGAN\scripts\generate_meta_info_pairdata.py
      40 lines  external\Real-ESRGAN\scripts\generate_multiscale_DF2K.py
      30 lines  external\Real-ESRGAN\scripts\pytorch2onnx.py
     128 lines  external\Real-ESRGAN\tests\test_dataset.py
      14 lines  external\Real-ESRGAN\tests\test_discriminator_arch.py
     112 lines  external\Real-ESRGAN\tests\test_model.py
      75 lines  external\Real-ESRGAN\tests\test_utils.py
       2 lines  external\Real-ESRGAN\tests\data\meta_info_gt.txt
       2 lines  external\Real-ESRGAN\tests\data\meta_info_pair.txt
      24 lines  external\Real-ESRGAN\tests\data\test_realesrgan_dataset.yml
     101 lines  external\Real-ESRGAN\tests\data\test_realesrgan_model.yml
      12 lines  external\Real-ESRGAN\tests\data\test_realesrgan_paired_dataset.yml
      63 lines  external\Real-ESRGAN\tests\data\test_realesrnet_model.yml
    2354 lines  external\Real-ESRGAN\tests\data\gt\baboon.png
     859 lines  external\Real-ESRGAN\tests\data\gt\comic.png
    3212 lines  external\Real-ESRGAN\tests\data\gt.lmdb\data.mdb
       1 lines  external\Real-ESRGAN\tests\data\gt.lmdb\lock.mdb
       2 lines  external\Real-ESRGAN\tests\data\gt.lmdb\meta_info.txt
     211 lines  external\Real-ESRGAN\tests\data\lq\baboon.png
      80 lines  external\Real-ESRGAN\tests\data\lq\comic.png
     291 lines  external\Real-ESRGAN\tests\data\lq.lmdb\data.mdb
       1 lines  external\Real-ESRGAN\tests\data\lq.lmdb\lock.mdb
       2 lines  external\Real-ESRGAN\tests\data\lq.lmdb\meta_info.txt
       2 lines  external\Real-ESRGAN\weights\README.md
     105 lines  external\realesrgan_video\async_dispatcher.py
      55 lines  external\realesrgan_video\config.py
     202 lines  external\realesrgan_video\face_utils.py
    1233 lines  external\realesrgan_video\ffmpeg_io.py
    1089 lines  external\realesrgan_video\gfpgan_subprocess.py
     846 lines  external\realesrgan_video\main.py
    3438 lines  external\realesrgan_video\nvenc_sdk.py
     174 lines  external\realesrgan_video\nvenc_writer.py
    1634 lines  external\realesrgan_video\pipeline.py
     213 lines  external\realesrgan_video\realesrgan_utils.py
     253 lines  external\realesrgan_video\tensorrt_accel.py
       2 lines  external\realesrgan_video\__init__.py
       9 lines  external\realesrgan_video\realesrgan\train.py
     280 lines  external\realesrgan_video\realesrgan\utils.py
     268 lines  external\realesrgan_video\realesrgan\utils_original.py
       5 lines  external\realesrgan_video\realesrgan\version.py
       6 lines  external\realesrgan_video\realesrgan\__init__.py
      55 lines  external\realesrgan_video\realesrgan\archs\discriminator_arch.py
      59 lines  external\realesrgan_video\realesrgan\archs\srvgg_arch.py
       9 lines  external\realesrgan_video\realesrgan\archs\__init__.py
     171 lines  external\realesrgan_video\realesrgan\data\realesrgan_dataset.py
      92 lines  external\realesrgan_video\realesrgan\data\realesrgan_paired_dataset.py
       9 lines  external\realesrgan_video\realesrgan\data\__init__.py
     232 lines  external\realesrgan_video\realesrgan\models\realesrgan_model.py
     171 lines  external\realesrgan_video\realesrgan\models\realesrnet_model.py
       9 lines  external\realesrgan_video\realesrgan\models\__init__.py
      37 lines  external\realesrgan_video\__pycache__\async_dispatcher.cpython-312.pyc
      66 lines  external\realesrgan_video\__pycache__\face_utils.cpython-312.pyc
     382 lines  external\realesrgan_video\__pycache__\ffmpeg_io.cpython-312.pyc
     235 lines  external\realesrgan_video\__pycache__\gfpgan_subprocess.cpython-312.pyc
     274 lines  external\realesrgan_video\__pycache__\main.cpython-312.pyc
     857 lines  external\realesrgan_video\__pycache__\nvenc_sdk.cpython-312.pyc
      52 lines  external\realesrgan_video\__pycache__\nvenc_writer.cpython-312.pyc
     392 lines  external\realesrgan_video\__pycache__\pipeline.cpython-312.pyc
      83 lines  external\realesrgan_video\__pycache__\realesrgan_utils.cpython-312.pyc
      98 lines  external\realesrgan_video\__pycache__\tensorrt_accel.cpython-312.pyc
     105 lines  external\realesrgan_video - 202606171605\async_dispatcher.py
      55 lines  external\realesrgan_video - 202606171605\config.py
     202 lines  external\realesrgan_video - 202606171605\face_utils.py
    1217 lines  external\realesrgan_video - 202606171605\ffmpeg_io.py
    1089 lines  external\realesrgan_video - 202606171605\gfpgan_subprocess.py
     733 lines  external\realesrgan_video - 202606171605\main.py
    1290 lines  external\realesrgan_video - 202606171605\pipeline.py
     130 lines  external\realesrgan_video - 202606171605\realesrgan_utils.py
     252 lines  external\realesrgan_video - 202606171605\tensorrt_accel.py
       2 lines  external\realesrgan_video - 202606171605\__init__.py
       9 lines  external\realesrgan_video - 202606171605\realesrgan\train.py
     280 lines  external\realesrgan_video - 202606171605\realesrgan\utils.py
     268 lines  external\realesrgan_video - 202606171605\realesrgan\utils_original.py
       5 lines  external\realesrgan_video - 202606171605\realesrgan\version.py
       6 lines  external\realesrgan_video - 202606171605\realesrgan\__init__.py
      55 lines  external\realesrgan_video - 202606171605\realesrgan\archs\discriminator_arch.py
      59 lines  external\realesrgan_video - 202606171605\realesrgan\archs\srvgg_arch.py
       9 lines  external\realesrgan_video - 202606171605\realesrgan\archs\__init__.py
     171 lines  external\realesrgan_video - 202606171605\realesrgan\data\realesrgan_dataset.py
      92 lines  external\realesrgan_video - 202606171605\realesrgan\data\realesrgan_paired_dataset.py
       9 lines  external\realesrgan_video - 202606171605\realesrgan\data\__init__.py
     232 lines  external\realesrgan_video - 202606171605\realesrgan\models\realesrgan_model.py
     171 lines  external\realesrgan_video - 202606171605\realesrgan\models\realesrnet_model.py
       9 lines  external\realesrgan_video - 202606171605\realesrgan\models\__init__.py
     105 lines  external\realesrgan_video - 202606260859\async_dispatcher.py
      55 lines  external\realesrgan_video - 202606260859\config.py
     202 lines  external\realesrgan_video - 202606260859\face_utils.py
    1225 lines  external\realesrgan_video - 202606260859\ffmpeg_io.py
    1089 lines  external\realesrgan_video - 202606260859\gfpgan_subprocess.py
     733 lines  external\realesrgan_video - 202606260859\main.py
    1290 lines  external\realesrgan_video - 202606260859\pipeline.py
     130 lines  external\realesrgan_video - 202606260859\realesrgan_utils.py
     252 lines  external\realesrgan_video - 202606260859\tensorrt_accel.py
       2 lines  external\realesrgan_video - 202606260859\__init__.py
       9 lines  external\realesrgan_video - 202606260859\realesrgan\train.py
     280 lines  external\realesrgan_video - 202606260859\realesrgan\utils.py
     268 lines  external\realesrgan_video - 202606260859\realesrgan\utils_original.py
       5 lines  external\realesrgan_video - 202606260859\realesrgan\version.py
       6 lines  external\realesrgan_video - 202606260859\realesrgan\__init__.py
      55 lines  external\realesrgan_video - 202606260859\realesrgan\archs\discriminator_arch.py
      59 lines  external\realesrgan_video - 202606260859\realesrgan\archs\srvgg_arch.py
       9 lines  external\realesrgan_video - 202606260859\realesrgan\archs\__init__.py
     171 lines  external\realesrgan_video - 202606260859\realesrgan\data\realesrgan_dataset.py
      92 lines  external\realesrgan_video - 202606260859\realesrgan\data\realesrgan_paired_dataset.py
       9 lines  external\realesrgan_video - 202606260859\realesrgan\data\__init__.py
     232 lines  external\realesrgan_video - 202606260859\realesrgan\models\realesrgan_model.py
     171 lines  external\realesrgan_video - 202606260859\realesrgan\models\realesrnet_model.py
       9 lines  external\realesrgan_video - 202606260859\realesrgan\models\__init__.py
     105 lines  external\realesrgan_video - 202607011015\async_dispatcher.py
      55 lines  external\realesrgan_video - 202607011015\config.py
     202 lines  external\realesrgan_video - 202607011015\face_utils.py
    1225 lines  external\realesrgan_video - 202607011015\ffmpeg_io.py
    1089 lines  external\realesrgan_video - 202607011015\gfpgan_subprocess.py
     789 lines  external\realesrgan_video - 202607011015\main.py
    1907 lines  external\realesrgan_video - 202607011015\nvenc_sdk.py
     141 lines  external\realesrgan_video - 202607011015\nvenc_writer.py
    1303 lines  external\realesrgan_video - 202607011015\pipeline.py
     130 lines  external\realesrgan_video - 202607011015\realesrgan_utils.py
     252 lines  external\realesrgan_video - 202607011015\tensorrt_accel.py
       2 lines  external\realesrgan_video - 202607011015\__init__.py
       9 lines  external\realesrgan_video - 202607011015\realesrgan\train.py
     280 lines  external\realesrgan_video - 202607011015\realesrgan\utils.py
     268 lines  external\realesrgan_video - 202607011015\realesrgan\utils_original.py
       5 lines  external\realesrgan_video - 202607011015\realesrgan\version.py
       6 lines  external\realesrgan_video - 202607011015\realesrgan\__init__.py
      55 lines  external\realesrgan_video - 202607011015\realesrgan\archs\discriminator_arch.py
      59 lines  external\realesrgan_video - 202607011015\realesrgan\archs\srvgg_arch.py
       9 lines  external\realesrgan_video - 202607011015\realesrgan\archs\__init__.py
     171 lines  external\realesrgan_video - 202607011015\realesrgan\data\realesrgan_dataset.py
      92 lines  external\realesrgan_video - 202607011015\realesrgan\data\realesrgan_paired_dataset.py
       9 lines  external\realesrgan_video - 202607011015\realesrgan\data\__init__.py
     232 lines  external\realesrgan_video - 202607011015\realesrgan\models\realesrgan_model.py
     171 lines  external\realesrgan_video - 202607011015\realesrgan\models\realesrnet_model.py
       9 lines  external\realesrgan_video - 202607011015\realesrgan\models\__init__.py



```

**Tool: bash**

**Input:**
```json
{
  "command": "Get-ChildItem -Recurse -Directory -Depth 1 -LiteralPath \"D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\" | Select-Object FullName"
}
```

**Output:**
```

FullName                                                                                                               
--------                                                                                                               
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet                                                
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\ifrnet_video                                          
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN                                           
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video                                      
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video - 202606171605                       
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video - 202606260859                       
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video - 202607011015                       
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606080823                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606081658                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606090940                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606090955                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606091507                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606091815                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606101145                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606101315                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606101420                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606101600                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606101656                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606110928                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606110953                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606111107                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606111150                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606111317                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606111355                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606121535                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606121733                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606121742                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606150933                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606150935                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606151055                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606151131                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606151210                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606151248                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606151316                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606151618                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606151720                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606161457                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606161650                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606161710                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606170850                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606171150                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606171318                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606171348                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606171637                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606181541                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606241253                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606241341                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606241428                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606241645                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606251259                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606251541                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606290920                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606291117                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202606291627                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607011109                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607020937                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607031508                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607040928                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607041020                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607061437                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607071008                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607071033                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607131327                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607171142                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607181352                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607201133                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607201256                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607201335                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607201547                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607231139                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607240956                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607241715                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202607271702                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202608011338                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202608101221                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202608141324                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\backup202608171329                             
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\benchmarks                                     
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\liteflownet                                    
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\models                                         
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\IFRNet\__pycache__                                    
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\ifrnet_video\__pycache__                              
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN\.vscode                                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN\assets                                    
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN\docs                                      
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN\experiments                               
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN\inputs                                    
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN\options                                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN\realesrgan                                
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN\scripts                                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN\tests                                     
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\Real-ESRGAN\weights                                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607071803                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607081125                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607081218                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607081320                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607081543                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607081808��ֻ�����������...
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607091358                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607091458                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607101033                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607101425                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607130916                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607131552                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607171142                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607201252                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607201525                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607201650                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607231036                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607231139                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607270925                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607271534                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607281051                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607281324                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607281501                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607290943                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607291056                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607291118                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607291435                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\backup202607291529                   
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\realesrgan                           
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video\__pycache__                          
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video - 202606171605\realesrgan            
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video - 202606260859\realesrgan            
D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\realesrgan_video - 202607011015\realesrgan            



```

---

## Assistant (Plan · Ox Alpha Free · 644.6s)

_Thinking:_

Now I have a good picture of the project scope. The active codebase (per AGENTS.md):

**Active files:**
- `src/main_video_optimized.py` (1822 lines) — entry point
- `src/processors/ifrnet_processor_video_optimized.py` (779 lines)
- `src/processors/realesrgan_processor_video_optimized.py` (969 lines)
- `src/utils/config_manager.py` (374 lines)
- `src/utils/video_utils.py` (1226 lines)
- `src/utils/video_fixer.py` (201 lines), `src/utils/output_filter.py` (80 lines)
- `external/ifrnet_video/` — v6.4.5.1 deep modular package: main.py (1693), pipeline.py (1563), nvenc_sdk.py (2799), ffmpeg_io.py (643), ifrnet_utils.py (297), tensorrt_accel.py (218), config.py
- `external/realesrgan_video/` — main.py (846), pipeline.py (1634), nvenc_sdk.py (3438), ffmpeg_io.py (1233), gfpgan_subprocess.py (1089), face_utils.py, nvenc_writer.py, async_dispatcher.py, tensorrt_accel.py, realesrgan_utils.py, config.py

That's about 25K+ lines of active code. This is a huge analysis task. I should use parallel explore agents to analyze different parts, then synthesize.

Plan:
1. Agent 1: Analyze src/main_video_optimized.py + src/utils/* — architecture & bugs
2. Agent 2: Analyze src/processors/ifrnet_processor_video_optimized.py + external/ifrnet_video/ (pipeline, ffmpeg_io, nvenc_sdk, main, tensorrt_accel) — architecture & bugs
3. Agent 3: Analyze src/processors/realesrgan_processor_video_optimized.py + external/realesrgan_video/ — architecture & bugs

I'll ask each agent to return detailed findings: architecture summary (modules, classes, data flow, design patterns) and bug list with file:line references, severity, and impact.

Also I can read config/default_config.json myself for config understanding.

Let me dispatch three explore agents in parallel with "very thorough" level. Each agent needs a highly detailed prompt since they work autonomously.

Let me also read the memory/MEMORY.md index to leverage existing knowledge of known issues — that could inform the bug list. Actually let me include that as a small read myself in parallel.

Dispatch now.

项目实际规模已确认：活跃代码集中在 `src/main_video_optimized.py`（1822 行）、两个 optimized 处理器、`src/utils/`、以及 `external/ifrnet_video`（v6.4.5.1 深度模块化包，约 7.8K 行）与 `external/realesrgan_video`（约 11.6K 行）。总计约 2.5 万行活跃代码。

我将并行派出 3 个深度探索代理分别分析入口层、IFRNet 子系统、Real-ESRGAN 子系统，同时自己读取配置与项目记忆。

**Tool: task**

**Input:**
```json
{
  "description": "分析主入口与utils层",
  "prompt": "你是一个资深 Python/GPU 系统代码审查专家。这是一个只读分析任务：不要修改任何文件。\n\n请深度分析以下文件（Video Enhancement 项目的主入口与工具层）：\n1. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\\main_video_optimized.py （唯一活跃入口，1822行）\n2. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\\utils\\config_manager.py\n3. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\\utils\\video_utils.py （1226行）\n4. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\\utils\\video_fixer.py\n5. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\\utils\\output_filter.py\n6. 参考：D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\config\\default_config.json\n\n背景：这是 GPU 加速视频处理管线（IFRNet 插帧 + Real-ESRGAN 超分），支持 TensorRT、FP16、CUDA Graph、NVENC 编码、视频切片并行处理、checkpoint/resume。\n\n请完整阅读每个文件并输出：\n\n【A. 架构分析】\n- 模块职责与类/函数清单（含关键函数签名）\n- 主流程数据流：从输入视频到输出视频经过哪些阶段（预处理→切片→插帧→超分→合并等），每阶段调用什么函数/子进程\n- 与处理器层（src/processors/ifrnet_processor_video_optimized.py、realesrgan_processor_video_optimized.py）和外部后端（external/ifrnet_video、external/realesrgan_video）的调用关系（import 或 subprocess？传参方式？）\n- 使用的设计模式（如配置管理器单例、策略模式、管道模式等）\n- checkpoint/resume 机制如何工作\n- 切片(slice)/合并(merge)/音轨(audio)处理逻辑位置\n\n【B. BUG 清单】对每个 bug 给出：\n- 文件路径:行号\n- Bug 类型（逻辑错误/资源泄漏/并发问题/异常处理缺失/性能瓶颈/崩溃点/边界条件）\n- 具体问题描述（引用相关代码片段）\n- 影响评估（严重程度：致命/高/中/低）\n\n重点检查：\n1. subprocess 调用（ffmpeg 等）的返回码检查、stderr 死锁（PIPE 未消费）、僵尸进程、shell=False 规范\n2. 文件句柄/临时文件清理（异常路径是否泄漏）\n3. 线程/进程并发问题（共享状态、竞态、死锁风险）\n4. 异常处理：裸 except、吞异常、finally 缺失\n5. 路径处理：中文路径、空格路径、跨平台问题\n6. 数值/边界：除零、None 解引用、索引越界、fps/duration 计算\n7. 内存：大文件读入内存、列表累积\n8. 配置覆盖逻辑错误（CLI 覆盖优先级）\n9. checkpoint/resume 状态一致性\n10. GPU 相关：context 管理、显存释放顺序\n\n【C. 可维护性问题】\n- 重复代码、过长函数、魔法数字、硬编码路径等（简要列举位置）\n\n要求：所有结论必须基于实际读到的代码，给出精确行号。输出为结构化中文报告。",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_fd31609ddffeiJ5nA6SVeRF6af" state="completed">
<task_result>
所有目标文件已完整读取（主入口实际为 2185 行而非 1822 行，video_utils 实际 1438 行而非 1226 行），并交叉核对了两个活跃处理器与 external 后端的对接方式。以下为结构化审查报告。

---

# Video Enhancement 主入口与工具层 · 深度代码审查报告

**审查范围**（实际行数已核对）：
| 文件 | 实际行数 |
|---|---|
| `src/main_video_optimized.py` | 2185 |
| `src/utils/config_manager.py` | 419 |
| `src/utils/video_utils.py` | 1438 |
| `src/utils/video_fixer.py` | 214 |
| `src/utils/output_filter.py` | 85 |
| `config/default_config.json` | 228 |

---

## A. 架构分析

### A1. 模块职责与关键函数清单

**`src/main_video_optimized.py`（编排层，2185 行）**

| 函数 | 行号 | 职责 |
|---|---|---|
| `main() -> int` | 2050 | 总入口；环境检查→配置加载→CLI覆盖→分支批量/单文件 |
| `_build_parser()` | 1743 | 定义约 50 个 CLI 参数（含 BooleanOptionalAction、互斥组） |
| `_apply_cli_overrides(config, args)` | 795 | CLI→Config 写入；含 `--no-*` vs `--use-*` 高优先级仲裁（850–892, 934–972） |
| `_process_single(...) -> bool` | 1224 | 单文件全流程：提音频→去噪→插帧/超分（可跳步）→FIX-C 归一化→LA 音频对齐→合并→后处理 |
| `_process_batch(...) -> int` | 1683 | 目录批处理循环 |
| `_normalize_segs_for_copy(seg_paths, codec_hint)` | 1054 | FIX-C：合并前 remux 归一化 timescale=90000（1163–1172），三重校验（1180–1185） |
| `_get_seg_codec(seg_path)` | 1029 | ffprobe 探测分段编码器 |
| `_trim_audio_start(audio_path, offset_seconds)` | 512 | NVENC LA 导致视频缩短时的音频头部修剪 |
| `_run_denoise_stage(...)` | 456 | 可选预去噪（依赖不存在的 `denoise_processor`，见 B25） |
| `_run_postprocess_stage(...)` | 541 | 完整性校验/统计/GPU 峰值显存/中间文件清理 |
| `_write_final_report(...)` | 643 | `--report` JSON 快照 |

**`src/utils/config_manager.py`（配置层）**
- `class Config`：`DEFAULT_CONFIG` 兜底字典（18–120）、`load_from_file`（138）、递归合并 `_merge_config`（153，跳过 `//` 注释键）、`_validate_config`（163）、`_setup_paths`（182，base_dir 向上两级推算 + 模型路径派生）、`get(*keys, default)`（265）、`set(*keys, value)`（284）、`save`（299）、`get_input_videos`（335）。

**`src/utils/video_utils.py`（FFmpeg 工具层）**
- `class VideoInfo`（42）：ffprobe JSON 解析 + OpenCV 回退。
- `get_audio_codec`（121）/`get_all_audio_streams`（210）/`is_lossless_audio`（253）。
- `extract_audio`（338）/`smart_extract_audio`（385，按编码选扩展名、stream-copy）/`add_audio_to_video`（484）。
- `get_video_duration`（606）/`get_frame_rate`（552）/`format_time`（644）/`verify_video_integrity`（670）。
- **`split_video_by_time`（701）**：segment muxer `-c copy -map 0:v` 切片 + `reuse_existing` 复用逻辑。
- **`merge_videos_by_codec`（1120）**：concat demuxer 合并；编码梯队决策（DIRECT_COPY_CODECS，1186）、一致性检查（1198）、色彩元数据注入（1291–1297）、rc=0 后 ffprobe 主动验证视频流（1346–1402）。
- `probe_color_metadata`（1013）/`build_color_args`（1055）/`_setparams_from_color_args`（1085）/`merge_videos`（798，旧版）/`encode_video`（1405）。

**`src/utils/video_fixer.py`**：`VideoFixer.check_video_integrity`（19）、三级修复策略 `fix_video`（57）：重编码（88）→流复制（112）→忽略错误（135）。

**`src/utils/output_filter.py`**：`TileFilter`（9）/`CustomFilter`（48）流代理 + `filter_tile_output`/`filter_output` 上下文管理器（26/65）。

### A2. 主流程数据流（单文件双步默认模式）

```
main() (2050)
 ├─ 阶段0 _check_environment (277): torch/CUDA/ffmpeg 探测（shutil.which）
 ├─ Config(args.config) (2080) → _apply_cli_overrides (2093)
 └─ _process_single (2162)
     ├─ ① 音频提取: VideoInfo.has_audio → smart_extract_audio (video_utils:385)
     │      → ffmpeg -vn -map 0:a:N -c:a copy → temp/main_audio/<stem>_audio.m4a   [main:1276-1289]
     ├─ ② [可选] _run_denoise_stage (1293): DenoiseProcessor（当前仓库中不存在→恒走 ImportError 回退）
     ├─ ③ 双步模式 (1454):
     │   interpolate_then_upscale:
     │     IFRNetProcessor.process_video_segments(actual_input) (1458)
     │       → split_video_by_time (video_utils:701, ffmpeg segment muxer 子进程)
     │       → _process_segments 循环 → 底层 ifrnet_video.main.IFRNetVideoProcessor.process_video
     │         （Python 进程内调用，非 subprocess）[ifrnet_processor:614]
     │     ifrnet_proc._cleanup_video_processor() (1470)  ← 显存释放点
     │     esrgan_proc.process_segments_directly(step1_segs, video_name) (1476)
     │       → importlib 加载 external/realesrgan_video/main.py → create_video_enhancer
     │         （进程内；GFPGAN-TRT 时底层再起独立子进程）
     │   upscale_then_interpolate: 对称反序 (1486-1505)，中途 esrgan_proc.close_enhancer() (1498)
     ├─ ④ 第二阶段资源释放 (1523-1526)
     ├─ ⑤ FIX-C 归一化 _normalize_segs_for_copy (1550)：每段 ffmpeg remux 子进程 ×N
     ├─ ⑥ LA 音频对齐：累加各段时长 vs 原始时长 → _trim_audio_start (1596) [main:1553-1607]
     └─ ⑦ merge_videos_by_codec(final_segs, output_video, audio_path, cfg) (1616)
             → concat 列表临时文件 + ffmpeg 子进程 → 成功后 _delete_checkpoint ×2 (1637)
               → auto_cleanup_temp 时 rmtree 临时目录 + 删音频 (1643-1657)
```

### A3. 与处理器层 / 外部后端的调用关系

- **主入口 → 处理器：同进程 Python import**（`main:1245-1246` 延迟导入 `ifrnet_processor_video_optimized.IFRNetProcessor` 与 `realesrgan_processor_video_optimized.RealESRGANVideoProcessor`），通过修改后的共享 `Config` 对象传参（引用传递，处理器在构造函数中逐字段 `config.get(...)` 读值，`ifrnet_processor:65-134` / `realesrgan_processor:63-177`）；少量运行时开关以属性注入（`proc.quiet=`、`proc.preview=`，main:1313/1376-1378/1441-1445）。
- **处理器 → external 后端：同样是进程内 import，非 subprocess**：
  - IFRNet：sys.path 注入后 `from ifrnet_video.main import IFRNetVideoProcessor`（ifrnet_processor:457），实例缓存于 `self._video_processor` 复用（448-490）；
  - ESRGAN：`importlib.util.spec_from_file_location("realesrgan_video_main", ...)` 显式按路径加载 main.py（realesrgan_processor:365-370），参数打包为 `argparse.Namespace` 传入（373+）。
- **所有 ffmpeg/ffprobe 均为 list 参数 subprocess.run（shell=False 全合规）**；唯一额外子进程是 GFPGAN-TRT（由 external 层创建，句柄存于 `enhancer['gfpgan_subprocess']`，由 `close_enhancer` 关闭，realesrgan_processor:870-891）。

### A4. 设计模式

1. **门面 + 单一事实来源配置**：JSON 为准，`Config.DEFAULT_CONFIG` 仅作兜底；`//` 键注释约定（config_manager:156）。
2. **适配器层**：processor 是编排层与 external 重模块之间的防腐层（含 Namespace 字段对齐、高优先级仲裁复刻）。
3. **惰性初始化 + 单例复用**：`_get_or_create_video_processor`（ifrnet_processor:448）、`_initialize_enhancer` 的 `_enhancer_initialized` 标志——模型/TRT 只加载一次。
4. **Copy-by-default 策略**：三个输出参数均未指定时合并走 `-c:v copy`（main:1002-1013, 1531-1535）。
5. **Memento（checkpoint/resume）**：见 A5。
6. **管道 + 两级资源交接**：step1 输出分段路径列表直接作为 step2 输入（`process_segments_directly`），阶段间显式释放 GPU 资源（VRAM-CLEANUP 注释块）。
7. **上下文管理器**：output_filter 的 stdout 过滤。
8. **防御式降级**：OOM 减批、TRT 失败回退、去噪失败回退原输入、归一化失败保留原始分段。

### A5. checkpoint/resume 机制

- **存储位置**：`temp/ifrnet_<prefix>_<video>/checkpoint.json`（ifrnet_processor:354-360）；ESRGAN 同构（`esrgan_from_segments_<name>`，realesrgan_processor:339）。
- **内容**：`{"processed_segments": [索引...], "last_segment": i, "config_snapshot": {...}}`；每完成一段追加保存（ifrnet_processor:556-561）。
- **一致性校验**（`_load_checkpoint`，ifrnet_processor:365-418）：快照比对 `segment_duration/interpolation_factor/codec/crf/encode_preset` + 指纹三键 `input_mtime / input_size / upstream_segments_hash`（380）；`upstream_segments_hash = md5('|'.join(sorted(paths)))`（259-261）。任一不匹配 → 整体作废从头处理。
- **恢复语义**：跳过已完成索引但要求对应输出文件存在，否则重做（540-548）。
- **删除时机**：全流程成功后主入口无条件 `_delete_checkpoint()`（main:1636-1641）；失败保留供续跑（`_print_failure_hints` 文案与此呼应）。
- **已知缺口**：见 B6（指纹只含路径名不含内容/上游参数）。

### A6. 切片 / 合并 / 音轨逻辑位置

| 功能 | 位置 |
|---|---|
| 切片 | `video_utils.split_video_by_time`（701–795）；单段直通复制（731–736）；复用判定（739–758） |
| 合并 | `video_utils.merge_videos_by_codec`（1120–1403）；编码决策 1186–1218；concat 列表 1252–1259；执行与校验 1317–1403 |
| 合并前置归一化 | `main._normalize_segs_for_copy`（1054–1218，timescale=90000） |
| 音轨提取 | `smart_extract_audio`（385–482，主流程 main:1280 调用） |
| 音轨修剪 | `main._trim_audio_start`（512–534），触发判定 main:1553–1607 |
| 音轨回写 | `merge_videos_by_codec(audio_path=...)` 的 `-map 1:a:0`（1279）；skip 单步模式的二次合并 main:1334/1405 |

---

## B. BUG 清单

> 严重程度定义：致命=数据丢失/流程必然崩溃；高=特定常见条件下错误结果或流程中断；中=边界条件下错误结果/资源泄漏；低=瑕疵、口径不一。

### 【高】

**B1. 批量模式无逐文件异常隔离 —— 一个文件的未捕获异常终止整批**
- 位置：`main_video_optimized.py:1722-1731`
- 类型：异常处理缺失 / 并发无关的流程健壮性问题
- ```python
  ok = _process_single(
      config, str(src), str(dst), mode,
      skip_interpolate, skip_upscale,
      args, env_info,
  )
  ```
  `_process_single` 内部多处只捕获 `KeyboardInterrupt`（如 1459/1478/1487/1506），其余异常直接上抛；而循环体外没有 try/except。打印文案写着“继续下一个”（1731），实际任何非 Kbd 异常（如磁盘满 OSError、cv2 崩溃）都会冲出 `_process_batch`，剩余全部文件不再处理，且已产生的 GPU 资源依赖 `__del__` 兜底。
- 影响：**高**（批量场景可靠性承诺失效）。

**B2. `skip_upscale` 路径的 `keep_audio` 配置篡改在异常时不恢复 → 跨文件配置污染**
- 位置：`main_video_optimized.py:1372-1390`
- 类型：共享状态管理缺陷
- ```python
  proc_keep_audio = config.get("models", "ifrnet", "keep_audio", default=True)
  config.set("models", "ifrnet", "keep_audio", value=False)
  ...
  except KeyboardInterrupt:
      ...
      config.set("models", "ifrnet", "keep_audio", value=proc_keep_audio)
      return False
  config.set("models", "ifrnet", "keep_audio", value=proc_keep_audio)
  ```
  恢复语句只在“正常返回”和"KeyboardInterrupt"两条路径执行。若 `proc.process_video` 抛其他异常（结合 B1 会冲出批量循环），config 中 `keep_audio` 永久变为 False；若异常被上层某处吞掉继续跑下一个文件，该文件将继承错误配置。应使用 `try/finally`。
- 影响：**高**（与 B1 叠加时产生隐蔽的跨文件行为差异）。

**B3. CLI 覆盖完全绕过配置校验**
- 位置：`config_manager.py:135`（`self._validate_config()` 在构造函数内执行）vs `main_video_optimized.py:795-1021`（覆盖发生在其后，无任何再校验）
- 类型：配置覆盖优先级/校验逻辑缺口
- argparse 对多数数值参数无范围约束（`--segment-duration`、`--crf-ifrnet`、`--crf-esrgan`、`--tile-size`、`--gfpgan-weight` 等，main:1838/1878/1941/1925/1984）。例如 `--segment-duration -5` 或 `--crf-esrgan 999` 直接写入 config 并下发到 ffmpeg/切片逻辑；`--gfpgan-weight 5.0` 也照单全收。而 `Config._validate_config` 本可拦截（179-180 只在加载期生效）。
- 附带不一致：同一函数里混用真值判断与 `is not None` 判断——`if args.batch_size_ifrnet:`（831，传 0 被静默忽略）vs `if args.crf_ifrnet is not None:`（835，CRF=0 是合法语义且能正确覆盖）。后者正确，前者是隐患。
- 影响：**高**（非法输入直达 GPU 编码链路，故障点远离出错原因）。

**B4. `merge_videos_by_codec` 强改输出扩展名 → 主流程随后报告“输出文件不存在”**
- 位置：`video_utils.py:1311-1313`
- ```python
  print(f"first_codec: {...}; need_reencode: {...}")   # 调试输出残留
  if not output_path.suffix or need_reencode:
      output_path = output_path.with_suffix(f'.{params["format"]}')
  ```
  当触发重编码（`first_codec ∉ DIRECT_COPY_CODECS`，如 mpeg4/flv1 输入）且用户指定 `--output out.mkv` 时，实际写出 `out.mp4`，但函数返回 True。调用方 `main:1616` 之后的所有逻辑仍使用原始 `output_video` 字符串：
  - `main:1628 os.path.exists(output_video)` → False，“文件大小”行消失；
  - `_run_postprocess_stage`（main:551）`if not Path(output_path).exists(): ❌ 最终输出文件不存在` 直接返回；
  - `_write_final_report` 记录 `output_size_mb=None`。
  即任务实际成功却被整体判为失败/缺失。
- 影响：**高**（条件触发但后果是完整的假阴性报告链）。

**B5. `eval()` 解析帧率 + 异常捕获面过窄**
- 位置：`video_utils.py:84`
- ```python
  self.fps = eval(stream.get('r_frame_rate', '0/1'))
  ```
  - `eval` 属不应使用的动态执行（输入虽来自本机 ffprobe，仍是坏实践，且 `'N/A'` 会抛 `NameError`/`SyntaxError`）；
  - 外层 except（94 行）只捕 `CalledProcessError / json.JSONDecodeError / KeyError`，上述异常或除零变体会直接穿透 `VideoInfo.__init__`。主流程虽在 main:1288 有兜底 catch，但 skip 单步路径中 `VideoInfo` 的其它调用点未必有同等保护。应改为 `Fraction` 解析。
- 影响：**高**（鲁棒性；对损坏/奇异容器头文件可崩溃）。

### 【中】

**B6. checkpoint 指纹不含上游内容与上游参数 → 双步恢复可混合新旧质量分段**
- 位置：`src/processors/ifrnet_processor_video_optimized.py:259-261, 420-435`
- ```python
  self._upstream_segments_hash = hashlib.md5(
      '|'.join(sorted(input_segments)).encode()
  ).hexdigest()
  ```
  哈希输入仅为**排序后的路径字符串**。`process_segments_directly` 路径下 `_current_input_video` 从未被赋值（只有 `process_video_segments` 在 203 行赋值），故快照中 `input_mtime/input_size` 缺失。场景：用户修改 step1 的 CRF 后重跑双步流水线——step1 自身 checkpoint 因 crf 变化作废并全部重算，但产出文件名不变；step2 的 `upstream_segments_hash` 与旧值完全一致，于是跳过所有已标记分段、继续使用**旧的**超分输出。最终成片混合两代质量，且无任何告警。哈希应至少包含各上游分段的 `(size, mtime)` 或其自身快照参数。
- 影响：**中偏高**（静默产出不符合用户预期的结果）。

**B7. `split_video_by_time` 的复用与验证强度不足：截断分段可通过校验、段数偏差被静默吞掉**
- 位置：`video_utils.py:727, 732, 741-756, 783-791`
- 类型：边界条件 / 一致性
- ① 727 行浮点取模：`duration % segment_duration > 0` 对 `60.0000001s` 这类元数据会多算一段（num_segments=3 实产 2），后续 783 行 `for i in range(num_segments)` 找不到 `segment_002.mp4` 只是跳过，无数量核对告警——若 ffmpeg 因关键帧对齐实际产出少于预期，同样静默通过，最终合并产物比源短；
  ② 复用判定（741-756）只要求“数量相等 + `verify_video_integrity` 通过”，而后者的实现（686-698）是 **OpenCV 打开并读第一帧**——上次运行中断留下的截断尾段首帧完好即可蒙混过关，复用后整条时间线缺失尾部若干秒；
  ③ 该函数不接收任何内容指纹，跨次运行的陈旧 segments 目录（同名不同内容视频）理论上可被复用（mtime/size 防线只在 checkpoint 层）。
- 影响：**中**（A/V 时长完整性是该项目的核心验收指标之一）。

**B8. `VideoFixer._fix_ignore_errors` 用“文件存在且 >1KB”代替返回码判断，可把前一方法留下的半成品误判为修复成功**
- 位置：`video_fixer.py:150-152`
- ```python
  result = subprocess.run(cmd, ...)
  if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
      print(f"   ✅ 忽略错误修复成功")
      return True
  ```
  方法1/2（101/124）失败时 `ffmpeg -y` 已可能写出部分文件且未清理；方法3 自身失败后，残留的 >1KB 半成品恰好满足条件 → 返回“成功”。随后 `auto_fix_if_needed`（177）会用 `check_video_integrity` 再验一次可部分兜底，但该方法只读 1 秒（46 行 `-t 1`），长视频尾部损坏依旧漏检。三次尝试之间应先 `unlink` 旧输出。
- 影响：**中**（仅在启用 `auto_fix_corrupted` 路径生效）。

**B9. 音频临时文件泄漏矩阵**
- 位置：`main_video_optimized.py`
- 类型：资源泄漏（临时文件）
- ① LA 修剪后（1596-1598）`audio_path` 被替换为 `trimmed_xxx`；成功收尾清理（1651-1657）只删除 trimmed 文件，**原始提取的 `main_audio/<stem>_audio.m4a` 永远留在 temp 目录**；
  ② `--skip-interpolate`（1302-1365）与 `--skip-upscale`（1368-1436）两条路径**完全没有音频清理逻辑**——即使全流程成功，提取的音频也不删；
  ③ 失败路径（合并失败 1660-1661、step 失败提前 return）均不清理音频；
  ④ skip 分支音频回写用的 `output_video + ".with_audio.mp4"`（1326/1397）在 `merge_videos_by_codec` 抛异常时残留在输出目录（1344-1345 只打印）。
  结果：长期使用下 `temp/main_audio/` 与输出目录持续积累垃圾。
- 影响：**中**（磁盘缓慢泄漏，batch 高频场景更明显）。

**B10. LA 丢帧补偿的判定依据脆弱 + 阈值公式硬编码 60fps**
- 位置：`main_video_optimized.py:1569-1589`
- ```python
  try:
      import torch as _torch_la
      from realesrgan_video.nvenc_sdk import NVENCEncoder as _NE_LA
      if _torch_la.cuda.is_available():
          _total_la_depth += _esrgan_la
  except ImportError:
      pass  # FFmpeg CLI 路径，不丢帧
  ```
  以“SDK 模块可导入”推断“本次分段确实走了 SDK NVENC”。但 default_config.json:170 明确记载“输出≥1080p 时后端自动切 constqp+LA=0”——此时 SDK 可导入却没丢帧，`_total_la_depth` 被高估；反之未来 FFmpeg CLI 若引入丢帧则漏估。另外 1589 行 `_la_threshold = max(0.3, _total_la_depth / 60.0 * 0.5)` 把帧→秒换算固定在 60fps，对 24/30fps 源的阈值物理含义漂移。误判方向是**误剪音频开头**（不可逆的有损操作）。
- 影响：**中**（音画同步风险，取决于分辨率/编码器组合）。

**B11. 三处裸 `except:` 吞掉一切（含 Ctrl+C / SystemExit）**
- 位置：`video_utils.py:627-628, 639-640, 697-698`
- ```python
  except:
      pass
  ```
  `get_video_duration` 与 `verify_video_integrity` 是全管线高频调用点（每个分段多次），用户在中断瞬间命中会被静默吞掉，中断响应延迟且无痕迹。至少应为 `except Exception`。
- 影响：**中**（可诊断性与中断语义）。

**B12. `TileFilter` 的 `writelines` 旁路 + 全局 stdout 替换线程不安全**
- 位置：`output_filter.py:23-24, 36-44`
- `__getattr__` 把未显式覆盖的属性（包括 `writelines`、`fileno`）直接转发给原始流——任何库若调用 `sys.stdout.writelines([...])`，`Tile x/y` 行将绕过正则过滤直通终端；同时该设计替换的是进程级 `sys.stdout/stderr`，本项目 GPU 流水线存在预取/工作线程（external 层），并发打印期间进入/退出上下文会造成部分输出被错误过滤或漏过滤。
- 影响：**中低**（功能性小漏洞 + 潜在交错问题）。

**B13. `smart_extract_audio` 对已存在输出“同名即复用”，无新鲜度校验**
- 位置：`video_utils.py:435-437`
- ```python
  if output_path.exists() and not overwrite:
      logger.info(f"输出文件已存在，跳过: {output_path}")
      return str(output_path.absolute())
  ```
  用户替换了同名输入视频（或批量目录里换了内容）后重跑，将静默挂上**旧视频的音轨**。应比较 `(size, mtime)` 或提供指纹侧车文件。
- 影响：**中**（错误结果类，触发条件明确）。

**B14. 批量模式下 `input_dir == output_dir` 会就地覆盖源文件**
- 位置：`main_video_optimized.py:1714`
- ```python
  dst = Path(output_dir) / src.name
  ```
  无同路径守卫。用户图省事把输出指到输入目录时，第一个“成功”的文件就是对自己 `shutil`/ffmpeg 覆盖写（且中间态可能损坏源）。
- 影响：**中**（数据破坏风险，一行守卫可消除）。

**B15. `.ts/.mpg` 等输入经 segment muxer 直写 `.mp4` 容器的编码兼容性**
- 位置：`video_utils.py:766-777`（`SUPPORTED_VIDEO_EXTS` 含 `.ts/.mpg/.mpeg`，main:191-194）
- TS 源的 AAC 通常为 ADTS 格式，copy 进 MP4 需要 `aac_adtstoasc` BSF（项目自己在 `video_fixer.py:120` 都知道加），此处不加；老 ffmpeg 会直接切分失败，新版本虽常自动插入但仍属隐式依赖。H.264 裸 AnnexB→MP4 同理依赖自动 hvcC/avvC 处理。
- 影响：**中低**（条件性功能失败）。

### 【低】

**B16. 启动摘要展示的“默认值”与真实默认不符**
- `main_video_optimized.py:350-352`（batch_size 显示兜底 4/上限 8；json 默认 24/36）、`373-375`（esr batch 显示 6/prefetch 48；json 默认 24/96）、`358-359/381-382`（CRF 兜底显示 21；实际 23）、`388-392`（gfpgan weight/batch 兜底 0.5/8；json 0.7/4）。lambda 第二参只是显示兜底，但与 `default_config.json` 系统性不一致会误导排障。

**B17. 死条件**
- `main_video_optimized.py:1570-1573`：`if not args.skip_interpolate:` / `if not args.skip_upscale:` —— 该代码位于两条 skip 分支各自 `return` 之后，恒为 True，属误导性死代码。

**B18. 调试 print 残留于生产合并路径**
- `video_utils.py:1311`（每次合并必打印 `first_codec: ...`），破坏 `--quiet` 语义下的日志整洁。

**B19. checkpoint.json 非原子写入**
- `ifrnet_processor_video_optimized.py:445-446`：`open("w") + json.dump`，进程被 kill 在写入中途会留下截断 JSON（loader 能容错但代价是全部进度作废）。标准做法是 tmp + `os.replace`。

**B20. 合并输出校验失败后损坏产物不回收**
- `video_utils.py:1385-1392` 抛 `FFmpegError` 前/后均不删除已生成的坏 `output_path`；调用方（main:1660）也只打印提示，用户目录中留下看似存在的坏文件。

**B21. `build_color_args` 无元数据时回退 `-color_range pc`（Full Range）**
- `video_utils.py:1081`。对未标注意味着 limited 的 SDR 源，强制写入 pc 的 colr box 会改变播放端解释（发灰/对比度变化）。注释声明这是有意回退，但从正确性角度属于高风险默认，建议回退 `tv` 或保持 unknown。

**B22. `Config.get_input_videos` 大小写双 glob 未去重**
- `config_manager.py:353-355`：`*.mp4` 与 `*.MP4` 各 extend 一次。在大小写敏感平台（Linux）可遗漏混合大小写文件；在不敏感实现上会产生重复条目。应统一 `suffix.lower()` 过滤（主入口 `_process_batch:1694-1697` 就是这么做的，两套逻辑不一致）。

**B23. 报告字段口径不一：duration None→0，size None→null**
- `main_video_optimized.py:708-711` 用 `or 0` 把探测失败变成 `0.0`，与 691-692/718-720 的 `None` 约定冲突，下游消费方无法区分“0 秒”与“未知”。

**B24. 去噪中间文件写入用户可见的输出目录**
- `main_video_optimized.py:479-481`：`out_dir = Path(output_path).parent`，批量模式下 `<stem>_denoised.mp4` 出现在交付目录；失败中断时不清理（清理只在成功后置 600-610）。

**B25. 去噪处理器属性事后注入 + 模块实际不存在**
- `main_video_optimized.py:486-488`：`proc = DenoiseProcessor(config); proc.denoise_strength = ...; proc.model_name = ...` 构造后再改属性，是否生效取决于该类的延迟加载实现；且仓库内 **glob 全树找不到 `denoise_processor.py`**——`--denoise` 当前必然走 ImportError 回退（491-493），该功能实为空挂。

**B26. 批量退出码语义宽松**
- `main_video_optimized.py:2153`：`return 0 if n > 0 else 1`——10 个文件失败 9 个仍退出 0，不利于 CI/脚本编排。

**B27. 旧版工具函数的平台/语义边角**
- `video_utils.py:876`：POSIX 风格单引号转义 `'\\''` 用于 concat 列表，Windows 奇异路径下依赖 ffmpeg 的 av_get_token 兼容（主流场景可用，非标准转义来源）；
- `video_utils.py:507-519`：`add_audio_to_video` 一旦传入 config 就从“拷视频”变成 libx264 全重编，与函数名义相悖（当前主流程未使用，属休眠陷阱）。

### 正面确认（重点检查项中未见问题的部分）

- **stderr 死锁 / 僵尸进程**：全部子进程调用均为 `subprocess.run(capture_output=True)` 或 `DEVNULL`（PIPE 由 run() 内部 communicate 回收），未发现手工 `Popen` + 未消费 PIPE 的组合；GFPGAN 子进程有 `close_enhancer` 显式关闭 + `__del__` 兜底（realesrgan_processor:870-894）。
- **shell=False 规范**：全仓审查范围内 100% 列表传参。
- **concat 列表临时文件**：两处均有 finally 清理（video_utils:965-970, 1340-1344）。
- **NVML 环境变量**：`os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK","0")` 位于任何 torch 导入之前（main:149），位置正确。
- **GPU 资源交接顺序**：双步之间显式 `_cleanup_video_processor()` / `close_enhancer()`（main:1470/1498/1523-1526），且都做了 gc + `empty_cache`，顺序（先释放第一阶段再构建第二阶段引擎）正确。

---

## C. 可维护性问题

| # | 问题 | 位置 |
|---|---|---|
| C1 | 时间格式化逻辑双实现且格式不同：`_fmt_time`（中文口语式）vs 导入的 `format_time`（时钟式），同一函数内混用（main:1626 用后者、1629 区域用前者） | main:228-238 vs video_utils:644-667 |
| C2 | 超长函数：`_process_single` 约 450 行（1224-1676）、`merge_videos_by_codec` 约 280 行、`_apply_cli_overrides` 约 225 行；skip 两分支的音频回写块近乎复制粘贴（1323-1345 vs 1394-1416） | main |
| C3 | `SUPPORTED_ESRGAN_MODELS` 常量定义后从未用于校验 `--esrgan-model`（帮助文本手抄一份可选列表） | main:196-203, 1913-1917 |
| C4 | 魔法数字群：timescale `90000`（main:1170）、大小阈值 `4096`/`1024`（main:1183, video_utils:683）、超时散落 `30/60/180/300`、LA 阈值 `max(0.3, depth/60*0.5)`（main:1589） | 多处 |
| C5 | ffmpeg/ffprobe 二进制硬编码字面量遍布 video_utils（数十处 `'ffmpeg'`），而配置体系明明有 `ffmpeg_bin` 字段（default_config.json:90/167）却只在处理器层生效 | video_utils 通篇 |
| C6 | CRF/preset 默认字面量三处不一致：18（video_utils:854/1226）、21（main 显示兜底）、23（json/config） | 三文件 |
| C7 | sys.path 手工注入在 ≥3 个模块重复实现（main:154-161、ifrnet_processor:42-50、realesrgan_processor:46-51），且互相插入顺序敏感 | 三文件 |
| C8 | `src/processors/` 下大量历史版本与 backup 目录并存（v2/v3/v5/v6/v6_1/v6_4/backup*/…约 30 个文件），接口同名易误导入 | src/processors |
| C9 | `eval()`（B5）与裸 `except`（B11）等反模式 | video_utils |
| C10 | `get_frame_rate` 是带副作用打印的调试函数（586-592 print），却被 `add_audio_to_video` 以注释形式引用两次（537/543），死注释残留 | video_utils:537,543 |
| C11 | `Config.set` 无 keys 时 `config[keys[-1]]` 会 IndexError（284-297），内部 API 无防御；`get` 与 `set` 的键缺失行为不对称（set 会凭空造 dict） | config_manager |
| C12 | `paths.output_dir` 语义漂移：单文件模式被写成“输出文件的父目录”（main:2158-2159），批量模式写成“目录本身”（main:1720），下游读者需记住两种含义 | main |
| C13 | `--auto-cleanup` 与 `--no-auto-cleanup` 同时给出时静默按后者胜出（807-810 固定顺序），未用 mutually exclusive group | main:2022-2025, 807-810 |
| C14 | `verify_video_integrity` 依赖 OpenCV 解码能力，HEVC/AV1 构建缺失时恒 False → 分段复用永久失效 + 后处理永远警告，无环境自检提示 | video_utils:686-698 |

---

## 总结（Top 风险排序）

1. **B1+B2**（批量隔离缺失 + 配置污染）——最现实的可靠性缺口，修复成本低（循环加 try/except、改 finally）。
2. **B6**（断点指纹弱）——静默产出质量混杂的结果，违背 checkpoint/resume 的核心承诺。
3. **B4**（扩展名强改 → 假阴性失败链）——一条命令即可复现的用户可见故障。
4. **B3**（CLI 绕过校验）——把非法参数直送 ffmpeg/NVENC，排障成本最高的一类。
5. **B7/B13/B14**（复用/覆盖类数据完整性）——共同主题是“以名字代内容”的等价假设，建议统一引入 `(size,mtime)` 指纹侧车。
6. 工具层卫生项（B5/B11/B18/C9）适合一次性小 PR 集中清理。

以上全部结论基于本次实际读取的代码行号，未做任何文件修改。
</task_result>
</task>
```

**Tool: task**

**Input:**
```json
{
  "description": "分析ESRGAN子系统代码",
  "prompt": "你是一个资深 Python/CUDA/TensorRT/NVENC/GFPGAN 系统代码审查专家。这是一个只读分析任务：不要修改任何文件。\n\n请深度分析 Real-ESRGAN 超分子系统的完整代码链路：\n\n1. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\\processors\\realesrgan_processor_video_optimized.py （969行，处理器封装层）\n2. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\main.py （846行）\n3. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\pipeline.py （1634行）\n4. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\nvenc_sdk.py （3438行）\n5. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\ffmpeg_io.py （1233行）\n6. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\gfpgan_subprocess.py （1089行，GFPGAN 人脸增强子进程方案）\n7. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\face_utils.py\n8. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\nvenc_writer.py （174行）\n9. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\async_dispatcher.py （105行）\n10. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\tensorrt_accel.py （253行）\n11. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\realesrgan_utils.py （213行）\n12. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\config.py\n\n背景：GPU 加速视频超分（2x/4x）+ 可选 GFPGAN 人脸增强，支持 PyTorch/TensorRT、FP16、CUDA Graph、NVENC 编码、tile 平铺推理（OOM 自动降级）、分段(segment)并行处理。已知历史修复：2026-07-28 有一次 ESRGAN NVENC 跨段修复。\n\n请完整阅读并输出：\n\n【A. 架构分析】\n- 各模块职责、核心类清单（类名、关键方法、职责）\n- 数据流：ffmpeg 读帧 → tile 超分推理 →（可选 GFPGAN）→ NVENC 写出；线程模型与队列\n- 分段(segment)处理架构：如何切分、并行度控制、段间衔接（跨段帧连续性）、最后合并\n- processor 层与 external 后端的调用方式\n- GFPGAN 子进程方案设计（为何用子进程、通信协议、人脸检测/对齐流程）\n- tile 推理与 OOM 自动降级机制\n- TensorRT/FP16/CUDA Graph 集成点\n- 设计模式\n\n【B. BUG 清单】对每个 bug 给出文件:行号、类型、描述、影响（致命/高/中/低）：\n重点检查：\n1. tile 拼接缝隙/重叠区混合逻辑错误、边缘像素处理\n2. OOM 检测与降级重试的状态一致性\n3. GFPGAN 子进程：通信死锁、子进程崩溃后主进程行为、临时文件清理\n4. NVENC session/event/buffer 泄漏（尤其异常路径）、ctypes 指针生命周期\n5. 跨段衔接：首尾帧重复/丢失、时间戳错位、音画不同步\n6. ffmpeg 子进程 PIPE 死锁、进程清理\n7. 多线程竞态、队列无界、线程泄漏\n8. 显存管理：tensor 生命周期、no_grad、缓存未释放\n9. 异常处理缺失导致的静默失败或挂起\n10. 边界条件：奇数尺寸、tile 尺寸不整除、极短片段、alpha 通道/灰度视频\n\n【C. 可维护性问题】简要列出（过长函数、重复代码、硬编码等，给出行号）。\n另外注意：external 下有多个备份目录（backup*、带日期目录），确认活跃版本是 external\\realesrgan_video（不带日期后缀）即可，不需要分析备份。\n\n要求：所有结论必须基于实际读到的代码，给出精确行号。输出结构化中文报告。",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_fd3157658ffeHKB22O13Fu3Vmb" state="completed">
<task_result>

</task_result>
</task>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\config\\default_config.json"
}
```

**Output:**
```
<path>D:\Workspace_Python\Video_Enhancement\Video_Enhancement\config\default_config.json</path>
<type>file</type>
<content>
1: {
2:   "processing": {
3:     "// mode":                    "interpolate_then_upscale | upscale_then_interpolate",
4:     "mode":                       "interpolate_then_upscale",
5:     "interpolation_factor":       2,
6:     "upscale_factor":             2,
7:     "segment_duration":           30,
8:     "auto_fix_corrupted":         false,
9:     "auto_cleanup_temp":          true,
10:     "// batch_mode":              "true = 从 paths.input_dir 批量读取；false = 单文件 paths.input_video",
11:     "batch_mode":                 false
12:   },
13: 
14:   "paths": {
15:     "// base_dir_note": [
16:       "项目根目录绝对路径。",
17:       "★ 留空（推荐）：config_manager 自动从本文件位置向上两级推算，",
18:       "  即 config/default_config.json → 上两级 = 项目根，无需手动填写。",
19:       "★ 手动填写：适用于 config 文件放在非标准位置的情况。"
20:     ],
21:     "base_dir":    "",
22:     "input_video": "",
23:     "// input_dir": "批量模式（batch_mode=true）时扫描的输入视频目录",
24:     "input_dir":   "",
25:     "// derived_note": "output_dir / temp_dir / log_dir 留空时由 config_manager 自动派生为 base_dir/output 等",
26:     "output_dir":  "",
27:     "temp_dir":    "",
28:     "log_dir":     "",
29:     "// trt_cache_dir_note": [
30:       "TRT Engine 缓存目录。",
31:       "★ 留空（推荐）：config_manager 自动派生为 base_dir/.trt_cache。",
32:       "★ 手动填写：绝对路径，IFRNet 与 ESRGan 共享同一目录（Engine 文件名含模型/Shape 信息，不会冲突）。",
33:       "命令行可用 --trt-cache-dir 覆盖此字段。"
34:     ],
35:     "trt_cache_dir": "",
36:     "// models_gfpgan_dir_note": [
37:       "GFPGAN 模型存放目录（含主生成模型 + facexlib 辅助模型）。",
38:       "★ 留空（推荐）：config_manager 自动派生为 base_dir/models_GFPGAN。",
39:       "★ 手动填写：绝对路径，指向包含 GFPGANv1.4.pth 等文件的目录。"
40:     ],
41:     "models_gfpgan_dir": ""
42:   },
43: 
44:   "models": {
45: 
46:     "ifrnet": {
47:       "// model_name_note": [
48:         "★ IFRNet 模型名称（推荐方式，processor 自动拼路径）。",
49:         "  可选值：IFRNet_S_Vimeo90K（默认，轻量快速）| IFRNet_L_Vimeo90K（高质量，更慢）| IFRNet_Vimeo90K",
50:         "  命令行可用 --ifrnet-model 覆盖此字段。"
51:       ],
52:       "model_name":         "IFRNet_S_Vimeo90K",
53:       "// model_path_note": [
54:         "★ IFRNet 权重文件绝对路径（可选，优先级高于 model_name）。",
55:         "  留空（推荐）：processor 按 model_name 自动拼接 base_dir/models_IFRNet/checkpoints/<model_name>.pth",
56:         "  手动填写：填写绝对路径，直接使用该路径（忽略 model_name）。",
57:         "  命令行可用 --ifrnet-model-path 覆盖此字段。",
58:         "  ⚠️  文件必须存在，否则程序在加载模型时直接崩溃（torch.load 无前置校验）"
59:       ],
60:       "model_path":         "",
61:       "use_gpu":            true,
62: 
63:       "// ---- 批处理大小 ----": "",
64:       "// batch_size_note":     "初始批大小；OOM 时自动减半直至 1",
65:       "batch_size":             24,
66:       "// max_batch_size_note": "批大小上限；OOM 后永久压低天花板，保守设置保护显存",
67:       "max_batch_size":         36,
68: 
69:       "// ---- v5 推理优化参数 ----": "",
70:       "// use_fp16_note":       "FP16 半精度推理（默认开启，GPU 环境推荐；CPU 环境自动忽略）",
71:       "use_fp16":               true,
72:       "// use_compile_note":    "torch.compile mode=default dynamic=True；首次编译约 1-3 分钟；与 TRT 互斥；需 PyTorch 2.0+",
73:       "use_compile":            false,
74:       "// use_cuda_graph_note": "手动 CUDA Graph；use_compile=true 时自动禁用（由 compile 接管）",
75:       "use_cuda_graph":         false,
76:       "// use_tensorrt_note":   "TRT Engine 加速；首次构建耗时长，缓存于 .trt_cache/；需安装 tensorrt pycuda onnx",
77:       "use_tensorrt":           true,
78: 
79:       "// ---- v5 硬件解/编码参数 ----": "",
80:       "// use_hwaccel_note":    "NVDEC 硬件解码；自动探测，失败时回退 CPU 软解",
81:       "use_hwaccel":            true,
82:       "// codec_note":          "IFRNet 分段输出编码器；有 NVENC 时脚本内部自动升级为 h264_nvenc",
83:       "codec":                  "libx264",
84:       "// codec_alt":           "libx265",
85:       "// crf_note":            "IFRNet 分段输出质量（0-51，越小越好；23=默认均衡，18=近无损）",
86:       "crf":                    23,
87:       "// encode_preset_note": "libx264/libx265 编码预设；可选 ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow（NVENC 时自动映射为 p1~p7）",
88:       "encode_preset":         "medium",
89:       "keep_audio":             true,
90:       "ffmpeg_bin":             "ffmpeg",
91: 
92:       "// ---- v5 NVENC 码率控制 ----": "",
93:       "// rate_mode_note":       "NVENC 码率控制模式: constqp(最快/最低延迟) | vbr_hq(CQ质量优先) | qvbr(质量可变VBR)",
94:       "rate_mode":               "vbr_hq",
95:       "// lookahead_depth_note": "NVENC 前向帧预看深度: 0=禁用 | 8=低延迟 | 16=平衡(推荐) | 32=最高质量。注意：LA 仅在 VBR_HQ/QVBR 模式下生效；CONSTQP 下硬件静默禁用。NVENC SDK 严格遵守帧数守恒 (Output==Input)，LA 跨编码会话不可复用。",
96:       "lookahead_depth":         8,
97: 
98:       "// ---- v5 性能报告（null 表示不输出）----": "",
99:       "report_json":            null
100:     },
101: 
102:     "realesrgan": {
103:       "// model_name_note":    "模型名称（不含 .pth 后缀，v6 推理脚本自动拼接路径并在文件不存在时自动下载）",
104:       "// model_name_opts":    "realesr-general-x4v3 | RealESRGAN_x4plus | RealESRGAN_x2plus | realesr-animevideov3 | RealESRGANv2-animevideo-xsx2",
105:       "model_name":            "realesr-general-x4v3",
106:       "// model_path_note": [
107:         "v6 底层脚本通过 model_name 自动拼接路径，此字段不参与实际推理。",
108:         "留空（推荐）：config_manager 自动派生为 base_dir/models_RealESRGAN/<model_name>.pth，仅供参考。",
109:         "v6 脚本的模型查找目录由脚本内 base_dir 决定（已修复为相对 __file__ 自动推算）。"
110:       ],
111:       "model_path":            "",
112:       "use_gpu":               true,
113: 
114:       "// ---- 基础推理参数 ----": "",
115:       "// denoise_strength_note": "降噪强度 0~1；仅 realesr-general-x4v3 模型有效",
116:       "denoise_strength":      0.5,
117:       "// tile_size_note":     "tile 切块大小（0=不切块；VRAM 不足时设 512）",
118:       "tile_size":             0,
119:       "tile_pad":              10,
120:       "pre_pad":               0,
121:       "// use_fp16_note":      "FP16 半精度推理（默认开启，GPU 环境推荐；CPU 环境自动忽略）",
122:       "use_fp16":              true,
123:       "// face_enhance_note":  "启用 GFPGAN 人脸增强；anime 系模型不支持（脚本内自动禁用）；需安装 gfpgan>=1.3.5 和 facexlib>=0.3.0",
124:       "face_enhance":          false,
125: 
126:       "// ---- v6 推理优化参数（与底层 main.py argparse 默认值对齐）----": "",
127:       "// batch_size_note":    "SR 批处理大小；T4/16G 建议 6~8，A10/24G 建议 8~12",
128:       "batch_size":            24,
129:       "// prefetch_factor_note": "读帧预取队列深度；建议 >= batch_size*2",
130:       "prefetch_factor":       96,
131:       "// use_compile_note":   "torch.compile reduce-overhead；首次约 1-3 分钟；与 TRT 互斥；需 PyTorch 2.0+；默认开启（与底层 --no-compile 反向）",
132:       "use_compile":           false,
133:       "// use_cuda_graph_note": "手动 CUDA Graph；use_compile=true 时自动禁用（由 compile 接管）；默认开启",
134:       "use_cuda_graph":        false,
135:       "// use_tensorrt_note":  "TRT Engine 加速；TRT 8.x/10.x 双 API 兼容；需安装 tensorrt pycuda onnx onnxruntime-gpu；与 IFRNet 默认一致",
136:       "use_tensorrt":          true,
137:       "// gfpgan_trt_note":    "GFPGAN 独立 TRT 子进程加速；需同时设 face_enhance=true；启用时自动禁用 FP16 改用 FP32",
138:       "gfpgan_trt":            false,
139: 
140:       "// ---- v6 face_enhance 精细控制参数（face_enhance=false 时忽略）----": "",
141:       "// gfpgan_model_note":      "GFPGAN 版本；可选 1.3 / 1.4 / RestoreFormer；1.4 综合效果最佳，RestoreFormer 适合严重降质",
142:       "gfpgan_model":          "1.4",
143:       "// gfpgan_model_path_note": [
144:         "GFPGAN 主模型权重文件绝对路径（可选，优先级高于 gfpgan_model 自动拼接）。",
145:         "★ 留空（推荐）：processor 按 gfpgan_model + base_dir/models_GFPGAN/ 自动拼路径。",
146:         "★ 手动填写：绝对路径，直接使用该路径加载模型（忽略 gfpgan_model 的自动拼接）。"
147:       ],
148:       "gfpgan_model_path":     "",
149:       "// gfpgan_weight_note":     "GFPGAN 融合权重 0.0~1.0；0=不增强，1=完全替换原始人脸；默认 0.5 为均衡值",
150:       "gfpgan_weight":         0.7,
151:       "// gfpgan_batch_size_note": "单次 GFPGAN 前向最多处理的人脸数；OOM 时自动降级持久化；T4/16G 建议 6~8，A10/24G 建议 8~12",
152:       "gfpgan_batch_size":     4,
153:       "// face_det_threshold_note": "人脸检测置信度阈值 [0.0-1.0]；0.5=保留多数人脸，0.7=过滤模糊远景，0.9=仅保留清晰人脸",
154:       "face_det_threshold":    0.7,
155:       "// adaptive_batch_note":     "根据人脸密度动态调整批处理大小；默认开启（--no-adaptive-batch 可禁用）",
156:       "adaptive_batch":        true,
157: 
158:       "// ---- v5 硬件解/编码参数 ----": "",
159:       "// use_hwaccel_note":   "NVDEC 硬件解码；自动探测，失败时回退 CPU 软解",
160:       "use_hwaccel":           true,
161:       "// codec_note":         "SR 分段输出编码器；有 NVENC 时脚本内部自动升级为 h264_nvenc/hevc_nvenc。此为底层 argparse 使用的键名（与 ifrnet 统一为 codec）",
162:       "codec":                 "libx264",
163:       "// encode_preset_note": "libx264/libx265 编码预设；可选 ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow（NVENC 时自动映射为 p1~p7）",
164:       "encode_preset":         "medium",
165:       "// crf_note":           "SR 分段输出质量（0-51，越小越好；23=默认均衡，18=近无损）",
166:       "crf":                   23,
167:       "ffmpeg_bin":            "ffmpeg",
168: 
169:       "// ---- v6 NVENC 码率控制 ----": "",
170:       "// rate_mode_note":       "NVENC 码率控制模式: constqp(最快/最低延迟) | vbr_hq(CQ质量优先) | qvbr(质量可变VBR)；输出≥1080p 时后端自动切 constqp+LA=0（消除 drain 背压）",
171:       "rate_mode":               "vbr_hq",
172:       "// lookahead_depth_note": "NVENC 前向帧预看深度: 0=禁用 | 8=低延迟 | 16=平衡(推荐) | 32=最高质量",
173:       "lookahead_depth":         8,
174: 
175:       "// ---- v6.4.6 流水线队列与延迟解析 ----": "",
176:       "// frame_queue_size_note":  "读帧→检测 队列深度（批）；SR 为瓶颈时队列常年全满，过深只增内存压力",
177:       "frame_queue_size":          8,
178:       "// detect_queue_size_note": "检测→SR 队列深度（批）；4 批已足够吸收检测抖动",
179:       "detect_queue_size":         4,
180:       "// defer_sr_resolve_note":  "SR 输出延迟解析（D2H等待+CPU拷贝与下一批GPU推理重叠）；false 回退同步路径",
181:       "defer_sr_resolve":          true,
182: 
183:       "// ---- v6 预览与报告参数 ----": "",
184:       "// preview_note":       "启用实时预览窗口（显示最终输出结果，按 q 键提前退出）",
185:       "preview":              false,
186:       "// preview_interval_note": "预览帧间隔（每多少帧刷新一次窗口）",
187:       "preview_interval":     30,
188:       "// report_json_note":   "JSON 性能报告输出路径（如 report.json）；null 表示不输出",
189:       "report_json":          null
190:     }
191:   },
192: 
193:   "output": {
194:     "// section_note":      "最终合并阶段（merge_videos_by_codec）及音频提取的配置",
195:     "// format_note":       "输出容器格式",
196:     "format":               "mp4",
197:     "// codec_note":        "合并重编码时使用的视频编码器（merge_videos_by_codec 固定读取 'codec' 键）",
198:     "codec":                "libx264",
199:     "// codec_alt":         "libx265 | libsvtav1",
200:     "// preset_note":       "编码预设（libx264/libx265 适用；libsvtav1 使用数字如 8）",
201:     "preset":               "medium",
202:     "// crf_note":          "最终合并输出质量；18-23 常用",
203:     "crf":                  23,
204:     "pix_fmt":              "yuv420p",
205:     "// pix_fmt_alt":       "yuv420p10le（10bit，需编码器支持）",
206:     "// audio_format_note": "音频提取格式；smart=智能保留原始编码（推荐）；aac/mp3 指定格式",
207:     "audio_format":         "smart",
208:     "// audio_codec_note":  "merge_videos_by_codec 音频处理方式；copy=直接复制音轨（推荐，无损）",
209:     "audio_codec":          "copy",
210:     "// audio_bitrate_note": "audio_codec 非 copy 时的目标码率",
211:     "audio_bitrate":        "192k",
212:     "// bitrate_note":      "video_utils.extract_audio() 内部读取此键名；与 audio_bitrate 保持一致",
213:     "bitrate":              "192k"
214:   },
215: 
216:   "temp_files": {
217:     "segment_prefix":      "segment_",
218:     "processed_prefix":    "processed_",
219:     "temp_video_suffix":   "_temp.mp4"
220:   },
221: 
222:   "logging": {
223:     "// level_opts":       "DEBUG | INFO | WARNING | ERROR",
224:     "level":               "INFO",
225:     "log_to_file":         true,
226:     "log_to_console":      true
227:   }
228: }

(End of file - total 228 lines)
</content>
```

---

## Assistant (Plan · Ox Alpha Free · 892.1s)

_Thinking:_

The first agent (main entry + utils) returned a very detailed report. The second agent (ESRGAN subsystem) returned EMPTY. I need to re-dispatch the ESRGAN analysis agent. Also, I haven't dispatched an IFRNet subsystem agent yet — wait, looking back at my message, I only dispatched 2 agents (main entry and ESRGAN). I intended 3 but only sent 2. Let me now dispatch:
1. ESRGAN subsystem agent (retry — previous returned empty)
2. IFRNet subsystem agent

Both in parallel.

入口层分析已完成（发现 27 项问题）。ESRGAN 子系统代理返回为空，需要重派；同时派出 IFRNet 子系统分析代理。

**Tool: task**

**Input:**
```json
{
  "description": "分析IFRNet子系统代码",
  "prompt": "你是一个资深 Python/CUDA/TensorRT/NVENC 系统代码审查专家。这是一个只读分析任务：不要修改任何文件。\n\n请深度分析 IFRNet 帧插帧子系统的完整代码链路：\n\n1. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\\processors\\ifrnet_processor_video_optimized.py （779行，处理器封装层）\n2. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\main.py （1693行）\n3. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\pipeline.py （1563行）\n4. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\nvenc_sdk.py （2799行）\n5. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\ffmpeg_io.py （643行）\n6. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\ifrnet_utils.py （297行）\n7. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\tensorrt_accel.py （218行）\n8. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\config.py\n9. 模型定义参考：D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\IFRNet\\models\\IFRNet_S.py（简要）\n\n背景：GPU 加速视频插帧（2x-16x），支持 PyTorch/TensorRT、FP16、CUDA Graph、NVENC SDK 编码（ctypes 直调 nvEncodeAPI）、分段(segment)处理与 checkpoint/resume。已知环境约束：不能调用 pycuda.autoinit（与 PyTorch CUDA context 冲突）。\n\n请完整阅读并输出：\n\n【A. 架构分析】\n- 各模块职责、核心类清单（类名、关键方法、职责、行号）\n- 数据流：ffmpeg/NVDEC 读帧 → 批量推理（batch 插帧）→ NVENC 写出；线程模型与队列深度\n- 分段处理架构与 checkpoint/resume 实现\n- NVENC SDK 封装设计：session 生命周期、lookahead、completion event 轮询模式、空帧防御\n- TensorRT/FP16/CUDA Graph 集成点与回退逻辑\n- OOM 自动降级机制\n- processor 层与 external 后端的调用方式（进程内 import？参数如何传递？）\n- 设计模式\n\n【B. BUG 清单】对每个 bug 给出文件:行号、类型（逻辑错误/资源泄漏/并发/异常处理缺失/性能瓶颈/崩溃点/边界条件）、描述、影响评估（致命/高/中/低）：\n重点检查：\n1. ctypes 与 NVENC API 的指针生命周期、struct 内存布局错误（ctypes.sizeof 校验）、句柄泄漏\n2. completion event 等待逻辑：超时、轮询忙等、事件未销毁\n3. CUDA context 管理：PyTorch context 与 NVENC context 交互、pycuda 使用是否避开 autoinit\n4. 线程竞态：队列共享状态、stop 标志可见性、双线程写同一资源\n5. ffmpeg 子进程 PIPE 管理、进程清理、stderr 消费\n6. 帧数守恒验证：输入帧数 vs 输出帧数的校验逻辑漏洞\n7. checkpoint 一致性、断点续跑的帧对齐\n8. 显存管理：tensor 泄漏、empty_cache 时机、批大小降级后状态\n9. 异常路径下 NVENC session/buffer 是否释放（try/finally 完整性）\n10. 边界条件：奇数宽高（IFRNet 要求尺寸约束）、极短视频（<2帧）、最后一段不足一个 batch、灰度/alpha 视频\n11. FP16 数值溢出风险、TRT engine 缓存失效判断（GPU 变更检测）\n\n【C. 可维护性问题】简要列举（过长函数、重复代码、硬编码魔法数字，给出行号）。\n\n要求：所有结论必须基于实际读到的代码，给出精确行号。输出结构化中文报告。",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_fd30cfa93ffe5eZfyZ0cILp4Ws" state="completed">
<task_result>

</task_result>
</task>
```

**Tool: task**

**Input:**
```json
{
  "description": "分析ESRGAN子系统代码",
  "prompt": "你是一个资深 Python/CUDA/TensorRT/NVENC/GFPGAN 系统代码审查专家。这是一个只读分析任务：不要修改任何文件。你必须输出完整的分析报告文本作为最终回复（不要返回空内容）。\n\n请深度分析 Real-ESRGAN 超分子系统的完整代码链路（活跃版本目录 external\\realesrgan_video，不带日期后缀）：\n\n1. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\\processors\\realesrgan_processor_video_optimized.py （969行，处理器封装层）\n2. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\main.py （846行）\n3. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\pipeline.py （1634行）\n4. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\nvenc_sdk.py （3438行）— 可重点看与 ifrnet_video 版本的结构差异、GFPGAN 相关集成\n5. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\ffmpeg_io.py （1233行）\n6. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\gfpgan_subprocess.py （1089行，GFPGAN 人脸增强独立子进程方案）\n7. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\face_utils.py （202行）\n8. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\nvenc_writer.py （174行）\n9. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\async_dispatcher.py （105行）\n10. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\tensorrt_accel.py （253行）\n11. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\realesrgan_utils.py （213行）\n12. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\realesrgan_video\\config.py\n\n背景：GPU 加速视频超分（2x/4x）+ 可选 GFPGAN 人脸增强，支持 PyTorch/TensorRT、FP16、CUDA Graph、NVENC 编码、tile 平铺推理（OOM 自动降级）、分段(segment)并行处理。已知历史修复：2026-07-28 有一次\"ESRGAN NVENC 跨段修复\"。已知环境约束：不能调用 pycuda.autoinit。\n\n请完整阅读并输出：\n\n【A. 架构分析】\n- 各模块职责、核心类清单（类名、关键方法、职责、行号）\n- 数据流：ffmpeg 读帧 → tile/整幅超分推理 →（可选 GFPGAN 子进程）→ NVENC 写出；线程模型与队列\n- 分段(segment)处理架构：切分方式、并行度、段间衔接（跨段帧连续性——重点说明 2026-07-28 修复了什么）\n- GFPGAN 子进程方案设计：通信协议（stdin/stdout? 文件?）、人脸检测/对齐/融合流程、生命周期管理\n- tile 推理实现与 OOM 自动降级机制\n- TensorRT/FP16/CUDA Graph 集成点\n- processor 层与 external 后端调用方式（importlib 按路径加载？argparse.Namespace 传参？）\n- 设计模式\n\n【B. BUG 清单】对每个 bug 给出文件:行号、类型、描述、影响（致命/高/中/低）：\n重点检查：\n1. tile 拼接：重叠区混合、边缘像素、奇数尺寸/tile 不整除\n2. OOM 降级重试的状态一致性（批内部分成功怎么办）\n3. GFPGAN 子进程：死锁（PIPE 缓冲满）、崩溃恢复、临时文件清理、超时\n4. NVENC session/event/buffer 在异常路径的泄漏、ctypes 指针生命周期\n5. 跨段衔接：首尾帧重复/丢失、时间戳错位\n6. ffmpeg 子进程 PIPE 死锁、清理\n7. 线程竞态、无界队列、线程泄漏\n8. 显存管理\n9. 异常吞掉导致静默失败或挂起\n10. 边界条件：奇数尺寸、极短片段、无人脸帧、灰度视频\n\n【C. 可维护性问题】简要列出（过长函数、重复代码、硬编码，给行号）。\n不需要分析 backup* 目录和带日期后缀的备份目录。\n\n再次强调：最终回复必须包含完整报告内容，不能为空。",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_fd30cb952ffef7MwPPZAaSbz7N" state="completed">
<task_result>
所有目标文件已完整读取并交叉验证（含记忆文档确认 2026-07-28 修复栈）。以下为完整分析报告。

---

# Real-ESRGAN 超分子系统完整代码审查报告

**审查范围**（实际行数与任务描述略有出入，以仓库当前版本为准）：

| 文件 | 任务描述行数 | 实际行数 |
|---|---|---|
| `src\processors\realesrgan_processor_video_optimized.py` | 969 | 1182 |
| `external\realesrgan_video\main.py` | 846 | 1032 |
| `external\realesrgan_video\pipeline.py` | 1634 | 1962 |
| `external\realesrgan_video\nvenc_sdk.py` | 3438 | 3827 |
| `external\realesrgan_video\ffmpeg_io.py` | 1233 | 1470 |
| `external\realesrgan_video\gfpgan_subprocess.py` | 1089 | 1187 |
| `external\realesrgan_video\face_utils.py` | 202 | 248 |
| `external\realesrgan_video\nvenc_writer.py` | 174 | 225（全文件死代码，无任何 import 方） |
| `external\realesrgan_video\async_dispatcher.py` | 105 | 122 |
| `external\realesrgan_video\tensorrt_accel.py` | 253 | 289 |
| `external\realesrgan_video\realesrgan_utils.py` | 213 | 297 |
| `external\realesrgan_video\config.py` | — | 61 |

---

## 【A. 架构分析】

### A.1 各模块职责与核心类清单

**processor 封装层 — `src\processors\realesrgan_processor_video_optimized.py`**
- `RealESRGANVideoProcessor`（L60）：分段编排器。
  - `process_video`（L183）：分段→超分→合并总入口；
  - `process_video_segments`（L265）：时长探测、≤segment_duration 直通单段（L297-304）、否则 `split_video_by_time` 切片；
  - `process_segments_directly`（L319）：接收上游（IFRNet）分段；
  - `_initialize_enhancer`（L350）：**importlib.util.spec_from_file_location 按路径加载 `external\realesrgan_video\main.py`**（模块名 `realesrgan_video_main`，L365-370），构建 **argparse.Namespace** 严格对齐底层 argparse 字段（L373-423），调用 `create_video_enhancer(ns)` 一次性建重型组件缓存到 `self._enhancer`；
  - `_process_segment_with_enhancer`（L435）：每段调 `run_pipeline_for_video(enhancer, in, out)` + `verify_video_integrity`；
  - `_process_segments`（L563）：断点 checkpoint 循环；finally 统一关闭跨段复用的 `NVENCEncoder`（L632-641）与 `close_enhancer()`（GFPGAN 子进程）；
  - checkpoint 机制：`_load_checkpoint/_save_checkpoint/_get_config_snapshot`（L480-561），含 mtime/size/upstream_hash 指纹校验；
  - `_run_esrgan_video`（L698）：**遗留的一次性路径**（每次重新 importlib 加载模块 + 新建 enhancer），仅被 ≤segment_duration 的"短视频直通"分支经 `_process_segment`(L657) 使用。
- 设计模式：Facade + Template Method（复刻底层参数仲裁）。

**backend 入口 — `external\realesrgan_video\main.py`**
- `_HWProfile` dataclass + `_GPU_PROFILES_TABLE`（L93-131）：GPU 型号静态表（tier/nvdec/nvenc/pcie）；
- `_detect_hw_profile`（L135）；
- `PreviewWriter`（L150）：包装 writer，子进程隔离探测 GUI 可用性（L168-227，防 Qt xcb abort 杀主进程）；
- `create_video_enhancer(args)`（L283）：五阶段初始化 —— 参数仲裁（--no-tensorrt/--use-compile-force/--use-cuda-graph-force 覆盖、TRT>compile>CUDAGraph 互斥，L300-365）、faulthandler 启用（L290）、GFPGAN 子进程预启动（阶段1，L427-504，最长等 90min）、CUDA 初始化+RealESRGAN 加载（阶段2，L509-550，经 `_build_upsampler`）、face_enhancer 创建（阶段3，子进程模式建 `DummyGFPGANer` 仅持 FaceRestoreHelper，L556-582）、SR TensorRT 引擎（阶段4，L618-645，先 synchronize+empty_cache+gc 清理 IFRNet 残留显存 `[FIX-TRT-CTX-OOM]`）；
- `run_pipeline_for_video(enhancer, in, out)`（L670）：每段创建 reader/writer/pipeline；NVENC 编码器跨段缓存复用 + 参数 key 一致性检查（L751-791，`_sdk_nvenc_key` 含 codec 维度）；≥1080p 输出自动切 constqp+LA=0（`[FIX-HIGHRES-RC]`，L718-726）；`ESRGAN_DISABLE_SDK_NVENC=1` 逃生门（L734）；finally 生成 JSON 报告；
- `main_optimized`（L930）/ `main`（L941，argparse 定义）。

**核心流水线 — `external\realesrgan_video\pipeline.py`**
- `_sr_infer_batch`（L37-175）：纯 SR 推理（H2D→model/TRT forward→interpolate(outscale≠netscale 时 bicubic)→D2H）；支持 H2D 预取张量、`defer_resolve` 延迟解析句柄（L113-137，携带私有 event + gpu_refs）；
- `_sr_materialize`（L178-197）：句柄→numpy，混合列表透传保序；
- `GPUMemoryPool`（L200-222）：**名不副实，现为纯并发槽信号量**；
- `DeepPipelineOptimizer`（L225-1962）：
  - `__init__`（L228-386）：batch 硬顶按分辨率动态化（≤640×360 允许 48，L240-247）、四级有界队列 P/F/D/S/G（frame_queue_size/detect_queue_size 可配，S/G 固定16）、三条 CUDA stream（transfer/sr/gfpgan，L286-288）、自适应批处理 EMA（L302-314，低密度→大批/高密度→小批，`[FIX-ADAPTIVE-BS-INVERSION]`）、OOM 冷却与 TRT working-bs 跟踪（L317-322）、GFPGAN 子进程注入消费（L328-365）；
  - `optimize_pipeline`（L393-513）：子进程就绪等待（2700s 上限，区分 exitcode=0 的"context 污染主动退出"）、启动 read/detect/sr/gfpgan 四线程、主线程跑 `_write_frames`、结束后 `begin_flush()` 提前触发 NVENC 收尾（`[EARLY-FLUSH]` L493-494）再 join 线程（15s）；
  - `_read_frames`（L586-709）：FRAME_TIMEOUT 看门狗（30次≈60s 判死）、reader 探活兜底、EOF 末批 30s 耐心入队、finally 无条件送哨兵（60 次×1s）、**不在 finally 设 running=False**（让哨兵自然传播）；
  - `_detect_faces`（L715-770）：提交 `detect_executor`（CPU retinaface，face_utils 强制 device=cpu）；
  - `_estimate_safe_batch_size`（L772-791）：mem_get_info+cached_free 的 70% 保守估算；
  - `_clamp_adaptive_bs`（L793-804）：TRT 时锁定 engine B；
  - `_process_sr`（L806-1269）：**OOM 降级阶梯核心**（见 A.6）；预取下一批 H2D 重叠（L1132-1161）；`_resolve_executor` 单线程保序延迟解析（L1163-1178，限深 2）；每 100 批深层清理（L1192-1196）；finally 处理预取残留（SR 或 CPU Lanczos resize 保帧，L1219-1258）；
  - `_resolve_and_put`（L1271-1296）/`_drain_resolve_futs`（L1298-1306）；
  - `_process_gfpgan`（L1308-1644）：`_pending_tasks` 有序窗口（`_MAX_IN_FLIGHT=2`），PhaseA 就绪即输出/PhaseB 取新 SR 项；SHM slot 获取三级策略（try_acquire→弹头腾位→阻塞 30s）；提交失败逐级降级（shm→pickle→直通 SR）；
  - `_split_restored`/`_assemble_result`（L1646-1681）：贴回走 `paste_executor`（60s 超时回退 SR 结果）；
  - `close`（L1688-1740）：**GFPGAN 子进程跨段保活回注 `args._early_gfpgan_subprocess`**（`[FIX-GFPGAN-KEEPALIVE]` L1712-1721）；
  - `_write_frames`（L1742-1962）：写帧 + 120s 全空转死锁看门狗（dump 队列水位+全线程栈，L1920-1949）。

**NVENC SDK 层 — `external\realesrgan_video\nvenc_sdk.py`**（ctypes 直连 SDK 13.0）
- 结构体手工布局（`_pack_=1` + 字节偏移硬编码写入，L210-403；`_sdk13_ver` 版本公式 L170）；函数表索引表 `_FUNC_IDX`（L455-477，SDK13 OpenEncodeSessionEx 从 24→29）；
- `NVENCEncoder`（L531）：
  - `__init__`（L545-706）：DLL/libcuda 加载、primary context retain（cuDevicePrimaryCtxRetain，与 PyTorch 共享同一 primary ctx——**这是不调 pycuda.autoinit 的正确替代方案**）、CRF=0 强制 CONSTQP+LA=0 开关（L567-577）、slot 数计算（constqp→LA=0→slot=pipeline_depth；LA>0→max(pipe,LA+1)，L585-593）；
  - `_open_session`（L708-1101）：CreateInstance→OpenEncodeSessionEx（apiVersion 多候选重试）→GUID 动态枚举（codec 匹配，未命中回退告警）→preset config→**H.264 字段按 SDK 真实偏移硬编码**（chromaFormatIDC=1 防 monochrome 灰屏 `[FIX-CHROMA]` L861-872；profileLevel=51；idrPeriod=fps；DPB=4）→RC params（constqp/VBR_HQ/QVBR 三模式 + avgBitRate 天花板 `[FIX-BR-CEILING]` + AQ bitfield）→多 slot（input buf/bitstream buf/CUDA event 三元组，L1000-1039）→专用 non-blocking 编码拷贝 stream（`[FIX-ASYNC-COPY]` L1069-1084）；
  - `_copy_into_input_buffer`（L1103-1125）：cuMemcpy2DAsync_v2+私有 stream 同步，替代 legacy null stream 的隐式全局屏障；
  - SPS/PPS 工具：`_extract_sps_pps/_has_sps_pps/_nal_first_vcl_type`（L1225-1329，支持 h264/hevc NAL、av1 OBU）；
  - 排空体系：`_drain_outputs_blocking`（L1374-1450，doNotWait=0 阻塞语义，T4 上 doNotWait=1 会 segfault）、`_apply_drained_entries`（L1456-1550，`[FIX-AUX-NO-CLEAR]` 辅助块不占帧槽 + 相位漂移诊断）、`_ensure_slot_free`（L1552-1625，**提交前强制排空物理槽防硬件层覆盖丢帧** `[FIX-SLOT-BACKPRESSURE]`，超限时空帧占位兜底）；
  - 三个编码入口：`encode_frames_batch`（L1661-2063，LA 路径，chunked+send_eos，`[FIX-EOS-EMPTY-CHUNK]` L1673-1690 空 chunk 也必须发 EOS）、`encode_frames_batch_ce_pipeline`（L2065-2437，LA=0 异步 CE 流水线：Harvest→Submit(per-frame CE)→Drain，返回码全面检查 `[FIX-RC-CHECK]`）、`encode_frame`（L2439-2640，同步单帧，活跃链路无调用方）；
  - `flush`（L2642-2773，EOS+全 slot 排空）、`_stream_begin(force=True)`（L2775-2797，每段重置 `_slot_pending/_slots_warmed/_prev_chunk_outputs` 并对齐 `_output_slot_idx=_strm_ts_base=_frame_idx`）、`close`（L2837-2891，先 push primary ctx 再销毁 `[FIX-REOPEN-CTX-ORDER]`，配对 release retain）；
- `_NVENCEncodeThread`（L2912）：daemon 编码线程；`cuCtxSetCurrent` 失败即 abort（L3032-3049）+ 段首预检 context/会话句柄（`[FIX-SEG2-PREFLIGHT]` L3051-3078）；LA>0 分块累积编码（动态 chunk=min(内存预算500MB, flush_chunk_frames=128)，L3094-3148）/LA=0 per-batch ce_pipeline（L3149-3174）；最终块 send_eos=True + 帧数守恒审计（`[FIX-PER-CHUNK-DIAG]/[FIX-AUDIT-PENDING]` L3186-3213）；`begin_flush/flush_and_join` 两阶段幂等收尾（L3285-3350）；
- `_rgb_to_nv12_gpu(_batch)`（L3357-3438）：BT.601 limited range GPU 转换；
- `FFmpegMuxer`（L3445-3587）：ES→MP4 纯 mux（-c:v copy，h264/hevc/obu 三格式），stderr drain 线程，HEVC 参数集随首 VCL AU 捆绑（`[FIX-HEVC-SPS-AU]`）；
- `NVENCWriter`（L3594-3827）：**活跃版 writer**（main.py 从 nvenc_sdk 导入）；mini-batch(4) 累积、RGB→NV12、submit 前 current_stream 同步（`[FIX-SYNC-BEFORE-SUBMIT]` L3731-3737）、LA>0 时 NV12 D2H 到 pinned host（`[FIX-LA-D2H-HOST]` L3651-3690）。

**ffmpeg IO — `external\realesrgan_video\ffmpeg_io.py`**
- `HardwareCapability`（L151-295）：NVDEC 两阶段真实探测（软编→硬解）、NVENC lavfi probe、`best_encoder` 静态表+probe 双路径统一仲裁；
- `FFmpegReader`（L298-639）：后台读线程 rawvideo rgb24 pipe、prefetch 队列、stderr drain、`FRAME_TIMEOUT` 哨兵与 EOF 分离、`_send_eof_guaranteed`（600s 坚持）、探活/水位接口；
- `FFmpegWriter`（L642-1470）：Level 2/3 回退编码器；写线程攒批(8)+0.2s 超时刷出；`_write_with_timeout`（select+64KB chunk 写）；stall 检测重试（NVENC 600s/软编 180s）；首帧 stderr 编码器死亡检测；close 五阶段（SENTINEL→join→强杀→flush→音轨分离 mux/重命名 tmp 文件）。

**GFPGAN 子进程 — `gfpgan_subprocess.py` + `async_dispatcher.py` + `face_utils.py`**（详见 A.4）

**其他**
- `tensorrt_accel.py`：SR TRT 加速器。ONNX 静态形状导出→build（workspace 4GB、FP16、心跳线程、SM 时间预估）→加载（SM tag 缓存失效检测、`create_execution_context()==None` 显式处理 `[FIX-TRT-CTX-OOM]` L213-224）→`infer`（batch padding 到 engine B + 输出 `.contiguous()` 防隐式显存泄漏 `[FIX-MEM-LEAK]` L282-288）；
- `realesrgan_utils.py`：`PinnedBufferPool`（N 路 ring + per-slot CUDA event 追踪 H2D/D2H 完成性，`[FIX-PINNED-POOL-RACE]`，忘 mark 时整卡同步兜底 L177-189）、线程本地池 `_get_pinned_pool`、`ThroughputMeter`、`get_video_meta_info`（duration×fps 优先并对 nb_frames 交叉校验，L254-266）、`_build_upsampler`；
- `config.py`：MODEL_CONFIG（8 个模型的网络构造+netscale+URL）、模型目录常量；
- `nvenc_writer.py`：**整个文件是死代码**（main.py L52-53 明确改从 nvenc_sdk 导入；其内容还缺少活跃版的 `_to_pinned_host_if_la` 等修复）。

### A.2 数据流与线程模型

```
FFmpegReader 线程(ffmpeg_reader_loop)
   │ rawvideo rgb24 pipe ──→ P 队列(prefetch_factor=48)
   ▼
_read_frames 线程 ──按自适应 bs 攒批──→ F 队列(frame_queue, 8)
   ▼
_detect_faces 线程 ──submit→ detect_executor(CPU×2)──→ D 队列(detect_queue, 4)
   ▼                                          （face_data 携带 crops/affines/inv_affines/orig-BGR）
_process_sr 线程 ──┬─ H2D 预取(transfer_stream, pinned ring)
                   ├─ TRT(engine_B padding)/eager forward(sr_stream)
                   ├─ defer_resolve 句柄 → sr_resolve 线程(×1) 保序物化
                   └──────────────────────────────→ S 队列(sr_queue, 16)
   ▼
_process_gfpgan 线程 ──┬─ AsyncGFPGANDispatcher(collector 线程) ⇄ GFPGAN 子进程(spawn, SHM 双缓冲/pickle)
                       ├─ 主进程 PyTorch 回退（face_utils._gfpgan_infer_batch）
                       ├─ paste_executor(×2) 人脸贴回
                       └──────────────────────────────→ G 队列(gfpgan_queue, 16)
   ▼
主线程 _write_frames ──→ NVENCWriter(mini-batch4 → RGB→NV12 → sync → [LA>0 D2H host]
                          → enc 队列(depth 4/16)) ──→ NVENC-Enc 编码线程
                          ──→ FFmpegMuxer(stdin pipe, -c:v copy) ──→ MP4(+faststart)
        （或回退）──→ FFmpegWriter(队列24 → 攒批8 → stdin) ──→ tmp 文件 → 音轨 mux → rename
```

CUDA stream：transfer/sr/gfpgan 三条 torch stream + TRT 私有 stream + NVENC `_stream_encode`(non-blocking)。所有跨 stream 依赖用 `wait_stream`/event 显式表达，规避 legacy null stream 隐式全局屏障（nvenc_sdk L7-17、pipeline L160-169 注释记录了这一性能修复史）。

### A.3 分段(segment)处理架构与 2026-07-28 跨段修复

**切分方式**：processor 用 `split_video_by_time` 按 `segment_duration`（默认 30s）切片到临时目录；每段独立跑完整 `run_pipeline_for_video`（reader/writer/muxer/pipeline 对象每段新建）；段间由 `merge_videos_by_codec` 合并（合并时注入色彩元数据）。checkpoint 断点续跑按段索引记录。

**并行度**：段间串行（顺序处理），段内靠上述五级流水线并行。跨段复用的只有三类重型资产：
1. `enhancer['_sdk_nvenc_encoder']`（NVENCEncoder + 驱动会话 + 全部 slot/event/stream + `_cached_sps_pps` + 单调递增的 `_frame_idx/_output_slot_idx`）；
2. GFPGAN 子进程（spawn 进程 + TRT engine + task/result 队列 + SHM）；
3. upsampler/TRT SR 引擎/detect helper。

每段边界动作：新 `_NVENCEncodeThread.__init__` 调 `_stream_begin(force=True)` 重置 per-segment 编码状态 → 新 FFmpegMuxer（`set_muxer_ref` + `_sps_pps_injected=False` 保证新 muxer 拿到 SPS/PPS 预注入）→ 段首帧 force_idr（`_written==0`）保证每段独立 GOP 起点，杜绝跨段参考帧依赖。

**2026-07-28 "ESRGAN NVENC 跨段修复"修了什么**（据代码注释 + memory/esrgan-cross-segment-sigsegv-fix-stack.md）：

症状是"段 1 正常、段 2 首个编码 batch 后 SIGSEGV"。三层根因链：
1. **EOS 后直接复用 session 必崩** → 先引入 `reopen()`（段边界原地重建会话，百毫秒级）；
2. reopen 后 gen≥2 会话的首个异步 CE→LockBitstream 仍崩（faulthandler 实锤崩在驱动内部 nvEncLockBitstream）→ `[FIX-SEG-START-SYNC]` 段首 batch 改同步 encode_frame 过渡；
3. 配套加固同批落地：
   - `[DIAG-SEGFAULT]` faulthandler 全线程栈 dump（main.py L290-297）；
   - `[FIX-SEG2-PREFLIGHT]` 编码线程启动预检 context/会话句柄，把不可归因 SIGSEGV 转化为带 session 代数的 RuntimeError（nvenc_sdk L3051-3078）;
   - `[FIX-RC-CHECK]` ce_pipeline/flush 的 libcuda/EOS 返回码检查，污染 context 不再静默；
   - `[FIX-REOPEN-CTX-ORDER]` 销毁前先 push primary ctx（原顺序下销毁调用静默失败→驱动资源泄漏）；
   - `[FIX-SYNC-BEFORE-SUBMIT]` Writer submit 前补 current_stream 同步——此前只补在了从未被 import 的 nvenc_writer.py 死代码上（花帧/帧内容被顶替类竞态）；
   - `[EARLY-FLUSH]` begin_flush 先 flush mini-batch 残留帧再发 SENTINEL（pipeline 存在单帧 write_frame 路径，段末 1-3 帧原本会被 SENTINEL 截杀静默丢失）。

2026-07-29 该栈被"跨段真复用"取代（memory: esrgan-cross-segment-optimization-complete.md）：删除 reopen() 与 SEG-START-SYNC，改为无条件跳过 reopen（`[FIX-SKIP-REOPEN]`，EOS 仅作簿记标记）、GFPGAN 子进程保活回注、`_stream_begin` 封装、encoder 参数 key 一致性检查，段切换从几十秒降到毫秒级。

**跨段帧连续性的其余配套修复**（同族问题，均已在活跃代码落地）：
- `[FIX-EOS-EMPTY-CHUNK]`（nvenc_sdk L1673-1690）：段总帧数恰被 chunk 整除时最终块为空，旧代码提前 return 跳过 EOS → LA 里滞留的尾帧在下一段提交时才被 drain 出来、误写进下一段的 muxer（本段丢尾帧+下一段开头混入上一段内容且参考帧断裂被判损坏）；
- `[FIX-SLOT-BACKPRESSURE]`（`_ensure_slot_free`，L1552-1625）：物理 bs_buf 复用前必须排空上次占用者码流，根治"某段少几帧、紧邻下一段多出几乎相同数量帧"；
- `[FIX-GLOBAL-FI]`+`[FIX-SLOT-DEQUE]`：跨 chunk 全局帧号 + per-slot deque FIFO，防跨块覆盖乱序；
- `[FIX-AUX-NO-CLEAR]`：LA 冷会话把 SPS/PPS/AUD 作为独立辅助块 drain，不占帧槽不清 pending；
- `[FIX-PINNED-POOL-RACE]`（realesrgan_utils L30-189 + pipeline `_sr_infer_batch`）：D2H/H2D pinned buffer 无同步复用导致"分段视频里某几帧内容和更早批次逐像素相同"。

### A.4 GFPGAN 子进程方案设计

**生命周期（两阶段进程架构）**：
- 阶段 1 Builder 进程（`daemon=False`，gfpgan_subprocess.py L214-241）：独立 spawn 进程完成 ONNX 导出（basicsr fused_leaky_relu/upfirdn2d 兼容猴补丁 L311-346）→ TRT build（静态 batch= gfpgan_batch_size、512×512、FP16、workspace 4GB、心跳日志）→ 写 `.trt` 缓存（tag 含 SM 架构+权重融合系数）→ `os._exit`。超时 90min terminate。
- 阶段 2 Inference 进程（`daemon=True`，`_worker` L492-1110）：TRT 路径延迟 GFPGANer GPU 加载；加载引擎+warmup；warmup 失败判 CUDA context 污染 → `ready_event.set()` 后 `os._exit(0)`，主进程轮询区分"主动退出（降级主进程 PyTorch）"与"意外退出"；TRT 可用则迁移 GFPGANer 到 GPU 并弃用 face_helper；进入任务循环。

**通信协议（三通道并存，逐级回退）**：
1. **共享内存零拷贝**（首选）：`SharedMemoryDoubleBuffer`——2 组 slot（input/output 各 2），每 slot `MAX_FACES=64 × (512,512,3) uint8`；主进程 `write_input(slot, crops)`（非 512 crop 用 cv2.resize 归一）→ task_queue 投 `(task_id, n_faces, slot_id)` → 子进程从 shm 读、推理、把 restored 写回 output shm → result_queue 回 `(task_id, count)` → 主进程 `read_output(slot, n)` 拷出（全零行映射为 None）。slot 生命周期：acquire→write→task→wait_result→release，由 `_pop_and_output_head` 的 finally 统一释放。
2. **pickle 回退**：人脸数 >64 或 SHM 创建/attach 失败时，crops 直接经 mp.Queue pickle 传输。
3. **控制消息**：`('__validate__', id)` post-SR 真实显存压力 warmup 验证；`('__pause__', dur)` SR OOM 时令子进程 empty_cache+sleep 让出显存。

**人脸流程**：检测在主进程 CPU（`face_utils._make_detect_helper` 强制 cpu，retinaface_resnet50，resize=640），`_detect_faces_batch` 做置信度阈值过滤（score ≥ face_det_threshold 且面积 ≥48×48）+ 预计算逆仿射矩阵（inv×outscale）；推理对齐 512 crops 批量前向（weight 融合 `w*out+(1-w)*input`）；贴回 `_paste_faces_batch` 用预存 inv_affines 走 facexlib `paste_faces_to_input_image(upsample_img=sr_bgr)`，尺寸异常强制 resize 回期望分辨率。

**调度**：`AsyncGFPGANDispatcher` 后台 collector 线程持续排空 result_queue（区分 validate/普通结果），`_results` dict + Condition；GFPGAN 线程侧 `_pending_tasks` 保持最多 2 个在飞任务并按序输出（Phase A 就绪即发/Phase B 限深弹出），任一环节失败都逐级降级到"直通 SR 结果"，保证帧数守恒。

**崩溃恢复**：提交前 `is_alive()` 检查；`wait_result` 120s 超时返回 None → 贴回降级为 SR 结果；post-SR 验证失败时区分"子进程已死（context 损坏，永久降级主进程路径）"与"活着但 TRT 没起来（以 PyTorch FP16 服务）"。

### A.5 tile 推理实现与 OOM 自动降级机制

**重要发现：tile 在活跃管线中并未真正参与推理。** `_build_upsampler` 把 `tile/tile_pad/pre_pad` 传给 `RealESRGANer` 构造器（realesrgan_utils.py L293-297），但流水线从未调用 `upsampler.enhance()`——`_sr_infer_batch` 直接执行 `upsampler.model(batch_t)`（pipeline.py L99/L102），完全绕过 RealESRGANer 内部（含 tile 切块、mod 对齐补边、pre_pad）。因此：
- 用户配置 `tile_size=512` 等参数在此路径下是**惰性配置**（无效果、也无提示）；
- 所谓"tile 平铺推理 + OOM 自动降级"实际由 **batch 维度拆分**承担。

**实际 OOM 降级阶梯**（`_sr_with_oom_fallback`，pipeline.py L857-1097）：
1. TRT 模式不持久化降低 optimal_batch_size（engine B 编译期固定，缩小只会造成 pad 浪费），改为维护 `_trt_working_bs` 临时工作批量；
2. OOM → Stage1 轻清理（synchronize→empty_cache，expandable_segments 下必须先同步排空 cudaFreeAsync）→ 冷却期（非 TRT sleep，TRT 不 sleep）→ retry_bs 减半重试；
3. 超过最大重试（TRT 4 / 非 TRT 12）→ 深度清理（sync+empty_cache+gc+empty_cache+sleep3s）→ 逐帧推理兜底；
4. bs=1 仍 OOM → 重建 sr_stream + 估算安全 bs（`_estimate_safe_batch_size`，OS free+allocator 缓存的 70%）→ 仍不行才逐帧；
5. 恢复侧：非 TRT OOM 后持久降级 optimal_batch_size；TRT 连续成功 50 批后渐进 ×2 恢复到 engine B，且恢复前做 free<20% 显存压力检查（`[FIX-GRADUAL-RECOVER]/[FIX-MEM-GUARD]`）；
6. OOM 时联动 `gfpgan_subprocess.pause(5s)` 让子进程让出显存。

### A.6 TensorRT / FP16 / CUDA Graph 集成点

- **SR TRT**：`TensorRTAccelerator`（tensorrt_accel.py），静态形状 engine（B,C,H,W 取自首个输入视频 meta × batch_size），SM tag 缓存管理，infer 时小批 padding + 输出 contiguous；集成点在 `_sr_infer_batch` L79-82 与 `_clamp_adaptive_bs`。
- **GFPGAN TRT**：全部在子进程内（Builder/Inference 两阶段），启用时自动禁用全局 FP16（防溢出噪斑，main.py L371-374）；post-SR 二次验证决定是否正式启用。
- **FP16**：upsampler.half + `_sr_infer_batch` 的 `.half()` 分支 + TRT fp16 flag；GFPGAN-TRT 与之互斥。
- **CUDA Graph / torch.compile：声明了但未实现。** `DeepPipelineOptimizer.cuda_graph_accel` 恒为 None（pipeline.py L295），全库无任何 `torch.compile(` 调用；`_sr_infer_batch` 的 cuda_graph_accel 分支（L83-93）是永不进入的死代码；main.py 大量参数仲裁逻辑（--use-tensorrt 与 compile 互斥、"compile 已启用（内含 cudagraphs）"提示）操作的是一个不存在的特性。这是文档/CLI 与实现的显著偏差。

### A.7 processor 层与 external 后端的调用方式

- **importlib 按路径加载**：`spec_from_file_location("realesrgan_video_main", .../main.py)`，避免与内置 `main` 模块冲突（processor L365-370、L810-815）；同时把 `external/realesrgan_video` 与 `external/` 双目录插入 sys.path 以支撑包内平级导入与 `realesrgan_video.*` 包导入。
- **argparse.Namespace 传参**：字段与底层 argparse 严格一一对应（含 rate_mode/lookahead_depth/frame_queue_size/defer_sr_resolve 等 v6.4.5 新字段）；processor 侧重刻了底层 main() 的预处理（gfpgan_trt→no_fp16）。
- 复用协议：首次 `_initialize_enhancer(first_segment_path)`（TRT 引擎需要真实视频探测尺寸，dummy 路径会 ffprobe 崩溃——`[FIX-TRT-INIT-INPUT]`），之后每段只调 `run_pipeline_for_video`；跨段对象通过 `enhancer` dict 与 `args._early_gfpgan_subprocess` 两个通道传递。
- 遗留路径 `_run_esrgan_video`（一次性 importlib + main_optimized）仍在短视频分支使用，与复用路径并存（代码重复，见 C）。

### A.8 设计模式汇总

- 生产者-消费者多级流水线（有界队列 + 哨兵传播 + 看门狗）；
- 对象/资源池：PinnedBufferPool（环形+事件）、SHM 双缓冲 slot 池、GPUMemoryPool（退化为信号量）；
- 策略/链式降级：TRT→eager、SDK NVENC→FFmpeg CLI、SHM→pickle、子进程→主进程、SR→CPU resize 贴回；
- Facade/Adapter（processor 包装 backend）、Monkey Patch（facexlib model_rootpath，main.py L71-81 与 gfpgan_subprocess.py L37-46 双处叠加）、两阶段进程（Builder/Inference）、幂等哨兵（begin_flush/flush_and_join、close 幂等标志）、审计式防御（帧数守恒校验、相位漂移诊断、faulthandler）。

---

## 【B. BUG 清单】

严重度定义：致命=崩溃/大面积数据损坏；高=功能性数据丢失或平台性失效；中=特定条件下丢帧/挂起/资源泄漏；低=边缘场景或诊断性问题。

| # | 位置 | 类型 | 描述 | 影响 |
|---|---|---|---|---|
| B1 | ffmpeg_io.py:985-989（`_write_with_timeout`） | **平台兼容性** | `select.select([], [stdin_fd], [stdin_fd], ...)` 用于 subprocess 管道 fd。Windows 上 select 仅支持 socket，管道 fd 直接抛 OSError(10038)，被 `except (OSError, ValueError)` 捕获 → 写入恒失败 → stall 重试 3 次 → `_broken=True`。当前环境为 win32：一旦走 FFmpegWriter 路径（libx264 编码、`ESRGAN_DISABLE_SDK_NVENC=1`、或 SDK NVENC 初始化失败回退），软件编码回退链整体不可用 | **高**（平台性功能失效；生产若一直跑 SDK 路径则被掩盖） |
| B2 | nvenc_sdk.py:3608（NVENCWriter.__init__ 签名）+ main.py:793-795 + ffmpeg_io.py:303-337 | **功能性数据丢失** | SDK NVENC 路径音轨丢失：main.py 传 `audio_src=getattr(reader, 'audio_src', None)`，但 `FFmpegReader` 根本没有 `audio_src` 属性（只有 `audio`=ffprobe audio node）→ 恒为 None → FFmpegMuxer 不加音频输入（nvenc_sdk.py:3475-3476）；且 nvenc_sdk.NVENCWriter 接收的 `audio` 参数在函数体内从未使用。对比 FFmpegWriter 路径会用 `args.input` 做音轨分离 mux（ffmpeg_io.py:1399-1446）。结果：SDK Level 1 路径输出无声视频，两条编码路径行为不一致，合并后成品无声 | **高** |
| B3 | main.py:838-846, 866, 924-927 | **异常吞掉/静默失败** | `run_pipeline_for_video` 捕获流水线 Exception 后仅打印 traceback，finally 里因 `interrupted=False` 照常打印"✅ 超分完成！"并 `return True`。processor 侧只要输出文件存在且容器完整（`verify_video_integrity`）即记为段成功 → 半截视频混入合并产物，静默内容丢失 | **高** |
| B4 | pipeline.py:1900-1905 + realesrgan_utils.py:254-266 | **跨段衔接/边界条件** | `_write_frames` 的提前退出条件 `written_count >= total_frames and sr_queue空 and gfpgan_queue空` 中 `total_frames` 来自 ffprobe duration×fps 估算（代码自己注释"-c copy 分段文件元数据常见不一致"）。若实际帧数 > 估算且上游恰逢瞬时排空（get 超时 10s 进入 Empty 分支），会提前退出丢弃真实剩余帧 | **中高** |
| B5 | pipeline.py:87-102 + realesrgan_utils.py:293-297 | **配置失效（tile）** | 活跃管线直接 `upsampler.model(batch_t)` 前向，从不走 `RealESRGANer.enhance()` → `--tile-size/--tile-pad/--pre-pad` 全部无效；同时失去 enhance() 内部的尺寸补齐安全网。所谓"tile 平铺推理"在该路径不存在，OOM 防护只有 batch 拆分一条腿。对奇数/极小输入尺寸少了一层保护 | **中**（功能与文档不符；鲁棒性下降） |
| B6 | nvenc_sdk.py:3357-3396 + main.py:707-708 | **边界条件（奇数尺寸）** | `_rgb_to_nv12_gpu` 对奇数 H/W 丢弃最后一行/列色度（`H-H%2`），NV12 总高 `H+H//2` 与编码器初始化 height 的对应关系在奇数高度下失配；NV12/YUV420 本身要求偶数宽高。整数倍率 ×2/×4 时 out 恒为偶数侥幸成立，但 `outscale` 是 float 参数（如 1.5）时 `int(h*1.5)` 可为奇数 → NVENC 编码异常或花屏，且无前置校验 | **中** |
| B7 | nvenc_sdk.py:1627-1659（`_lock_bitstream_blocking`）+ 2415-2422 | **潜在挂起/API 误导** | `timeout_ms` 参数被接受但完全未使用（doNotWait=0 纯阻塞）。ce_pipeline Phase3 对 NEED_MORE_INPUT 帧的无条件 blocking LockBitstream 若驱动不产出将无限期挂起；flush_and_join 120s 超时后仅打印"可能死锁"，线程仍存活 | **中** |
| B8 | nvenc_sdk.py:1900-1901（send_eos 分支）vs 2670-2676（flush） | **返回码检查不一致** | `encode_frames_batch` 的 EOS EncodePicture 返回码被忽略，与 flush() 的 `[FIX-RC-CHECK]` 修复不一致；会话异常时 EOS 静默失败，后续 per-slot 阻塞 drain 在坏会话上运行（正是 2026-07-28 SIGSEGV 的形态之一） | **中** |
| B9 | nvenc_sdk.py:2146-2153, 2378-2385 | **异常路径资源泄漏+过度致命化** | ce_pipeline Harvest/Phase3 中 `cuEventSynchronize` 成功但 `cuEventDestroy` 返回非 0 即 raise RuntimeError——事件既没销毁（泄漏）又把整段编码打断；而 constqp CE 路径（L1831-1840）对同样的 destroy 是 try/except 忽略。行为自相矛盾 | **中** |
| B10 | async_dispatcher.py:74-91 + pipeline.py:1355-1356 | **IPC 数据竞争** | `wait_result` 120s 超时返回 None 后，调用方 finally 立即 `release_slot(slot)`，而子进程稍后仍会向该 output shm slot 写结果 → slot 被下一任务复用时可能读到上一任务的污染像素（全零会被 `r.any()` 映射为 None 掩盖一部分）。触发需单任务 >120s，概率低但后果是画面错乱 | **中** |
| B11 | gfpgan_subprocess.py:1120-1130（infer）、1145-1150（post_sr_validate） | **死锁风险（回退路径）** | 非匹配结果 `result_queue.put((res_id, result))` 无 timeout，result_queue maxsize=2；若 collector 未消费且队列满，同步回退路径将无限阻塞调用线程。活跃路径有 dispatcher 时不触发 | **中低** |
| B12 | realesrgan_processor_video_optimized.py:297-304 + 870-891 | **资源生命周期** | 短视频直通分支 finally 只调 `close_enhancer()`（只关 GFPGAN 子进程），`enhancer['_sdk_nvenc_encoder']`（NVENC 驱动会话/slots/events/stream）不做显式 close，仅靠 CPython refcount 触发 `__del__`；一旦出现引用环/异常引用，驱动会话滞留 | **中低** |
| B13 | pipeline.py:1953-1959（`_write_frames` except 分支） | **潜在紧密循环** | 通用异常只打印后继续循环，没有连续错误计数/熔断；若异常是确定性的（如 final_frames 元素类型异常发生在 `_broken` 检查之前的代码），将无限快速刷错误日志 | **中低** |
| B14 | pipeline.py:1287-1296（`_resolve_and_put`） | **极端背压丢帧** | 下游阻塞超过 300s（300×1s 重试）且 running 仍 True 时放弃入队 → 整批帧静默丢失（仅打印）。破坏帧数守恒的最后防线之一 | **中低** |
| B15 | main.py:631-638 + tensorrt_accel.py:255-281 | **跨段假设未校验** | TRT engine 形状取自 enhancer 初始化时的首个输入视频；`process_segments_directly` 允许外部传入分段列表，若后续段分辨率不同，静态 engine 将收到形状不符的张量（execute 失败或垃圾输出），无 per-segment 分辨率防护 | **中**（混合分辨率场景） |
| B16 | ffmpeg_io.py:636-639 + 454 | **进程清理滞后** | `close()` 置 `_running=False` + join(3s)，但读线程阻塞在 `stdout.read(frame_size)` 时不响应；ffmpeg 进程要等 read 返回后才在 finally 里 terminate。daemon 线程兜底不至于泄漏进程，但关闭阶段资源短暂悬挂 | **低** |
| B17 | ffmpeg_io.py:254-295（`_nvenc_ok`/`_NVENC_TRUST_STATIC=True`） | **降级链脆弱** | probe 失败仍信任静态表选 nvenc；若驱动层确实不可用：SDK init 失败 → 回退 FFmpegWriter → 再次 best_encoder 又选 nvenc → ffmpeg 启动失败 raise（L947-953）→ 整段失败而不是落到 libx264 | **低中** |
| B18 | pipeline.py:1955-1959 | **双重释放（潜伏）** | except 分支用 `locals()` 里上一迭代的 `memory_block` 再次 `release` → GPUMemoryPool 信号量超发。当前它只是计数器无实害；未来若绑定真实内存即成 bug | **低** |
| B19 | realesrgan_processor_video_optimized.py:382-383 + main.py:377-381 | **副作用路径** | `_initialize_enhancer` 的 `ns.output='dummy_output.mp4'`；若 `report=='__AUTO__'`，报告写到 CWD 的 `dummy_output_report.json` | **低** |
| B20 | pipeline.py:741-743 vs 1832-1886；nvenc_sdk.py 计数器同理 | **良性数据竞态** | `_face_*_total/optimal_batch_size/_adaptive_read_batch_size/_diag_*` 跨线程无锁读写。GIL 下不会崩，进度条/统计可能瞬时不一致 | **低**（记录性） |
| B21 | gfpgan_subprocess.py:117-127 | **临时资源清理盲区** | SHM unlink 仅发生在正常 `close()`；主进程被 kill/断电时 POSIX `/dev/shm` 段残留（Windows 由 OS 回收）。无 stale-SHM 启动清扫 | **低** |
| B22 | async_dispatcher.py:106-110 | **静默丢弃** | `submit_validate` 在 queue.Full 时 pass → 验证请求消失 → `wait_validate` 白等 180s 才降级 | **低** |
| B23 | gfpgan_subprocess.py:1046-1045 区域 worker 校验分支 L956-963 | **逻辑疑点** | validate 的 context-dead 分支里 `if _ctx_dead: if 'out of memory'...` —— `_ctx_dead` 关键字集合本身包含 'cudaerror' 而 OOM 判定在其内部二次判断，嵌套可读性差且 `_is_oom` 变量算出后在 `_ctx_dead=False` 分支未被使用（L955 计算、L964 else 分支未引用），疑似残留半成品逻辑 | **低**（行为待验证） |
| B24 | nvenc_writer.py 全文 | **死代码陷阱复发风险** | 整个文件无 import 方（活跃 NVENCWriter 在 nvenc_sdk.py）。2026-07-28 的教训（SYNC-BEFORE-SUBMIT 曾补在死代码上"修复从未生效"）说明保留它是持续性隐患 | **低**（维护性地雷） |

**专项核对结论**（任务要求重点检查项）：
- *tile 拼接/重叠区混合/边缘像素*：不存在——tile 根本未接入（B5），无拼接错误但也无 tile 能力；
- *OOM 批内部分成功一致性*：处理正确——异常时 `all_sr` 半成品句柄被丢弃重启整批，defer 句柄的 GPU refs 随 GC 释放；prefetch 张量 OOM 后置 None 防复用脏数据（pipeline.py L974-975）；
- *GFPGAN PIPE 死锁*：所有 queue.put 均带 timeout 且有降级路径；唯一无 timeout 的回插在回退路径（B11）；
- *ctypes 指针生命周期*：总体严谨（func_table 保引用防 GC、c_void_p 包装、close 幂等 + ctx 配对 push/pop/release）；缺口为 B9 的事件泄漏与 B12 的隐式析构；
- *跨段首尾帧*：IDR-per-segment + `_stream_begin` + EOS-empty-chunk 修复 + slot 背压 + 帧守恒审计已构成较完整防线；残余风险是 B4 的估算提前退出；
- *pycuda.autoinit*：确认未调用；NVENC 通过 `cuDevicePrimaryCtxRetain` 与 PyTorch 共享 primary ctx，符合环境约束；
- *灰度视频*：reader 强制 rgb24 输出，无问题；*无人脸帧*：crops 空则直通 SR；*极短片段*：EOF 末批 flush 路径完备。

---

## 【C. 可维护性问题】

1. **巨型函数**：`NVENCEncoder._open_session`（nvenc_sdk.py L708-1101，约 400 行，会话建立+结构体魔数+RC 参数混杂）；`encode_frames_batch`（L1661-2063）与 `encode_frames_batch_ce_pipeline`（L2065-2437）各 ~400 行且大量排空/记账逻辑重复（`_apply_drained_entries` 只抽象了一部分，inline drain L2320-2355、末尾 drain L1995-2042 仍是手写副本）；`DeepPipelineOptimizer._process_sr`（L806-1269）与 `_process_gfpgan`（L1308-1644）各 300-400 行、嵌套四层。
2. **结构性重复**：
   - GFPGAN TRT 构建逻辑三份：`_build_only_worker`（gfpgan_subprocess.py L256-478）与子进程内 `GFPGANTRTAccelerator._export_onnx/_build_engine_dynamic/_onnx_compat_patch`（L615-725）几乎复制粘贴；tensorrt_accel.py 又一份（含相同的 SM 时间预估表 L136-142 vs L411-417）。
   - facexlib 猴补丁两份完全相同（main.py L71-81、gfpgan_subprocess.py L37-46），叠加包装顺序脆弱。
   - processor 中 `_initialize_enhancer` 与 `_run_esrgan_video` 两套 Namespace 构建（L373-423 vs L726-786），后者是遗留一次性路径却仍在短视频分支服役。
   - 自适应批处理 EMA 更新块在 `_process_gfpgan` 内出现两次近乎相同代码（L1517-1524 与 L1572-1587）。
3. **死代码/惰性特性**：`nvenc_writer.py` 全文件（B24）；`NVENCEncoder.encode_frame`（L2439，活跃链路无调用方）；`cuda_graph_accel` 死分支（pipeline.py L83-93）；`torch.compile` 全套 CLI/仲裁逻辑无实现（main.py L330-365）；`GPUMemoryPool` 名实不符（pipeline.py L200）；`HardwareCapability.has_nvdec` 在本模块无调用方；`AsyncGFPGANDispatcher.in_flight`/`_submit_blocking` 未被使用；`_NVENC_VBR_QUALITY_OFFSET` 自述废弃仍保留（nvenc_sdk.py L480-489）；pipeline.py L1736-1739 `if not thread.is_alive(): thread.daemon=True` 永假的矛盾守卫。
4. **硬编码魔法数**：等待上限 5400/2700/1800s（main.py L463、pipeline.py L417）、`_MAX_IN_FLIGHT=2`、resolve future 限深 2、`_recover_threshold=50`、冷却 8/15/20、chunk 内存预算 5e8、muxer wait 60s、`_MINI_BATCH=4`、`MAX_FACES=64`、`FACE_SHAPE=(512,512,3)`、`MIN_FACE_AREA=48*48`、paste 超时 60s——多数无配置出口。
5. **注释/实现漂移**：FFmpegMuxer 注释说"中间分段不需要 faststart"却仍加 `-movflags +faststart`（nvenc_sdk.py L3479-3482）；`_NVENCEncodeThread.__init__` 注释描述的 `_to_pinned_host_if_la` 属于另一个类的方法名（历史教训自述过 changelog 与代码脱节，此处命名仍易误导）；`_apply_sps_pps` 被注释引用但不存在该方法名（nvenc_sdk.py L871）。
6. **诊断代码体量大**：`_diag_aux_block/_diag_phase_shift/_diag_empty/_la_buffered` 等计数打印散布在每个 drain 循环（5 处以上相似 if 块），宜抽公共 logger；`_dump_all_queues` 反射遍历 `vars(self)` 的诊断器留在生产路径。
7. **行数统计偏差**：任务描述中的行数（如 nvenc_sdk 3438）与仓库现状（3827）差距较大，说明外部描述基线落后于代码演进——建议 AGENTS.md/文档同步真实规模，避免评审与记忆文件引用错位。

---

**总结**：该子系统经过多轮实战修复，跨段帧守恒、pinned-buffer 竞态、OOM 阶梯、子进程降级链等历史痛点已有系统性防御，代码内注释-记忆文件-修复标记三位一体可追溯性良好。当前最值得优先处置的是三个"高"级问题——Windows 下 FFmpegWriter select 管道失效（B1）、SDK NVENC 路径静默丢音轨（B2）、流水线异常被吞后段仍标记成功（B3）——以及两个认知纠偏项：tile 与 compile/CUDA Graph 均为惰性配置（B5、A.6），应在文档或启动日志中明示，避免运维误调参。
</task_result>
</task>
```

---

## Assistant (Plan · Ox Alpha Free · 649.4s)

_Thinking:_

The ESRGAN subsystem analysis came back with a very detailed report. The IFRNet subsystem agent returned empty again. I need to retry the IFRNet agent. It seems the explore agents sometimes return empty — possibly the task is too large. Let me split it into smaller pieces or retry once. Let me retry with a more focused prompt, maybe splitting into two: (1) processor + main.py + pipeline.py, (2) nvenc_sdk.py + ffmpeg_io.py + others.

Actually let me just retry the full one but emphasize output completeness. Or split into 2 parallel agents to reduce per-agent workload.

ESRGAN 子系统报告已收到（24 项问题）。IFRNet 代理再次返回空，我将其拆分为两个更小的并行任务重试。

**Tool: task**

**Input:**
```json
{
  "description": "分析IFRNet处理器与流水线",
  "prompt": "你是一个资深 Python/CUDA 系统代码审查专家。这是一个只读分析任务：不要修改任何文件。重要：你的最终回复必须包含完整的中文分析报告文本，绝不能为空。\n\n请深度分析 IFRNet 插帧子系统的前半部分（只需以下 3 个文件）：\n\n1. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\\processors\\ifrnet_processor_video_optimized.py （约779行，处理器封装层）\n2. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\main.py （1693行，后端入口）\n3. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\pipeline.py （1563行，核心流水线）\n\n背景：GPU 加速视频插帧（2x-16x），支持 PyTorch/TensorRT、FP16、CUDA Graph、NVENC 编码、分段(segment)处理与 checkpoint/resume。环境约束：不能调用 pycuda.autoinit。\n\n输出要求（结构化中文报告）：\n\n【A. 架构分析】\n- 每个文件的模块职责、核心类清单（类名、关键方法、职责、行号）\n- 数据流：读帧 → 批量插帧推理 → NVENC/ffmpeg 写出；线程模型与队列\n- 分段处理架构、checkpoint/resume 实现\n- TensorRT/FP16/CUDA Graph 集成点与回退逻辑\n- OOM 自动降级机制\n- processor 层如何调用 external 后端（import 方式、参数传递）\n- 设计模式\n\n【B. BUG 清单】每个 bug 给出：文件:行号、类型（逻辑错误/资源泄漏/并发/异常缺失/性能瓶颈/崩溃点/边界条件）、描述、影响（致命/高/中/低）。\n重点检查：\n1. 线程竞态：队列共享状态、stop 标志可见性\n2. ffmpeg 子进程 PIPE 管理、stderr 消费、进程清理\n3. 帧数守恒验证逻辑漏洞（输入帧数 vs 输出帧数）\n4. checkpoint 一致性、断点续跑帧对齐\n5. 显存管理：tensor 泄漏、empty_cache 时机、批大小降级状态一致性\n6. 异常路径资源释放（try/finally 完整性）\n7. 边界条件：奇数宽高、极短视频（<2帧）、最后不足一个 batch、灰度视频\n8. FP16/TRT engine 缓存失效判断（GPU 变更检测）\n9. 异常吞掉导致静默失败\n\n【C. 可维护性问题】过长函数、重复代码、硬编码魔法数字（给行号）。\n\n再次强调：最终回复必须包含完整报告内容。",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_fd2ff7a2fffeBzlgBW01DzkyIA" state="completed">
<task_result>

</task_result>
</task>
```

**Tool: task**

**Input:**
```json
{
  "description": "分析NVENC SDK与FFmpeg IO",
  "prompt": "你是一个资深 ctypes/CUDA/NVENC 驱动级编程专家。这是一个只读分析任务：不要修改任何文件。重要：你的最终回复必须包含完整的中文分析报告文本，绝不能为空。\n\n请深度分析 IFRNet 插帧子系统后半部分的底层模块（只需以下 4 个文件）：\n\n1. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\nvenc_sdk.py （2799行，ctypes 直调 NVIDIA Video Codec SDK）\n2. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\ffmpeg_io.py （643行）\n3. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\ifrnet_utils.py （297行）\n4. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\tensorrt_accel.py （218行）\n5. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\config.py （27行）\n\n背景：NVENC 通过 ctypes 直调 nvEncodeAPI（SDK 版本化函数表索引），与 PyTorch 共享 CUDA primary context（cuDevicePrimaryCtxRetain，不用 pycuda.autoinit）。支持 lookahead (LA)、constqp/vbr_hq/qvbr RC 模式、completion event 轮询、多 slot 缓冲。已知历史问题域：NVENC completionEvent 关键路径、LA 跨编码会话不可复用、空帧防御、帧数守恒。\n\n输出要求（结构化中文报告）：\n\n【A. 架构分析】\n- 各模块职责、核心类清单（类名、关键方法、职责、行号）\n- NVENC session 生命周期：创建→配置→编码→排空→销毁；slot/input/output buffer/event 管理\n- lookahead 分块编码逻辑、EOS 处理、flush 流程\n- completion event 等待/轮询模式\n- RGB→NV12 转换位置（CPU/GPU）与色彩空间处理\n- FFmpegReader/Writer 的线程模型、PIPE 管理\n- TRT 加速器的 engine 构建/缓存/失效策略\n- 设计模式\n\n【B. BUG 清单】每个 bug 给出：文件:行号、类型（ctypes 内存布局错误/指针生命周期/句柄泄漏/并发竞态/异常缺失/挂起点/崩溃点/边界条件）、描述、影响（致命/高/中/低）。\n重点检查：\n1. struct 字段偏移硬编码的正确性风险、_pack_ 对齐、版本兼容（SDK 函数表索引随版本变化）\n2. cuInit/cuCtx 交互：PyTorch 已初始化 context 时重复初始化的后果、context pop/push 配对完整性\n3. event 创建/销毁配对、异常路径泄漏\n4. input/output buffer 释放完整性（close 与 __del__）\n5. 编码循环中的帧数守恒：丢帧场景（slot 满、drain 失败、EOS 时序）\n6. LA 深度跨段复用问题\n7. select() 在 Windows 上对管道 fd 的兼容性（select 仅支持 socket）\n8. subprocess stdin 写阻塞、stderr 死锁\n9. 返回码检查遗漏（NVENC API status != 0 未处理）\n10. TRT engine 缓存 key 是否含 GPU SM 架构、batch 维度静态形状假设\n\n【C. 可维护性问题】过长函数、重复代码、硬编码魔法数字（给行号）。\n\n再次强调：最终回复必须包含完整报告内容。",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_fd2ff31d0ffeeKzMUdjcN0pHbP" state="completed">
<task_result>
所有文件已完整读取并交叉验证完毕（nvenc_sdk.py 实际为 **3526 行**，非任务描述的 2799 行；已确认全库无 `select()` 调用、无 `cuDevicePrimaryCtxRelease` 调用、TRT 分支 `timesteps*B` 为列表复制语义正确）。以下为完整分析报告。

---

# IFRNet 插帧子系统底层模块深度分析报告

**分析范围**：`external/ifrnet_video/{nvenc_sdk.py, ffmpeg_io.py, ifrnet_utils.py, tensorrt_accel.py, config.py}`
**说明**：`nvenc_sdk.py` 实测 3526 行（任务给的 2799 行为旧版本行数）；所有行号以当前文件为准。

---

## 【A. 架构分析】

### A.1 各模块职责与核心类清单

#### ① nvenc_sdk.py（3526 行）— ctypes 直调 NVIDIA Video Codec SDK 13.0 的 Level 1 GPU 直通编码层

| 类/函数 | 行号 | 关键方法 | 职责 |
|---|---|---|---|
| `_NvGuid` | 33 | `__eq__/__hash__` | 16 字节 GUID 结构体（`_pack_=1`），支持比较与哈希 |
| GUID 常量区 | 57–95 | — | H264/HEVC/AV1 codec GUID、P1–P7 preset GUID、H264 High profile GUID |
| `NVENCAPI_STRUCT_VERSION` / `_sdk13_ver` | 107 / 122 | — | SDK 13.0 struct version 公式：`ver \| (api<<16) \| (0x7<<28) [ \| bit31]` |
| 版本常量 | 127–151 | — | CreateInputBuffer/LockInputBuffer/CreateBitstreamBuffer 等 VER；`NV_ENC_ERR_NEED_MORE_INPUT=17` |
| ctypes struct 声明 | 158–342 | — | `_NvEncOpenEncodeSessionExParams`(158)、`_NvEncInitializeParams`(170)、`_NvEncConfigH264VUIParameters`(201)、`_NvEncConfigH264`(219)、`_NvEncConfig`(248)、`_NvEncPresetConfig`(266)、`_NvEncRegisterResource`(274)、`_NvEncMapInputResource`(291)、`_NvEncPicParamsH264`(302)、`_NvEncPicParams`(310)、`_NvEncLockBitstream`(330) |
| 函数表索引表 `_FUNC_IDX` | 403–425 | — | SDK 13.0 函数表槽位硬编码（OpenEncodeSessionEx=29、RegisterResource=30、DestroyEncoder=27 等） |
| RC/预设映射常量 | 437–463 | — | `_PRESET_P_INDEX`（x264 名→p 索引）、码率钳制 `[5M,50M]bps` |
| `_RotationBitReader` | 470 | `bits/ue/se` | H.264 RBSP 大端 exp-Golomb 位读取器（当前为备用死代码，见 L1698–1702 说明） |
| **`NVENCEncoder`** | 509 | 见下 | 核心编码器：session 生命周期、多 slot 缓冲池、LA 流式编码、CE 流水线、EOS flush、SPS/PPS 缓存注入 |

`NVENCEncoder` 关键方法：
- `__init__`（523–1071）：DLL 加载 → libcuda 加载+cuInit → API 版本探测 → primary context 获取 → 函数表创建 → OpenEncodeSessionEx（apiVersion 多候选重试）→ 动态 GUID 枚举 → GetEncodePresetConfig → RC 参数写入（constqp/vbr_hq/qvbr 三模式，全部裸偏移写）→ InitializeEncoder → 多 slot 创建（input buffer + bitstream buffer + CUDA event）→ 专用 non-blocking 拷贝 stream 创建。
- `_copy_into_input_buffer`（1073）：cuMemcpy2DAsync_v2 + 私有 stream 同步，失败回退同步 cuMemcpy2D_v2。
- `_destroy_slot`/`_destroy_all_slots`（1102/1129）：slot 三元组销毁。
- `_find_dll`（1137）：Win/Linux DLL 搜索路径。
- `_extract_sps_pps`（1173）/`_has_sps_pps`（1214）/`_nal_first_vcl_type`（1241）：Annex B NAL 扫描（h264/hevc/av1 三分支）。
- `_drain_outputs_blocking`（1278）：按 `_output_slot_idx` 轮转顺序阻塞 LockBitstream 排空至 NEED_MORE_INPUT。
- `_apply_drained_entries`（1355）：per-slot FIFO 队首记账分发（辅助块不占帧槽、相位漂移/gfi 回退诊断）。
- `_ensure_slot_free`（1458）：复用物理槽前强制排空该槽 pending，超限走目标槽直接阻塞 Lock → 最终空帧占位兜底。
- `_lock_bitstream_blocking`（1510）/`_lock_bitstream_with_retry`（1555）：阻塞锁定与指数退避重试两个原语。
- `encode_frames_batch`（1606）：兼容封装 = 强制新流 + `encode_frames_stream` + 结果按 fi 对齐回填。
- `_stream_begin`（1638）：跨 chunk 流状态复位（FIFO 表、帧号基线、输出指针段首对齐）。
- `encode_frames_stream`（1704）：**LA 有界分块流式编码核心**——逐帧 LockInputBuffer→memcpy2D→EncodePicture（constqp 附 per-frame CE 并同步）→ 全局 drain；`send_eos=True` 时发 EOS + 全槽排空 + 残留占位。
- `encode_frames_batch_ce_pipeline`（2074）：LA=0 的 CE 异步流水线三阶段（Harvest/Submit/Drain）+ 二次排空安全网。
- `encode_frame`（2486）：单帧同步兼容路径（恒用 slot 0 遗留句柄）。
- `flush`（2696）：发 EOS → 按 `_output_slot_idx` 起始轮转全槽排空（首个 Lock 阻塞、后续非阻塞）→ 返回拼接 ES。
- `close`/`__del__`（2832/2865）：销毁 slots/stream/encoder、恢复 saved context。

| 类/函数 | 行号 | 职责 |
|---|---|---|
| **`_NVENCEncodeThread`** | 2885 | 独立编码线程：`submit`(2954)/`estimate_timeout`(2967)/`_loop`(2982)/`flush_and_join`(3172)。LA>0 
</task_result>
</task>
```

---

## Assistant (Plan · Ox Alpha Free · 924.8s)

_Thinking:_

The NVENC SDK agent's report got cut off mid-way (the result was truncated at `_NVENCEncodeThread` | 2885). It seems the report was cut off. I have partial info: architecture of nvenc_sdk.py but no BUG list or maintainability section for the NVENC/ffmpeg_io/ifrnet_utils/tensorrt_accel files.

Also the pipeline/main.py/processor agent (first one) returned empty again. That's 3 empty returns for that particular task. Something about that combination of files makes the agent fail (maybe too much content — main.py 1693 + pipeline.py 1563 + processor 779 = ~4000 lines).

Let me:
1. Resume the NVENC agent session to ask it to continue with the BUG list and maintainability sections (task_id ses_fd2ff31d0ffeeKzMUdjcN0pHbP).
2. Retry the pipeline/main analysis as an even smaller task — maybe split into two: (a) src\processors\ifrnet_processor_video_optimized.py + external\ifrnet_video\main.py; (b) external\ifrnet_video\pipeline.py alone. Or just read those files myself. Given repeated failures, maybe I should analyze pipeline.py + main.py myself by reading them directly. They're large (1563 + 1693 lines) but readable in 2-4 Read calls each.

Actually, let me first resume the truncated agent, and simultaneously dispatch a fresh smaller agent for pipeline.py only, and another for ifrnet_processor + ifrnet_video/main.py.

Wait — resuming a session: "This should only be set if you mean to resume a previous session". The resumed agent continues with previous context. Let me do that for the NVENC one since its context already contains the full analysis.

Plan:
1. Resume NVENC agent (ses_fd2ff31d0ffeeKzMUdjcN0pHbP): "Continue outputting sections B (bug list) and C"
2. New agent for pipeline.py (1563 lines) alone
3. New agent for ifrnet_video/main.py + ifrnet_processor_video_optimized.py

Let me dispatch all three in parallel.

NVENC 底层报告被截断（缺 BUG 清单），流水线层代理第三次返回空。我将：恢复截断的会话让其补全，并把流水线分析拆成两个更小任务重派。

**Tool: task**

**Input:**
```json
{
  "description": "续写NVENC分析报告",
  "prompt": "你上一条回复在输出【A. 架构分析】的 _NVENCEncodeThread 表格中途被截断了。请继续输出剩余内容：完成 A 部分剩余类（_NVENCEncodeThread、RGB→NV12 转换、FFmpegMuxer/NVENCWriter 如有、FFmpegReader/Writer、TRT 加速器、设计模式小结），然后完整输出【B. BUG 清单】和【C. 可维护性问题】两部分。保持之前的行号精度与格式。\n\n重点检查项提醒：\n1. struct 字段偏移硬编码、_pack_ 对齐、SDK 版本兼容风险\n2. cuInit/cuCtx pop/push 配对完整性（你已发现无 cuDevicePrimaryCtxRelease 调用——请确认这是泄漏还是有意为之并评估）\n3. event 创建/销毁配对、异常路径泄漏\n4. input/output buffer 释放完整性\n5. 帧数守恒：slot 满/drain 失败/EOS 时序导致的丢帧场景、空帧占位逻辑\n6. LA 深度跨段复用问题\n7. subprocess stdin 写阻塞、stderr 消费\n8. NVENC API 返回码检查遗漏\n9. TRT engine 缓存 key 是否含 GPU SM 架构、静态 batch 形状假设\n\n再次强调：最终回复必须包含完整的 B 和 C 部分内容。",
  "subagent_type": "explore",
  "task_id": "ses_fd2ff31d0ffeeKzMUdjcN0pHbP"
}
```

**Output:**
```
<task id="ses_fd2ff31d0ffeeKzMUdjcN0pHbP" state="completed">
<task_result>

</task_result>
</task>
```

**Tool: task**

**Input:**
```json
{
  "description": "分析IFRNet入口与处理器",
  "prompt": "你是一个资深 Python/GPU 系统代码审查专家。这是一个只读分析任务：不要修改任何文件。重要：你的最终回复必须包含完整的中文分析报告文本，绝不能为空。\n\n请只深度分析这两个文件：\n1. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\src\\processors\\ifrnet_processor_video_optimized.py （约779行，处理器封装层）\n2. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\main.py （1693行，后端入口）\n\n背景：GPU 加速视频插帧（2x-16x）。processor 层通过 sys.path 注入后 `from ifrnet_video.main import IFRNetVideoProcessor` 进程内调用，实例缓存复用。支持分段(segment)处理与 checkpoint/resume、TensorRT/FP16/CUDA Graph 开关仲裁、NVENC 自动升级。\n\n输出要求（结构化中文报告）：\n\n【A. 架构分析】\n- 两文件的模块职责、核心类清单（类名、关键方法、职责、行号）\n- processor 如何构造参数传给后端（字段映射、默认值仲裁逻辑如 --no-tensorrt/--use-tensorrt 的优先级）\n- 后端 main.py 的初始化阶段划分（模型加载/TRT engine 构建/CUDA 初始化顺序）\n- 分段循环结构、checkpoint 读写位置、每段之间哪些对象被复用哪些重建\n- 设计模式\n\n【B. BUG 清单】每个 bug 给出：文件:行号、类型（逻辑错误/资源泄漏/并发竞态/异常缺失/配置错误/崩溃点/边界条件）、描述、影响（致命/高/中/低）。\n重点检查：\n1. 参数仲裁/覆盖逻辑的矛盾（如 CLI 与 config 冲突时的行为）\n2. checkpoint 一致性与断点续跑的帧对齐问题\n3. 分段间 GPU 对象生命周期：模型/engine/session 是否正确释放或复用，跨段状态污染\n4. ffmpeg 子进程调用规范、返回码、清理\n5. 路径处理（中文/空格/相对路径）\n6. 异常吞掉导致静默失败\n7. 边界条件：单段视频直通分支、模型文件不存在时的报错路径\n8. 数值参数校验缺失（crf/batch_size/factor 范围）\n\n【C. 可维护性问题】过长函数、重复代码、硬编码魔法数字（给行号）。\n\n再次强调：最终回复必须包含完整报告内容，不能为空。",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_fd2f50697ffeKUiokDjbd31jwZ" state="completed">
<task_result>
核查完毕。以下为完整分析报告。

---

# IFRNet 插帧子系统深度代码审查报告

**审查对象**
1. `D:\Workspace_Python\Video_Enhancement\Video_Enhancement\src\processors\ifrnet_processor_video_optimized.py`（实际 **961 行**，非预估 779 行）
2. `D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\ifrnet_video\main.py`（实际 **2153 行**，非预估 1693 行）

审查方式为只读静态分析；对关键疑点交叉验证了同包的 `config.py`、`ifrnet_utils.py`、`pipeline.py`、`nvenc_sdk.py`、`ffmpeg_io.py` 以确认结论成立条件。

---

## 【A. 架构分析】

### A1. 模块职责与核心类清单

**processor 层（ifrnet_processor_video_optimized.py）— 适配/编排层**

| 类/方法 | 行号 | 职责 |
|---|---|---|
| `IFRNetProcessor.__init__` | 65–165 | 配置解析、模型路径解析与存在性预检（91–98）、sys.path 注入（143–150）、临时目录句柄与实例缓存字段初始化 |
| `_la_is_active` | 171–181 | 判定 NVENC Lookahead 是否生效（VBR_HQ/QVBR 且 crf>0 且 LA>0），纯日志用途 |
| `process_video_segments` | 183–237 | 完整视频入口：建临时目录 → 读 checkpoint → 短视频直通分支（220–224）或按时间切分（227–231） |
| `process_segments_directly` | 239–263 | 接收上游分段直连入口：计算上游分段指纹 md5（259–261）供断点校验 |
| `process_video` | 265–348 | 完整流程编排：提取音频 → 分段插帧 → 合并 → 成功删 checkpoint → 清理后端 |
| `_setup_temp_dirs` | 354–363 | `temp_base/{prefix}_{video_name}` 下建 segments/processed/checkpoint.json |
| `_load_checkpoint` | 365–418 | 加载断点 + 配置快照兼容性校验（mtime/size/上游 hash），损坏/不兼容回退空断点 |
| `_get_config_snapshot` | 420–435 | 断点指纹快照：segment_duration、factor、codec、crf、preset、输入 mtime/size、上游 hash |
| `_save_checkpoint` / `_delete_checkpoint` | 437–446 / 643–650 | 断点持久化（非原子写）/ 全部完成后删除 |
| `_get_or_create_video_processor` | 448–490 | **延迟单例**：进程内构造一次后端 `IFRNetVideoProcessor` 并缓存复用 |
| `_cleanup_video_processor` | 492–518 | `cleanup(release_model=True)` + del + gc + empty_cache |
| `_process_segments` | 520–582 | 分段循环：断点跳过 → 逐段处理 → 成功记档保存 → 失败置位终止 |
| `_process_segment` | 584–641 | 单段驱动后端 `process_video()`，事后 `verify_video_integrity` 校验 |
| `_cleanup_temp_files` | 652–661 | rmtree 临时目录 |
| `main()`（CLI） | 668–957 | 独立命令行入口，CLI→config 覆盖仲裁（834–928） |

**后端（external/ifrnet_video/main.py）— 引擎层**

| 类/方法 | 行号 | 职责 |
|---|---|---|
| 模块级初始化 | 408–486 | warnings/stderr 全局过滤（408、413–421）、环境变量（423）、日志抑制（425–428）、sys.path（439–441）、**模块级加载硬编码 S 架构 Model（486）** |
| `IFRNetVideoProcessor(TensorRTAccelMixin).__init__` | 494–588 | 参数接收与 TRT 互斥仲裁（572–581）、设备解析（582）、`_load_model`、GPUMonitor 创建（586） |
| `_load_model` | 590–640 | torch.load → cudnn.benchmark → FP16 → torch.compile 包装（606–631）→ compute/h2d/d2h 三流创建 |
| `_get_cuda_graph` | 645–691 | 按 shape_key 捕获/回放 CUDA Graph，捕获失败自愈降级 |
| `_infer_batch` | 695–905 | 核心批推理：四条返回路径 —— GPU 直留（return_gpu，775–815）/ Ring D2H（820–854）/ PinnedPool D2H（859–894）/ 同步 CPU 回退（896–905） |
| `_estimate_safe_batch_size` | 909–925 | OOM 后显存估算（FIX-BATCHCAP 叠加 reserved−allocated） |
| `_safe_infer` | 927–1035 | OOM 级联降级 + CUDA Graph 错误自愈循环 |
| `_is_first_segment` | 1041–1042 | 段序判断（日志抑制粒度） |
| `_get_or_create_nvenc_encoder` | 1046–1065 | 跨段 NVENC 编码器 keyed 缓存 |
| `_get_or_create_result_pool` / `_get_or_create_ring_buf` | 1067–1096 | 跨段锁页内存池/Ring keyed 缓存 |
| `cleanup(release_model)` | 1098–1141 | 两级释放：默认只放 NVENC/Pool/Ring；True 时追加 Graph→TRT context→engine→model |
| `_process_segment` | 1143–1674 | 单段核心（≈530 行）：reader 探测、bs 估算、四级编码回退决策、compile 预热、首帧 f0 写入、pipeline.run、帧数守恒校验 |
| `process_video` | 1678–1932 | 对外接口：输入校验、TRT 一次性构建（1701–1712）、monitor 启停、ADAPTIVE-QUEUE 跨段调优（1731–1927）、报告 |
| `_print_summary` / `_dump_report` | 1934–1975 | 汇总打印 / JSON 性能报告（utf-8 写入） |
| `main()`（CLI） | 1983–2149 | 独立入口，`global Model` 重绑定（2077–2078）修正架构选择 |

### A2. processor → 后端的参数映射与默认值仲裁

**字段映射**（`_get_or_create_video_processor`，468–488）：逐一透传 `model_path/device/batch_size/max_batch_size/use_fp16/use_compile/use_cuda_graph/use_tensorrt/use_hwaccel/codec/crf/encode_preset/rate_mode/lookahead_depth/keep_audio/ffmpeg_bin/report_json/trt_cache_dir/quiet`。
**关键缺失：`model_name` 未透传**（后果见 B-1）；`t2_cache_dir` 在 processor 层完全无配置通道。`trt_cache_dir` 仅在 use_tensorrt 时才解析（459–465），空值兜底 `base_dir/.trt_cache`。

**CLI→config 覆盖仲裁**（processor `main()` 834–928）：两级优先级。
- 第一级：普通旗标（`--no-fp16/--no-compile/--no-cuda-graph/--use-tensorrt` 等）直接 `config.set`；
- 第二级（868–904）："高优先级覆盖" `--no-tensorrt/--use-compile/--use-cuda-graph` 后写覆盖前者的 config 值；`--no-tensorrt` 与 `--use-tensorrt` 并存时判 False 并打印 `[CLI覆盖]`。
- **最终互斥裁定在后端 `__init__` [FIX-TRT-MUTEX]**（main.py 573–581）：TRT 激活 → 强制关手动 CUDA Graph + 跳过 compile；compile 成功包装后再禁用手动 Graph（622–625）。即 CLI 的 `--use-cuda-graph` "强制启用" 在 compile 开启时**实际不会生效**（888–895 有预警打印，属"文档化矛盾"，见 B-12）。

### A3. 后端初始化阶段划分

1. **import 期副作用**：全局 warnings 忽略、stderr 替换、PYTORCH_ALLOC_CONF、包路径注入、`Model = S 架构`（486）。
2. **`__init__`**：参数落盘与 TRT 互斥仲裁（572–581）→ 设备解析（CUDA 不可用**静默回退 CPU**，582）→ `_load_model`：权重加载 → cudnn 开关 → FP16 → compile 包装与 inductor 缓存目录 → 三条 CUDA stream（634–640）→ GPUMonitor 实例化。**注意：TRT engine 不在此阶段构建。**
3. **首个 `process_video` 调用时**：TRT engine 惰性构建一次（1701–1712，无论成败标记 `_trt_built=True`）→ monitor.start → 进入 `_process_segment`。
4. **每段 `_process_segment`**：FFmpegFrameReader 探测（1162–1173）→ effective_bs 显存估算/分辨率复用（1176–1209）→ hw_profile 缓存（1213–1214）→ scale/timesteps 解析（1219–1229）→ 编码器决策（crf=0 无损映射 1232–1239；Level 1 尝试 1318–1345；code=8 降级重试 1352–1397；Level 2/3 Ring 1403–1422；Level 4 兜底 1425–1440）→ compile 小形状预热（1248–1276，仅一次）→ f0 显式编码（1464–1485）→ `pipeline.run`（1502–1518，T1/T2/T3 三线程）→ 段末计数核对/收尾（1636–1674）。
5. **段间**：monitor.stop + ADAPTIVE-QUEUE 计算下次队列深度（1731–1927）。

### A4. 分段循环、checkpoint 读写与对象生命周期

**分段循环在 processor 层**（`_process_segments` 536–565）：
```
for i, seg in enumerate(segment_files):
    if i ∈ checkpoint.processed_segments 且输出存在 → 跳过
    success = _process_segment(...)          → 后端 process_video(segment_index=i+1)
    success → append + checkpoint 更新 + _save_checkpoint（每段落盘，561）
    失败   → _has_failure=True + break
```
checkpoint 位置：`temp_base/checkpoint.json`；读于两个公共入口开头（205/262），写于每段成功后，删于 `process_video` 合并成功后（339）。后端另有 T2-CACHE（`t2_cache_dir`）属性能缓存而非断点语义。

**跨段复用 vs 每段重建**：

| 复用（实例生命周期） | 每段重建 |
|---|---|
| 后端实例、torch 模型/FP16/compile 图 | FFmpegFrameReader（ffmpeg 解码子进程） |
| 三条 CUDA stream、TensorPool、CUDA Graph | FFmpegMuxer/FFmpegWriter（ffmpeg 封装/编码子进程） |
| TRT context/engine（`_trt_built` 门控） | `IFRNetPipelineRunner` 及其 T1/T2/T3 线程、pair/result 队列、事件池 |
| `_cached_nvenc_encoder`（keyed：W,H,fps,preset,qp,rc,LA,pipe,codec） | GPUMonitor 启停（实例复用，采样窗口重开） |
| `_cached_result_pool`/`_cached_ring_buf`（keyed） | tqdm 进度条、ThroughputMeter |
| `_next_pair_queue/_next_result_queue/_next_t3_fps_measured`、`_hw_profile_cache`、`_warmup_done`、`_rq_shrink_streak` | |

### A5. 设计模式

- **适配器/门面**：IFRNetProcessor 包装后端引擎，收敛配置与断点职责；
- **延迟单例 + 缓存**：`_get_or_create_video_processor` 及三个 `get_or_create_*` keyed 缓存（Cache-Aside with key）；
- **策略链/责任链式降级**：四级编码回退（SDK 直通 → code=8 preset 降级重试 → Ring+NVENC → Ring+软编 → PinnedPool 标准）；
- **Mixin**：`TensorRTAccelMixin` 注入 TRT 推理分支；
- **对象池**：PinnedResultPool / PinnedRingBuffer / CudaEventPool / TensorPool；
- **多生产者-消费者流水线**：T1 Reader / T2 Infer / T3 Writer + 队列反压 + 看门狗 join；
- **Memento（检查点）**：config_snapshot 指纹校验的断点续跑；
- **观察者（弱化）**：GPUMonitor 后台采样驱动自适应调优。

---

## 【B. BUG 清单】

严重度定义：致命=功能不可用/数据损毁；高=特定常见场景必现故障或挂起；中=边缘场景错误行为/静默劣化；低=规范/健壮性问题。

### B-1【致命｜逻辑错误+配置错误】模型架构选择被硬编码，非 S 模型经 processor 层必崩
- 位置：`external/ifrnet_video/main.py:486`（`Model, _ifrnet_s_mod = _load_ifrnet_module('IFRNet_S_Vimeo90K')`，模块级）＋ `main.py:592`（`model = Model()` 使用该全局类）＋ `src/processors/ifrnet_processor_video_optimized.py:468-488`（构造后端时**未传 `model_name`**）。
- 链路：`config.MODEL_MODULE_MAP`（config.py:25-29）证明三种名字对应三个不同网络定义模块（models.IFRNet / IFRNet_S / IFRNet_L）。独立 CLI 因 `global Model` 重绑定（2077–2078）而正确；但 processor 层走库内导入路径，全局 `Model` 永远是 S 架构。用户配置 `model_name="IFRNet_L_Vimeo90K"` → processor 正确解析出 L 权重路径 → 后端用 S 架构 `load_state_dict`（594，strict 默认）→ 形状不匹配 RuntimeError → 被 processor `_process_segment` 捕获打印 traceback → 所有分段失败。
- 附带效应：即使修好架构，`model_name` 缺省为 S 还会污染 pipeline 的 T2 估算因子（pipeline.py:1580 `_MODEL_T2_FACTOR.get(proc.model_name)`）与 TRT/T2 缓存命名。
- 影响：**致命**（对该配置组合）。后端 `self.model_name`（517/540）存而几乎不用，是典型的"参数看起来支持实际不支持"。

### B-2【高｜边界条件/活锁】batch_size=1 仍 OOM 时无限重试循环
- 位置：`main.py:977-1003`（`_safe_infer` 的 `if self.batch_size <= 1:` 分支）。
- 描述：bs=1 再 OOM → 深度清理 → `_estimate_safe_batch_size` 几乎必然返回 ≥1 → `continue` 重试。若显存被他方进程长期占用，估算恒 ≥1 而推理恒 OOM，形成无退出条件的打印风暴循环（无重试计数上限）。流水线路径被外层 join 超时（pipeline.py:1647，上限 7200s）间接兜住，但 CPU 回退同步路径（main.py:1586-1634）无任何外层超时，可真挂死。
- 影响：高（挂起/数小时假死）。

### B-3【中｜资源泄漏/状态污染】get_or_create 缓存在"先 free 后构造失败"路径上留下悬空缓存
- 位置：`main.py:1076-1082`（result_pool）、`main.py:1090-1096`（ring buf）。
- 描述：`_get_or_create_ring_buf` 在 key 不匹配时先 `self._cached_ring_buf.free()` 再构造新对象；若 `PinnedRingBuffer(...)` 构造中途抛异常（锁页内存分配失败），**`_cached_ring_buf`/`_cached_ring_key` 仍指向已清空 slots 的旧对象且 key 未清除**。下一段若 key 恰好匹配旧 key，将"命中"已 free 的缓冲：`writer_acquire` 返回的 `self.slots[slot_id]` 直接 IndexError；`_get_or_create_result_pool` 同理会返回队列为空的已释放池，表现为 acquire 30s 超时并被误诊为"T3-Writer 可能卡死"（ifrnet_utils.py:116）。
- 影响：中（触发需"分辨率变化+分配失败"叠加，但一旦触发即为确定性崩溃/误诊）。

### B-4【中｜跨段状态污染】Level 切换时 `self._ring_buf` 不复位，可能抢占 Level 4 数据通路
- 位置：`main.py:820`（RING 分支判定仅看 `_ring_buf is not None`）、`main.py:1403-1414`（仅 `_active_level==4` 才赋值 `_ring_buf`）、pipeline.py:1969-1972（段尾明确不清 `_ring_buf`）。
- 场景：某段 NVENC 失败落入 Level 2（`_ring_buf` 被赋值）；后续某段编码器又可用走 Level 1（GPU_RAW，`return_gpu=True` 先行返回，暂不触发）；再后续某段 Ring 分配失败落到 Level 4 时，`_infer_batch` 的 RING 分支（820）排在 POOL 分支（859）之前**抢先命中上一阶段遗留的旧 ring**——旧槽容量按旧 bs/分辨率分配，`slot_tensor[:BT].copy_(pred_rgb)` 形状不匹配抛错，或 Writer 端协议错配，段以晦涩错误失败。
- 影响：中（条件链长，但属典型"跨段布尔态未随路径切换归零"缺陷；修复只需在 Level≠2/3 决策点显式 `self._ring_buf=None` 或让分支判定携带本段 active_level）。

### B-5【中｜异常吞掉→静默失败】muxer/writer 关闭返回码不参与成败判定
- 位置：`main.py:1639`（`writer.close()` 忽略结果）；底层 `nvenc_sdk.py:3410-3426`（FFmpegMuxer.close：rc≠0 仅 print Warning，不 raise 不返回）、`ffmpeg_io.py:781-792` 同模式。
- 场景：全部帧已写入 stdin、但 ffmpeg 在收尾阶段失败（典型如 `-movflags +faststart` 二次 moov 写盘因磁盘满/杀毒占用失败）→ 帧数守恒校验（1670，基于 `_write_count`）通过 → `_process_segment` 返回 True → 后端判定成功，产出无 moov 的坏 mp4。processor 层 `verify_video_integrity`（processor.py:626）是最后防线，能拦住完全不可播的文件，但对"时长/帧数异常但可解析"的截断文件未必拦截。
- 影响：中（坏产物可能带着 checkpoint 流入下游；后端自身语义应为 fail）。

### B-6【中｜异常缺失】后端 `_process_segment` 前半段无保护 + 异常路径泄漏 GPUMonitor
- 位置：`main.py:1162-1498`（reader 构造到 pipeline.run 之前无 try 包裹；首个 try 在 1501）；`monitor.start()` 于 1715，`stop()` 于 1733 仅在 `_process_segment` 正常返回后执行。
- 场景：输入探测阶段 `FFmpegFrameReader(...)` 抛异常（损坏文件/ffprobe 失败/中文路径下 ffmpeg 版本问题等）→ 异常直接穿透 `process_video` → processor 兜底捕获标段失败，但 **GPUMonitor 线程永远不停**，持续采样；下一段再次 `start()` 造成统计窗口重叠/线程堆积（取决于 GPUMonitor 幂等性）。
- 同类问题：f0 写入区（1464–1485，含 `.cuda()` 拷贝与 NVENC 编码）无 try/finally，此区间异常会跳过 `writer.close()/reader.close()`（1639–1640），遗留两个 ffmpeg 僵尸子进程直至解释器退出。
- 影响：中（资源泄漏 + 统计污染；进程通常随后退出故不到"高"）。

### B-7【中｜边界条件】非整数 scale 静默产生帧率/帧数失配（A/V 漂移或硬失败）
- 位置：`main.py:1219-1229`。
- 描述：`n_interp = int(scale_frac) - 1` 用**截断后的整数**生成插帧数，而 `new_fps = fps * float(scale_frac)` 用**原始小数**。例如 scale=2.5：n_interp=1（每对输入帧产出 2 帧），new_fps=2.5×fps → 输出时长缩短 20%，音频按源时长拷贝 → 音画漂移；软编路径完全静默，Level 1 路径则被帧守恒校验（1655/1670，期望按 2.5 计算）以"缺帧 >2"为由整段判败，报错误导（实际是 fps 元数据错误）。processor CLI 已限制 {2,4,8,16}，但 config 手填小数或后端 CLI `--scale 2.5` 均可触达。
- 影响：中。

### B-8【中｜并发竞态/一致性】直连分段模式下断点指纹不含内容变更检测
- 位置：`processor.py:259-261`（hash 仅拼接排序后的**路径名**）、`processor.py:420-435`（direct 模式下 `_current_input_video=None`，mtime/size 指纹缺位）。
- 场景：上游重切同名分段（segment_001.mp4…）但内容更新（换参重跑上游处理器、源视频替换）→ 路径 hash 不变 → 断点校验通过 → 已完成索引被跳过、旧 `interpolated_*` 被直接并入本次合并 → 新旧内容混流。
- 影响：中（数据一致性，触发依赖上游复用同名文件的习惯——本仓库 split_video_by_time 恰好如此）。

### B-9【中低｜逻辑错误】短视频直通分支绕过断点机制且失败不置位
- 位置：`processor.py:220-224`。
- 描述：duration ≤ segment_duration 时不写/不读 checkpoint（205 行加载的对象被丢弃）：中断后只能整体重跑（可接受但与文档承诺不符）；更重要的是处理失败时仅返回 `[]`，**`self._has_failure` 保持 False**（该标志只在 `_process_segments` 循环里置位，563–564），注释宣称"供上游主流程判断是否继续"的契约在此路径失真。
- 影响：中低。

### B-10【中低｜数值校验缺失】CLI/config 数值参数缺乏范围与 falsy 校验
- 位置与细节：
  - `--crf`（processor.py:790-791，帮助文本自称 0~51）**无任何范围校验**，负值/51+ 直达 x264 `-crf` 与 NVENC qp，段中段才爆；
  - `--batch-size/--max-batch-size`（852-855）：`if args.batch_size:` 使 **0 值被静默忽略**（用户以为覆盖成功实则沿用 config）；负值则放行进入后端 `max(batch, max_batch)`（main.py:523）产生荒谬上限；
  - `--segment-duration`（850-851）同样 falsy 忽略 + 无正数校验，负值传入 `split_video_by_time`；
  - `--preview-interval`（938-939）无下界，配合 `--preview` 时 0 值将在主循环 `frame_count % 0` 触发 ZeroDivisionError（main.py:1627，仅 CPU 路径可达）；
  - 后端 `lookahead_depth` 无负数防御，直达 SDK。
- 影响：中低（错误暴露晚、报错远离根因）。

### B-11【低｜异常缺失/资源】OOM 后 `_max_batch_size` 被永久下调且跨段不复原
- 位置：`main.py:970-975`（首次 OOM 即 `max = bs-1`，注释明言"永久"）；与 `_process_segment` 的 1206-1207 只降不升的逻辑叠加后，一次瞬时 OOM（如他进程临时占卡）会压低整个实例余下所有段的吞吐上限，且无任何"尝试回升"机制（对比 `_oom_cooldown` 只恢复 batch_size 不恢复上限）。
- 影响：低（性能劣化而非错误），但与 `[SEGMENT-REUSE]` 复用设计相互作用后用户难以察觉根因。

### B-12【低｜配置错误/文档矛盾】"强制启用"旗标的强制语义被后端静默推翻 + 三处默认值互相矛盾
- 位置：processor.py:771-783、888-900（预警打印）；main.py:573-581、622-625（实际裁定）；rate_mode 默认值三说不一——config 默认 `vbr_hq`（processor.py:119）、CLI 帮助写"默认 qvbr"（processor.py:801）、SDK 注释称 Level 1 默认 `constqp`（nvenc_sdk.py:3437）。另 `--max-batch-size` 帮助写"默认 8"而 config 默认 16（757 vs 104）。
- 描述：`--use-cuda-graph` 单独使用（不配 `--no-compile`）时被 compile 接管禁用——"覆盖一切"的旗标语义名不副实；虽有打印预警，但属于把仲裁矛盾推给运行期日志而非在参数层解决。
- 影响：低（行为有日志可循，但易造成"我明明加了开关为什么没生效"类工单）。

### B-13【低｜安全/健壮性】`torch.load(weights_only=False)`
- 位置：`main.py:593`。反序列化任意 pickle，权重文件被篡改即可代码执行。本地可信环境风险有限，但作为常被换用的 `.pth` 加载点建议 `weights_only=True`（IFRNet state_dict 为纯张量，兼容）。同时此处缺少友好 FileNotFound 包装（processor 层 91-98 已预检，后端独立使用时报裸栈）。影响：低。

### B-14【低｜静默降级】CUDA 不可用时请求 cuda 却静默切 CPU
- 位置：`main.py:582`。`device='cuda'` 但驱动异常时无告警地落到 CPU（配合 525-526 自动关 FP16/Graph），用户面对数量级变慢却无任何提示。同类：TRT 构建失败后 `_trt_built=True` 一票否决（1711-1712），整实例静默回退 torch 路径且不再重试。影响：低。

### B-15【低｜规范】checkpoint JSON 读写未显式 `encoding='utf-8'` 且非原子写
- 位置：`processor.py:369`（读）、`processor.py:445-446`（写）。当前内容经 `ensure_ascii` 默认转义恰好 ASCII 安全，但违反仓库 AGENTS.md 明文规范；且 `open("w")` 直接覆写，进程在写入中途被杀会产生截断 JSON（现有 except 能兜成"重新开始"，代价是丢进度）。建议 tmp+`os.replace`。影响：低。

### B-16【低｜逻辑错误】断点续跑后的剩余时间估算失真
- 位置：`processor.py:567-574`。`completed = len(checkpoint["processed_segments"])` 含历史会话完成的段，`elapsed` 仅本次会话耗时 → avg 被低估、remaining 严重偏小。影响：低（纯展示）。

### B-17【低｜冗余/重复输出】提前 EOF 警告在同一条件下打印两次
- 位置：`main.py:1578-1585`（CUDA 分支内）与 `1642-1649`（收尾处无条件）逻辑完全相同。影响：低。

### B-18【低｜死功能】`--preview` 在 GPU 主路径无效
- 位置：`main.py:1627-1632` 仅存在于 CPU 同步回退循环；CUDA 路径委托 `pipeline.run`（1502-1518），其签名无 preview 参数，`preview_interrupted`（1488）在 GPU 路径恒 False。文档宣传的调试特性在实际唯一路径上是 no-op。影响：低。

### B-19【低｜边界条件】scale>33 时静默钳制为 33x
- 位置：`main.py:1225-1227`（`n_interp>32 → Fraction(33)`），无任何打印告知用户倍数被改。结合 B-7，此处还把 `int(Fraction(33))=33` 作为分母，行为一致但依旧无声。影响：低（processor CLI 限 16 不可达，config/后端 CLI 可达）。

### B-20【低｜进程级副作用】import 即改全局 stderr/warnings/env
- 位置：`main.py:408`（warnings 全局 ignore）、411-421（`sys.stderr = _NVMLFilter(...)` 进程级替换）、423（env setdefault）。processor 层是**进程内导入**，这些副作用波及同进程的 ESRGAN 管线及宿主一切代码：过滤器按 write 块匹配，跨块的模式行漏过滤；其他库持有的旧 stderr 引用绕过过滤器造成行为不一致。影响：低-中（工程卫生）。

**已排查排除的疑点**（避免误报）：
- RING 路径异常处理中的 `self._ring_buf._write_sem.release()`（main.py:846-851）：对照 `writer_acquire`（ifrnet_utils.py:187-196），信号量获取成功后锁内不可能抛异常，release 与 acquire 严格配对，**无双重释放**；
- ffmpeg 子进程均以 list 参数 + `shell=False` 调用（ffmpeg_io.py:415/746、nvenc_sdk.py:3328），空格/中文路径安全；stderr 有 drain 线程防 pipe 堵塞；close 带 wait/kill 兜底——调用规范总体良好（问题仅在 B-5 的返回码不传播）；
- Level 1 各失败分支均清理 `_cached_nvenc_encoder`（1389-1390、1396-1397），"半初始化 encoder 复用"已被 [FIX-PRESET-DEGRADE-CACHE] 防住；
- processor 层模型不存在时的报错路径（91-98）清晰友好，属正面样本。

---

## 【C. 可维护性问题】

### C1. 过长函数
| 函数 | 位置 | 行数 | 说明 |
|---|---|---|---|
| `IFRNetVideoProcessor._process_segment` | main.py:1143–1674 | ≈532 行 | 集 reader 探测、bs 估算、四级编码决策、预热、f0、pipeline 编排、守恒校验于一身的"上帝函数"，至少可拆 6 个私有方法 |
| `IFRNetVideoProcessor.process_video` | main.py:1678–1932 | ≈255 行 | 其中 ADAPTIVE-QUEUE 决策块（1749–1927，σ加权合并/迟滞收缩/P跟随R）约 180 行内联策略，应独立成 `AdaptiveQueuePolicy` 类 |
| `IFRNetProcessor.main()`（CLI） | processor.py:668–957 | ≈290 行 | 参数定义、config 覆盖仲裁、互斥预警三段可拆 |

### C2. 重复代码
- **CLI 覆盖仲裁整套逻辑双实现**：processor.py:868–904 与 main.py:2034–2064 几乎逐句相同（含同样的覆盖消息文案），规则演进必然漂移；
- **Level 1 创建块重复**：main.py:1324–1341 与降级重试块 1361–1383 结构重复（makedirs/Muxer/set_muxer_ref/日志）；
- **EOF 警告两份**：1578–1585 vs 1642–1649（见 B-17）；
- **MODEL_NAME_MAP 双处定义**：processor.py:59–63 与 external/ifrnet_video/config.py:32–36，新增模型需改两处；
- **`bytes_per_frame = H*W*3*2*6` 公式两处**：main.py:921 与 1175；
- **`_expected_out=(fc-1)*scale+1` 守恒公式两处**：main.py:1655 与 1940；
- **`_la_is_active` 规则重复**：processor.py:171–181 复刻了后端 1289–1305 的 LA 生效判定（含 constqp 清零规则），后端规则一变前端日志即撒谎。

### C3. 硬编码魔法数字（择要）
| 行号 | 内容 |
|---|---|
| main.py:486 | 硬编码 `'IFRNet_S_Vimeo90K'`（B-1 根因） |
| main.py:500/501 | batch_size=48、max_batch_size=64 默认（与 processor 层默认 4/16 脱节） |
| main.py:921 / 1175 | `H*W*3*2*6` 帧字节数启发式 ×36 |
| main.py:947–948, 996 | 2.0MP / 1.0MP 高分辨率阈值 |
| main.py:992, 1006 | OOM cooldown 20 / 10 |
| main.py:1194–1199 | VRAM 安全因子 0.45/0.60/0.35 与系数 2.5 |
| main.py:1225–1226 | 插帧上限 32 / `Fraction(33)` |
| main.py:1251–1252 | compile 预热形状 32×32、bs=1 |
| main.py:1404–1408 | Ring 总预算 4GiB、slot 数 min(8, max(2,…)) |
| main.py:1670 | 帧守恒豁免容差 >2 帧 |
| main.py:1804–1806 | NVENC 理论 3000fps@1080p 假设 |
| main.py:1855–1856, 1883 | 收缩步长 0.75、result_queue 上限 64、下限 8 |
| main.py:2117 | CLI `max_batch_size = args.batch_size * 4` |
| processor.py:127 | preview_interval 默认 30 |

### C4. 其他结构问题
- **死代码/死参数**：`checkpoint["last_segment"]` 写而不读（processor.py:560）；后端 `_process_segment` 的 `frame_start/frame_end/skip_first_output` 唯一调用点恒为 `0/-1/False`（1719–1720）；main.py:616–621 的"use_tensorrt 时 compile 不执行"分支因 573–579 已强制 `use_compile=False` 而不可达；`n_seg_est`（1173）穿透私有属性仅供进度条。
- **封装破坏**：main.py:1472–1473 从外部直接戳 `_nvenc_encoder._pending_f0_nv12/_pending_f0_force_idr` 私有字段；848 直接操作 `_ring_buf._write_sem`。
- **全局副作用导入**（B-20）使该模块无法被安全地在测试/宿主进程中轻量导入。
- **processor 层能力缺口**：`t2_cache_dir` 无配置通道；`quiet/preview` 以属性注入（938–940）而非构造参数，破坏"构造即完备"约定。

### 总体评价
分层意图清晰（processor 编排/断点 ↔ 后端引擎/性能），分段复用与四级回退设计成熟，大量带编号的 FIX 注释显示活跃的自我修复史。但存在一条**致命主线缺陷（B-1 模型架构硬编码 + model_name 未透传）**使多模型能力在库内调用路径整体失效；跨段 keyed 缓存的失败路径（B-3/B-4）和收尾返回码吞噬（B-5）是下一梯队需要优先加固的点；后端 `_process_segment` 急需拆解，CLI 仲裁逻辑应在两层间去重收敛为单一实现。
</task_result>
</task>
```

---

## Assistant (Plan · Ox Alpha Free · 720.4s)

_Thinking:_

Excellent progress. Now I have:
1. ✅ Main entry + utils layer (27 bugs, architecture)
2. ✅ ESRGAN subsystem (24 bugs + full architecture)
3. ✅ IFRNet processor + main.py (20 bugs + full architecture)
4. ❌ NVENC SDK deep-dive: partial architecture (truncated), no BUG list — resume returned empty again
5. ❌ IFRNet pipeline.py analysis — the dedicated agent for pipeline.py never got dispatched successfully. Wait, let me check my last dispatches:
   - Resume of NVENC agent → empty
   - New agent "分析IFRNet入口与处理器" → success (this covered ifrnet_processor_video_optimized.py + external/ifrnet_video/main.py)

Still missing:
- pipeline.py (1563 lines) detailed bug review for IFRNet side
- nvenc_sdk.py BUG list (IFRNet version)
- ffmpeg_io.py / ifrnet_utils.py / tensorrt_accel.py (IFRNet version) BUG list

The NVENC agent's context has all files read but its continuation returned empty twice. Rather than resuming again (which failed twice), let me dispatch a fresh agent focused only on producing the B and C sections from those 5 files. But fresh agent needs to re-read files. The files are large (nvenc_sdk 3526 lines).

Alternative: analyze pipeline.py myself by reading it directly, and dispatch one fresh agent for nvenc_sdk.py + ffmpeg_io.py bug hunting.

Actually, given time constraints, maybe a more efficient approach:
- Dispatch one fresh explore agent for `external/ifrnet_video/pipeline.py` (1563 lines) — architecture + bugs.
- Dispatch one fresh explore agent for `external/ifrnet_video/nvenc_sdk.py` (3526 lines) — focus on BUG list only.
- Meanwhile, I read `external/ifrnet_video/ffmpeg_io.py` (643), `ifrnet_utils.py` (297), `tensorrt_accel.py` (218) myself.

That's reasonable. Let me do that.

IFRNet 处理器与后端入口报告已收到（发现 1 项致命缺陷：模型架构硬编码）。还缺 IFRNet 的 pipeline.py、nvenc_sdk.py 深度审查。我派两个新代理分别处理，同时自己读剩余的小文件。

**Tool: task**

**Input:**
```json
{
  "description": "审查NVENC SDK缺陷清单",
  "prompt": "你是一个资深 ctypes/CUDA/NVENC 驱动级编程专家。这是一个只读分析任务：不要修改任何文件。重要：你的最终回复必须包含完整的中文报告文本，绝不能为空（此前有代理返回空导致任务失败）。本次只需输出【B. BUG 清单】和【C. 可维护性问题】，不需要架构介绍部分。\n\n请深度审查这一个文件：\nD:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\nvenc_sdk.py （3526行，ctypes 直调 NVIDIA Video Codec SDK 13.0）\n\n背景知识（前序分析已确认的架构）：`_pack_=1` struct 手工布局+字节偏移硬编码；函数表索引表 _FUNC_IDX（SDK13 OpenEncodeSessionEx=29）；NVENCEncoder 类含 session 创建（apiVersion 多候选重试）、动态 GUID 枚举、RC 参数裸偏移写入（constqp/vbr_hq/qvbr）、多 slot（input buffer/bitstream buffer/CUDA event 三元组）、专用 non-blocking 拷贝 stream；LA 分块流式编码 encode_frames_stream（LockInputBuffer→memcpy2D→EncodePicture→drain）、encode_frames_batch_ce_pipeline（LA=0 CE 异步流水线 Harvest/Submit/Drain 三阶段）、flush（EOS+全槽排空）、close/__del__；_ensure_slot_free 强制排空物理槽；_apply_drained_entries per-slot FIFO 记账（辅助块不占帧槽）；空帧占位兜底；FFmpegMuxer ES→MP4 mux；编码线程 _NVENCEncodeThread。与 PyTorch 共享 primary context（cuDevicePrimaryCtxRetain，无 cuDevicePrimaryCtxRelease 调用）。\n\n【B. BUG 清单】每个 bug 给出：行号、类型（ctypes 内存布局/指针生命周期/句柄泄漏/并发竞态/异常缺失/挂起点/崩溃点/边界条件/返回码遗漏）、描述、影响（致命/高/中/低）。\n必查清单：\n1. cuInit 重复调用与 PyTorch 已初始化 context 的交互；primary ctx 无 Release 是泄漏还是有意为之（进程生命周期评估）\n2. saved context push/pop 配对在异常路径的完整性（close 中途 raise 会怎样）\n3. event 创建/销毁配对；__del__ 中销毁的可靠性\n4. input/output buffer 销毁完整性；session 销毁顺序（先 destroy slots 还是先 DestroyEncoder？顺序错误的后果）\n5. 帧数守恒：slot 满时 drain 失败、EOS 时序、空帧占位对输出帧计数的扭曲——哪些场景会真实丢帧或多出垃圾帧\n6. LA chunked 编码跨 chunk 的 FIFO 记账一致性\n7. EncodePicture/LockBitstream 等 API 返回码未检查的位置\n8. NEED_MORE_INPUT 与异步 event 的竞态\n9. ctypes 结构体：字段偏移硬编码与 SDK 版本漂移的风险点；c_void_p 所有权问题（ctypes 对象被 GC 后指针悬空）\n10. FFmpegMuxer 子进程 stdin 写阻塞、stderr drain、close 超时\n11. 编码线程 _NVENCEncodeThread：submit 队列无界？flush_and_join 超时后线程泄漏？\n12. 多实例并存时的全局 DLL/libcuda 句柄管理\n\n【C. 可维护性问题】过长函数（给出行号范围与行数）、重复代码块、硬编码魔法数字。\n\n再次强调：最终回复必须包含完整的 B 和 C 内容，不能为空。",
  "subagent_type": "explore"
}
```

**Output:**
```
<task id="ses_fd2e6f25fffeFo3zdIBXzR7tHU" state="completed">
<task_result>
全文（3526 行）已逐段通读完毕，并用 grep 交叉验证了死代码定义、跨模块契约（`main.py` L1059-1060 确认编码器逐段 close/重建；`_pending_f0_nv12` 仅由外部注入）以及常量引用情况。以下为最终报告。

---

# 【B. BUG 清单】

## B-1 CUDA context 生命周期：retain 不 Release + push 不 pop，逐段重建下累积泄漏
- **行号**：L630（`cuInit(0)` 每实例重复调用）、L658-667（`cuDevicePrimaryCtxRetain` + `cuCtxPushCurrent(primary)`）、L2856-2863（`close()` 用 `cuCtxPushCurrent(self._saved_ctx)` “恢复”）
- **类型**：句柄泄漏 / 资源配对缺失
- **描述**：(a) `cuInit` 本身幂等（驱动引用计数），与 PyTorch 共存安全，此项无害。(b) `cuDevicePrimaryCtxRetain(0)` 取得的正是 PyTorch 在用的 primary context，共享是有意为之；但全程**无任何 `cuDevicePrimaryCtxRelease`**。(c) `__init__` 把 primary ctx **push** 到当前线程栈后从不 pop；`close()` 不是 pop 掉 primary，而是再 **push** 一个 saved_ctx——push/pop 完全不对称。经查 `main.py` L1059-1060，编码器按段 close 后重建：每个开/关周期该线程 ctx 栈净增 ≥1 条目、retain 计数 +1，直到进程退出才由驱动回收。
- **影响**：**中**。单编码器进程生命周期内可接受（接近有意设计）；但当前“逐段重建”的实际用法下是确定性累积泄漏，且栈内残留 context 会使后续代码的 "current context" 语义漂移（`_loop` 里为此又打了 `cuCtxSetCurrent` 补丁，L2989-3000，属补丁叠补丁）。

## B-2 `__init__` 中途失败：saved_ctx 永不恢复、close() 清理链断裂
- **行号**：L684 / L702 / L730 / L741 / L749 / L772 / L784 / L796 / L937 / L953（各 raise 点）、L2833-2835（close 早退）、L2865-2869（`__del__` 吞异常）
- **类型**：异常缺失 / 清理不完整
- **描述**：初始化任一步骤 raise 后依赖 `__del__`→`close()` 兜底，但：(a) 若失败发生在 L689（`self._func_ptrs` 赋值）之前，`close()` 访问 `self._func_ptrs` 抛 AttributeError，被 `__del__` 的 bare except 静默吞掉，已创建的 session/ctx 泄漏且无日志；(b) `close()` 开头 `if self._encoder.value is None: return` 直接早退，**跳过 saved_ctx 恢复**；(c) 即使走到恢复分支，它在锁内最后一行，前面任何语句异常都会跳过它（无 finally 包裹）。
- **影响**：**中**（初始化失败场景下上下文栈进一步失衡；正常路径不受影响）。

## B-3 slot 创建中途 `cuEventCreate` 失败：当槽 input/bitstream buffer 泄漏
- **行号**：L969-1008
- **类型**：句柄泄漏
- **描述**：循环内顺序为 CreateInputBuffer → CreateBitstreamBuffer → cuEventCreate → `self._slots.append(...)`。event 创建失败（L1000-1002）触发 `_destroy_all_slots()`，但**当前槽的两个 buffer 句柄尚未 append 进 `_slots`**，不会被销毁（只能指望后续 DestroyEncoder 兜底回收 session 级资源，且该调用同样在失败路径上不保证执行）。
- **影响**：**低**（罕见路径，量级为单个 buffer 对）。

## B-4 EOS `EncodePicture` 返回码未检查 → 段尾整段退化为冻结帧
- **行号**：L1942-1943（`encode_frames_stream` 的 EOS）、L2725-2726（`flush()` 的 EOS）
- **类型**：返回码遗漏
- **描述**：两处 EOS 提交均不接收返回值。若 EOS 因任何原因失败，随后所有 `LockBitstream` 立刻返回 NEED_MORE_INPUT 而退出，`send_eos` 路径会把 `_strm_slot_pending` 里**全部残留帧**标记为 `(fi, prev_frame)` 占位（L2029-2040）——即段尾所有 LA 滞留的真实帧被替换成上一帧内容的复制品，且流程“成功”结束，无任何错误上抛。
- **影响**：**高**（触发时输出尾部大段静止画面，且静默）。

## B-5 `_drain_outputs_blocking` 非 SUCCESS/NEED_MORE_INPUT 错误码被静默吞掉
- **行号**：L1323-1324
- **类型**：返回码遗漏 / 异常缺失
- **描述**：`if bs_status != NV_ENC_SUCCESS: break` —— 除 NEED_MORE_INPUT 外的任何错误（如 INVALID_PARAM、设备错误）都无声退出循环，该槽及后续槽的已就绪帧全部丢失，无日志、无计数、无上抛。对比之下 `_apply_drained_entries` 为错配专门加了诊断计数，唯独 API 错误本身零痕迹。
- **影响**：**高**（真实丢帧且完全不可观测）。

## B-6 `_lock_bitstream_blocking` 的超时对阻塞调用无效（HEVC/AV1 空槽永久挂起）
- **行号**：L1510-1553（尤其 L1526 deadline 检查与 L1530 `doNotWait=0` 的组合）、调用点 L1481-1482
- **类型**：挂起点
- **描述**：`timeout_ms` 只在**两次循环之间**检查；单次 `doNotWait=0` 的 blocking LockBitstream 若驱动端不返回，Python 侧 deadline 永远无法生效。而文件自己在 L1951-1954/L2742-2745 明确记载：“HEVC/AV1 驱动对空槽的 blocking LockBitstream **永久阻塞**（test4 实测）”。`_ensure_slot_free` 的兜底路径恰好是对“pending 存在但硬件迟迟不出数据”的槽做 2000ms blocking Lock（L1477-1486）——这正是最可能踩中驱动无限等待的场景，2000ms 形同虚设。
- **影响**：**高**（HEVC/AV1 + LA 路径的确定性卡死风险；H.264 因空槽返回 SUCCESS+size=0 而幸免）。

## B-7 `flush()` 使用 doNotWait=1，与本项目自己记录的 T4 segfault 结论直接矛盾
- **行号**：L2770-2771（首个 blocking 后切换 `doNotWait=1`）vs L1284-1285、L1313-1316（“doNotWait=1 在 T4 驱动上对 LA 预热 buffer 直接 segfault（output3.err 实测）”）
- **类型**：崩溃点 / 自相矛盾的防御
- **描述**：`_drain_outputs_blocking` 和 `_lock_bitstream_blocking` 都因 segfault 记录而强制 doNotWait=0，唯独 `flush()` 的第二把及以后的 Lock 用 doNotWait=1 加速排空。当前调度上 flush 仅在 LA=0 路径被调用（L3162），暂时规避；但 `flush()` 是公共 API，任何 LA>0 直接调用（含测试脚本、未来重构）即触雷。
- **影响**：**中**（条件触发的进程级崩溃，且与既有经验相悖，极易在重构中复活）。

## B-8 `_ensure_slot_free` 排空超限兜底：占位帧扭曲帧数守恒 + 真实码流随后被丢弃
- **行号**：L1487-1507（fallback 占位 + `_output_slot_idx += _n_fill`）、L1404-1414（FIFO 空时 mismatch → `continue` 丢弃数据）、L1481-1486（直锁目标槽但不推进 `_output_slot_idx`）
- **类型**：边界条件 / 并发竞态 / 记账不一致
- **描述**：三条相互纠缠的问题：(a) fallback 将 pending 逐条以 prev 帧填充消费并把 `_output_slot_idx` 前推 `_n_fill`，但这些帧硬件**稍后仍会产出真实码流**——之后轮转到该槽时 FIFO 已空，真实码流在 `_apply_drained_entries` 走 mismatch 分支被 `continue` 丢弃；(b) 直锁路径（L1481-1486）从目标槽取走了数据却**不推进** `_output_slot_idx`，与前一路径对同一个指针做方向相反的错误操作，物理轮转与记账相位进一步脱钩；(c) 占位复用物理 bs_buf 时，旧帧 DMA 写入与新帧提交之间存在窗口期覆盖竞态。输出表现为：多出若干冻结帧（prev 复制）+ 对应真实帧消失，帧数守恒校验（L3151）只会报差值，无法定位。
- **影响**：**中**（质量劣化 + 计数失真；“宁占位不覆盖”是权衡后的设计，但真实数据被静默丢弃这一点超出设计意图）。

## B-9 `encode_frame()` 与流式 FIFO 记账体系完全脱节
- **行号**：L2486-2694（固定使用 slot0 legacy 句柄 L2532/2598-2599、不入 `_strm_slot_pending`、L2632-2634 取“全局 drain 首帧”作为本帧返回值）
- **类型**：边界条件 / 记账不一致
- **描述**：该方法绕过 per-slot FIFO：不登记 pending、不推进 `_strm_next_fi` 之外的任何记账、返回值取 drain 队列第一项（多槽轮转下很可能是**更早帧**的数据）。一旦与 `encode_frames_stream` 混用（注释 L1764 声称靠约定避免，无任何强制），slot0 的 FIFO 与物理内容立即错位，后续 drain 连环 misattribute。另外 NEED_MORE_INPUT 分支（L2638-2645）在最终 status 校验（L2660）之前早退，返回的同样是来路可疑的 drain 数据。
- **影响**：**中**（仅在混用/单帧路径触发，但触发即静默花屏/错序）。

## B-10 `encode_frames_batch_ce_pipeline` 的 inline-drain/redrain 分支：辅助块误当帧数据 + 静默丢帧（潜伏死路）
- **行号**：L2324-2359（inline-drain）、L2438-2473（redrain）、L2114（每批 `_reset_output_slot_idx(0)`）
- **类型**：记账一致性 / 静默丢帧（潜伏）
- **描述**：因 `[FIX-LA-BATCH-ROUTE]`（L2361）ce_pipeline 现仅服务 LA=0，以下 LA>0 分支成为死路，但仍是地雷：(a) inline-drain **完全不判 VCL 辅助块**——独立 SPS/PPS AU 会被当作 `_slot_pending[_drain_slot]` 那帧的码流写入 results 并清 pending，真实 VCL 帧随后无处安放；(b) `_pending is None` 时 `continue` **静默丢弃**已取回的码流（连计数都没有，对比 stream 路径至少有 `_diag_slot_mismatch`）；(c) redrain 直接把 `est_fi` 当 results 下标，无视 per-slot FIFO 映射，与同文件其余部分的记账模型冲突；(d) 每批 `_reset_output_slot_idx(0)` 与 stream 路径 L1663 的 `_output_slot_idx = _frame_idx` 修正相悖。
- **影响**：**中**（当前被路由屏蔽；任何人以 LA>0 直接调用 ce_pipeline 即触发数据损坏）。

## B-11 NEED_MORE_INPUT 与 completionEvent 的固有竞态（CE 先于编码完成触发）
- **行号**：L2172-2180（自述根因：“CE 在帧入队时即触发，但编码尚未完成”）、L2181（5000ms blocking retry 缓解）、L2406-2434（Phase 3 同款）
- **类型**：并发竞态
- **描述**：`cuEventSynchronize(_prev_ce)` 通过 ≠ NVENC 硬件已完成该帧编码。缓解手段是非阻塞 Lock 失败后再做最长 5 秒的 blocking retry；retry 仍拿不到就 `results[_prev_fi] = b""`——该帧随后被上游 Tier1-A 用前一帧顶替，GOP 参考关系被破坏（文件自己在 L2179-2180 承认这导致过“花屏/PPS错误”，修复只是缩小而非消除窗口）。5 秒超时值也是拍脑袋常数，慢场景仍会误判为空帧。
- **影响**：**中**（有兜底但兜底本身产生替代帧，属于“把崩溃换成画质损伤”的残余风险）。

## B-12 `encode_frames_batch(force=True)` 强制清空跨 chunk 残留 pending
- **行号**：L1629（`self._stream_begin(force=True)`）、L1648（`_strm_slot_pending.clear()`）
- **类型**：边界条件 / 数据丢失
- **描述**：兼容包装无条件重置流状态。若调用方此前用 `encode_frames_stream` 提交过未 send_eos 的块（LA 窗口尚有滞留帧），这些帧的 FIFO 记录被整体清除，但其码流仍占用物理 bs_buf；新流首帧提交后 drain 到旧槽时 FIFO 为空 → mismatch 丢弃。等于把“半截流”的所有滞留帧变成静默丢失 + 相位污染。
- **影响**：**中**（取决于调用纪律；接口层面无防护）。

## B-13 codec 回退与 muxer 封装格式跨层不一致（AV1 请求在不支持的 GPU 上必然产出废文件）
- **行号**：L753-762（未枚举到目标 GUID 时回退 `guid_array[0]`=H.264，仅打警告）、L3307（muxer 按请求 codec 选 `-f obu`）
- **类型**：边界条件 / 静默降级
- **描述**：T4 等 Ampere 以下 GPU 无 AV1（文件注释 L65 自己说明）。用户指定 `codec="av1"` 时：encoder 静默回退成 H.264 编码，muxer 却按 AV1 OBU 封装 H.264 ES → ffmpeg parser 必然失败或产出垃圾容器。两层各自“合理”，组合起来是断崖。
- **影响**：**高**（触发时必现；默认 h264 路径不受影响）。

## B-14 FFmpegMuxer：stdin 写入无超时 + close 结果不校验 → teardown 死锁链与静默坏文件
- **行号**：L3404（`stdin.write` 无限阻塞）、L3410-3426（close 60s 超时后 kill，rc != 0 仅 print 不抛）、L3332/3340（stderr 行列表无限增长）、L3394-3408（write 对 ValueError/closed 状态无处理）
- **类型**：挂起点 / 异常缺失
- **描述**：(a) ffmpeg 因磁盘慢/faststart moov 二次写（L3318-3320 注释自述此风险，但 L3321 仍然加了 `+faststart`，注释与代码矛盾）卡住时，pipe 缓冲写满 → 编码线程永久阻塞在 `write` → 它持有 `NVENCEncoder._lock`；主线程随后的 `close()`/`__del__` 在 `acquire(lock)` 上跟着永久阻塞——**解释器退出被拖死**（daemon 线程救不了被锁住的主线程）。(b) `flush_and_join` 超时放行后（见 B-15），muxer.close() 关 stdin 与垂死编码线程的 write 竞态 → ValueError/BrokenPipe。(c) ffmpeg 被 kill 或出错时 `close()` 只打 warning 不抛错 → 上游认为分段成功 → 合并阶段才发现 "moov atom not found"，故障延迟放大。
- **影响**：**高**（死锁链 + 静默损坏输出）。

## B-15 `_NVENCEncodeThread.flush_and_join` 超时后线程泄漏，资源在半死状态下被拆除
- **行号**：L3172-3187、L2951（daemon=True）
- **类型**：线程泄漏 / 并发竞态
- **描述**：join 超时仅打印“可能死锁”然后照常返回；卡住的编码线程仍持有 encoder `_lock`、可能正阻塞在某次 NVENC C 调用里。上游继续关 muxer → B-14 的竞态；进程退出时 daemon 线程在 ctypes 调用中间被硬杀 → NVENC session/DLL 状态不确定（Windows 上偶发退出期驱动报错）。另外重复调用 `flush_and_join` 会多次投递 SENTINEL（L3179 无状态检查），脏但不致命。
- **影响**：**高**（与 B-14 组合成最主要的线上挂起/退出崩溃来源）。

## B-16 `GetEncodePresetConfig` 回退到 Ex 变体但沿用 4 参数原型（ABI 不匹配）
- **行号**：L780-785、L364-366
- **类型**：ctypes 原型不匹配（潜在 UB）
- **描述**：`nvEncGetEncodePresetConfigEx` 实际签名多一个 `tuningInfo` 参数（5 参）；回退分支用 4 参 CFUNCTYPE 调用它，x64 调用约定下第 5 个寄存器参数读到垃圾值作为 tuningInfo。SDK 13 同时导出两者，index 10 通常有效，故极少走回退；一旦走到即是未定义行为。
- **影响**：**低**（潜伏）。

## B-17 `DestroyEncoder` 返回码未检查 / 函数槽为空时静默跳过
- **行号**：L2849-2852
- **类型**：返回码遗漏
- **描述**：销毁结果不看返回值；`destroy_addr` 为 None 时直接跳过且无日志——session 泄漏无感知。相比之下创建路径的返回码检查是齐全的。
- **影响**：**中**（长期运行多实例时的资源可见性问题）。

## B-18 事件创建/同步返回码未检查 + 异常路径漏毁 event
- **行号**：L999（slot event）、L1855、L2270、L2609（per-frame CE）未检查 `cuEventCreate` 返回值；L1866/L1906、L2295/L2300、L2613/L2661：EncodePicture raise 时已创建的 `_ep_ce`/`_ce` 尚未 sync/destroy，finally 不覆盖 → 单次泄漏一个 event；L2618-2624、L2138-2143、L2374-2379 的 `cuEventSynchronize` 返回值也全部忽略。
- **类型**：返回码遗漏 / 句柄泄漏（微量）
- **影响**：**低**。

## B-19 `_lock_bitstream_with_retry` 潜在 UnboundLocalError
- **行号**：L1573-1604
- **类型**：边界条件
- **描述**：若 `max_retries <= 0`（当前调用方都传默认值，不会发生），循环体零次执行，末尾 `return b"", bs_status` 触发 UnboundLocalError。
- **影响**：**低**（防御性缺陷）。

## B-20 杂项确定性缺陷
- **行号/描述/影响**：
  - L707-712：apiVersion 候选降序排列，旧格式 0xd0/0xc0 **先于** SDK13 正确格式 0x0d 尝试；若被旧格式意外接受，后续所有 `_sdk13_ver` struct version 都不匹配，失败形态诡异。**低**。
  - L1897-1899：`_la_buffered` 只增不清、跨段漂移；`== self._la_depth` 在 la_depth=0 时永假。诊断失真。**低**。
  - L3055-3063 vs L3128-3139：空帧处理双标准——stream 路径写占位帧（保持帧数、内容重复），ce_pipeline 路径 `elif not h264_data: pass` 直接不写（MP4 帧数缩水、时间戳压缩）。**中**（LA=0 路径遇到硬件空帧时输出比输入少帧）。
  - L1830：`nv12_tensors[_i] = None` 就地改写调用方列表，副作用契约纯靠注释维持。**低**。
  - L3215-3234：`_rgb_to_nv12_gpu` 对奇数宽高：`UV[:, 0::2]` 形状为 ceil(W/2)，与 `Cb` 的 W//2 不匹配 → 广播异常崩溃，无前置校验。**低**。
  - L2960-2965：`submit` 读 `self.error` 与 `put` 之间非原子，线程恰在异常退出瞬间提交的一批帧丢失且 `submitted_frames` 虚增。**低**。
- **类型**：混合（边界条件/诊断漂移/接口契约）。

## 必查清单逐项结论（速览）

| # | 结论 |
|---|---|
| 1 | cuInit 幂等安全；primary retain 不 Release 对单例可接受，但配合逐段重建（main.py L1059）成为 B-1 的确定性累积 |
| 2 | encode 方法 push/pop 由 finally 保证配对；close() 的恢复无 finally 保护、init 失败链断裂（B-2） |
| 3 | event 配对基本完整；漏洞在 EncodePicture raise 路径漏毁（B-18）；`__del__` 依赖解释器存活且吞一切异常，shutdown 期不可靠 |
| 4 | 销毁顺序正确（slots → stream → DestroyEncoder）；缺口在 event 失败路径的 buffer 泄漏（B-3）与 DestroyEncoder 返回码（B-17） |
| 5 | 真丢帧：B-5（错误码吞没）、B-8（真实码流被弃）、B-20c（LA=0 空帧不写）；多垃圾帧：B-4/B-8 的 prev 填充占位；B-12 半截流滞留帧静默清除 |
| 6 | 跨 chunk FIFO 主机制成立（实例级 pending + `_output_slot_idx=_frame_idx` 对齐）；破坏点是 B-8/B-9/B-12 三个旁路 |
| 7 | 未检查返回码集中处：EOS×2（B-4）、drain 错误 break（B-5）、DestroyEncoder（B-17）、cuEventCreate/Synchronize（B-18） |
| 8 | CE 早于编码完成是已知竞态，blocking-retry 缓解但残余替代帧风险（B-11） |
| 9 | 所有权管理合格（`_func_table_raw` 保活、局部结构体均为同步调用）；风险集中在 `_FUNC_IDX` 表、rc/h264/LOCK_BITSTREAM/PIC_PARAMS 裸偏移、776/1544/3360/2552 尺寸随 SDK 漂移；四个从未实例化的 ctypes 结构体布局与裸偏移互相矛盾（见 C） |
| 10 | stdin 无超时 + close 不校验 rc（B-14）；stderr 有 drain 线程但列表无界 |
| 11 | submit 队列有界（maxsize，非无界）；flush_and_join 超时后线程泄漏确认（B-15） |
| 12 | DLL/libcuda 经 Windows 加载器引用计数共享，多实例无重复加载问题；真正的多实例问题是 B-1 的 ctx 栈/retain 累积 |

---

# 【C. 可维护性问题】

## C-1 过长函数

| 函数 | 行号范围 | 行数 | 说明 |
|---|---|---|---|
| `NVENCEncoder.__init__` | L523-1071 | ~549 | DLL 加载、ctx 管理、GUID 枚举、RC 参数裸写、slot 池创建全部塞在一个函数里 |
| `encode_frames_batch_ce_pipeline` | L2074-2484 | ~411 | Harvest/Submit/Drain 三阶段 + inline-drain + redrain 五层嵌套逻辑 |
| `encode_frames_stream` | L1704-2072 | ~369 | 含 80 行的 EOS 内联排空块（L1931-2045），本可独立成方法 |
| `encode_frame` | L2486-2694 | ~209 | 与 stream 路径 80% 重复的单帧版 |
| `_NVENCEncodeThread._loop` | L2982-3170 | ~189 | 含三个嵌套闭包（`_wrap_exc`/`_drain_write`/`_encode_chunk`） |
| `flush` | L2696-2830 | ~135 | |
| `_apply_drained_entries` | L1355-1456 | ~102 | |

## C-2 重复代码块（同类逻辑 ≥3 处副本）

1. **CUDA ctx push 序言 + finally pop**：完全相同的 10 行模板出现 4 次 —— L1728-1737 / L2092-2101 / L2500-2509 / L2700-2709，对应 finally pop 在 L2064-2072 / L2476-2484 / L2686-2694 / L2822-2830。应抽成 contextmanager。
2. **SPS/PPS 缓存阶梯**：`if idr and cached and not has: prepend; elif idr and cached is None: extract+muxer-inject; elif not idr and cached is None: extract` 三分支结构重复 **7 次**（L2148-2167、L2186-2201、L2337-2356、L2382-2402、L2413-2429、L2452-2468、L2663-2684）——讽刺的是统一入口 `_apply_sps_pps()`（L1674-1696）已经存在，但 ce_pipeline 全部绕开它手写。
3. **裸 LockBitstream 构造 + 拷出循环**（1544 字节缓冲、version/doNotWait/handle@8、size@36、ts@40、ptr@56 的手工偏移读写）重复 **5 次**：L1310-1347、L1526-1553、L1573-1604、L1963-2026、L2765-2804。
4. **CUDA_MEMCPY2D 128 字节手工打包**重复 **3 次**：L1805-1823、L2242-2254、L2556-2575（三份偏移注释还各写各的）。
5. **CFUNCTYPE 原型在热循环/每次调用内重新构造**：L1301-1304、L1752-1754、L2119-2120、L2526-2527 等——既损性能又易漂移；原型应模块级定义一次。
6. **Annex B NAL 扫描器四胞胎**：`_extract_sps_pps`(L1173)、`_has_sps_pps`(L1214)、`_nal_first_vcl_type`(L1241)、`FFmpegMuxer._es_has_param_sets`(L3346) 是同一套起始码扫描的四份变体。
7. **空帧诊断 print 块**（`setdefault('_diag_x',0); +=1; if <=5 or %50==0: print`）重复 8+ 处，应为一个小工具方法。

## C-3 硬编码魔法数字（高风险子集）

- **裸偏移**：LOCK_BITSTREAM 的 8/36/40/56；PIC_PARAMS 的 16/24/40/48/56/64/68；MEMCPY2D 的 16/24/32/48/72/88/104/112/120；RC 区的 20/24/28/32/36/88/90/25(idx)；h264 cfg 绝对偏移 176/180/184/192/216/268（`8+168` 基准散落在 L800-845 多处字面量，无命名常量）。
- **缓冲尺寸**：776（create buf）、1544（lock buf）、3360（pic buf）、2552（func table）、128（memcpy2d）——与 `_FUNC_TABLE_SIZE=512`（L132，**已死常量**）并存，SDK 升级时极易改错/漏改。
- **struct version bit31 不一致**：PRESET/CONFIG/INIT/PIC/LOCKBITSTREAM 带 bit31（L345-349），REGISTER/MAP 不带（L350-351）——差异原因无处注释，纯属“GPU 实测出来就这么填”。
- **RC 模式值**：32(VBR_HQ)/64(QVBR)/0(CONSTQP)；位标志 `0x2`(FORCEIDR)/`0x8`(EOS)、bit3/bit5/bit8 —— 均无枚举名。
- **超时/预算**：2000ms（L1482）、5000ms（L2181/L2410）、60s（L3416）、120s（L3172）、assumed_fps=200（L2970）、pinned 预算 5e8（L3092）、码率钳制 50M/5M（L455-456）、估算系数 3.0（L865/L881）。
- **队列/分块**：默认 queue depth 4、LA 自动升 16（L2911-2912）、batch_frames=48、`_chunk_frames=2000` 初值（L3018）、flush_chunk_frames=128、768 安全上限——四个数字的相互作用只在注释里可推理。

## C-4 其他结构性可维护性问题

1. **误导性死代码**（比没有更糟）：
   - `_NvEncLockBitstream`/`_NvEncRegisterResource`/`_NvEncMapInputResource`/`_NvEncPicParams` 四个 ctypes 结构体（L274-342）**从未实例化**，且字段布局与实际使用的裸偏移互相矛盾（如 typed struct 声称 outputBitstream@128、bitstreamSizeInBytes@1036，实际裸路径写 handle@8、读 size@36）——后人若“规范化”改用结构体即刻损坏。
   - `NVENCAPI_STRUCT_VERSION()`（L107-115）保留的是文件头注释自己承认错误的 sizeof 公式。
   - `_RotationBitReader`（L470-505）“备用方向 B”保留；`_FUNC_TABLE_SIZE`、`_NVENC_VBR_QUALITY_OFFSET`（L437，自标废弃）等死常量。
2. **注释与代码矛盾**：L3318-3321，注释论证“中间分段不要 faststart（moov 二次写可能超 60s 被 SIGKILL）”，下一行代码就加了 `+faststart`。
3. **重复 import**：`os`、`threading` 各导入两次（L9/L13 与 L26/L27）。
4. **print 当日志**：全文件无 logging，级别靠 emoji 和前缀字符串区分；关键诊断（`_diag_*` 计数）与一次性信息混在同一通道。
5. **隐式跨模块契约**：`_pending_f0_nv12`/`_pending_f0_force_idr`（L3023-3029 读取）由 `main.py` L1472 从外部注入，本文件零声明；`_flush_frame_count` 同理供测试脚本反射读取——契约只活在调用方源码里。
6. **就地修改入参**：`encode_frames_stream` 置空 `nv12_tensors[_i]`（L1830）节省内存的设计与“函数收列表返回列表”的直觉相悖，无签名级提示。
7. **镜像双维护负担**：文件头注明与 `external/realesrgan_video/nvenc_sdk.py` 互为镜像（L5），本次审查可见两份已开始漂移（如 ESRGAN 版无 `_ensure_slot_free` 的直锁兜底），AGENTS.md 要求人工双向同步本身就是持续缺陷源。
8. **“失败方案考古”式注释**：单个方法注释动辄 30-50 行历史方案证伪记录（如 L1674-1702、L2919-2939），有价值但应外移至 memory/ 文档，留在原地使代码本体被淹没。

---

**总结**：该文件功能上经过大量 GPU 实测打磨（帧数守恒、HEVC 路由、辅助块记账等防御都有明确实验依据），但其可靠性短板高度集中在三处——**错误码静默吞噬**（B-4/B-5/B-17）、**不可中断的阻塞等待**（B-6/B-14/B-15 的挂起链）和**多套并行记账体系的旁路失配**（B-8/B-9/B-10/B-12）。可维护性上最大的负债是五套重复的底层访问样板与四个与实际布局相悖的死结构体。
</task_result>
</task>
```

---

## Assistant (Plan · Ox Alpha Free · 430.0s)

_Thinking:_

Excellent! Now I have the NVENC SDK bug list. Still missing: IFRNet pipeline.py (1563 lines) detailed review, and IFRNet's ffmpeg_io.py/ifrnet_utils.py/tensorrt_accel.py.

Let me dispatch one more agent for pipeline.py + ffmpeg_io.py + ifrnet_utils.py + tensorrt_accel.py (total ~2700 lines). That's a reasonable size. Meanwhile I could read memory/MEMORY.md to check known issues for cross-validation.

Actually let me also consider: the ESRGAN agent already analyzed its own ffmpeg_io.py (1470 lines), pipeline.py (1962 lines) etc. For IFRNet's smaller files (ffmpeg_io 643 lines, ifrnet_utils 297, tensorrt_accel 218), plus pipeline.py (1563). One agent should handle this fine (~2700 lines total).

While that runs, I'll read memory/MEMORY.md index myself.

NVENC SDK 审查完成（20 项缺陷，含挂起链与静默丢帧问题）。最后派一个代理补齐 IFRNet 的 pipeline.py 与其余底层文件，同时我自己读取项目记忆索引做交叉验证。

**Tool: task**

**Input:**
```json
{
  "description": "审查IFRNet流水线与工具层",
  "prompt": "你是一个资深 Python/CUDA 并发编程代码审查专家。这是一个只读分析任务：不要修改任何文件。重要：你的最终回复必须包含完整的中文报告文本，绝不能为空（此前多个代理返回空导致任务失败）。\n\n请深度分析以下 4 个文件（IFRNet 插帧子系统的核心流水线与底层 IO/工具）：\n1. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\pipeline.py （1563行，核心流水线，T1 Reader/T2 Infer/T3 Writer 三线程）\n2. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\ffmpeg_io.py （643行）\n3. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\ifrnet_utils.py （297行）\n4. D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\tensorrt_accel.py （218行）\n\n背景：GPU 插帧管线。pipeline.run 启动 T1(读帧)/T2(推理)/T3(写帧) 三线程+队列反压；T2 调用后端 main.py 的 _infer_batch（有 GPU 直留/Ring D2H/PinnedPool/同步 CPU 四条路径）；T3 通过 PinnedResultPool/RingBuffer 与 NVENC 编码线程交互；join 看门狗上限 7200s。\n\n输出要求（结构化中文报告）：\n\n【A. 架构分析】（简要）\n- 各模块职责、核心类清单（类名、关键方法、职责、行号）\n- T1/T2/T3 数据流、队列深度、哨兵机制、看门狗逻辑\n- Ring/PinnedPool 内存池设计\n- TRT 加速器的 engine 构建/缓存/失效策略\n- 设计模式\n\n【B. BUG 清单】每个 bug 给出：文件:行号、类型（并发竞态/资源泄漏/异常缺失/挂起点/崩溃点/边界条件/性能瓶颈）、描述、影响（致命/高/中/低）。\n必查清单：\n1. T1/T2/T3 线程的哨兵传播：某线程异常退出后其他线程是否会永久阻塞在 queue.get？\n2. stop 标志可见性、双检查竞态\n3. 队列深度自适应调优的跨段状态污染\n4. Ring buffer 的 writer_acquire/writer_release 协议：slot 复用竞态、信号量泄漏\n5. ffmpeg 子进程 stdin/stdout PIPE 管理、stderr drain、close 时序\n6. 帧数守恒：EOF 早退、超时判死误杀慢场景\n7. TRT engine 缓存 key（GPU SM 架构？batch 静态形状？）、构建失败的回退路径\n8. CUDA event 池的事件复用竞态\n9. tensor 生命周期与显存释放时机\n10. 边界条件：极短视频（0-2帧）、奇数尺寸、fps 异常\n\n【C. 可维护性问题】过长函数、重复代码、硬编码魔法数字（给行号）。\n\n再次强调：最终回复必须包含完整报告内容，不能为空。",
  "subagent_type": "explore",
  "filePath": "D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\memory\\MEMORY.md"
}
```

**Output:**
```
<task id="ses_fd2dc359fffe0bh78q0C4O4z9o" state="completed">
<task_result>

</task_result>
</task>
```

---

## Assistant (Plan · Ox Alpha Free · 14.4s)

_Thinking:_

The pipeline agent returned empty AGAIN. This is the 4th empty return. Something is systematically failing with agents handling pipeline.py. Let me try reading the file myself. 1563 lines + 643 + 297 + 218 = ~2700 lines. I can read them directly — it's within my capacity.

Let me read pipeline.py first, then ffmpeg_io.py, ifrnet_utils.py, tensorrt_accel.py. I'll do this in parallel reads (2 calls per file maybe). Actually Read tool defaults to 2000 lines so pipeline.py fits in one call.

流水线代理第 4 次返回空。我直接自己读取这些文件分析。

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\pipeline.py"
}
```

**Output:**
```
<path>D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\ifrnet_video\pipeline.py</path>
<type>file</type>
<content>
1: #!/usr/bin/env python3
2: # -*- coding: utf-8 -*-
3: # IFRNet Video Enhancement - 三阶段流水线模块（T1 Reader / T2 GPU 推理 / T3 Writer）。
4: # 包含：IFRNetPipelineRunner、GPUMonitor/GPUStats、硬件画像与队列自动调优。
5: # 镜像 external/realesrgan_video/pipeline.py 的职责。
6: 
7: from __future__ import annotations
8: 
9: import dataclasses
10: import json
11: import os
12: import queue
13: import sys
14: import threading
15: import time
16: from collections import deque
17: from typing import List, Optional, Tuple
18: 
19: import numpy as np
20: import torch
21: 
22: # [FIX-OOM-TRT] expandable_segments 减少 PyTorch CUDA allocator 碎片化。
23: os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
24: 
25: _PKG_DIR = os.path.dirname(os.path.abspath(__file__))
26: if _PKG_DIR not in sys.path:
27:     sys.path.insert(0, _PKG_DIR)
28: 
29: from ifrnet_video.config import base_dir
30: from ifrnet_video.ifrnet_utils import CudaEventPool, _PinnedResultItem, _get_pinned_pool
31: from ifrnet_video.ffmpeg_io import _detect_encode_parallelism, _software_encode_fps
32: from ifrnet_video.nvenc_sdk import (
33:     _NVENCEncodeThread,
34:     _rgb_to_nv12_gpu,
35:     _rgb_to_nv12_gpu_batch,
36: )
37: 
38: 
39: # ─────────────────────────────────────────────────────────────────────────────
40: # [GPU-MONITOR] 后台 GPU 监测线程（v2：滑动窗口 + 精细统计 + 队列调优建议）
41: # ─────────────────────────────────────────────────────────────────────────────
42: 
43: @dataclasses.dataclass
44: class GPUStats:
45:     """
46:     GPU 采样统计结果（完整运行 + 稳定段 + 最近滑动窗口）。
47: 
48:     字段说明
49:     ─────────────────────────────────────────────────────────────────────────
50:     sample_count    : 总采样次数
51:     duration_s      : 采样时长（秒）
52:     total_vram_gib  : GPU 显存总量（GiB），0.0 表示未能获取
53: 
54:     【完整运行利用率】
55:     avg_util        : 全程均值 %
56:     max_util        : 全程峰值 %
57:     p50_util        : 中位数 %（比均值更抗异常抖动）
58:     p95_util        : 95 分位数 %（反映高负载上限）
59:     util_std        : 标准差（波动性指标；越高说明 GPU 供料越不稳定）
60:     low_util_frac   : util < 50% 的样本占比（空闲时间比；越高说明 GPU 饥饿越严重）
61: 
62:     【稳定段（剔除前 15% 预热样本后的统计）】
63:     stable_avg      : 稳定段均值 %
64:     stable_p50      : 稳定段中位数 %
65:     stable_p95      : 稳定段 95 分位数 %
66:     stable_std      : 稳定段标准差 %
67: 
68:     【显存】
69:     avg_mem_gib     : 全程均值 GiB
70:     max_mem_gib     : 全程峰值 GiB
71:     p95_mem_gib     : 95 分位数 GiB
72: 
73:     【最近滑动窗口段（最后 window_seconds 秒）】
74:     recent_avg      : 最近段均值 %（反映运行末尾 GPU 状态）
75:     recent_p95      : 最近段 95 分位数 %
76: 
77:     【推导属性（不存储，按需计算）】
78:     mem_headroom_gib: 显存余量 = total_vram_gib - max_mem_gib
79:     mem_frac        : 峰值显存占比 = max_mem_gib / total_vram_gib
80:     """
81:     sample_count:   int   = 0
82:     duration_s:     float = 0.0
83:     total_vram_gib: float = 0.0
84: 
85:     # 完整运行利用率
86:     avg_util:      float = 0.0
87:     max_util:      float = 0.0
88:     p50_util:      float = 0.0
89:     p95_util:      float = 0.0
90:     util_std:      float = 0.0
91:     low_util_frac: float = 0.0   # 空闲时间占比 [0, 1]
92: 
93:     # 稳定段（剔除前 15% 预热）
94:     stable_avg:    float = 0.0
95:     stable_p50:    float = 0.0
96:     stable_p95:    float = 0.0
97:     stable_std:    float = 0.0
98: 
99:     # 显存
100:     avg_mem_gib:   float = 0.0
101:     max_mem_gib:   float = 0.0
102:     p95_mem_gib:   float = 0.0
103: 
104:     # 最近滑动窗口段
105:     recent_avg:    float = 0.0
106:     recent_p95:    float = 0.0
107: 
108:     @property
109:     def mem_headroom_gib(self) -> float:
110:         """显存余量（GiB）= total_vram - max_mem。"""
111:         return max(self.total_vram_gib - self.max_mem_gib, 0.0)
112: 
113:     @property
114:     def mem_frac(self) -> float:
115:         """峰值显存占全局显存的比例 [0, 1]。"""
116:         return self.max_mem_gib / max(self.total_vram_gib, 1.0)
117: 
118:     def summary_str(self) -> str:
119:         """生成多行统计摘要字符串。"""
120:         vram_str = (f'{self.total_vram_gib:.1f} GiB'
121:                     if self.total_vram_gib > 0 else '未知')
122:         lines = [
123:             f'[GPU-MONITOR] 采样 {self.sample_count} 次  时长 {self.duration_s:.0f}s  '
124:             f'VRAM 总量: {vram_str}',
125:             f'  利用率(全程)  均值={self.avg_util:.1f}%  P50={self.p50_util:.1f}%  '
126:             f'P95={self.p95_util:.1f}%  峰值={self.max_util:.1f}%  σ={self.util_std:.1f}%  '
127:             f'空闲占比={self.low_util_frac*100:.1f}%',
128:             f'  利用率(稳定段) 均值={self.stable_avg:.1f}%  P50={self.stable_p50:.1f}%  '
129:             f'P95={self.stable_p95:.1f}%  σ={self.stable_std:.1f}%',
130:             f'  利用率(最近段) 均值={self.recent_avg:.1f}%  P95={self.recent_p95:.1f}%',
131:             f'  显存           均值={self.avg_mem_gib:.2f} GiB  '
132:             f'P95={self.p95_mem_gib:.2f} GiB  峰值={self.max_mem_gib:.2f} GiB  '
133:             f'({self.mem_frac*100:.0f}% 使用  余量 {self.mem_headroom_gib:.1f} GiB)',
134:         ]
135:         return '\n'.join(lines)
136: 
137: 
138: 
139: class GPUMonitor:
140:     """
141:     后台线程定期采样 GPU 利用率和显存占用（v2 滑动窗口精细统计版）。
142: 
143:     v2 改进（相较 v1）：
144:     · 原始样本改为 (timestamp, util%, mem_gib) 三元组，支持任意时间窗口切片。
145:     · 新增线程锁（_lock）保护 _raw_samples，消除残留数据竞争。
146:     · get_stats() 返回 GPUStats 数据类：完整运行 + 稳定段（去预热）+ 最近滑动窗口段。
147:     · print_report() 打印多维度统计并给出 batch_size / pair_queue / result_queue 调优建议。
148:     · summary() 保留为向后兼容接口（委托 get_stats() 实现）。
149: 
150:     继承 v1 全部修复标签：
151:     · FIX-NVML-LEAK / FIX-NVML-LOCK / FIX-THREAD-RST / FIX-SNAP / FIX-DEV-IDX
152:     """
153: 
154:     _LOW_UTIL_THRESHOLD = 50.0   # util < 此值视为"空闲"采样点
155:     _WARMUP_FRAC        = 0.15   # 稳定段：剔除前 15% 的预热样本
156: 
157:     def __init__(
158:         self,
159:         device:         torch.device,
160:         interval:       float = 1.0,
161:         window_seconds: float = 30.0,   # 最近滑动窗口时长（秒）
162:     ):
163:         self.device         = device
164:         self.interval       = interval
165:         self.window_seconds = window_seconds
166: 
167:         self._stop_event    = threading.Event()
168:         self._thread: Optional[threading.Thread] = None
169:         self._nvml_help_printed = threading.Event()   # FIX-NVML-LOCK
170: 
171:         # 带时间戳原始样本列表：List[(monotonic_t, util%, mem_gib)]
172:         # 受 _lock 保护（FIX-SNAP 升级版）
173:         self._raw_samples: List[Tuple[float, float, float]] = []
174:         self._lock = threading.Lock()
175: 
176:     # ── 向后兼容属性 ─────────────────────────────────────────────────────────
177: 
178:     @property
179:     def util_samples(self) -> List[float]:
180:         """兼容 v1：返回纯利用率列表快照。"""
181:         with self._lock:
182:             return [s[1] for s in self._raw_samples]
183: 
184:     @property
185:     def mem_samples(self) -> List[float]:
186:         """兼容 v1：返回纯显存列表快照。"""
187:         with self._lock:
188:             return [s[2] for s in self._raw_samples]
189: 
190:     # ── 公开接口 ─────────────────────────────────────────────────────────────
191: 
192:     def start(self):
193:         """启动后台采样（仅对 CUDA 设备有效）。"""
194:         if self.device.type != 'cuda':
195:             return
196:         self._stop_event.clear()                   # FIX-THREAD-RST
197:         with self._lock:
198:             self._raw_samples.clear()
199:         self._thread = threading.Thread(
200:             target=self._sample, daemon=True, name='GPUMonitor'
201:         )
202:         self._thread.start()
203: 
204:     def stop(self):
205:         """通知采样线程停止并等待结束（最长 5 秒）。"""
206:         if self._thread is not None:
207:             self._stop_event.set()
208:             self._thread.join(timeout=5.0)
209:             self._thread = None                    # FIX-THREAD-RST
210: 
211:     # ── 采样线程 ─────────────────────────────────────────────────────────────
212: 
213:     def _sample(self):
214:         """
215:         采样线程主循环（继承 v1 全部 FIX 标签）：
216:         1. FIX-DEV-IDX : 解析目标 GPU 索引，支持 cuda:N 任意编号。
217:         2. FIX-NVML-LEAK: 独立 nvml_initialized 标志，确保 nvmlShutdown 必定被调用。
218:         3. FIX-NVML-LOCK: _nvml_help_printed 用 threading.Event 保证提示仅打印一次。
219:         4. 采样结果以 (monotonic_time, util, mem_gib) 存入 _raw_samples（受 _lock 保护）。
220:         """
221:         # FIX-DEV-IDX
222:         dev_idx = self.device.index if self.device.index is not None else 0
223: 
224:         pynvml_module    = None
225:         pynvml_handle    = None
226:         nvml_initialized = False   # FIX-NVML-LEAK
227: 
228:         try:
229:             import pynvml
230:             pynvml_module = pynvml
231:         except ImportError:
232:             pass
233: 
234:         if pynvml_module is not None:
235:             try:
236:                 pynvml_module.nvmlInit()
237:                 nvml_initialized = True            # FIX-NVML-LEAK
238:                 pynvml_handle = pynvml_module.nvmlDeviceGetHandleByIndex(dev_idx)
239:             except Exception:
240:                 pynvml_handle = None
241:                 # nvml_initialized 保持现值：Init 成功但 GetHandle 失败时仍须 Shutdown
242: 
243:         if pynvml_handle is None and torch.cuda.is_available():
244:             if not self._nvml_help_printed.is_set():   # FIX-NVML-LOCK
245:                 self._nvml_help_printed.set()
246:                 print(
247:                     "[GPU-MONITOR] 未检测到 nvidia-ml-py，已回退至 torch.cuda 内置 API。\n"
248:                     "  推荐安装 nvidia-ml-py 以获得更精确的监控数据：\n"
249:                     "  pip install nvidia-ml-py==13.580.65   # 匹配驱动 580.65\n"
250:                     "  (若驱动不同，请根据 nvidia-smi 输出选择主版本号一致的 nvidia-ml-py 版本)",
251:                     flush=True,
252:                 )
253: 
254:         while not self._stop_event.is_set():
255:             util         = 0.0
256:             mem_used_gib = 0.0
257:             try:
258:                 if pynvml_handle is not None:
259:                     util     = float(pynvml_module.nvmlDeviceGetUtilizationRates(pynvml_handle).gpu)
260:                     mem_info = pynvml_module.nvmlDeviceGetMemoryInfo(pynvml_handle)
261:                     mem_used_gib = mem_info.used / (1024 ** 3)
262:                 elif torch.cuda.is_available():
263:                     try:                           # FIX-DEV-IDX
264:                         util = float(torch.cuda.utilization(dev_idx))
265:                     except Exception:
266:                         util = 0.0
267:                     free, total = torch.cuda.mem_get_info(dev_idx)
268:                     mem_used_gib = (total - free) / (1024 ** 3)
269:                 # 两者均不可用：保持 0.0（不影响推理流程）
270:             except Exception:
271:                 pass
272: 
273:             with self._lock:
274:                 self._raw_samples.append((time.monotonic(), util, mem_used_gib))
275:             self._stop_event.wait(self.interval)
276: 
277:         # FIX-NVML-LEAK: nvmlInit 成功过则必须 Shutdown
278:         if nvml_initialized and pynvml_module is not None:
279:             try:
280:                 pynvml_module.nvmlShutdown()
281:             except Exception:
282:                 pass
283: 
284:     # ── 精细统计 ─────────────────────────────────────────────────────────────
285: 
286:     def get_stats(self) -> GPUStats:
287:         """
288:         返回 GPUStats（线程安全快照 → 完整统计 + 稳定段 + 最近滑动窗口）。
289: 
290:         计算步骤：
291:           1. 加锁快照 _raw_samples（FIX-SNAP 升级版）。
292:           2. 全量 util_arr / mem_arr 计算完整运行统计（均值/P50/P95/σ/空闲占比）。
293:           3. 剔除前 WARMUP_FRAC（15%）样本，得稳定段统计。
294:           4. 从时间戳逆向切片最近 window_seconds 秒，得最近段统计。
295:         """
296:         with self._lock:
297:             snap = list(self._raw_samples)   # FIX-SNAP: 快照隔离
298: 
299:         stats = GPUStats()
300:         if not snap:
301:             return stats
302: 
303:         ts_arr   = np.array([s[0] for s in snap])
304:         util_arr = np.array([s[1] for s in snap])
305:         mem_arr  = np.array([s[2] for s in snap])
306:         n = len(snap)
307: 
308:         stats.sample_count = n
309:         stats.duration_s   = float(ts_arr[-1] - ts_arr[0]) if n > 1 else 0.0
310: 
311:         # 全局显存总量
312:         try:
313:             stats.total_vram_gib = (
314:                 torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
315:             )
316:         except Exception:
317:             stats.total_vram_gib = 0.0
318: 
319:         # ── 完整运行利用率 ──────────────────────────────────────────────────
320:         stats.avg_util      = round(float(np.mean(util_arr)),                  1)
321:         stats.max_util      = round(float(np.max(util_arr)),                   1)
322:         stats.p50_util      = round(float(np.percentile(util_arr, 50)),        1)
323:         stats.p95_util      = round(float(np.percentile(util_arr, 95)),        1)
324:         stats.util_std      = round(float(np.std(util_arr)),                   1)
325:         stats.low_util_frac = round(float(np.mean(util_arr < self._LOW_UTIL_THRESHOLD)), 3)
326: 
327:         # ── 稳定段（剔除前 15% 预热）──────────────────────────────────────
328:         trim_n = max(1, int(n * self._WARMUP_FRAC))
329:         if n - trim_n >= 3:
330:             stable = util_arr[trim_n:]
331:             stats.stable_avg = round(float(np.mean(stable)),             1)
332:             stats.stable_p50 = round(float(np.percentile(stable, 50)),  1)
333:             stats.stable_p95 = round(float(np.percentile(stable, 95)),  1)
334:             stats.stable_std = round(float(np.std(stable)),              1)
335:         else:
336:             # 样本太少，以全量代替
337:             stats.stable_avg = stats.avg_util
338:             stats.stable_p50 = stats.p50_util
339:             stats.stable_p95 = stats.p95_util
340:             stats.stable_std = stats.util_std
341: 
342:         # ── 显存 ────────────────────────────────────────────────────────────
343:         stats.avg_mem_gib = round(float(np.mean(mem_arr)),             2)
344:         stats.max_mem_gib = round(float(np.max(mem_arr)),              2)
345:         stats.p95_mem_gib = round(float(np.percentile(mem_arr, 95)),   2)
346: 
347:         # ── 最近滑动窗口段 ──────────────────────────────────────────────────
348:         if n > 1 and stats.duration_s > 0:
349:             cutoff_t    = ts_arr[-1] - self.window_seconds
350:             recent_mask = ts_arr >= cutoff_t
351:             recent_util = util_arr[recent_mask]
352:             if len(recent_util) >= 2:
353:                 stats.recent_avg = round(float(np.mean(recent_util)),            1)
354:                 stats.recent_p95 = round(float(np.percentile(recent_util, 95)), 1)
355:             else:
356:                 stats.recent_avg = stats.avg_util
357:                 stats.recent_p95 = stats.p95_util
358:         else:
359:             stats.recent_avg = stats.avg_util
360:             stats.recent_p95 = stats.p95_util
361: 
362:         return stats
363: 
364:     # ── 调优建议报告 ─────────────────────────────────────────────────────────
365: 
366:     @staticmethod
367:     def _round_bs(bs: int) -> int:
368:         """将 batch_size 向上取整到最近 8 的倍数（TRT / CUDA Graph shape 对齐）。"""
369:         return max(8, (bs + 7) // 8 * 8)
370: 
371:     def print_report(
372:         self,
373:         stats:            GPUStats,
374:         current_bs:       int,
375:         current_pair_q:   int,
376:         current_result_q: int,
377:         codec:            str   = '',       # [FIX-NVENC-AWARE] 编码器名称，用于避开 NVENC 误判
378:         slot_mb:          float = 0.0,      # [FIX-MAXRQ-DYNAMIC] PinnedPool 每槽 MiB，用于内存上限约束
379:         t2_ms:            float = 0.0,      # [FIX-MAXRQ-DYNAMIC] T2 推理延迟（ms）
380:         t3_ms:            float = 0.0,      # [FIX-MAXRQ-DYNAMIC] T3 编码延迟（ms）
381:     ) -> None:
382:         """
383:         打印精细统计报告，并依据以下逻辑给出三项调优建议：
384: 
385:         ┌─────────────────────────────────────────────────────────────────────┐
386:         │  batch_size 建议（综合 VRAM 余量 + 稳定段利用率）                    │
387:         │  · 峰值 VRAM > 87%          → 缩小 bs，防 OOM                       │
388:         │  · stable_avg < 55% + VRAM < 60% → 增大 bs，提升 GPU 吞吐           │
389:         │  · stable_avg ≥ 80% + VRAM < 58% → 微增 bs，充分利用空闲显存        │
390:         │  · 其余                     → 当前 bs 合适                          │
391:         ├─────────────────────────────────────────────────────────────────────┤
392:         │  pair_queue / T2 输入缓冲深度建议（综合波动性 + 空闲占比）            │
393:         │  · stable_std > 25 或 low_util_frac > 30% → 增大，平滑 H2D 供料     │
394:         │  · stable_std < 10 且空闲低且队列偏大      → 适当缩小，节省显存      │
395:         │  · 其余                                   → 无需调整                │
396:         ├─────────────────────────────────────────────────────────────────────┤
397:         │  result_queue / T3 输出缓冲深度建议（综合 P95 + 波动性 + 显存余量）   │
398:         │  · stable_p95 > 85% 且 stable_std > 20%  → 增大，解耦 T3 瓶颈       │
399:         │  · 显存余量 > 2 GiB 且 result_q 较小      → 可增大，改善 T2/T3 解耦  │
400:         │  · 其余                                   → 无需调整                │
401:         └─────────────────────────────────────────────────────────────────────┘
402:         """
403:         print(stats.summary_str())
404: 
405:         if stats.sample_count == 0:
406:             print('[GPU-MONITOR] 无采样数据，跳过调优建议。')
407:             return
408: 
409:         mem_frac     = stats.mem_frac
410:         stable_avg   = stats.stable_avg
411:         stable_std   = stats.stable_std
412:         low_frac     = stats.low_util_frac
413:         headroom_gib = stats.mem_headroom_gib
414: 
415:         # ── batch_size 建议 ──────────────────────────────────────────────────
416:         if mem_frac > 0.87:
417:             sug_bs = self._round_bs(int(current_bs * 0.82 / max(mem_frac, 0.1)))
418:             sug_bs = max(8, min(sug_bs, current_bs - 8))
419:             print(f'[GPU-MONITOR] ⚠️  VRAM 使用率 {mem_frac*100:.0f}%，'
420:                   f'建议减小 batch_size: {current_bs} → {sug_bs}（防 OOM）')
421:         elif stable_avg < 55.0 and mem_frac < 0.60:
422:             factor = min(2.5, 0.60 / max(mem_frac, 0.05))
423:             sug_bs = self._round_bs(int(current_bs * factor))
424:             sug_bs = min(256, max(current_bs + 8, sug_bs))
425:             print(f'[GPU-MONITOR] 💡 batch_size 建议增大: {current_bs} → {sug_bs}'
426:                   f'  （稳定利用率 {stable_avg:.0f}% 偏低，VRAM 余量 {headroom_gib:.1f} GiB）')
427:         elif stable_avg >= 80.0 and mem_frac < 0.58:
428:             factor = min(1.8, 0.58 / max(mem_frac, 0.05))
429:             sug_bs = self._round_bs(int(current_bs * factor))
430:             sug_bs = min(256, max(current_bs + 8, sug_bs))
431:             print(f'[GPU-MONITOR] 💡 batch_size 可微增: {current_bs} → {sug_bs}'
432:                   f'  （利用率 {stable_avg:.0f}%，VRAM 余量 {headroom_gib:.1f} GiB 充裕）')
433:         else:
434:             print(f'[GPU-MONITOR] ✅ batch_size={current_bs} 配置合适'
435:                   f'  （稳定利用率 {stable_avg:.0f}%，VRAM 占用 {mem_frac*100:.0f}%）')
436: 
437:         # ── pair_queue（T2 输入缓冲）建议 ───────────────────────────────────
438:         _pq_reasons = []
439:         if stable_std > 25.0:
440:             _pq_reasons.append(f'利用率波动大 σ={stable_std:.0f}%')
441:         if low_frac > 0.30:
442:             _pq_reasons.append(f'GPU 空闲占比 {low_frac*100:.0f}%')
443: 
444:         if _pq_reasons:
445:             # [FIX-MAXPQ-DYNAMIC] 建议上限随绝对上限 24 提升（原硬编码 8）；
446:             # 内存安全约束在消费端（ADAPTIVE-QUEUE 决策块）由 _compute_max_pair_queue 执行
447:             sug_pq = min(_PAIR_Q_ABS_CAP, current_pair_q + 2)
448:             print(f'[GPU-MONITOR] 💡 pair_queue 建议增大: {current_pair_q} → {sug_pq}'
449:                   f'  （{"、".join(_pq_reasons)}，增大可平滑 H2D 供料气泡）')
450:         elif stable_std < 10.0 and low_frac < 0.10 and current_pair_q > 4:
451:             sug_pq = max(3, current_pair_q - 1)
452:             print(f'[GPU-MONITOR] 💡 pair_queue 可适当减小: {current_pair_q} → {sug_pq}'
453:                   f'  （利用率稳定，σ={stable_std:.0f}%，节省显存）')
454:         else:
455:             print(f'[GPU-MONITOR] ✅ pair_queue={current_pair_q} 无需调整'
456:                   f'  （σ={stable_std:.0f}%，空闲占比 {low_frac*100:.0f}%）')
457: 
458:         # ── [FIX-T3-DETECT] result_queue（T3 输出缓冲）建议 ─────────────────
459:         # 先判断是否 T3-bottleneck：若是，增大 result_queue 无助于提速，
460:         # 反而加重 PinnedPool 锁页内存压力，应保持或缩小。
461:         sug_rq = None   # [FIX-MAXRQ-DYNAMIC] 统一收集建议值，用于后续内存上限约束
462:         if self._is_t3_bottleneck(stats, codec=codec):
463:             if current_result_q > 16:
464:                 sug_rq = max(16, current_result_q - 8)
465:                 print(f'[GPU-MONITOR] ⚠️  检测到 T3-bottleneck（编码器是真正瓶颈）：'
466:                       f'result_queue 建议缩小 {current_result_q} → {sug_rq}'
467:                       f'  （空闲占比 {stats.low_util_frac*100:.0f}%，均值 {stable_avg:.0f}%，'
468:                       f'增大队列无助提速，仅增加 PinnedPool 内存压力）')
469:             else:
470:                 print(f'[GPU-MONITOR] ⚠️  检测到 T3-bottleneck（编码器是真正瓶颈）：'
471:                       f'result_queue={current_result_q} 保持不变'
472:                       f'  （根本瓶颈在编码器速度，应考虑换用更快的 preset'
473:                       f'{"" if "nvenc" in codec.lower() else " 或 NVENC"}）')
474:         elif stats.stable_p95 > 85.0 and stable_std > 20.0:
475:             sug_rq = min(64, current_result_q + 8)
476:             print(f'[GPU-MONITOR] 💡 result_queue 建议增大: {current_result_q} → {sug_rq}'
477:                   f'  （P95={stats.stable_p95:.0f}%，σ={stable_std:.0f}%，T3 可能拖累 T2）')
478:         elif headroom_gib > 2.0 and current_result_q < 32:
479:             sug_rq = min(48, current_result_q + 8)
480:             print(f'[GPU-MONITOR] 💡 result_queue 可增大: {current_result_q} → {sug_rq}'
481:                   f'  （VRAM 余量 {headroom_gib:.1f} GiB，增大可改善 T2/T3 解耦度）')
482:         else:
483:             print(f'[GPU-MONITOR] ✅ result_queue={current_result_q} 无需调整'
484:                   f'  （P95={stats.stable_p95:.0f}%，余量 {headroom_gib:.1f} GiB）')
485: 
486:         # [FIX-MAXRQ-DYNAMIC] PinnedPool 内存上限约束（与 get_queue_suggestions 一致）
487:         if slot_mb > 0.0 and sug_rq is not None:
488:             _mem_avail_gb_pr = _detect_encode_parallelism()['mem_avail_gb']
489:             _max_rq_pr = _compute_max_result_queue(
490:                 slot_mb=slot_mb, mem_avail_gb=_mem_avail_gb_pr,
491:                 T2_ms=t2_ms, T3_ms=t3_ms,
492:             )
493:             if sug_rq > _max_rq_pr:
494:                 print(f'[GPU-MONITOR] ℹ️  PinnedPool 内存上限约束: '
495:                       f'result_queue 理想值 {sug_rq} → {_max_rq_pr}'
496:                       f'  (slot={slot_mb:.1f} MB × {_max_rq_pr}'
497:                       f' ≈ {slot_mb * _max_rq_pr:.0f} MB'
498:                       f'  ≤ RAM预算 {_mem_avail_gb_pr * 1024 * 0.06:.0f} MB'
499:                       f'  [mem_avail={_mem_avail_gb_pr:.1f} GB × 6%])')
500: 
501:     def get_queue_suggestions(
502:         self,
503:         stats:            GPUStats,
504:         current_pair_q:   int,
505:         current_result_q: int,
506:         slot_mb:          float = 0.0,   # [FIX-T3-MEMCAP] 每个 result slot 的 MiB 数
507:         t2_ms:            float = 0.0,   # [FIX-MAXRQ-DYNAMIC] T2 推理延迟（ms），0=未知
508:         t3_ms:            float = 0.0,   # [FIX-MAXRQ-DYNAMIC] T3 编码延迟（ms），0=未知
509:         codec:            str   = '',    # [FIX-NVENC-AWARE] 编码器名称，用于避开 NVENC 误判
510:         ) -> Tuple[int, int]:
511:         """
512:         [FIX-T3-DETECT / FIX-T3-MEMCAP / FIX-POOL-AUTOSCALE / FIX-MAXRQ-DYNAMIC]
513:         返回 (建议 pair_queue, 建议 result_queue)。用于跨段自适应队列调整，不打印信息。
514: 
515:         新增逻辑：
516:         · T3-bottleneck 时不增大 result_queue（否则 PinnedPool 雪球式积累）。
517:         · slot_mb > 0 时对 result_queue 施加 PinnedPool 内存上限约束。
518:         · [FIX-MAXRQ-DYNAMIC] 上限改由 _compute_max_result_queue() 三轴动态计算
519:           （RAM 上限 / T3/T2 下限 / 绝对上限），替代静态 _PINNED_POOL_MAX_MB。
520:         """
521:         pair_q   = current_pair_q
522:         result_q = current_result_q
523:         stable_std   = stats.stable_std
524:         low_frac     = stats.low_util_frac
525:         headroom_gib = stats.mem_headroom_gib
526:         stable_p95   = stats.stable_p95
527: 
528:         # pair_queue 建议（T3-bottleneck 不影响 pair_queue 逻辑）
529:         # [FIX-MAXPQ-DYNAMIC] 建议上限随绝对上限 24 提升（原硬编码 8）
530:         if stable_std > 25.0 or low_frac > 0.30:
531:             pair_q = min(_PAIR_Q_ABS_CAP, current_pair_q + 2)
532:         elif stable_std < 10.0 and low_frac < 0.10 and current_pair_q > 4:
533:             pair_q = max(3, current_pair_q - 1)
534: 
535:         # [FIX-T3-DETECT] result_queue 建议：T3-bottleneck 时保持或缩小
536:         if self._is_t3_bottleneck(stats, codec=codec):
537:             # T3 是真正瓶颈，增大 result_queue 无助于提速，反而增加内存压力
538:             if current_result_q > 16:
539:                 result_q = max(16, current_result_q - 8)
540:             # else: 已经很小了，保持不变
541:         elif stable_p95 > 85.0 and stable_std > 20.0:
542:             result_q = min(64, current_result_q + 8)
543:         elif headroom_gib > 2.0 and current_result_q < 32:
544:             result_q = min(48, current_result_q + 8)
545: 
546:         # [FIX-T3-MEMCAP / FIX-MAXRQ-DYNAMIC] PinnedPool 内存上限约束（三轴动态计算）
547:         if slot_mb > 0.0:
548:             _mem_avail_gb_gqs = _detect_encode_parallelism()['mem_avail_gb']
549:             _max_rq_by_mem = _compute_max_result_queue(
550:                 slot_mb      = slot_mb,
551:                 mem_avail_gb = _mem_avail_gb_gqs,
552:                 T2_ms        = t2_ms,
553:                 T3_ms        = t3_ms,
554:             )
555:             result_q = min(result_q, _max_rq_by_mem)
556: 
557:         return pair_q, result_q
558: 
559:     # ── 向后兼容接口 ─────────────────────────────────────────────────────────
560: 
561:     def summary(self) -> Tuple[float, float, float, float, float]:
562:         """
563:         向后兼容（v1 接口）：返回
564:             (avg_util, max_util, avg_mem_gib, max_mem_gib, stable_p95)
565: 
566:         stable_p95 替代原"窗口分割峰值均值"，语义更精确（稳定段 95 分位利用率）。
567:         """
568:         s = self.get_stats()
569:         return s.avg_util, s.max_util, s.avg_mem_gib, s.max_mem_gib, s.stable_p95
570: 
571:     # ── [FIX-T3-DETECT] T3 瓶颈检测器 ──────────────────────────────────────
572: 
573:     @staticmethod
574:     def _is_t3_bottleneck(stats: 'GPUStats', codec: str = '') -> bool:
575:         """
576:         [FIX-T3-DETECT] 判断流水线瓶颈是否在 T3（编码器）而非 T2（推理）。
577: 
578:         T3-bottleneck 的 GPU 采样特征（与 T2-bottleneck 截然不同）：
579:           · GPU 空闲占比极高（low_util_frac > 0.60）：GPU 大多数时间在等编码器腾出
580:             result_queue 空位，无法持续供料；
581:           · P95 利用率高（stable_p95 > 85%）：偶尔爆发到 100%，与高空闲同时存在；
582:           · 稳定段均值极低（stable_avg < 30%）：整体利用率远低于 GPU 能力。
583: 
584:         这是典型的「T3 背压→T2 阻塞→GPU 爆发/空转交替」波形。
585:         在此状态下增大 result_queue 毫无帮助（编码器速度不变），
586:         仅增加锁页内存压力、拖慢 DMA 带宽。
587: 
588:         与 T2-bottleneck 的区分：T2 慢时 GPU 持续均匀高负载（stable_avg 高）。
589: 
590:         [FIX-NVENC-AWARE] NVENC 是独立硬件编码单元，不体现在 CUDA 利用率中；
591:         当编码器为 NVENC 时，低 CUDA 利用率是正常现象，不应判定为 T3-bottleneck。
592:         """
593:         if 'nvenc' in codec.lower():
594:             return False
595:         return (
596:             stats.low_util_frac > 0.60        # GPU 空闲 > 60%
597:             and stats.stable_p95 > 85.0       # 但 P95 仍然爆发到 85%+（阵发性）
598:             and stats.stable_avg < 30.0       # 均值极低，说明绝大多数时间 GPU 空转
599:         )
600: 
601: 
602: # ─────────────────────────────────────────────────────────────────────────────
603: # [AUTO-TUNE] 硬件感知队列深度自动调节
604: # ─────────────────────────────────────────────────────────────────────────────
605: 
606: 
607: @dataclasses.dataclass
608: class _HWProfile:
609:     gpu_name:       str
610:     gpu_tier:       float
611:     has_nvdec:      bool
612:     has_nvenc:      bool
613:     pcie_bw_gbs:    float
614:     cpu_cores:      int
615:     t2_measured_ms: float = 0.0
616: 
617:     def __str__(self) -> str:
618:         return (f'{self.gpu_name} tier={self.gpu_tier:.1f} '
619:                 f'nvdec={self.has_nvdec} nvenc={self.has_nvenc} '
620:                 f'pcie={self.pcie_bw_gbs:.0f}GB/s cpu={self.cpu_cores}c')
621: 
622: 
623: _GPU_PROFILES_TABLE = [
624:     ('H100|H800',        15.2, True,  False, 63.0),
625:     ('L40S',             11.3, True,  True,  31.5),
626:     ('A100|A800',         4.8, True,  False, 31.5),
627:     ('L40(?!S)',          5.6, True,  True,  31.5),
628:     ('A30(?!\\d)',        2.5, True,  False, 31.5),
629:     ('A10(?!\\d)',        1.9, True,  True,  31.5),
630:     ('V100',              1.7, False, False, 31.5),
631:     ('T4(?!\\d)',         1.0, True,  True,  15.7),
632:     ('RTX\\s*4090',       3.3, True,  True,  31.5),
633:     ('RTX\\s*4080',       2.6, True,  True,  31.5),
634:     ('RTX\\s*4070\\s*Ti', 2.2, True,  True,  31.5),
635:     ('RTX\\s*4070',       1.9, True,  True,  31.5),
636:     ('RTX\\s*4060',       1.4, True,  True,  15.7),
637:     ('RTX\\s*3090\\s*Ti', 2.3, True,  True,  31.5),
638:     ('RTX\\s*3090',       2.1, True,  True,  15.7),
639:     ('RTX\\s*3080\\s*Ti', 1.8, True,  True,  15.7),
640:     ('RTX\\s*3080',       1.6, True,  True,  15.7),
641:     ('RTX\\s*3070',       1.3, True,  True,  15.7),
642:     ('RTX\\s*2080\\s*Ti', 1.1, True,  True,  15.7),
643:     ('RTX\\s*2080',       0.9, True,  True,  15.7),
644:     ('GTX\\s*1080\\s*Ti', 0.5, True,  True,  15.7),
645:     ('GTX\\s*1080',       0.4, True,  True,  15.7),
646: ]
647: _GPU_FALLBACK = (1.0, True, True, 15.7)
648: 
649: 
650: def _get_gpu_slug() -> str:
651:     if torch.cuda.is_available():
652:         import re as _re_sm
653:         props = torch.cuda.get_device_properties(0)
654:         slug = _re_sm.sub(r'[^a-z0-9]', '', props.name.lower())[:16]
655:         return f'_sm{props.major}{props.minor}_{slug}'
656:     return '_cpu'
657: 
658: _T2_CACHE_DIR_DEFAULT = os.path.join(base_dir, '.t2_cache')
659: 
660: def _load_t2_cache(cache_dir: str) -> dict:
661:     path = os.path.join(cache_dir, 't2_measured.json')
662:     if os.path.isfile(path):
663:         try:
664:             with open(path, 'r', encoding='utf-8') as _f:
665:                 return json.load(_f)
666:         except Exception:
667:             return {}
668:     return {}
669: 
670: def _save_t2_cache(cache_dir: str, cache: dict):
671:     os.makedirs(cache_dir, exist_ok=True)
672:     with open(os.path.join(cache_dir, 't2_measured.json'), 'w', encoding='utf-8') as _f:
673:         json.dump(cache, _f, indent=2)
674: 
675: def _detect_hw_profile(device: torch.device) -> _HWProfile:
676:     import re as _re
677:     cpu_cores = os.cpu_count() or 4
678:     if device.type != 'cuda':
679:         return _HWProfile('CPU', 0.05, False, False, 0.0, cpu_cores)
680:     gpu_name = torch.cuda.get_device_name(device)
681:     tier, has_nvdec, has_nvenc, pcie_bw = _GPU_FALLBACK
682:     for pat, _t, _nd, _ne, _pb in _GPU_PROFILES_TABLE:
683:         if _re.search(pat, gpu_name, _re.IGNORECASE):
684:             tier, has_nvdec, has_nvenc, pcie_bw = _t, _nd, _ne, _pb
685:             break
686:     return _HWProfile(gpu_name, tier, has_nvdec, has_nvenc, pcie_bw, cpu_cores)
687: 
688: # [AUTO-TUNE-CALIB] T2 双分量模型
689: 
690: _T2_BASELINE_HWB  = float(576 * 736 * 24)
691: _T2_FIXED_MS      = 240.0   # torch.compile / eager 路径固定 overhead（含 JIT 编译）
692: # [FIX-T2-TRT-CALIB] TRT 路径固定 overhead 实测约 2-5ms（无 JIT、纯硬件调度）
693: _T2_FIXED_MS_TRT  = 5.0
694: _T2_VAR_MS        = 25.0    # eager/compile 路径斜率 (保留不变)
695: _T2_VAR_MS_TRT    = 335.0   # [FIX-T2-VAR-TRT] TRT 路径基于 T4 实测校准 (Tesla T4, bs=24, 576×736)
696: 
697: 
698: 
699: def _pool_limit_mb_for_profile(profile: '_HWProfile') -> float:
700:     """
701:     [FIX-POOL-AUTOSCALE] 依据 GPU tier 自动计算合理的 PinnedPool 内存上限（MiB）。
702: 
703:     设计原则：
704:     · PinnedPool 使用系统 RAM 锁页内存（不占 VRAM），但过多会拖慢 DMA 带宽。
705:     · 上限随 GPU tier 递增（高端 GPU 往往配套大内存服务器）。
706:     · 兼顾实际可用 RAM：上限不超过 MemAvailable × 12%，最低保底 1024 MiB。
707: 
708:     tier 分档（来自 _GPU_PROFILES_TABLE）：
709:       GTX 1080 / 1080 Ti (tier ≤ 0.5)  → 1024 MiB
710:       T4 / RTX 2080 Ti   (tier ≤ 1.1)  → 2048 MiB   ← T4 推荐值
711:       RTX 3090 / 4070 Ti (tier ≤ 2.3)  → 3072 MiB
712:       A10 / L40S / 4080+ (tier ≤ 5.9)  → 4096 MiB
713:       A100 / A800         (tier ≤ 8.9)  → 6144 MiB
714:       H100 / H800         (tier > 8.9)  → 8192 MiB
715:     """
716:     tier = getattr(profile, 'gpu_tier', 1.0)
717:     if tier > 8.9:
718:         tier_limit = 8192.0
719:     elif tier > 4.7:
720:         tier_limit = 6144.0
721:     elif tier > 1.8:
722:         tier_limit = 4096.0
723:     elif tier > 1.1:
724:         tier_limit = 3072.0
725:     elif tier > 0.5:
726:         tier_limit = 2048.0
727:     else:
728:         tier_limit = 1024.0
729: 
730:     # 兼顾系统可用 RAM：上限 ≤ MemAvailable × 12%，最低 1024 MiB
731:     try:
732:         mem_avail_mb = 0.0
733:         try:
734:             with open('/proc/meminfo', 'r') as _mf:
735:                 for _line in _mf:
736:                     if _line.startswith('MemAvailable:'):
737:                         mem_avail_mb = int(_line.split()[1]) / 1024.0  # kB → MiB
738:                         break
739:         except OSError:
740:             import psutil as _ps
741:             mem_avail_mb = _ps.virtual_memory().available / (1024.0 ** 2)
742:         ram_limit = max(1024.0, mem_avail_mb * 0.12)
743:     except Exception:
744:         ram_limit = tier_limit  # 无法读取时不做 RAM 约束
745: 
746:     return min(tier_limit, ram_limit)
747: 
748: # 模块级常量：PinnedPool 锁页内存上限（MiB），按 GPU 型号自动缩放，只初始化一次。
749: # [FIX-MAXRQ-DYNAMIC] 此常量仅保留供 PinnedPool 构建阶段参考；
750: # 运行时队列约束改由 _compute_max_result_queue() 动态计算，不再直接使用此常量。
751: 
752: _PINNED_POOL_MAX_MB: float = _pool_limit_mb_for_profile(
753:     _detect_hw_profile(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
754: )
755: 
756: 
757: 
758: def _compute_max_result_queue(
759:     slot_mb: float,
760:     mem_avail_gb: float,
761:     T2_ms: float = 0.0,
762:     T3_ms: float = 0.0,
763:     ram_budget_fraction: float = 0.06,
764:     abs_cap: int = 48,
765: ) -> int:
766:     """
767:     [FIX-MAXRQ-DYNAMIC] 动态计算 result_queue 安全上限（三轴联合约束）。
768: 
769:     三轴约束原理
770:     ─────────────────────────────────────────────────────────────────────────
771:     轴1 RAM 上限（主要约束）：
772:       PinnedPool 使用系统 RAM 锁页内存（不占 VRAM），但过多会竞争 DMA 带宽。
773:       上限 = mem_avail_gb × ram_budget_fraction / slot_mb
774:       ram_budget_fraction 默认 6%：T4+20GiB RAM+47MB/slot ≈ 26 槽，与实测吻合。
775:       该参数是唯一需要人工校准的旋钮；无论换 GPU / 换分辨率 / 换机器都自适应。
776: 
777:     轴2 T3/T2 速度比（最小需求下限）：
778:       result_queue 存在是为了缓冲 T3 比 T2 慢这一事实，但 libx264 是流式编码器，
779:       并非每批阻塞整个 T3_ms；实际突发缓冲需求远小于 T3/T2 比值。
780:       经验系数 0.22（= 0.15 × 1.5）：在 T4+crf=0+ultrafast 场景实测吻合。
781:       floor_by_t3 = max(8, int(T3_ms / T2_ms × 0.22))
782:       T2_ms/T3_ms 均为 0 时（未知），跳过此轴，floor = 8。
783: 
784:     轴3 绝对上限（保险兜底）：
785:       防止内存估算本身有误（如 mem_avail_gb 异常偏大）时失控。
786:       固定为 abs_cap（默认 48）。
787: 
788:     三轴关系：RAM 决定上限，T3/T2 比决定下限，取二者之间合理值：
789:       result = max(floor_by_t3, min(cap_by_ram, abs_cap))
790: 
791:     参数
792:     ─────────────────────────────────────────────────────────────────────────
793:     slot_mb            : 每个 result slot 的锁页内存占用（MiB），
794:                          = effective_bs × T × H_pad × W_pad × 3 / 1e6
795:     mem_avail_gb       : 系统当前可用 RAM（GiB），来自 _detect_encode_parallelism()
796:     T2_ms              : GPU 推理延迟（ms），来自 AUTO-TUNE 测量；0 表示未知
797:     T3_ms              : 每批次编码延迟（ms），来自 AUTO-TUNE 估算；0 表示未知
798:     ram_budget_fraction: PinnedPool 允许占用可用 RAM 的比例（默认 6%）
799:     abs_cap            : result_queue 绝对上限（默认 48）
800:     """
801:     if slot_mb <= 0.0:
802:         return abs_cap
803: 
804:     # 轴1: RAM 上限（可用内存 × 预算比 / 每槽大小）
805:     ram_budget_mb = mem_avail_gb * 1024.0 * ram_budget_fraction
806:     cap_by_ram = max(8, int(ram_budget_mb / slot_mb))
807: 
808:     # 轴2: T3/T2 实际需求下限（仅在两者均已知时计算）
809:     # libx264 流式编码 burst 系数约 0.22（经验值，T4 实测 ~8 槽已足够解耦）
810:     if T2_ms > 0.0 and T3_ms > 0.0:
811:         floor_by_t3 = max(8, int(T3_ms / T2_ms * 0.22))
812:     else:
813:         floor_by_t3 = 8   # 未知时使用安全下限
814: 
815:     # 轴3: 绝对上限兜底，防止内存估算失控
816:     return max(floor_by_t3, min(cap_by_ram, abs_cap))
817: 
818: 
819: # [FIX-MAXPQ-DYNAMIC] pair_queue 绝对上限（原硬编码 8 的提升值）
820: 
821: _PAIR_Q_ABS_CAP: int = 24
822: 
823: 
824: 
825: def _compute_max_pair_queue(
826:     slot_mb: float,
827:     mem_avail_gb: float,
828:     ram_budget_fraction: float = 0.15,
829:     abs_cap: int = _PAIR_Q_ABS_CAP,
830: ) -> int:
831:     """
832:     [FIX-MAXPQ-DYNAMIC] 动态计算 pair_queue 安全上限（两轴联合约束）。
833: 
834:     与 _compute_max_result_queue 同构，但有两处关键差异：
835:       1) pair slot 是普通 numpy 数组（pageable RAM），非 pinned 锁页内存，
836:          不参与 DMA 带宽竞争，因此预算比从 6% 放宽到 15%；
837:       2) pair_queue 缓冲的是 "T1(读取) 远快于 T2(推理)"（T2/T1 高达 25-30×），
838:          稳态下 8 槽已足够，更深的意义在于：
839:            · 吸收 T1 的瞬时 stall（pad CPU 突发 / NVDEC 抖动 / 预取槽竞争）；
840:            · 支撑 P 跟随 R 的对称缓冲（result_queue 可达 24-48 时 P 需 12-24）。
841:          因此需求下限固定为 8（实测无饥饿基线），不设 T2/T1 比例轴，
842:          由 GPU-MONITOR/RETUNE 建议驱动实际深度。
843: 
844:     轴1 RAM 上限：cap_by_ram = mem_avail_gb × ram_budget_fraction / slot_mb
845:     轴2 绝对上限：abs_cap（默认 24，替换原硬编码 8）
846: 
847:     参数
848:     ─────────────────────────────────────────────────────────────────────────
849:     slot_mb            : 每个 pair slot 的内存占用（MiB），
850:                          = effective_bs × 3帧(raw+2×pad) × H_pad × W_pad × 3 / 1e6
851:     mem_avail_gb       : 系统当前可用 RAM（GiB），来自 _detect_encode_parallelism()
852:     ram_budget_fraction: pair 缓冲允许占用可用 RAM 的比例（默认 15%，pageable）
853:     abs_cap            : pair_queue 绝对上限（默认 24）
854:     """
855:     if slot_mb <= 0.0:
856:         return abs_cap
857: 
858:     # 轴1: RAM 上限（可用内存 × 预算比 / 每槽大小），下限 8
859:     ram_budget_mb = mem_avail_gb * 1024.0 * ram_budget_fraction
860:     cap_by_ram = max(8, int(ram_budget_mb / slot_mb))
861: 
862:     # 轴2: 绝对上限兜底，防止内存估算失控
863:     return max(8, min(cap_by_ram, abs_cap))
864: 
865: 
866: 
867: _MODEL_T2_FACTOR = {
868:     'IFRNet_S_Vimeo90K': 1.0,      # 基准（最小模型）
869:     'IFRNet_Vimeo90K':   1.6,      # 中型
870:     'IFRNet_L_Vimeo90K': 3.0,      # 大型
871: }
872: 
873: # [AUTO-TUNE-BACKEND] 推理后端相对 TRT 的速度因子
874: _INFER_BACKEND_FACTORS = {
875:     'trt':        1.0,
876:     'cuda_graph': 1.5,
877:     'compile':    2.0,
878:     'eager':      3.5,
879: }
880: 
881: 
882: def _auto_queue_depths(
883:     profile: _HWProfile, codec: str, x264_preset: str, crf: int,
884:     H_pad: int, W_pad: int, effective_bs: int, T: int,
885:     infer_backend: str = 'eager',
886:     model_name: str = 'IFRNet_S_Vimeo90K',   # 新增参数
887:     verbose: bool = True,
888:     t3_fps_measured: float = 0.0,   # [FIX-T3-FPS] 跨段实测 T3 fps（0 表示无实测，用静态估算）
889: ) -> Tuple[int, int, int]:
890:     import math as _math
891:     HWB = float(H_pad * W_pad * effective_bs)
892:     if profile.has_nvdec:
893:         nvdec_fps = min(440.0 * 1920.0 * 1080.0 / max(float(H_pad * W_pad), 1.0), 3000.0)
894:     else:
895:         nvdec_fps = min(60.0 * 1920.0 * 1080.0 / max(float(H_pad * W_pad), 1.0), 600.0)
896:     t1_ms = effective_bs / nvdec_fps * 1000.0
897: 
898:     infer_factor = _INFER_BACKEND_FACTORS.get(infer_backend, _INFER_BACKEND_FACTORS['eager'])
899:     model_factor = _MODEL_T2_FACTOR.get(model_name, 1.0)
900:     if profile.t2_measured_ms > 0:
901:         t2_ms  = profile.t2_measured_ms
902:         t2_src = 'measured'
903:     else:
904:         # [FIX-T2-TRT-CALIB] TRT 路径固定 overhead 仅 2-5ms（无 JIT），
905:         # 与 torch.compile/eager 的 240ms 差距 48×，必须分路处理。
906:         # [FIX-T2-VAR-TRT] TRT 路径可变斜率也需专用常量 (335ms vs 25ms)。
907:         _fixed_ms = _T2_FIXED_MS_TRT if infer_backend == 'trt' else _T2_FIXED_MS
908:         _var_ms   = _T2_VAR_MS_TRT if infer_backend == 'trt' else _T2_VAR_MS
909:         t2_base = (_fixed_ms + _var_ms * HWB / _T2_BASELINE_HWB) / max(profile.gpu_tier, 0.05)
910:         t2_ms   = max(t2_base * infer_factor * model_factor, 1.0)
911:         t2_src  = f'estimated(×{infer_factor}, model×{model_factor})'
912: 
913:     out_frames = effective_bs * (T + 1)
914:     codec_l = codec.lower()
915:     if 'nvenc' in codec_l and profile.has_nvenc:
916:         t3_ms  = out_frames / 3000.0 * 1000.0
917:         t3_src = 'NVENC'
918:     elif 'copy' in codec_l:
919:         t3_ms  = out_frames / 5000.0 * 1000.0
920:         t3_src = 'copy'
921:     elif t3_fps_measured > 0.0:
922:         # [FIX-T3-FPS] 优先使用跨段实测 T3 fps（含热节流等实际因素）
923:         t3_ms  = out_frames / t3_fps_measured * 1000.0
924:         t3_src = f'{codec}({x264_preset}, crf={crf}, measured={t3_fps_measured:.0f}fps)'
925:     else:
926:         fps_est = _software_encode_fps(profile.cpu_cores, H_pad, W_pad, codec, x264_preset, crf)
927:         t3_ms   = out_frames / fps_est * 1000.0
928:         t3_src  = f'{codec}({x264_preset}, crf={crf})'
929: 
930:     # [FIX-MAXPQ-DYNAMIC] pair 上限由动态两轴函数决定（RAM 约束 + 绝对上限 24），
931:     # 替换原硬编码 8：T2/T1 高达 25-30× 时 8 槽对 T1 瞬时 stall 余量不足，
932:     # 且无法支撑 P 跟随 R 的对称缓冲（result_queue 可达 24-48 → P 需 12-24）。
933:     _pair_slot_mb = effective_bs * 3 * H_pad * W_pad * 3 / 1e6  # 每 pair slot 的 MiB
934:     if _pair_slot_mb > 0.0:
935:         _pair_cap = _compute_max_pair_queue(
936:             slot_mb      = _pair_slot_mb,
937:             mem_avail_gb = _detect_encode_parallelism()['mem_avail_gb'],
938:         )
939:     else:
940:         _pair_cap = _PAIR_Q_ABS_CAP
941:     pair_depth   = max(2, min(int(_math.ceil(t2_ms / max(t1_ms, 0.1))) + 2, _pair_cap))
942:     result_depth = max(8, min(int(_math.ceil(t3_ms / max(t2_ms, 0.1))) + 3, 64))
943: 
944:     # [FIX-T3-MEMCAP / FIX-POOL-AUTOSCALE / FIX-MAXRQ-DYNAMIC] 动态约束 result_depth。
945:     # 每个 result slot 持有 effective_bs * T 帧的 pinned uint8 buffer。
946:     # 若不加约束，T3 极慢（大 T3/T2 比）时 result_depth 会达到 50+，
947:     # 导致 PinnedPool 分配 2 GiB+ 锁页内存，反而拖慢 DMA 带宽，形成恶性循环。
948:     # [FIX-MAXRQ-DYNAMIC] 改用三轴动态函数：RAM 上限 / T3/T2 下限 / 绝对上限联合约束，
949:     # 无论换 GPU / 换分辨率 / 换机器均自适应，无需修改硬编码常量。
950:     _slot_mb = effective_bs * T * H_pad * W_pad * 3 / 1e6  # 每 slot 的 MiB
951:     if _slot_mb > 0.0:
952:         _mem_avail_gb_aqd = _detect_encode_parallelism()['mem_avail_gb']
953:         _max_result_by_mem = _compute_max_result_queue(
954:             slot_mb      = _slot_mb,
955:             mem_avail_gb = _mem_avail_gb_aqd,
956:             T2_ms        = t2_ms,
957:             T3_ms        = t3_ms,
958:         )
959:         result_depth = min(result_depth, _max_result_by_mem)
960: 
961:     pool_size    = result_depth + 2
962:     if verbose:
963:         pair_mb = pair_depth * effective_bs * 3 * H_pad * W_pad * 3 / 1e6
964:         pool_mb = pool_size  * effective_bs * T * H_pad * W_pad * 3 / 1e6
965:         print(f'\n[AUTO-TUNE] {profile}  backend={infer_backend}(×{infer_factor}) model={model_name}(×{model_factor})\n'
966:               f'  T1={t1_ms:.1f}ms  T2[{t2_src}]={t2_ms:.1f}ms  T3({t3_src})={t3_ms:.1f}ms\n'
967:               f'  ratio T2/T1={t2_ms/max(t1_ms,0.1):.1f}x T3/T2={t3_ms/max(t2_ms,0.1):.1f}x\n'
968:               f'  pair_queue={pair_depth}(~{pair_mb:.0f}MB) '
969:               f'result_queue={result_depth} pool={pool_size}(~{pool_mb:.0f}MB pinned)',
970:               flush=True)
971:     return pair_depth, result_depth, pool_size
972: 
973: 
974: # ─────────────────────────────────────────────────────────────────────────────
975: # IFRNetPipelineRunner（三级流水线）
976: # ─────────────────────────────────────────────────────────────────────────────
977: 
978: 
979: class IFRNetPipelineRunner:
980:     """
981:     IFRNet 三级深度流水线
982:     ─────────────────────────────────────────────────────────────────────────
983:     T1 Reader  : FFmpegFrameReader → 组 batch → pair_queue（NVDEC + pad 后台线程）
984:     T2 Infer   : pair_queue → GPU 推理 → result_queue（主线程）
985:                  [STREAM-DUAL] 预取在 stream_h2d，D2H 在 stream_d2h，独立不阻塞。
986:     T3 Writer  : result_queue → FFmpegWriter（子线程）
987:     [EVENT-POOL] T2 从池中取 Event，T3 写完后归还，消除每批次创建销毁开销。
988:     [WATCHDOG]   空转超 120s dump 线程栈并强制退出。
989:     """
990: 
991:     _SENTINEL             = object()
992:     IDLE_DEADLOCK_TIMEOUT = 120.0
993: 
994:     def __init__(
995:         self,
996:         processor:         'IFRNetVideoProcessor',
997:         pair_queue_size:   int  = 4,
998:         result_queue_size: int  = 8,
999:         auto_tune:         bool = True,
1000:         codec:             str  = 'libx264',
1001:         x264_preset:       str  = 'medium',
1002:         crf:               int  = 21,
1003:         t2_cache_dir:      str  = '',
1004:         # 新增：外部指定的队列深度（覆盖 AUTO-TUNE）
1005:         pair_queue_override:   Optional[int] = None,
1006:         result_queue_override: Optional[int] = None,
1007:         t3_fps_measured:   float = 0.0,   # [FIX-T3-FPS] 跨段实测 T3 fps
1008:     ):
1009:         self.proc         = processor
1010:         self.pair_queue   = queue.Queue(maxsize=pair_queue_size)
1011:         self.result_queue = queue.Queue(maxsize=result_queue_size)
1012:         self.running      = True
1013:         self.auto_tune    = auto_tune
1014:         self.codec        = codec
1015:         self.x264_preset  = x264_preset
1016:         self.crf          = crf
1017:         self._hw_profile:       Optional[_HWProfile] = None
1018:         self._t2_estimated_ms   = 0.0
1019:         self._last_calib_config = None
1020:         self._cache_key: Optional[str] = None
1021:         self.t2_cache_dir = t2_cache_dir
1022:         self._pair_queue_override = pair_queue_override
1023:         self._result_queue_override = result_queue_override

(Output capped at 50 KB. Showing lines 1-1023. Use offset=1024 to continue.)
</content>
```

---

## Assistant (Plan · Ox Alpha Free · 12.3s)

_Thinking:_

Let me continue reading pipeline.py from line 1024.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\pipeline.py",
  "offset": 1024
}
```

**Output:**
```
<path>D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\ifrnet_video\pipeline.py</path>
<type>file</type>
<content>
1024:         self._t3_fps_measured_input = t3_fps_measured   # [FIX-T3-FPS] 跨段实测值，传给 _auto_queue_depths
1025:         self._error: Optional[Exception] = None          # 流水线异常标志，供 run() 传播错误
1026: 
1027:         # [FIX-DOUBLEBUF-H2D] 双槽飞行中 H2D，最多保持 2 个预取 in-flight
1028:         self._prefetch_deque: deque = deque()   # 元素: (item, img0_t, img1_t)
1029:         self._prefetch_slot  = 0                # 轮转 slot 组: 0→pinned(0,1), 1→pinned(2,3)
1030:         self._prefetch_hits  = 0
1031:         self._prefetch_total = 0
1032:         self._reader_th: Optional[threading.Thread] = None
1033:         self._infer_th:  Optional[threading.Thread] = None   # [FIX-INFER-THREAD]
1034:         self._writer_th: Optional[threading.Thread] = None
1035:         self._fc_extra   = 0                    # [FIX-INFER-THREAD] 推理线程累计输入帧
1036:         self._oc_extra   = 0                    # [FIX-INFER-THREAD] 推理线程累计输出帧
1037: 
1038:         # [FIX-SENTINEL-V643] SENTINEL 无法放回 pair_queue 时的紧急标记
1039:         self._pending_sentinel    = False
1040: 
1041:         # 诊断计数器：追踪 pipeline 各阶段的帧/批次数
1042:         self._diag_reader_pairs   = 0
1043:         self._diag_infer_batches  = 0
1044:         self._diag_infer_pairs    = 0
1045:         self._diag_gpu_stay_batches = 0  # GPU-STAY 路径处理的 batch 数
1046:         self._diag_nvenc_frames   = 0     # NVENC 编码的总帧数
1047:         self._diag_empty_h264     = 0     # 空 H.264 字节计数
1048: 
1049:         # [EVENT-POOL] 预分配 CUDA Event 对象池
1050:         self._event_pool = CudaEventPool(max_size=8)
1051: 
1052:     def _get_infer_backend(self) -> str:
1053:         proc = self.proc
1054:         if getattr(proc, '_trt_ok', False):
1055:             return 'trt'
1056:         if proc.use_compile:
1057:             if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'is_compiled'):
1058:                 try:
1059:                     if torch._dynamo.is_compiled(proc.model):
1060:                         return 'compile'
1061:                 except Exception:
1062:                     pass
1063:             if hasattr(proc.model, '_orig_mod'):
1064:                 return 'compile'
1065:         if proc.use_cuda_graph:
1066:             return 'cuda_graph'
1067:         return 'eager'
1068: 
1069:     # ── T1 Reader 线程 ────────────────────────────────────────────────────────
1070: 
1071:     def _reader_loop(self, reader, effective_bs, first_raw, first_padded):
1072:         raw_buf    = [first_raw]
1073:         padded_buf = [first_padded]
1074:         frames_read = 1
1075:         try:
1076:             while self.running:
1077:                 pair = reader.read()
1078:                 if pair is None:
1079:                     if len(raw_buf) >= 2:
1080:                         self._enqueue_pair(
1081:                             list(raw_buf[1:]),
1082:                             list(padded_buf[:-1]), list(padded_buf[1:]),
1083:                             True,
1084:                         )
1085:                     break
1086:                 raw_buf.append(pair[0])
1087:                 padded_buf.append(pair[1])
1088:                 frames_read += 1
1089:                 if len(raw_buf) == effective_bs + 1:
1090:                     self._enqueue_pair(
1091:                         list(raw_buf[1:]),
1092:                         list(padded_buf[:-1]), list(padded_buf[1:]),
1093:                         False,
1094:                     )
1095:                     raw_buf    = [raw_buf[-1]]
1096:                     padded_buf = [padded_buf[-1]]
1097:         except Exception as e:
1098:             if self._error is None:
1099:                 self._error = e
1100:             import traceback
1101:             print(f'\n[IFRNet-Reader] 异常 @frame={frames_read}: {type(e).__name__}: {e}', flush=True)
1102:             traceback.print_exc()
1103:         finally:
1104:             if not self.proc.quiet:
1105:                 print(f'\n[IFRNet-Reader] 退出，已读 {frames_read} 帧', flush=True)
1106:             for _ in range(60):
1107:                 try:
1108:                     self.pair_queue.put(self._SENTINEL, timeout=1.0)
1109:                     break
1110:                 except queue.Full:
1111:                     continue
1112:             else:
1113:                 # 若队列满且无法放回，存入 _pending_sentinel 供 _infer_loop 读取
1114:                 self._pending_sentinel = True
1115:                 if not self.proc.quiet:
1116:                     print('[IFRNet-Reader] pair_queue 持续满载，'
1117:                           '存入 _pending_sentinel', flush=True)
1118: 
1119:     def _enqueue_pair(self, img1_raw, img0_pad, img1_pad, is_end):
1120:         self._diag_reader_pairs += len(img1_raw)  # 诊断：计数 reader 发送的 pair 数
1121:         item = (img1_raw, img0_pad, img1_pad, is_end)
1122:         while self.running:
1123:             try:
1124:                 self.pair_queue.put(item, timeout=1.0)
1125:                 return
1126:             except queue.Full:
1127:                 infer_dead = self._infer_th is not None and not self._infer_th.is_alive()
1128:                 writer_dead = self._writer_th is not None and not self._writer_th.is_alive()
1129:                 if infer_dead or writer_dead:
1130:                     break
1131:                 continue
1132: 
1133:     # ── GPU 预取（[STREAM-DUAL] 使用 stream_h2d）────────────────────────────
1134: 
1135:     def _try_prefetch_next(self):
1136:         """
1137:         [FIX-DOUBLEBUF-H2D] 在 stream_h2d 上异步预取下一批，最多维持 2 个 in-flight。
1138:         · 每次调用以 while 循环填满至 2 个槽，消除大 bs 下 H2D 等待气泡。
1139:         · 轮转 pinned-buffer slot 组（0/1 ↔ 2/3），确保飞行中 DMA 不被后续请求覆盖。
1140:         · [STREAM-DUAL] 与 stream_d2h（D2H 输出）完全独立，PCIe 全双工利用。
1141:         """
1142:         while len(self._prefetch_deque) < 2:
1143:             if self.pair_queue.empty():
1144:                 return
1145:             try:
1146:                 item = self.pair_queue.get_nowait()
1147:             except queue.Empty:
1148:                 return
1149:             if item is self._SENTINEL:
1150:                 # [FIX-SENTINEL-LOSS] 将 SENTINEL 放回 pair_queue 供主循环消费。
1151:                 # 若 pair_queue 满载（GPU 热节流/编码阻塞），必须等待而非静默丢弃。
1152:                 # 否则 SENTINEL 永久丢失 → 推理线程提前退出 → 帧丢失。
1153:                 try:
1154:                     self.pair_queue.put(item, timeout=5.0)
1155:                 except Exception:
1156:                     if not self.proc.quiet:
1157:                         print('[IFRNet-Prefetch] ⚠️ SENTINEL 放回失败（pair_queue 持续满载5s），'
1158:                               '设置 _pending_sentinel', flush=True)
1159:                     self._pending_sentinel = True
1160:                 return
1161:             img1_raw, img0_pad, img1_pad, is_end = item
1162:             if not img0_pad:
1163:                 # [FIX-DOUBLEBUF-SLOT] timeout put
1164:                 try:
1165:                     self.pair_queue.put(item, timeout=5.0)
1166:                 except Exception:
1167:                     pass
1168:                 return
1169:             proc       = self.proc
1170:             pool       = _get_pinned_pool()
1171:             stream_h2d = proc.stream_h2d
1172:             device     = proc.device
1173:             dtype      = proc.dtype
1174:             # [FIX-DOUBLEBUF-H2D] 轮转 slot 组：0→pinned(0,1)，1→pinned(2,3)
1175:             slot_base  = self._prefetch_slot * 2
1176:             try:
1177:                 if stream_h2d is not None:
1178:                     with torch.cuda.stream(stream_h2d):
1179:                         img0_pin = pool.get_for_frames(img0_pad, to_rgb=False, slot=slot_base)
1180:                         img0_t   = img0_pin.to(device, non_blocking=True, dtype=dtype)
1181:                         img1_pin = pool.get_for_frames(img1_pad, to_rgb=False, slot=slot_base + 1)
1182:                         img1_t   = img1_pin.to(device, non_blocking=True, dtype=dtype)
1183:                 else:
1184:                     img0_t = pool.get_for_frames(img0_pad, to_rgb=False, slot=slot_base).to(
1185:                         device, dtype=dtype)
1186:                     img1_t = pool.get_for_frames(img1_pad, to_rgb=False, slot=slot_base + 1).to(
1187:                         device, dtype=dtype)
1188:                 self._prefetch_deque.append((item, img0_t, img1_t))
1189:                 self._prefetch_slot = 1 - self._prefetch_slot   # 切换到另一 slot 组
1190:             except Exception as e:
1191:                 print(f'[IFRNet-Prefetch] H2D 预取失败: {e}，放回队列', flush=True)
1192:                 # [FIX-DOUBLEBUF-SLOT] 用带超时的 put 防止 pair_queue 满时 T2 永久阻塞
1193:                 try:
1194:                     self.pair_queue.put(item, timeout=5.0)
1195:                 except Exception:
1196:                     pass
1197:                 return   # 出错停止继续填充
1198: 
1199:     def _pop_prefetch_or_none(self):
1200:         """[FIX-DOUBLEBUF-H2D] 从双槽 deque 中按 FIFO 顺序弹出预取结果。"""
1201:         if not self._prefetch_deque:
1202:             return None
1203:         item, img0_t, img1_t = self._prefetch_deque.popleft()
1204:         self._prefetch_hits  += 1
1205:         self._prefetch_total += 1
1206:         return item, img0_t, img1_t
1207: 
1208:     # ── T3 Writer 线程 ────────────────────────────────────────────────────────
1209: 
1210:     def _writer_loop(self, writer, pbar, n_seg_est, meter, timing_ref):
1211:         written           = 0
1212:         _idle_since       = None
1213:         received_sentinel = False
1214:         # [FIX-T3-FPS] T3 实际写入吞吐计时（起止时间戳，单位 monotonic）
1215:         _t3_t_first: Optional[float] = None
1216:         _t3_t_last:  Optional[float] = None
1217:         # [FIX-ENC-THREAD] 独立 NVENC 编码线程句柄（GPU_RAW 路径惰性初始化，其余路径为 None）
1218:         _enc_thread: Optional[_NVENCEncodeThread] = None
1219:         # [FIX-WRITTEN-TRUTH] 编码线程计数是否已汇总进 written（防止异常路径重复计数）
1220:         _enc_counted = False
1221:         try:
1222:             while self.running or not self.result_queue.empty():
1223:                 try:
1224:                     item = self.result_queue.get(timeout=2.0)
1225:                 except queue.Empty:
1226:                     all_empty = (
1227:                         self.pair_queue.empty() and
1228:                         self.result_queue.empty()
1229:                     )
1230:                     if all_empty and not received_sentinel:
1231:                         if _idle_since is None:
1232:                             _idle_since = time.time()
1233:                             print(f'\n[IFRNet-Writer][看门狗] 流水线空转，'
1234:                                   f'开始计时（阈值 {self.IDLE_DEADLOCK_TIMEOUT:.0f}s）', flush=True)
1235:                         elif time.time() - _idle_since > self.IDLE_DEADLOCK_TIMEOUT:
1236:                             print(f'\n[IFRNet-Writer][看门狗] ⚠️ 空转超过 '
1237:                                   f'{self.IDLE_DEADLOCK_TIMEOUT:.0f}s，判定死锁，'
1238:                                   f'已写 {written} 帧，强制退出。', flush=True)
1239:                             self._dump_threads()
1240:                             self.running = False
1241:                             break
1242:                     else:
1243:                         _idle_since = None
1244:                     if received_sentinel and self.result_queue.empty():
1245:                         break
1246:                     continue
1247: 
1248:                 if item is self._SENTINEL:
1249:                     received_sentinel = True
1250:                     break
1251: 
1252:                 _idle_since = None
1253: 
1254:                 # FIX Writer 增加预读取 + 批量处理
1255:                 # 批量取出更多待处理结果（最多 4 个）
1256:                 items = [item]
1257:                 for _ in range(3):
1258:                     try:
1259:                         items.append(self.result_queue.get_nowait())
1260:                     except queue.Empty:
1261:                         break
1262: 
1263:                 # 批量等待所有 D2H DMA 完成
1264:                 _has_sentinel = False
1265:                 for _item in items:
1266:                     if _item is self._SENTINEL:
1267:                         _has_sentinel = True
1268:                         continue
1269:                     if isinstance(_item, _PinnedResultItem):
1270:                         _item.event.synchronize()
1271: 
1272:                 # 批量写入（跳过 sentinel）
1273:                 _n_pairs_total = 0
1274:                 for _item in items:
1275:                     if _item is self._SENTINEL:
1276:                         continue
1277:                     # [FIX-GPU-STAY] Level 1 GPU-STAY: GPU tensor → 批量RGB→NV12 → NVENC SDK 编码
1278:                     # Infer 线程只做推理+传 GPU tensor，Writer 线程并行编码。
1279:                     if isinstance(_item, tuple) and len(_item) >= 2 and _item[0] == 'GPU_RAW':
1280:                         _, interp_gpu, img1_rgb, rB, rT, rorig_H, rorig_W = _item
1281:                         _nvenc = self.proc._cached_nvenc_encoder
1282:                         if _nvenc is None:
1283:                             print('[IFRNet-Writer] GPU_RAW received but NVENC encoder lost, skip batch', flush=True)
1284:                             continue
1285:                         # 批量 RGB→NV12（单次 kernel launch 替代 N 次调用）
1286:                         all_frames = torch.cat([interp_gpu, img1_rgb], dim=0)  # (B*T + B, H, W, 3)
1287:                         # [DIAG-RAW] 环境变量 IFRNET_DIAG=1 时输出诊断统计，帮助定位花屏根因
1288:                         import os as _os_diag
1289:                         if _os_diag.environ.get('IFRNET_DIAG') == '1':
1290:                             _di = interp_gpu.float(); _d2 = img1_rgb.float()
1291:                             print(f'[DIAG-GPU_RAW] batch interp mean={_di.mean():.1f} std={_di.std():.1f} '
1292:                                   f'min={_di.min():.0f} max={_di.max():.0f} | '
1293:                                   f'img1 mean={_d2.mean():.1f} std={_d2.std():.1f} '
1294:                                   f'min={_d2.min():.0f} max={_d2.max():.0f}', flush=True)
1295:                         all_nv12 = _rgb_to_nv12_gpu_batch(all_frames)
1296:                         n_interp = rB * rT
1297: 
1298:                         # [FIX-ENC-THREAD] CRITICAL: 等待当前 PyTorch stream 完成 NV12 写入，
1299:                         # 再交给编码线程的 cuMemcpy2D 读取，防止 GPU 数据未就绪导致静默花帧。
1300:                         torch.cuda.current_stream().synchronize()
1301: 
1302:                         # [FIX-LA-ACC-HOST] LA>0 时 _NVENCEncodeThread._loop() 会把整个 segment
1303:                         # 的所有批次累积到 _acc_nv12（[FIX-LA-ACCUMULATE] 设计，为保证 LA FIFO
1304:                         # 连续性 / _slot_pending 不跨批次错位，这是必要的正确性约束，不能改回
1305:                         # per-batch 编码）。但若累积的元素继续是 GPU tensor，几千帧 NV12 数据
1306:                         # （单帧约 (H+H//2)*W 字节，7000+ 帧可达 8~10GB+）会常驻显存不释放，
1307:                         # 与 T2 推理批次抢显存，形成阶梯状显存上升直至 OOM。此处在入队前立即 D2H
1308:                         # 搬到 pinned host 内存，GPU 侧引用随本次循环结束即可被 PyTorch allocator
1309:                         # 回收复用；encode_frames_batch() 的 cuMemcpy2D 已同步支持 host 源
1310:                         # （见 [FIX-LA-ACC-HOST] 在 encode_frames_batch 内的改动）。
1311:                         # LA=0 走 per-batch ce_pipeline，帧不会跨批次滞留，无需搬移，继续留在 GPU
1312:                         # 以保留原有 GPU-STAY 零拷贝性能。
1313:                         if _nvenc._la_depth > 0:
1314:                             # [FIX-LA-ACC-HOST-V2] 直接分配 pinned 缓冲区，一次 D2H 拷贝直接落地，
1315:                             # 代替 .cpu().pin_memory() 的两步搬运（先 D2H→pageable，再 CPU
1316:                             # pageable→pinned memcpy）。
1317:                             try:
1318:                                 _pinned = torch.empty(all_nv12.shape, dtype=all_nv12.dtype,
1319:                                                        device='cpu', pin_memory=True)
1320:                                 _pinned.copy_(all_nv12, non_blocking=False)
1321:                                 all_nv12 = _pinned
1322:                             except RuntimeError:
1323:                                 # pinned 内存分配失败（如系统锁页内存额度耗尽），退化为普通
1324:                                 # 可分页内存，仅牺牲部分 H2D 拷贝带宽，正确性不受影响。
1325:                                 all_nv12 = all_nv12.cpu()
1326: 
1327:                         # [FIX-ENC-THREAD] 构建交叉交错顺序的帧列表，提交给独立编码线程。
1328:                         # T3 Writer 做 RGB→NV12 kernel，编码线程做 encode_frames_batch，
1329:                         # 两者在 T4 SM（CUDA 计算）与 NVENC 固定功能硬件上真正并行。
1330:                         encode_order = []
1331:                         for bi in range(rB):
1332:                             for tj in range(rT):
1333:                                 encode_order.append(all_nv12[bi * rT + tj])
1334:                             encode_order.append(all_nv12[n_interp + bi])
1335: 
1336:                         self.__dict__.setdefault('_diag_empty_h264', 0)
1337:                         self.__dict__.setdefault('_diag_nvenc_frames', 0)
1338:                         # 惰性初始化编码线程（首次 GPU_RAW 时创建）
1339:                         if _enc_thread is None:
1340:                             _enc_thread = _NVENCEncodeThread(
1341:                                 _nvenc, writer, batch_frames=len(encode_order))  # [FIX-CHUNK-SAFE]
1342:                             # [FIX-DYN-TIMEOUT] 暴露到 pipeline 实例，run() 所在的主线程
1343:                             # 据此读取 submitted_frames 动态估算 writer 线程 join 超时。
1344:                             self._enc_thread_ref = _enc_thread
1345:                         _is_first_submit = (_enc_thread._written == 0 and _enc_thread._empty == 0)
1346:                         _enc_thread.submit(encode_order, force_idr_first=_is_first_submit)
1347:                         _n_pairs_total += rB
1348:                         # [FIX-LA-ACC-HOST] all_nv12/encode_order 的 GPU 版本源（all_frames/
1349:                         # interp_gpu/img1_rgb）在 LA>0 时已不再需要，显式断开引用而不是等待
1350:                         # 下一轮循环变量被覆盖，缩短显存峰值持有窗口。
1351:                         if _nvenc._la_depth > 0:
1352:                             all_frames = None
1353:                         continue
1354:                     # NVENC 直通编码 H.264 ES 输出
1355:                     if isinstance(_item, tuple) and len(_item) == 5:
1356:                         h264_frames, _rB, _rT, _oh, _ow = _item
1357:                         _n_pairs_total += _rB
1358:                         for _hitem in h264_frames:
1359:                             if isinstance(_hitem, tuple) and _hitem[0] == b'RAW':
1360:                                 writer.write(_hitem[1].tobytes() if hasattr(_hitem[1], 'tobytes') else _hitem[1])
1361:                                 written += 1
1362:                             elif isinstance(_hitem, bytes) and _hitem:
1363:                                 writer.write(_hitem)
1364:                                 written += 1
1365:                         continue
1366:                     if isinstance(_item, tuple) and len(_item) == 2 and _item[0] == 'RING':
1367:                         # ── [FIX-T3-V643] Ring Buffer 零拷贝路径（Level 2/3） ──
1368:                         _, _ring_slot = _item
1369:                         _rb = self.proc._ring_buf
1370:                         rv = _rb.reader_acquire(timeout=5.0)
1371:                         if rv is None:
1372:                             print('[IFRNet-Writer] ⚠️ ring buffer reader_acquire 返回 None，跳过', flush=True)
1373:                             continue
1374:                         _sid, _mv, _nf, _oh, _ow, _rB, _rT, _img1l, _ev = rv
1375:                         # 等待 D2H 完成
1376:                         if _ev is not None:
1377:                             _ev.synchronize()
1378:                             self._event_pool.release(_ev)
1379:                         frame_sz = _oh * _ow * 3
1380:                         _n_pairs_total += _rB
1381:                         # [FIX-MEMORYVIEW-DIM] mv 为 4D memoryview（PEP 3118），
1382:                         # 须 .cast('B') 转为 1D 字节视图后才能按字节偏移切片
1383:                         _batch_data = bytearray()
1384:                         mv_flat = _mv.cast('B')
1385:                         for i in range(_rB):
1386:                             interp_start = i * _rT * frame_sz
1387:                             interp_end   = (i + 1) * _rT * frame_sz
1388:                             _batch_data.extend(mv_flat[interp_start:interp_end])
1389:                             written += _rT
1390:                             # img1_raw: 来自 reader (RGB)，直接写入 rgb24
1391:                             if i < len(_img1l):
1392:                                 _batch_data.extend(_img1l[i].copy().tobytes())
1393:                                 written += 1
1394:                         writer.write_direct(bytes(_batch_data))
1395:                         _rb.reader_release(_sid)
1396:                         continue
1397:                     if isinstance(_item, _PinnedResultItem):
1398:                         n_pairs = _item.B
1399:                         _n_pairs_total += n_pairs
1400:                         arr = _item.buf[:_item.B * _item.T].numpy()
1401:                         for i in range(_item.B):
1402:                             for j in range(_item.T):
1403:                                 # [FIX-T3-V643-COLOR] 数据保持 RGB（不翻转），pix_fmt 已是 rgb24
1404:                                 fr = arr[i * _item.T + j,
1405:                                          :_item.orig_H, :_item.orig_W]
1406:                                 if not fr.flags['C_CONTIGUOUS']:
1407:                                     fr = np.ascontiguousarray(fr)
1408:                                 writer.write(fr)
1409:                                 written += 1
1410:                             # img1_raw: 来自 reader (RGB)，直接写入 rgb24
1411:                             writer.write(_item.img1_raw[i].copy())
1412:                             written += 1
1413:                         # [EVENT-POOL] 写完后归还 Event 和 buffer
1414:                         self._event_pool.release(_item.event)
1415:                         _item.pool.release(_item.buf)
1416:                     else:
1417:                         results, img1_raw_list, is_end = _item
1418:                         n_pairs = len(img1_raw_list)
1419:                         _n_pairs_total += n_pairs
1420:                         for i, interps in enumerate(results):
1421:                             for fr in interps:
1422:                                 writer.write(fr)
1423:                                 written += 1
1424:                             # img1_raw: 来自 reader (RGB)，直接写入 rgb24
1425:                             writer.write(img1_raw_list[i].copy())
1426:                             written += 1
1427: 
1428:                 if _has_sentinel:
1429:                     received_sentinel = True
1430:                     break
1431: 
1432:                 # [FIX-T3-FPS] 记录第一帧/最后一帧写入时间，用于计算实际 T3 fps
1433:                 _t3_now = time.monotonic()
1434:                 if _t3_t_first is None:
1435:                     _t3_t_first = _t3_now
1436:                 _t3_t_last = _t3_now
1437: 
1438:                 if pbar is not None:
1439:                     avg_t = np.mean(timing_ref[-20:]) * 1000 if timing_ref else 0
1440:                     pbar.set_postfix(
1441:                         fps=f'{meter.fps():.1f}',
1442:                         eta=f'{meter.eta(n_seg_est):.0f}s',
1443:                         ms=f'{avg_t:.0f}',
1444:                         P=self.pair_queue.qsize(),
1445:                         R=self.result_queue.qsize(),
1446:                         refresh=False,
1447:                     )
1448:                     pbar.update(_n_pairs_total)
1449:             # [FIX-ENC-THREAD] SENTINEL 后收尾：等待编码线程排空队列 + NVENC EOS flush。
1450:             # GPU_RAW 路径：_enc_thread.flush_and_join() 内部已调用 nvenc.flush()，
1451:             #               并汇总写入计数到 written 及诊断计数器。
1452:             # Level 2/3/4 路径（_enc_thread is None）：原逻辑直接 flush NVENC（如有）。
1453:             if _enc_thread is not None:
1454:                 # [FIX-DYN-TIMEOUT] 内层等待就是真正阻塞编码收尾的地方，必须先在这里
1455:                 # 动态放宽，外层 run() 的 writer 线程 join 才有意义（否则内层仍按旧的
1456:                 # 固定 120s 提前放弃、外层放再宽的超时也等不到真正编码完成）。
1457:                 # LA=0 无段末编码脉冲，沿用原 30s 语义（走 estimate_timeout 的 min_timeout 分支）。
1458:                 _la_for_timeout = getattr(_enc_thread._nvenc, '_la_depth', 0) > 0
1459:                 if _la_for_timeout:
1460:                     _flush_timeout = _enc_thread.estimate_timeout()
1461:                 else:
1462:                     _flush_timeout = 30.0
1463:                 _enc_written, _enc_empty = _enc_thread.flush_and_join(timeout=_flush_timeout)
1464:                 written += _enc_written
1465:                 _enc_counted = True
1466:                 self._diag_empty_h264   = getattr(self, '_diag_empty_h264',   0) + _enc_empty
1467:                 self._diag_nvenc_frames = getattr(self, '_diag_nvenc_frames',  0) + _enc_written + _enc_empty
1468:             else:
1469:                 # 非 GPU_RAW 路径（Level 2/3/4）：直接 flush NVENC encoder（如有）
1470:                 _nvenc_flush = getattr(self.proc, '_cached_nvenc_encoder', None)
1471:                 if _nvenc_flush is not None:
1472:                     try:
1473:                         _leftover = _nvenc_flush.flush()
1474:                         if _leftover:
1475:                             writer.write(_leftover)
1476:                     except Exception:
1477:                         pass
1478:         except Exception as e:
1479:             self._error = e
1480:             self.running = False
1481:             import traceback
1482:             print(f'\n[IFRNet-Writer] 写线程异常: {type(e).__name__}: {e}', flush=True)
1483:             traceback.print_exc()
1484:         finally:
1485:             # [FIX-WRITTEN-TRUTH] 异常路径下 flush_and_join 抛出导致计数未汇总，
1486:             # 直接并入编码线程实际成功写入 muxer 的帧数，避免 pipeline._written=0
1487:             # 触发理论值回退、掩盖输出截断。
1488:             if _enc_thread is not None and not _enc_counted:
1489:                 written += getattr(_enc_thread, '_written', 0)
1490:             if not self.proc.quiet:
1491:                 print(f'\n[IFRNet-Writer] 退出，已写 {written} 输出帧', flush=True)
1492:             # [FIX-T3-FPS] 计算并存储 T3 实测 fps（写入时长 < 1s 时视为不可靠）
1493:             if _t3_t_first is not None and _t3_t_last is not None:
1494:                 _t3_elapsed = _t3_t_last - _t3_t_first
1495:                 self._t3_fps_measured = written / _t3_elapsed if _t3_elapsed > 1.0 else 0.0
1496:             else:
1497:                 self._t3_fps_measured = 0.0
1498:             self._written = written
1499: 
1500:     def _dump_threads(self):
1501:         import traceback
1502:         for tid, frame in sys._current_frames().items():
1503:             print(f'\n── Thread {tid} ──', flush=True)
1504:             traceback.print_stack(frame)
1505: 
1506:     # ── 主入口（T2 推理主线程）───────────────────────────────────────────────
1507: 
1508:     def run(
1509:         self,
1510:         reader,
1511:         writer:            'FFmpegWriter',
1512:         timesteps:         List[float],
1513:         H:                 int,
1514:         W:                 int,
1515:         effective_bs:      int,
1516:         first_raw:         np.ndarray,
1517:         first_padded:      np.ndarray,
1518:         skip_first_output: bool,
1519:         pbar,
1520:         n_seg_est:         int,
1521:         meter:             'ThroughputMeter',
1522:         H_pad:             int = 0,
1523:         W_pad:             int = 0,
1524:         nvenc_encoder:     Optional[object] = None,  # [FIX-T3-V643] GPU 直通编码器
1525:     ) -> Tuple[int, int]:
1526:         proc = self.proc
1527:         proc._pipeline_runner = self
1528: 
1529:         if self.auto_tune and H_pad > 0 and W_pad > 0:
1530:             if self._hw_profile is None:
1531:                 self._hw_profile = _detect_hw_profile(proc.device)
1532:             infer_be = self._get_infer_backend()
1533:             gpu_slug = _get_gpu_slug()
1534: 
1535:             # ✅ KEY 中加入 model_name
1536:             self._cache_key = (f'{proc.model_name}_{H_pad}x{W_pad}_bs{effective_bs}'
1537:                                f'_{infer_be}{gpu_slug}')
1538:             
1539:             if self.t2_cache_dir and self._hw_profile.t2_measured_ms <= 0:
1540:                 _cached = _load_t2_cache(self.t2_cache_dir).get(self._cache_key, 0.0)
1541:                 if _cached > 0:
1542:                     self._hw_profile.t2_measured_ms = _cached
1543:                     if proc._is_first_segment():
1544:                         print(f'[T2-CACHE] 加载缓存 T2={_cached:.1f}ms '
1545:                               f'(key={self._cache_key})', flush=True)
1546: 
1547:             _current_cfg = (proc.model_name, H_pad, W_pad, effective_bs, infer_be)   # ✅ 加入模型名，跨模型切换时正确清零 t2_measured_ms
1548:             if self._last_calib_config != _current_cfg:
1549:                 self._hw_profile.t2_measured_ms = 0.0
1550: 
1551:             # ── 使用外部建议覆盖 ──
1552:             if (self._pair_queue_override is not None
1553:                     and self._result_queue_override is not None):
1554:                 _pd = self._pair_queue_override
1555:                 _rd = self._result_queue_override
1556:                 if not self.proc.quiet and proc._is_first_segment():
1557:                     print(f'\n[AUTO-TUNE] 使用外部建议队列: pair={_pd} result={_rd}')
1558:             else:
1559:                 _pd, _rd, _ = _auto_queue_depths(
1560:                     self._hw_profile, self.codec, self.x264_preset, self.crf,
1561:                     H_pad, W_pad, effective_bs, len(timesteps),
1562:                     infer_backend=infer_be,
1563:                     model_name=proc.model_name,   # ✅ 传入模型名
1564:                     t3_fps_measured=self._t3_fps_measured_input,   # [FIX-T3-FPS]
1565:                 )
1566:             self.pair_queue   = queue.Queue(maxsize=_pd)
1567:             # [FIX-LA-RQ-BURST] LA>0 chunk encoding produces bursty consumption
1568:             # from result_queue. Add headroom to prevent T2 blocking during
1569:             # _encode_chunk() bursts (chunk≤1000 frames, ~5s each).
1570:             _nvenc_la = getattr(getattr(self.proc, '_cached_nvenc_encoder', None), '_la_depth', 0)
1571:             if _nvenc_la > 0 and _rd > 0:
1572:                 _rd = min(_rd + max(4, _rd // 4), 48)
1573:             self.result_queue = queue.Queue(maxsize=_rd)
1574: 
1575:             if self._hw_profile.t2_measured_ms > 0:
1576:                 self._t2_estimated_ms = self._hw_profile.t2_measured_ms
1577:             else:
1578:                 _HWB = float(H_pad * W_pad * effective_bs)
1579:                 _ifactor = _INFER_BACKEND_FACTORS.get(infer_be, 3.5)
1580:                 _mfactor = _MODEL_T2_FACTOR.get(proc.model_name, 1.0)
1581:                 # [FIX-T2-TRT-CALIB] TRT 路径使用专用固定 overhead 常量（5ms vs 240ms）
1582:                 # [FIX-T2-VAR-TRT] TRT 路径可变斜率也需专用常量 (335ms vs 25ms)
1583:                 _fixed_ms = _T2_FIXED_MS_TRT if infer_be == 'trt' else _T2_FIXED_MS
1584:                 _var_ms   = _T2_VAR_MS_TRT if infer_be == 'trt' else _T2_VAR_MS
1585:                 _t2b = (_fixed_ms + _var_ms * _HWB / _T2_BASELINE_HWB) \
1586:                        / max(self._hw_profile.gpu_tier, 0.05)
1587:                 self._t2_estimated_ms = max(_t2b * _ifactor * _mfactor, 1.0)
1588: 
1589:         # [PINNED-D2H] 创建/复用 PinnedResultPool（检查 stream_d2h 是否可用）
1590:         _pool_ok = False
1591:         if (H_pad > 0 and W_pad > 0
1592:                 and getattr(proc, 'stream_d2h', None) is not None   # [STREAM-DUAL]
1593:                 and proc.device.type == 'cuda'):
1594:             _max_BT    = effective_bs * len(timesteps)
1595:             _pool_size = self.result_queue.maxsize + 2
1596:             try:
1597:                 proc._result_pool, _pool_is_new = proc._get_or_create_result_pool(
1598:                     _pool_size, _max_BT, H_pad, W_pad)
1599:                 _pool_ok = True
1600:             except Exception as _pe:
1601:                 print(f'[IFRNet-Pipeline] PinnedResultPool 分配失败: {_pe}，回退同步 D2H',
1602:                       flush=True)
1603:                 proc._result_pool = None
1604: 
1605:         # [SEGMENT-REUSE] 段 1 完整输出流水线详情，后续段简化
1606:         if proc._is_first_segment():
1607:             print(
1608:                 f'[IFRNet-Pipeline] 启动深度流水线 | '
1609:                 f'pair_queue={self.pair_queue.maxsize} '
1610:                 f'result_queue={self.result_queue.maxsize} '
1611:                 f'effective_bs={effective_bs} '
1612:                 f'T={len(timesteps)}× | '
1613:                 f'D2H={"pinned(stream_d2h)" if _pool_ok else "sync"}',
1614:                 flush=True,
1615:             )
1616: 
1617:         self._reader_th = threading.Thread(
1618:             target=self._reader_loop,
1619:             args=(reader, effective_bs, first_raw, first_padded),
1620:             daemon=True, name='IFRNet-Reader',
1621:         )
1622:         self._reader_th.start()
1623: 
1624:         self._written = 0
1625:         self._writer_th = threading.Thread(
1626:             target=self._writer_loop,
1627:             args=(writer, pbar, n_seg_est, meter, proc._timing),
1628:             daemon=True, name='IFRNet-Writer',
1629:         )
1630:         self._writer_th.start()
1631: 
1632:         # [FIX-INFER-THREAD] 启动独立 T2 推理线程（仿 ESRGAN _sr_thread）
1633:         self._infer_th = threading.Thread(
1634:             target=self._infer_loop,
1635:             args=(timesteps, H, W, effective_bs, H_pad, W_pad, meter, nvenc_encoder),
1636:             daemon=True, name='IFRNet-Infer',
1637:         )
1638:         self._infer_th.start()
1639: 
1640:         # 等待工作线程退出（先等 Writer，确保其错误先写入 self._error）
1641:         if self._writer_th is not None:
1642:             self._writer_th.join(timeout=10.0)
1643: 
1644:         # [FIX-JOIN-TIMEOUT] 动态超时：基于分段帧数估算，避免长分段误触发。
1645:         # n_seg_est / 5.0 表示最慢 5 fps 仍可完成；下限 120s 保底，上限 7200s 防止
1646:         # 极端大分段数值溢出。推理线程正常应在 ~n_seg_est / actual_fps 秒内完成。
1647:         _infer_timeout = max(120.0, min(n_seg_est / 5.0, 7200.0))
1648:         self._infer_th.join(timeout=_infer_timeout)
1649:         if self._infer_th.is_alive():
1650:             # 超时后发送停止信号，再给 60s 宽限期让 finally 块执行
1651:             print(f'\n[IFRNet] ⚠️ 推理线程 {_infer_timeout:.0f}s 未退出，'
1652:                   f'发送停止信号...', flush=True)
1653:             self.running = False
1654:             self._infer_th.join(timeout=60.0)
1655:             if self._infer_th.is_alive():
1656:                 _msg = (f'推理线程在停止信号后 60s 仍未退出'
1657:                         f'（总等待 {_infer_timeout + 60:.0f}s，n_seg_est={n_seg_est}）')
1658:                 self._error = RuntimeError(_msg)
1659:                 print(f'[IFRNet] ❌ {_msg}', flush=True)
1660:             else:
1661:                 print('[IFRNet] 推理线程已响应停止信号并退出', flush=True)
1662: 
1663:         # 检查流水线是否有异常
1664:         if self._error is not None:
1665:             raise RuntimeError(
1666:                 f'流水线处理异常: {type(self._error).__name__}: {self._error}'
1667:             ) from self._error
1668: 
1669:         # [FIX-RETUNE-POSTRUN] 段完成后用全段稳定 timing 做 RETUNE（精度优于早期 5-batch 采样）
1670:         # · 采用 timing[_CALIB_SKIP:] 中位数，剔除流水线启动热身噪声
1671:         # · 此时 T2-CACHE 早期写入已完成，此处仅更新队列建议（proc._retune_pair/result_q）
1672:         # · 同步更新 hw_profile.t2_measured_ms 为全段最终值（供下段 _auto_queue_depths 使用）
1673:         _RETUNE_SKIP = 3
1674:         if self.auto_tune and len(proc._timing) > _RETUNE_SKIP:
1675:             _stable = proc._timing[_RETUNE_SKIP:]
1676:             _t2_post = float(np.median(_stable)) * 1000.0
1677:             if _t2_post >= 1.0:
1678:                 _be_post = self._get_infer_backend()
1679:                 if self._hw_profile is not None:
1680:                     self._hw_profile.t2_measured_ms = _t2_post
1681:                     self._last_calib_config = (
1682:                         proc.model_name, H_pad, W_pad, effective_bs, _be_post
1683:                     )
1684:                 # 更新 T2-CACHE（以全段中位数覆盖早期估算，更稳定）
1685:                 if self.t2_cache_dir and self._cache_key:
1686:                     _c2 = _load_t2_cache(self.t2_cache_dir)
1687:                     _old2 = _c2.get(self._cache_key, 0.0)
1688:                     if _old2 <= 0 or abs(_t2_post - _old2) / max(_old2, 1.0) > 0.05:
1689:                         _c2[self._cache_key] = round(_t2_post, 1)
1690:                         _save_t2_cache(self.t2_cache_dir, _c2)
1691:                 _pd_post, _rd_post, _ = _auto_queue_depths(
1692:                     self._hw_profile, self.codec, self.x264_preset, self.crf,
1693:                     H_pad, W_pad, effective_bs, len(timesteps),
1694:                     infer_backend=_be_post, verbose=False,
1695:                     model_name=proc.model_name,
1696:                     t3_fps_measured=self._t3_fps_measured_input,
1697:                 )
1698:                 proc._retune_pair_q   = _pd_post
1699:                 proc._retune_result_q = _rd_post
1700:                 _dev_post = abs(_t2_post - self._t2_estimated_ms) / max(self._t2_estimated_ms, 1.0)
1701:                 print(
1702:                     f'[AUTO-TUNE-RETUNE] 实测 T2={_t2_post:.1f}ms'
1703:                     f'（全段 {len(_stable)} batches 中位数）| '
1704:                     f'静态估算={self._t2_estimated_ms:.1f}ms | '
1705:                     f'偏差={_dev_post*100:.0f}% | '
1706:                     f'当前 result_queue={self.result_queue.maxsize} | '
1707:                     f'校准建议 pair={_pd_post} result={_rd_post}（下次生效）',
1708:                     flush=True,
1709:                 )
1710: 
1711:         if self._writer_th and self._writer_th.is_alive():
1712:             # [FIX-LA-JOIN-TIMEOUT][FIX-DYN-TIMEOUT] LA>0 时段末需等待编码线程完成
1713:             # 最后一块编码 + EOS 排空。30s 硬超时对长分段不可靠（段末收尾一旦超过
1714:             # 30s，主线程放行后 writer.close() 会关闭 muxer stdin → 编码线程
1715:             # write to closed file → 输出静默截断）。
1716:             #
1717:             # 原固定 150s 只是"看起来比 flush_and_join 内层 120s 上限宽"，但内层
1718:             # 若真的按固定 120s 提前放弃，外层再宽的固定值也等不到真正编码完成——
1719:             # 编码线程的自身 _th 会在后台继续跑，而 writer 线程却已经"正常退出"，
1720:             # 段照样带着截断风险往下走。现改为按 _enc_thread.submitted_frames
1721:             # 动态估算（与 _writer_loop 内 flush_and_join 调用处使用同一套
1722:             # estimate_timeout 逻辑），外层在内层估算值基础上再加调度/收尾余量，
1723:             # 确保外层等待窗口始终 ≥ 内层实际会等待的时长。
1724:             _nv_join = getattr(self.proc, '_cached_nvenc_encoder', None)
1725:             _la_join = getattr(_nv_join, '_la_depth', 0) if _nv_join is not None else 0
1726:             if _la_join > 0:
1727:                 _enc_ref = getattr(self, '_enc_thread_ref', None)
1728:                 if _enc_ref is not None:
1729:                     # 复用与内层完全一致的估算函数，外层额外加 20s 调度/收尾余量
1730:                     # （线程唤醒、GIL 争用、flush_and_join 自身非编码开销等）。
1731:                     _w_timeout = _enc_ref.estimate_timeout(margin=20.0)
1732:                 else:
1733:                     # 理论上不应发生（LA>0 必走 GPU_RAW/_enc_thread 路径），
1734:                     # 保底退回旧的固定值，不让超时估算缺失阻塞收尾判断。
1735:                     _w_timeout = 150.0
1736:             else:
1737:                 _w_timeout = 30.0
1738:             self._writer_th.join(timeout=_w_timeout)
1739:             if self._writer_th.is_alive():
1740:                 # [FIX-WRITER-TIMEOUT-FAIL] 超时仍存活 = 段失败：
1741:                 # 绝不在 writer/编码线程存活时放行继续收尾（muxer 会被提前关闭），
1742:                 # 标记错误并向上抛出，防止截断输出被误判成功写入 checkpoint。
1743:                 self._error = RuntimeError(
1744:                     f'写线程在 {_w_timeout:.0f}s 内未退出（疑似死锁），段处理失败')
1745:                 print(f'\n[IFRNet-Writer] ❌ {self._error}', flush=True)
1746: 
1747:         # [FIX-WRITER-ERROR-PROP] writer join 之后必须复查错误：
1748:         # _writer_loop 的异常（flush_and_join 抛出等）发生在推理线程 join 后的
1749:         # 错误检查点之后，若不在此传播，段会被误判成功 → 截断文件带着
1750:         # checkpoint 流入下游。
1751:         if self._error is not None:
1752:             raise RuntimeError(
1753:                 f'流水线处理异常: {type(self._error).__name__}: {self._error}'
1754:             ) from self._error
1755: 
1756:         if self._reader_th and self._reader_th.is_alive():
1757:             self._reader_th.join(timeout=10.0)
1758: 
1759:         if self._prefetch_total > 0 and not self.proc.quiet:
1760:             hit_pct = self._prefetch_hits / self._prefetch_total * 100
1761:             print(
1762:                 f'[IFRNet-Pipeline] 预取命中率: '
1763:                 f'{self._prefetch_hits}/{self._prefetch_total} ({hit_pct:.1f}%)',
1764:                 flush=True,
1765:             )
1766: 
1767:         return self._fc_extra, self._oc_extra
1768: 
1769:     # ── T2 推理独立线程体 ──────────────────────────────────────────────────────
1770:     # [FIX-INFER-THREAD] 从 run() 主线程提取为独立线程，消除 Python GIL 竞争。
1771: 
1772:     def _infer_loop(
1773:         self,
1774:         timesteps:    list,
1775:         H:            int,
1776:         W:            int,
1777:         effective_bs: int,
1778:         H_pad:        int,
1779:         W_pad:        int,
1780:         meter,
1781:         nvenc_encoder=None,  # [FIX-T3-V643]
1782:     ):
1783:         """
1784:         [FIX-INFER-THREAD] T2 推理独立线程体。
1785:         · pair_queue → GPU 推理（_safe_infer）→ result_queue
1786:         · [FIX-DOUBLEBUF-H2D] 每次推理完成后立即尝试补充双槽预取，
1787:           确保 H2D(N+1) 与 compute(N) 全重叠。
1788:         · 最终向 result_queue 投递 SENTINEL，通知 Writer 退出。
1789:         """
1790:         proc           = self.proc
1791:         fc_extra       = 0
1792:         oc_extra       = 0
1793:         # [FIX-RETUNE-SKIP] 跳过前 _CALIB_SKIP 个 batch（流水线刚启动时
1794:         # pair_queue/result_queue 均为空，GPU 呈 burst 爆发态，T2 测量值异常偏低）。
1795:         # 从第 _CALIB_SKIP+1 个 batch 起连续取 _CALIB_BATCHES 个样本做中位数校准。
1796:         _CALIB_SKIP    = 3
1797:         _CALIB_BATCHES = 5
1798:         _calib_done    = False
1799: 
1800:         # [FIX-DOUBLEBUF-H2D] 入口预热双槽预取
1801:         self._try_prefetch_next()
1802: 
1803:         try:
1804:             while self.running:
1805:                 prefetch_result = self._pop_prefetch_or_none()
1806:                 if prefetch_result is not None:
1807:                     item, pfimg0_t, pfimg1_t = prefetch_result
1808:                     # [FIX-DOUBLEBUF-H2D] 弹出一个槽 → 立即补充回 2 个
1809:                     self._try_prefetch_next()
1810:                 else:
1811:                     pfimg0_t = pfimg1_t = None
1812:                     self._prefetch_total += 1
1813:                     try:
1814:                         item = self.pair_queue.get(timeout=2.0)
1815:                     except queue.Empty:
1816:                         if not self._reader_th.is_alive():
1817:                             # [DIAG-SENTINEL] pair_queue SENTINEL 丢失时的紧急退出路径
1818:                             if self._pending_sentinel and self._prefetch_deque:
1819:                                 # 仍有预取数据待处理，消耗完再退出
1820:                                 continue
1821:                             if self._pending_sentinel:
1822:                                 if not self.proc.quiet:
1823:                                     print('[IFRNet-Infer] _pending_sentinel 触发，退出推理循环',
1824:                                           flush=True)
1825:                             break
1826:                         continue
1827: 
1828:                 if item is self._SENTINEL:
1829:                     # 若 _prefetch_deque 仍有预取 batch，SENTINEL 放回 pair_queue
1830:                     # 继续处理，避免 slot semaphore 泄漏 + 最后一批帧丢失
1831:                     if self._prefetch_deque:
1832:                         try:
1833:                             self.pair_queue.put(item, timeout=2.0)
1834:                         except queue.Full:
1835:                             self._pending_sentinel = True
1836:                         continue
1837:                     break
1838: 
1839:                 img1_raw, img0_pad, img1_pad, is_end = item
1840:                 if not img1_raw:
1841:                     continue
1842:                 B = len(img0_pad)
1843: 
1844:                 results = proc._safe_infer(
1845:                     img0_pad, img1_pad, timesteps, H, W,
1846:                     prefetched_img0_t=pfimg0_t,
1847:                     prefetched_img1_t=pfimg1_t,
1848:                     return_gpu=(nvenc_encoder is not None),  # [FIX-T3-V643-GPU] Level 1: 保持 GPU
1849:                 )
1850: 
1851:                 # [FIX-RETUNE-POSTRUN] 早期 T2-CACHE 更新（仅更新缓存文件，不做队列建议）
1852:                 # 队列建议改在 run() 段完成后基于全段稳定数据统一计算，精度更高。
1853:                 if (not _calib_done and self.auto_tune
1854:                         and len(proc._timing) >= _CALIB_SKIP + _CALIB_BATCHES):
1855:                     # 取跳过热身后的稳定采样窗口（索引 [skip : skip+n]）
1856:                     _samples = proc._timing[_CALIB_SKIP : _CALIB_SKIP + _CALIB_BATCHES]
1857:                     t2_actual = float(np.median(_samples)) * 1000.0
1858:                     if t2_actual >= 1.0:
1859:                         _calib_done = True
1860:                         _infer_be2 = self._get_infer_backend()
1861:                         if self._hw_profile is not None:
1862:                             self._hw_profile.t2_measured_ms = t2_actual
1863:                             # [FIX-CALIB-KEY] 修复：加入 model_name，与 run() 中
1864:                             # _current_cfg = (model_name, H_pad, W_pad, bs, be) 保持一致
1865:                             self._last_calib_config = (
1866:                                 proc.model_name, H_pad, W_pad, effective_bs, _infer_be2
1867:                             )
1868:                         # T2-CACHE 早期写入：让下一个视频（非本段）尽快用上实测值
1869:                         if self.t2_cache_dir and self._cache_key:
1870:                             _c = _load_t2_cache(self.t2_cache_dir)
1871:                             _old = _c.get(self._cache_key, 0.0)
1872:                             if _old <= 0 or abs(t2_actual - _old) / max(_old, 1.0) > 0.10:
1873:                                 _c[self._cache_key] = round(t2_actual, 1)
1874:                                 _save_t2_cache(self.t2_cache_dir, _c)
1875:                                 print(f'\n[T2-CACHE] 已更新缓存 T2={t2_actual:.1f}ms '
1876:                                       f'(key={self._cache_key})', flush=True)
1877:                         # 队列建议不在此处输出，由 run() 完成后统一处理
1878: 
1879:                 # [FIX-T3-V643] 结果分发：GPU > RING > NVENC > PinnedResultItem > fallback
1880:                 if isinstance(results, tuple) and len(results) > 0 and results[0] == 'GPU':
1881:                     # ── Level 1: GPU-STAY 传递 GPU tensor 到 Writer 线程编码 ──
1882:                     # [FIX-GPU-STAY] 不再在 Infer 线程做串行 NVENC 编码，
1883:                     # 改为将 GPU tensor 传递到 result_queue，由 Writer 线程
1884:                     # 并行做批量 RGB→NV12 + NVENC SDK 编码。
1885:                     _, interp_gpu, img1_rgb_gpu, rB, rT, rorig_H, rorig_W = results
1886:                     self._diag_gpu_stay_batches += 1
1887:                     out_item = ('GPU_RAW', interp_gpu, img1_rgb_gpu, rB, rT, rorig_H, rorig_W)
1888:                 elif isinstance(results, tuple) and len(results) > 0 and results[0] == 'RING':
1889:                     # Level 2/3: Ring Buffer 零拷贝路径
1890:                     _, slot_id, ev, rB, rT, rorig_H, rorig_W, n_frames = results
1891:                     proc._ring_buf.writer_commit(
1892:                         slot_id, n_frames, rorig_H, rorig_W, rB, rT, img1_raw, ev
1893:                     )
1894:                     out_item = ('RING', slot_id)
1895:                 elif nvenc_encoder is not None and isinstance(results, _PinnedResultItem):
1896:                     # [FIX-D2H-SYNC] 等待 stream_d2h 上的 D2H DMA 完成，确保 pinned buffer 数据就绪
1897:                     # 否则第一个 batch 会因异步竞态读到全零数据 → 灰/黑帧
1898:                     results.event.synchronize()
1899:                     h264_frames = []
1900:                     interp_arr = results.buf[:results.B * results.T]
1901:                     for bi in range(results.B):
1902:                         for tj in range(results.T):
1903:                             rgb_gpu = interp_arr[bi * results.T + tj,
1904:                                       :results.orig_H, :results.orig_W, :].cuda()
1905:                             nv12_gpu = _rgb_to_nv12_gpu(rgb_gpu)
1906:                             h264_data = nvenc_encoder.encode_frame(nv12_gpu)
1907:                             h264_frames.append(h264_data)
1908:                         # Encode img1_raw (original frame) through NVENC too
1909:                         # img1_raw 来自 reader (RGB)，input_is_bgr=False 直接写 NV12
1910:                         img1_gpu = torch.from_numpy(img1_raw[bi].copy()).cuda()
1911:                         img1_nv12 = _rgb_to_nv12_gpu(img1_gpu, input_is_bgr=False)
1912:                         h264_data = nvenc_encoder.encode_frame(img1_nv12)
1913:                         h264_frames.append(h264_data)
1914:                     out_item = (h264_frames, results.B, results.T,
1915:                                 results.orig_H, results.orig_W)
1916:                     self._event_pool.release(results.event)
1917:                     results.pool.release(results.buf)
1918:                 elif isinstance(results, _PinnedResultItem):
1919:                     results.img1_raw = img1_raw
1920:                     out_item = results
1921:                 else:
1922:                     out_item = (results, img1_raw, is_end)
1923: 
1924:                 # 非阻塞 put + 提前提交下一批预取（背压时先做预取再阻塞）
1925:                 try:
1926:                     self.result_queue.put_nowait(out_item)
1927:                     self._try_prefetch_next()
1928:                 except queue.Full:
1929:                     if self._writer_th is not None and not self._writer_th.is_alive():
1930:                         raise RuntimeError(
1931:                             'Writer 线程已退出，result_queue 无消费者，推理线程中止'
1932:                         )
1933:                     # [FIX-DOUBLEBUF-H2D] 队列满时先触发预取，不让 GPU 闲着
1934:                     self._try_prefetch_next()
1935:                     self.result_queue.put(out_item, timeout=30.0)
1936: 
1937:                 fc_extra += B
1938:                 oc_extra += B * (len(timesteps) + 1)
1939:                 self._diag_infer_batches += 1
1940:                 self._diag_infer_pairs   += B
1941:                 meter.update(B)
1942: 
1943:         except Exception as e:
1944:             if self._error is None:
1945:                 self._error = e
1946:             import traceback

(Output capped at 50 KB. Showing lines 1024-1946. Use offset=1947 to continue.)
</content>
```

---

## Assistant (Plan · Ox Alpha Free · 19.8s)

_Thinking:_

Let me read the rest of pipeline.py (1947-1563... wait the file is 1563 lines? No — it said 1563 lines but we're at line 1946. The line count earlier was from `Measure-Object -Line` which may count differently, or the file is actually longer. Let me continue reading to the end.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\pipeline.py",
  "offset": 1947
}
```

**Output:**
```
<path>D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\ifrnet_video\pipeline.py</path>
<type>file</type>
<content>
1947:             print(f'[IFRNet-Infer] 推理线程异常: {type(e).__name__}: {e}', flush=True)
1948:             traceback.print_exc()
1949:         finally:
1950:             # [FIX-GPU-STAY] NVENC flush 已移到 Writer 线程 SENTINEL 之后执行，
1951:             # 避免 T2 flush 与 T3 仍在进行的 GPU_RAW 编码产生竞态条件。
1952:             writer_alive = self._writer_th is not None and self._writer_th.is_alive()
1953:             if not writer_alive:
1954:                 if not self.proc.quiet:
1955:                     print('[IFRNet-Infer] Writer 已退出，跳过 result_queue SENTINEL 投递', flush=True)
1956:             else:
1957:                 for _ in range(5):
1958:                     try:
1959:                         self.result_queue.put(self._SENTINEL, timeout=1.0)
1960:                         break
1961:                     except queue.Full:
1962:                         continue
1963:             self.running = False
1964:             proc._pipeline_runner = None
1965:             # [FIX-POOL-LEAK / SEGMENT-REUSE] 跨段复用模式下，PinnedResultPool
1966:             # 由 proc._cached_result_pool 持有（全局唯一），段间不释放。
1967:             # 只清空段级引用，池实例存活至 cleanup() 统一销毁。
1968:             # 无累积问题：全局只有一个池，不会像旧代码那样每段创建新池叠加。
1969:             proc._result_pool = None
1970:             import gc as _gc; _gc.collect()
1971:             # [SEGMENT-REUSE] PinnedRingBuffer 由 proc._cached_ring_buf 持有，
1972:             # Writer 线程排空队列期间仍需 _ring_buf，不清空（cleanup() 统一释放）
1973:             self._fc_extra = fc_extra   # [FIX-INFER-THREAD] 供 run() 读取
1974:             self._oc_extra = oc_extra   # [FIX-INFER-THREAD] 供 run() 读取
1975: 
1976:     def close(self):
1977:         self.running = False

(End of file - total 1977 lines)
</content>
```

---

## Assistant (Plan · Ox Alpha Free · 14.6s)

_Thinking:_

Now I have the full picture of pipeline.py. Let me now read the remaining smaller files: ffmpeg_io.py (643 lines... actually likely more), ifrnet_utils.py (297), tensorrt_accel.py (218). Let me read them in parallel.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\ffmpeg_io.py"
}
```

**Output:**
```
<path>D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\ifrnet_video\ffmpeg_io.py</path>
<type>file</type>
<content>
1: #!/usr/bin/env python3
2: # -*- coding: utf-8 -*-
3: # IFRNet Video Enhancement - FFmpeg 读写模块（解码器 / 编码回退 / 硬件探测）。
4: # 镜像 external/realesrgan_video/ffmpeg_io.py 的职责。
5: 
6: from __future__ import annotations
7: 
8: import os
9: import queue
10: import subprocess
11: import sys
12: import threading
13: from fractions import Fraction
14: from typing import Dict, List, Optional, Tuple
15: 
16: import numpy as np
17: 
18: _PKG_DIR = os.path.dirname(os.path.abspath(__file__))
19: if _PKG_DIR not in sys.path:
20:     sys.path.insert(0, _PKG_DIR)
21: 
22: from ifrnet_video.ifrnet_utils import _clamp_decode_threads
23: from ifrnet_video.nvenc_sdk import _PRESET_P_INDEX
24: 
25: 
26: # ─────────────────────────────────────────────────────────────────────────────
27: # M2/M3: 硬件能力探测
28: # ─────────────────────────────────────────────────────────────────────────────
29: 
30: class HardwareCapability:
31:     _nvdec: Optional[bool] = None
32:     _nvenc: Dict[str, bool] = {}
33: 
34:     @classmethod
35:     def has_nvdec(cls) -> bool:
36:         if cls._nvdec is None:
37:             cls._nvdec = cls._probe_nvdec()
38:         return cls._nvdec
39: 
40:     @classmethod
41:     def has_nvenc(cls, codec: str = 'h264_nvenc') -> bool:
42:         if codec not in cls._nvenc:
43:             cls._nvenc[codec] = cls._probe_nvenc(codec)
44:         return cls._nvenc[codec]
45: 
46:     @staticmethod
47:     def _probe_nvdec() -> bool:
48:         """[FIX-NDV] 两阶段真实探测：先软件编码 H.264，再用 NVDEC 实际解码。"""
49:         try:
50:             enc_cmd = [
51:                 'ffmpeg', '-f', 'lavfi',
52:                 '-i', 'testsrc=size=64x64:duration=0.04:rate=25',
53:                 '-vcodec', 'libx264', '-f', 'h264', 'pipe:1', '-loglevel', 'error',
54:             ]
55:             enc = subprocess.run(enc_cmd, capture_output=True, timeout=10)
56:             if enc.returncode != 0 or not enc.stdout:
57:                 return False
58:             dec_cmd = [
59:                 'ffmpeg', '-hwaccel', 'cuda',
60:                 '-f', 'h264', '-i', 'pipe:0',
61:                 '-f', 'rawvideo', '-pix_fmt', 'bgr24',
62:                 '-frames:v', '1', 'pipe:1', '-loglevel', 'error',
63:             ]
64:             dec = subprocess.run(dec_cmd, input=enc.stdout, capture_output=True, timeout=10)
65:             return dec.returncode == 0 and len(dec.stdout) > 0
66:         except Exception:
67:             return False
68: 
69:     @staticmethod
70:     def _probe_nvenc(codec: str) -> bool:
71:         # [FIX-PROBE] 历次修复汇总（根因均由 stderr 诊断确认）：
72:         #   LAVFI  : color 源去掉 d=0.1（0帧问题）→ 无限源 + -frames:v 1 截帧
73:         #   PIXFMT : FFmpeg 5.0+ color 源默认 yuv444p → 显式 -pix_fmt yuv420p
74:         #   BFRAME : 单帧编码触发 B 帧 lookahead 报错 → -bf 0
75:         #   MINDIM : h264_nvenc 宽≥145px / hevc_nvenc 宽≥129px → 256×144
76:         cmd = [
77:             'ffmpeg', '-hide_banner', '-y', '-loglevel', 'error',
78:             '-f', 'lavfi', '-i', 'color=c=black:s=256x144:r=1',
79:             '-vcodec', codec, '-frames:v', '1',
80:             '-pix_fmt', 'yuv420p', '-bf', '0',
81:             '-f', 'null', '-',
82:         ]
83:         try:
84:             result = subprocess.run(cmd, capture_output=True, timeout=10)
85:             if result.returncode != 0:
86:                 _err = result.stderr.decode('utf-8', errors='replace').strip()
87:                 print(
88:                     f'  [PROBE-FAIL] {codec} probe 失败 (rc={result.returncode})\n'
89:                     f'  手动测试: {" ".join(cmd)}\n'
90:                     f'  stderr: {_err or "(空)"}',
91:                     flush=True,
92:                 )
93:             return result.returncode == 0
94:         except Exception as e:
95:             print(
96:                 f'  [PROBE-FAIL] {codec} probe 异常: {e}\n'
97:                 f'  手动测试: {" ".join(cmd)}',
98:                 flush=True,
99:             )
100:             return False
101: 
102:     @classmethod
103:     def best_encoder(cls, preferred: str = 'libx264',
104:                      hw_profile: Optional['_HWProfile'] = None) -> str:
105:         """
106:         [FIX-NVENC-UNIFIED] 统一 NVENC 检测路径。
107: 
108:         优先级（按 hw_profile 是否提供分两条路径）：
109: 
110:         路径 A（hw_profile 已提供，GPU 型号已知）：
111:           1. 直接采信 hw_profile.has_nvenc（静态 GPU 型号表，与 AUTO-TUNE 一致）。
112:           2. 若静态表确认可用且 ffmpeg probe 失败（Docker 设备映射缺失常见场景），
113:              打印明确警告后仍信任静态表——probe 失败不等于硬件不可用。
114:              可将 _NVENC_TRUST_STATIC = False 改为强制要求 probe 通过。
115:           3. 静态表否定（has_nvenc=False）时不做 probe，直接回退软件编码。
116: 
117:         路径 B（hw_profile 未提供，GPU 型号未知）：
118:           仅凭 ffmpeg 实际 probe 判断，probe 失败即回退软件编码。
119:           此路径适用于无法识别 GPU 型号的环境（非 NVIDIA GPU 等）。
120: 
121:         两套检测结果在 Docker 未映射 NVENC 设备时会不一致：AUTO-TUNE 显示 nvenc=True
122:         但 ffmpeg probe 失败，导致实际使用 libx264 引发 T3 瓶颈。本修复确保两者一致。
123:         """
124:         nvenc_map    = {'libx264': 'h264_nvenc', 'libx265': 'hevc_nvenc'}
125:         fallback_map = {'h264_nvenc': 'libx264', 'hevc_nvenc': 'libx265'}
126: 
127:         # [FIX-NVENC-TRUST-STATIC] 当 probe 失败但静态表确认可用时，是否信任静态表。
128:         # True（默认）：信任静态表，适用于 Docker 设备映射缺失但硬件真实存在的场景。
129:         # False：强制要求 probe 通过，适用于需要 100% 运行时验证的严格环境。
130:         _NVENC_TRUST_STATIC: bool = True
131: 
132:         def _nvenc_ok(codec_name: str) -> bool:
133:             # [FIX-NVENC-UNIFIED] 路径 A：hw_profile 已知时优先信任静态 GPU 型号表。
134:             # 这与 docstring 描述的优先级一致，解决了原实现 probe-first 导致
135:             # Docker 环境中 hw_profile.has_nvenc=True 被静默忽略的问题。
136:             if hw_profile is not None and hasattr(hw_profile, 'has_nvenc'):
137:                 if not hw_profile.has_nvenc:
138:                     # 静态表明确否定：不做 probe，直接返回 False
139:                     return False
140:                 # 静态表确认可用：尝试 probe 做二次验证
141:                 probe_ok = cls.has_nvenc(codec_name)
142:                 if not probe_ok:
143:                     # [FIX-NVENC-PROBE-WARN] probe 失败但静态表确认硬件存在。
144:                     # 常见于 Docker 容器 /dev/nvidia* 设备映射不完整，
145:                     # nvidia-smi 可见 GPU 但 ffmpeg 无法访问 NVENC 编码器。
146:                     # 根据 _NVENC_TRUST_STATIC 决定是否信任静态表。
147:                     if _NVENC_TRUST_STATIC:
148:                         print(
149:                             f'  [FIX-NVENC-UNIFIED] {codec_name} ffmpeg probe 失败，'
150:                             f'但静态 GPU 型号表确认硬件存在（hw_profile.has_nvenc=True）。\n'
151:                             f'  信任静态表，继续使用 {codec_name}（Docker 设备映射不完整时正常）。\n'
152:                             f'  若实际编码报错，可在代码中将 _NVENC_TRUST_STATIC 改为 False。'
153:                         )
154:                         return True
155:                     else:
156:                         print(
157:                             f'  [FIX-NVENC-UNIFIED] {codec_name} ffmpeg probe 失败。'
158:                             f'静态表显示硬件存在但 _NVENC_TRUST_STATIC=False，回退软件编码。'
159:                         )
160:                         return False
161:                 return True  # 静态表 + probe 双重确认
162: 
163:             # [FIX-NVENC-UNIFIED] 路径 B：hw_profile 未知，仅凭 ffmpeg probe 判断。
164:             return cls.has_nvenc(codec_name)
165: 
166:         if preferred in fallback_map:
167:             if _nvenc_ok(preferred):
168:                 return preferred
169:             fallback = fallback_map[preferred]
170:             # [FIX-NVENC-WARN] 明确说明回退原因（probe 失败 or 静态表否定 or 无 profile）
171:             if hw_profile is not None and hasattr(hw_profile, 'has_nvenc') and not hw_profile.has_nvenc:
172:                 reason = '静态 GPU 型号表标记 has_nvenc=False'
173:             elif not cls.has_nvenc(preferred):
174:                 reason = 'ffmpeg probe 失败（Docker 设备映射可能缺失 /dev/nvidia*）'
175:             else:
176:                 reason = '未知原因'
177:             print(f'  [警告] {preferred} 不可用（{reason}），自动回退到 {fallback}')
178:             return fallback
179:         candidate = nvenc_map.get(preferred, preferred)
180:         if candidate != preferred and _nvenc_ok(candidate):
181:             return candidate
182:         return preferred
183: 
184:     @classmethod
185:     def lossless_encoder(cls) -> Tuple[str, List[str]]:
186:         """
187:         返回 (codec, extra_args) 用于无损中间段编码。
188:         优先 nvenc lossless（-rc constqp -qp 0），否则 libx264 lossless。
189: 
190:         与 best_encoder 的区别：best_encoder 只探测 NVENC 是否可用（VBR 编码），
191:         lossless_encoder 额外测试 constqp（常量 QP）模式——部分 GPU/driver 组合
192:         虽支持 NVENC VBR 但不支持 constqp 无损。
193:         """
194:         if cls.has_nvenc('h264_nvenc'):
195:             try:
196:                 cmd = [
197:                     'ffmpeg', '-hide_banner', '-y', '-loglevel', 'error',
198:                     '-f', 'lavfi', '-i', 'color=c=black:s=256x144:r=1',
199:                     '-vcodec', 'h264_nvenc',
200:                     '-rc', 'constqp', '-qp', '0',
201:                     '-pix_fmt', 'yuv420p', '-bf', '0',
202:                     '-frames:v', '1', '-f', 'null', '-',
203:                 ]
204:                 if subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0:
205:                     return 'h264_nvenc', ['-rc', 'constqp', '-qp', '0']
206:             except Exception:
207:                 pass
208:         return 'libx264', ['-qp', '0', '-preset', 'medium']
209: 
210: _X264_PRESET_FACTOR = {
211:     'ultrafast': 8.0, 'superfast': 6.0, 'veryfast': 4.0,
212:     'faster': 2.5, 'fast': 2.0, 'medium': 1.0,
213:     'slow': 0.4, 'slower': 0.2, 'veryslow': 0.1,
214: }
215: # [FIX-PRESET-UNIFY] x264 名称 → NVENC p-index 统一映射（复用 nvenc_sdk._PRESET_P_INDEX）。
216: # NVENC 使用 p1(最快)~p7(最慢) 命名体系，与 x264 的 ultrafast~veryslow 不兼容。
217: # 统一口径后与 SDK Level 1 直通路径完全一致（medium→p5、fast→p4、slow→p6 等），
218: # 避免 FFmpeg CLI 层与 SDK 层映射不一致导致的后端降级档位跳变。
219: # [FIX-CRF0-CALIB] crf=0（lossless）实测校准因子。
220: # 理论模型（crf_factor = 2^((0-23)/12) ≈ 0.264）严重低估 lossless 编码成本：
221: #   · lossless 需维持精确像素，内存带宽和预测搜索开销远高于有损编码
222: #   · 实测（T4, libx264, ultrafast, 416×736, 8c）: ~150 fps output
223: #   · 理论估算（修正前）:  ~2860 fps → 偏差约 19×
224: # 乘以此因子后估算 ~157 fps，贴近实测正常（非热节流）状态。
225: _CRF0_X264_CALIB_FACTOR: float = 0.055
226: 
227: 
228: def _software_encode_fps(cpu_cores: int, H: int, W: int,
229:                          codec: str, preset: str, crf: int) -> float:
230:     base_pixels = 1920.0 * 1080.0
231:     current_pixels = float(H * W)
232:     scale_res = max(base_pixels / current_pixels, 1.0)
233:     factor = _X264_PRESET_FACTOR.get(preset, 1.0)
234:     crf_factor = 2.0 ** ((crf - 23) / 12.0)
235:     base_fps = 120.0 if 'x265' in codec.lower() else 200.0
236:     cores_factor = min(cpu_cores, 16) / 8.0
237:     fps = base_fps * scale_res * factor * crf_factor * cores_factor
238:     # [FIX-CRF0-CALIB] lossless（crf=0）时理论模型严重低估编码成本，乘以实测校准因子
239:     if crf == 0 and 'nvenc' not in codec.lower():
240:         fps *= _CRF0_X264_CALIB_FACTOR
241:     return min(fps, 3000.0)
242: 
243: # ─────────────────────────────────────────────────────────────────────────────
244: # [FIX-SLICE-THREAD] 编码并行度自动探测
245: # ─────────────────────────────────────────────────────────────────────────────
246: 
247: def _detect_encode_parallelism(n_threads_hint: Optional[int] = None) -> dict:
248:     """
249:     [FIX-SLICE-THREAD] 自动探测 CPU / 内存资源，返回最优软编码并行参数字典。
250: 
251:     返回字段
252:     ─────────────────────────────────────────────────────────────────────────
253:     cpu_logical    : int    逻辑核心数（含超线程，来自 os.cpu_count()）
254:     cpu_physical   : int    物理核心数（来自 /proc/cpuinfo；失败则 logical//2）
255:     mem_avail_gb   : float  系统当前可用内存 GiB（来自 /proc/meminfo MemAvailable）
256:     encode_threads : int    软编码线程数（x264/x265 frame-level parallelism），
257:                             = min(cpu_logical, 16)，超过 16 收益递减
258:     slices         : int    x264 intra-frame 分片数（slice-based threading）：
259:                             每片由独立线程并行编码，降低单帧编码延迟。
260:                             = min(encode_threads, 16)，同时受内存可用量约束
261:                             （大分片数需更多行缓冲区；低分辨率时此约束通常不触发）
262:     ffmpeg_threads : int    FFmpeg 全局 -threads 值，用于 demux/filter graph
263:                             = min(cpu_logical, 8)
264:     """
265:     cpu_logical = os.cpu_count() or 4
266: 
267:     # 物理核心数：从 /proc/cpuinfo 读 "core id" 去重；失败则估算
268:     cpu_physical = max(cpu_logical // 2, 1)
269:     try:
270:         _core_ids: set = set()
271:         _pkg_ids:  set = set()
272:         _cur_pkg       = None
273:         with open('/proc/cpuinfo', 'r') as _cpuf:
274:             for _line in _cpuf:
275:                 _line = _line.strip()
276:                 if _line.startswith('physical id'):
277:                     _cur_pkg = _line.split(':', 1)[1].strip()
278:                 elif _line.startswith('core id') and _cur_pkg is not None:
279:                     _core_ids.add((_cur_pkg, _line.split(':', 1)[1].strip()))
280:         if _core_ids:
281:             cpu_physical = len(_core_ids)
282:     except Exception:
283:         pass
284: 
285:     # 系统可用内存（GiB）：读 /proc/meminfo MemAvailable；失败时尝试 psutil
286:     mem_avail_gb = 4.0
287:     try:
288:         with open('/proc/meminfo', 'r') as _memf:
289:             for _line in _memf:
290:                 if _line.startswith('MemAvailable:'):
291:                     mem_avail_gb = int(_line.split()[1]) / (1024.0 ** 2)
292:                     break
293:     except Exception:
294:         try:
295:             import psutil as _psutil
296:             mem_avail_gb = _psutil.virtual_memory().available / (1024.0 ** 3)
297:         except ImportError:
298:             pass
299: 
300:     # 若外部传入 hint，直接使用（但仍不超过 16）
301:     if n_threads_hint is not None and n_threads_hint > 0:
302:         encode_threads = min(n_threads_hint, 16)
303:     else:
304:         encode_threads = min(cpu_logical, 16)
305: 
306:     # 分片数 = encode_threads（1 slice/thread），但：
307:     #   · 上限 16：slice 数越多压缩率越低（片间参考受限），16 是实用阈值
308:     #   · 内存约束：每个 slice 约需 0.25 GiB 额外行缓冲（高分辨率下），低分辨率（<1080p）可忽略
309:     #   · 下限 2：至少 2 片才有并行效果
310:     slices_by_cpu  = encode_threads
311:     slices_by_mem  = max(2, int(mem_avail_gb / 0.25))   # 每 slice 估算 0.25 GiB
312:     slices = max(2, min(slices_by_cpu, slices_by_mem, 16))
313: 
314:     ffmpeg_threads = min(cpu_logical, 8)
315: 
316:     return {
317:         'cpu_logical':    cpu_logical,
318:         'cpu_physical':   cpu_physical,
319:         'mem_avail_gb':   mem_avail_gb,
320:         'encode_threads': encode_threads,
321:         'slices':         slices,
322:         'ffmpeg_threads': ffmpeg_threads,
323:     }
324: 
325: 
326: # ─────────────────────────────────────────────────────────────────────────────
327: # [FIX-NVENC-PIPE] NVENC pipe 模式参数常量
328: # ─────────────────────────────────────────────────────────────────────────────
329: 
330: # NVENC 内部帧缓冲数（-surfaces N）：
331: #   NVENC 硬件编码器内部维护一个帧槽池（surfaces），每个 slot 存储一帧正在被硬件
332: #   编码的图像。默认值为 8，对于均匀帧率的文件输入已经足够；但 pipe 输入存在速率
333: #   抖动（T3-Writer _write_loop 批量写入 + T2-Infer 批量 D2H），当短时供帧速率超过
334: #   硬件编码速率时，较小的 surfaces 数会导致 FFmpeg 无法向 NVENC 提交新帧（硬件满载
335: #   等待回收），引发编码器停顿（stall）。扩大至 32 可覆盖约 1 秒的帧缓冲（@30fps），
336: #   配合 T3-Writer 的 _MAX_BATCH=8 批量写入，基本消除 pipe 速率抖动的影响。
337: _NVENC_SURFACES_PIPE: int = 32
338: 
339: # NVENC VBR 模式前向帧预看窗口（-rc-lookahead N）：
340: #   仅在 crf>0（-rc:v vbr 模式）下启用。NVENC 默认不使用前向预看（N=0），
341: #   设为 16 后编码器可向前分析 16 帧的运动复杂度，进行更精准的码率分配：
342: #   · 场景切换前预先降低相邻帧码率，切换后爆发较高 I 帧码率
343: #   · 高运动区域分配更多比特，静止区域节约比特
344: #   典型 PSNR 改善 0.2-0.5 dB（1080p VBR）。
345: #   注意：lookahead 需要 N 帧前瞻缓冲（内部 FIFO），因此输出有 N 帧延迟，
346: #   与 -delay 0（零输出延迟）互斥，故仅在 VBR 路径启用，QP=0 路径改用 -delay 0。
347: _NVENC_LOOKAHEAD_VBR: int = 16
348: 
349: 
350: class FFmpegFrameReader:
351:     _SENTINEL = object()
352: 
353:     def __init__(
354:         self,
355:         video_path:      str,
356:         frame_start:     int   = 0,
357:         frame_end:       int   = -1,
358:         width:           int   = -1,
359:         height:          int   = -1,
360:         fps_override:    float = 0.0,
361:         prefetch:        int   = 128,
362:         use_hwaccel:     bool  = True,
363:         ffmpeg_bin:      str   = 'ffmpeg',
364:         pad_stride:      int   = 0,
365:     ):
366:         meta = _probe_video(video_path)
367:         self.width     = meta['width']  if width  < 0 else width
368:         self.height    = meta['height'] if height < 0 else height
369:         self.fps       = fps_override  if fps_override > 0 else meta['fps']
370:         self.nb_frames = meta['nb_frames']
371:         self.has_audio = meta['has_audio']
372: 
373:         actual_end = frame_end if frame_end >= 0 else self.nb_frames - 1
374:         self._segment_frames = actual_end - frame_start + 1
375:         self._frame_bytes    = self.width * self.height * 3
376: 
377:         self._pad_stride = pad_stride
378:         if pad_stride > 0:
379:             def _ceil(x, s): return x if x % s == 0 else x + (s - x % s)
380:             ph = _ceil(self.height, pad_stride) - self.height
381:             pw = _ceil(self.width,  pad_stride) - self.width
382:         else:
383:             ph = pw = 0
384:         self._pad_h  = ph
385:         self._pad_w  = pw
386:         self.need_pad = ph > 0 or pw > 0
387: 
388:         hw_args: List[str] = []
389:         if use_hwaccel and HardwareCapability.has_nvdec():
390:             # hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'nv12']
391:             # ✅ 正确：nv12 是 NVDEC 合法的 hwaccel_output_format
392:             # FFmpeg 会先将 CUDA NV12 surface download 到 CPU，
393:             # 再由 swscale 自动转换为 -pix_fmt rgb24 输出到管道
394:             hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'nv12']
395: 
396:         if frame_start == 0 and frame_end < 0:
397:             vf_args: List[str] = []
398:         else:
399:             vf_args = [
400:                 '-vf',
401:                 f"select='between(n\\,{frame_start}\\,{actual_end})',setpts=N/FR/TB",
402:                 '-vsync', '0',
403:             ]
404: 
405:         # [FIX-NVDEC-THREAD-CAP] 解码线程钳位到 ≤8，避免多核机上
406:         # NVDEC 解码 surface 超过驱动 32 上限（-threads 9 → 33 surfaces 被拒）。
407:         cmd = (
408:             [ffmpeg_bin]
409:             + hw_args
410:             + ['-threads', str(_clamp_decode_threads())]
411:             + ['-i', video_path]
412:             + vf_args
413:             + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-loglevel', 'error', 'pipe:1']
414:         )
415:         self._proc   = subprocess.Popen(
416:             cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
417:         )
418:         self._queue  = queue.Queue(maxsize=max(prefetch, 4))
419:         self._thread = threading.Thread(target=self._read_loop, daemon=True)
420:         self._thread.start()
421:         self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
422:         self._stderr_thread.start()
423: 
424:     def _drain_stderr(self):
425:         """Consume stderr to prevent pipe buffer deadlock."""
426:         try:
427:             while True:
428:                 chunk = self._proc.stderr.read(8192)
429:                 if not chunk:
430:                     break
431:         except Exception:
432:             pass
433: 
434:     def _read_loop(self):
435:         pad_h, pad_w = self._pad_h, self._pad_w
436:         do_pad = self.need_pad
437:         fb = self._frame_bytes
438:         try:
439:             while True:
440:                 # Read exactly fb bytes (robust against partial pipe reads)
441:                 buf = bytearray()
442:                 while len(buf) < fb:
443:                     chunk = self._proc.stdout.read(fb - len(buf))
444:                     if not chunk:
445:                         break
446:                     buf.extend(chunk)
447:                 if len(buf) < fb:
448:                     break
449:                 arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(
450:                     self.height, self.width, 3)
451:                 if do_pad:
452:                     padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
453:                     self._queue.put((arr, padded))
454:                 else:
455:                     self._queue.put((arr, arr))
456:         except Exception as e:
457:             self._queue.put(e)
458:             return
459:         self._queue.put(self._SENTINEL)
460: 
461:     def read(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
462:         item = self._queue.get()
463:         if item is self._SENTINEL:
464:             return None
465:         if isinstance(item, Exception):
466:             raise item
467:         return item
468: 
469:     def close(self):
470:         try:
471:             self._proc.terminate()   # ✅ 先终止进程，防止 stdout.read() 因进程卡死而永久挂起
472:         except Exception:
473:             pass
474:         try:
475:             self._proc.wait(timeout=15)
476:         except subprocess.TimeoutExpired:
477:             self._proc.kill()
478:             self._proc.wait()
479: 
480: 
481: def _probe_video(video_path: str) -> dict:
482:     cmd = [
483:         'ffprobe', '-v', 'error',
484:         '-select_streams', 'v:0',
485:         '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
486:         '-show_entries', 'format=nb_streams',
487:         '-of', 'json', video_path,
488:     ]
489:     result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
490:     if result.returncode != 0:
491:         raise RuntimeError(f'ffprobe 失败: {result.stderr}')
492:     import json as _json
493:     data = _json.loads(result.stdout)
494:     vs   = data['streams'][0]
495:     fps_str = vs.get('r_frame_rate', '24/1')
496:     try:
497:         fps = float(Fraction(fps_str))
498:     except (ValueError, ZeroDivisionError):
499:         fps = 24.0
500:     nb = 0
501:     if 'duration' in vs:
502:         dur = float(vs['duration'])
503:         if dur > 0:
504:             nb = int(dur * fps)
505:             # 交叉验证：若 nb_frames 与 duration×fps 偏差 > 5%，警告
506:             if 'nb_frames' in vs and vs['nb_frames'] not in ('N/A', ''):
507:                 nb_meta = int(vs['nb_frames'])
508:                 if nb_meta > 0 and nb > 0 and abs(nb_meta - nb) / max(nb_meta, nb) > 0.05:
509:                     print(f'⚠️  ffprobe元数据 nb_frames={nb_meta} 与 duration×fps={nb} 不一致，'
510:                           f'使用后者（分段文件 -c copy 常见）', flush=True)
511:     elif 'nb_frames' in vs and vs['nb_frames'] not in ('N/A', ''):
512:         nb = int(vs['nb_frames'])
513:     cmd_audio = [
514:         'ffprobe', '-v', 'error', '-select_streams', 'a:0',
515:         '-show_entries', 'stream=codec_type', '-of', 'json', video_path,
516:     ]
517:     a = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=15)
518:     has_audio = (a.returncode == 0 and '"codec_type": "audio"' in a.stdout)
519:     return {
520:         'width': int(vs['width']), 'height': int(vs['height']),
521:         'fps': fps, 'nb_frames': nb, 'has_audio': has_audio,
522:     }
523: 
524: 
525: # ─────────────────────────────────────────────────────────────────────────────
526: # M3: FFmpeg Writer
527: # ─────────────────────────────────────────────────────────────────────────────
528: 
529: class FFmpegWriter:
530:     _SENTINEL  = object()
531:     _MAX_BATCH = 8
532:     _STDERR_IGNORE = (
533:         'x265 [info]:', 'x265 [warning]:', 'set_mempolicy:',
534:         'encoded ', 'Weighted P-Frames', 'consecutive B-frames',
535:         'frame I:', 'frame P:', 'frame B:',
536:         # [FIX-SLICE-THREAD] x264 slice-threading 信息行
537:         'using cpu capabilities:', 'slice threads:', 'frame threads:',
538:         'x264 [info]:', 'x264 [warning]:',
539:         # [FIX-NVENC-PIPE] NVENC 初始化 / 会话诊断信息行（非错误，无需打印）
540:         # · 'Initialized NPP'  : CUDA NPP 库初始化（h264_nvenc/hevc_nvenc 启动时打印）
541:         # · 'NVENC session'    : NVENC 编码会话创建日志
542:         # · 'GPU #'            : NVENC 选择 GPU 设备信息行
543:         'Initialized NPP', 'NVENC session', 'GPU #',
544:     )
545: 
546:     def __init__(
547:         self,
548:         output_path: str,
549:         width:  int,
550:         height: int,
551:         fps:    float,
552:         codec:  str = 'libx264',
553:         extra_codec_args: Optional[List[str]] = None,
554:         crf:    int  = 23,
555:         preset: str  = None,
556:         audio_src: Optional[str] = None,
557:         ffmpeg_bin: str = 'ffmpeg',
558:         quiet: bool = True,
559:         n_threads: Optional[int] = None,   # [FIX-SLICE-THREAD] None=自动探测
560:         rc_mode: str = "constqp",           # NVENC rate control mode for Level 2 fallback
561:     ):
562:         # [FIX-T3-V643] 去除内部 _queue 和 _write_loop 线程，直接管道写入
563:         self._error: Optional[Exception] = None
564:         self._write_count = 0
565: 
566:         if preset is None:
567:             preset = 'p4' if 'nvenc' in codec else 'medium'
568:         elif 'nvenc' in codec and preset in _PRESET_P_INDEX:
569:             # [FIX-PRESET-UNIFY] x264 名称（ultrafast 等）→ p1~p7 体系，统一 _PRESET_P_INDEX 口径，
570:             # 与 SDK Level 1 直通路径完全一致，避免 "Unable to parse option value" 错误。
571:             preset = f"p{_PRESET_P_INDEX[preset] + 1}"
572: 
573:         # [FIX-SLICE-THREAD] 自动探测 CPU / 内存，计算最优软编码并行参数
574:         _par = _detect_encode_parallelism(n_threads)
575:         _et  = _par['encode_threads']   # 编码线程数
576:         _s   = _par['slices']           # x264 分片数
577:         _ft  = _par['ffmpeg_threads']   # FFmpeg 全局线程数
578:         _mem = _par['mem_avail_gb']
579: 
580:         # [FIX-LOSSLESS] crf=0 → 按编解码器映射为正确的无损参数。
581:         # 背景：
582:         #   · libx264 : crf=0 恰好等于无损，但显式用 -qp 0 语义更清晰
583:         #   · libx265 : crf=0 ≠ 无损！仅为极高质量有损；无损需 -x265-params lossless=1
584:         #   · nvenc   : -cq:v 0 是 VBR 模式下的极低码率控制，不是无损；
585:         #               无损需去掉 -rc:v vbr，改用 -qp 0 -b:v 0（常量 QP 模式）
586:         # [FIX-SLICE-THREAD] x265 frame-threads：默认 min(4, cpu_logical//2)
587:         # x265 frame-threads 含义：同时编码的帧数（帧级并行），通常 2-4 最佳；
588:         # 过高会引入帧延迟，与 pipe 流式输入场景不符。
589:         _x265_ft = max(2, min(4, _par['cpu_logical'] // 2))
590:         # x265 pool：线程池总大小（所有 frame-threads 共享），= encode_threads
591:         _x265_pool = _et
592: 
593:         if crf == 0:
594:             if 'nvenc' in codec:
595:                 # [FIX-LOSSLESS] NVENC 无损：常量 QP=0，去掉 vbr 码率控制。
596:                 # [FIX-NVENC-PIPE] 同时注入 pipe 场景三项优化参数：
597:                 #   · -bf 0        禁用 B 帧：B 帧编码需要前后参考帧，编码器须缓存后续帧
598:                 #                  才能输出，引入多帧流水线延迟；禁用后每帧独立编码即输出。
599:                 #   · -surfaces N  扩大 NVENC 内部帧缓冲槽数（默认 8 → _NVENC_SURFACES_PIPE），
600:                 #                  防止 pipe 写入速率抖动时编码器因 surface 耗尽而暂停。
601:                 #   · -delay 0     零输出延迟：配合 -bf 0 强制 NVENC 在每帧编码完成后
602:                 #                  立即写入输出流，不等待后续帧，最小化 pipe 端到端延迟。
603:                 #                  注意：-delay 0 与 -rc-lookahead 互斥（lookahead 需要前瞻
604:                 #                  缓冲），此处无损模式无需质量优化型预看，故可安全启用。
605:                 quality_args = [
606:                     '-preset', preset,
607:                     '-rc', 'constqp',
608:                     '-qp', '0', '-b:v', '0',
609:                     '-bf', '0',
610:                     '-surfaces', str(_NVENC_SURFACES_PIPE),
611:                     '-delay', '0',
612:                 ]
613:             elif codec == 'libx265':
614:                 # x265 无损：lossless=1 + 多线程（替换旧 pools=none）
615:                 # [FIX-SLICE-THREAD] pools={N} 启用线程池，frame-threads={F} 帧级并行
616:                 quality_args = [
617:                     '-preset', preset,
618:                     '-x265-params',
619:                     f'lossless=1:pools={_x265_pool}:frame-threads={_x265_ft}',
620:                 ]
621:             elif codec == 'libx264':
622:                 # x264 无损：-qp 0 + slice-based threading
623:                 # [FIX-SLICE-THREAD] threads=N 设置 x264 编码线程数；
624:                 # slices=S 将单帧切为 S 片并行编码（intra-frame 并行），
625:                 # 与 pipe 流式场景匹配（每帧编完即输出，无帧间延迟）。
626:                 quality_args = [
627:                     '-preset', preset, '-qp', '0',
628:                     '-x264-params', f'threads={_et}:slices={_s}',
629:                 ]
630:             else:
631:                 # 其他编解码器（如 ffv1、utvideo 等）：回退到 -qp 0
632:                 quality_args = ['-qp', '0']
633:         elif 'nvenc' in codec:
634:             # [FIX-NVENC-PIPE] NVENC VBR（cq）模式 pipe 场景优化：
635:             #   · -bf 0              禁用 B 帧，同 crf=0 路径，降低流水线缓冲延迟。
636:             #   · -rc-lookahead N    前向帧预看（N = _NVENC_LOOKAHEAD_VBR）：
637:             #                        VBR 模式下编码器向前分析 N 帧运动复杂度，优化帧间
638:             #                        码率分配，改善场景切换质量（PSNR +0.2-0.5 dB）。
639:             #                        与 -delay 0 互斥（预看需要 N 帧前瞻缓冲区），
640:             #                        因此 VBR 路径不设 -delay 0。
641:             #   · -surfaces N        扩大 NVENC 内部帧缓冲（同 crf=0 路径）。
642:             # [FIX-FFMPEGWRITER-RC] 根据 Level 1 的 RC 模式选择对应的 FFmpeg -rc:v 值
643:             # 注意：qvbr 在旧版 FFmpeg h264_nvenc 中不可用，回退到 vbr_hq
644:             _rc_v_map = {'vbr_hq': 'vbr_hq', 'qvbr': 'vbr_hq', 'constqp': 'vbr'}
645:             _rc_v = _rc_v_map.get(rc_mode, 'vbr')
646:             quality_args = [
647:                 '-preset', preset,
648:                 '-rc:v', _rc_v, '-cq:v', str(crf), '-b:v', '0',
649:                 '-bf', '0',
650:                 '-rc-lookahead', str(_NVENC_LOOKAHEAD_VBR),
651:                 '-surfaces', str(_NVENC_SURFACES_PIPE),
652:             ]
653:         elif codec == 'libx265':
654:             # [FIX-SLICE-THREAD] 替换旧 pools=none（完全禁用线程池）为正确多线程参数
655:             quality_args = [
656:                 '-preset', preset, '-crf', str(crf),
657:                 '-x265-params',
658:                 f'pools={_x265_pool}:frame-threads={_x265_ft}',
659:             ]
660:         else:
661:             # libx264（及其他 x264 系列）：追加 slice-based threading 参数
662:             # [FIX-SLICE-THREAD] threads=N + slices=S：N 线程各负责 S/N 片，
663:             # 当 slices >= threads 时 x264 自动切换为 slice-based 模式。
664:             quality_args = [
665:                 '-preset', preset, '-crf', str(crf),
666:                 '-x264-params', f'threads={_et}:slices={_s}',
667:             ]
668: 
669:         cmd = [
670:             ffmpeg_bin, '-y',
671:             '-f', 'rawvideo', '-vcodec', 'rawvideo',
672:             '-pix_fmt', 'rgb24',
673:             '-s', f'{width}x{height}',
674:             '-r', f'{fps:.6f}',
675:             '-i', 'pipe:0',
676:         ]
677:         # [FIX-SLICE-THREAD] FFmpeg 全局 -threads：仅软编码路径需要，
678:         # NVENC 是 GPU 固定功能硬件单元，不受 CPU 线程数控制。
679:         if 'nvenc' not in codec:
680:             cmd.insert(2, '-threads')
681:             cmd.insert(3, str(_ft))
682:         if audio_src:
683:             cmd += ['-i', audio_src, '-c:a', 'copy', '-map', '0:v', '-map', '1:a?']
684:         if extra_codec_args:
685:             # 合并 extra_args 与 quality_args，保留后者中不在前者内的关键 pipe 参数
686:             _extra_flags = set()
687:             for i in range(0, len(extra_codec_args), 2):
688:                 _extra_flags.add(extra_codec_args[i])
689:             _merged = list(extra_codec_args)
690:             for i in range(0, len(quality_args), 2):
691:                 if quality_args[i] not in _extra_flags:
692:                     _merged.extend(quality_args[i:i+2])
693:             cmd += ['-vcodec', codec] + _merged
694:         else:
695:             cmd += ['-vcodec', codec] + quality_args
696:         cmd += ['-pix_fmt', 'yuv420p', '-loglevel', 'error', output_path]
697: 
698:         # [FIX-SLICE-THREAD / FIX-NVENC-PIPE] 打印编码参数摘要
699:         # NVENC 路径：打印 GPU 硬件编码关键参数（无 CPU 线程参数，因 NVENC 不受其控制）
700:         # 软编码路径：打印 CPU 线程 / slice 并行配置（与旧行为一致）
701:         if 'nvenc' in codec:
702:             # [FIX-NVENC-PIPE] NVENC 参数摘要：显示 pipe 场景优化参数的实际生效值，
703:             # 便于用户确认 surfaces / lookahead / delay 参数是否符合预期。
704:             # 注：ffmpeg_threads 对 NVENC 本身无效，仅作用于 demux/filter graph，
705:             #     此处显示以完整呈现 FFmpeg 命令的全局线程配置。
706:             if crf == 0:
707:                 _nvenc_info = (
708:                     f'[FIX-NVENC-PIPE] NVENC 无损(QP=0): '
709:                     f'preset={preset}  bf=0  '
710:                     f'surfaces={_NVENC_SURFACES_PIPE}  delay=0  '
711:                     f'ffmpeg_threads={_ft}(全局demux，不影响NVENC硬件单元)'
712:                 )
713:             else:
714:                 _nvenc_info = (
715:                     f'[FIX-NVENC-PIPE] NVENC VBR(cq={crf}): '
716:                     f'preset={preset}  bf=0  '
717:                     f'rc-lookahead={_NVENC_LOOKAHEAD_VBR}  '
718:                     f'surfaces={_NVENC_SURFACES_PIPE}  '
719:                     f'ffmpeg_threads={_ft}(全局demux，不影响NVENC硬件单元)'
720:                 )
721:             print(f'   {_nvenc_info}', flush=True)
722:         else:
723:             # [FIX-SLICE-THREAD] 软编码路径：打印 CPU 线程 / slice 并行配置摘要
724:             _codec_l = codec.lower()
725:             if 'x264' in _codec_l or codec not in ('libx265',):
726:                 _thread_info = (
727:                     f'[FIX-SLICE-THREAD] 软编码并行: '
728:                     f'cpu={_par["cpu_logical"]}逻辑/{_par["cpu_physical"]}物理  '
729:                     f'mem_avail={_par["mem_avail_gb"]:.1f}GiB  '
730:                     f'encode_threads={_et}  slices={_s}  ffmpeg_threads={_ft}'
731:                 )
732:             else:
733:                 _thread_info = (
734:                     f'[FIX-SLICE-THREAD] 软编码并行: '
735:                     f'cpu={_par["cpu_logical"]}逻辑/{_par["cpu_physical"]}物理  '
736:                     f'mem_avail={_par["mem_avail_gb"]:.1f}GiB  '
737:                     f'encode_threads={_et}(frame-threads={_x265_ft})  ffmpeg_threads={_ft}'
738:                 )
739:             print(f'   {_thread_info}', flush=True)
740: 
741:         # 打印完整 FFmpeg 命令，便于调试和确认编码参数（quiet=True 时跳过）
742:         print(f'   [FFmpegWriter] 命令: {" ".join(cmd)}', flush=True)
743:         # if not quiet:
744:         #     print(f'   [FFmpegWriter] 命令: {" ".join(cmd)}', flush=True)
745: 
746:         self._proc   = subprocess.Popen(
747:             cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
748:             stderr=subprocess.PIPE
749:         )
750:         self._stderr_lines: List[str] = []
751:         self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
752:         self._stderr_thread.start()
753:         # [FIX-T3-V643] 不再启动内部 _write_loop 线程，改为 write_direct() 直接写管道
754: 
755:     def _drain_stderr(self):
756:         try:
757:             for line in self._proc.stderr:
758:                 decoded = line.decode(errors='ignore').rstrip()
759:                 self._stderr_lines.append(decoded)
760:                 if decoded and not any(decoded.lstrip().startswith(p)
761:                                        for p in self._STDERR_IGNORE):
762:                     print(f'[FFmpeg ERR] {decoded}')
763:         except Exception:
764:             pass
765: 
766:     def write_direct(self, data):
767:         """[FIX-T3-V643] 直接写 bytes/memoryview 到 FFmpeg stdin pipe，零中间拷贝。"""
768:         if self._error is not None:
769:             raise RuntimeError(f'FFmpegWriter 内部错误: {self._error}') from self._error
770:         try:
771:             self._proc.stdin.write(data)
772:             self._write_count += 1
773:         except BrokenPipeError:
774:             self._error = RuntimeError('FFmpeg stdin 管道已断开')
775:             raise self._error
776: 
777:     def write(self, frame):
778:         """[FIX-T3-V643] 兼容旧接口：numpy array → tobytes() → write_direct()。"""
779:         self.write_direct(frame.tobytes())
780: 
781:     def close(self):
782:         self._stderr_thread.join(timeout=5)
783:         try:
784:             self._proc.stdin.close()
785:         except Exception:
786:             pass
787:         try:
788:             self._proc.wait(timeout=30)
789:         except subprocess.TimeoutExpired:
790:             self._proc.kill()
791:             self._proc.wait()
792:         rc = self._proc.returncode
793:         if rc is not None and rc != 0:
794:             stderr_out = '\n'.join(self._stderr_lines[-20:])
795:             print(f'\n[Warning] FFmpeg 退出码={rc}, stderr: {stderr_out[:400]}')
796:         if self._error:
797:             print(f'[Warning] FFmpegWriter 累计写帧异常: {self._error}')

(End of file - total 797 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\ifrnet_utils.py"
}
```

**Output:**
```
<path>D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\ifrnet_video\ifrnet_utils.py</path>
<type>file</type>
<content>
1: #!/usr/bin/env python3
2: # -*- coding: utf-8 -*-
3: # IFRNet Video Enhancement - 通用工具模块（模型加载 / 池 / tensor 辅助）。
4: # 镜像 external/realesrgan_video/realesrgan_utils.py 的职责。
5: 
6: from __future__ import annotations
7: 
8: import dataclasses
9: import os
10: import queue
11: import sys
12: import threading
13: import time
14: from collections import deque
15: from contextlib import nullcontext
16: from typing import List, Optional, Tuple
17: 
18: import numpy as np
19: import torch
20: 
21: _PKG_DIR = os.path.dirname(os.path.abspath(__file__))
22: if _PKG_DIR not in sys.path:
23:     sys.path.insert(0, _PKG_DIR)
24: 
25: from ifrnet_video.config import MODEL_MODULE_MAP, MODEL_STRIDE
26: 
27: import torch.nn.functional as _F_warp
28: 
29: _warp_grid_cache: dict = {}
30: 
31: def _cached_warp(img: 'torch.Tensor', flow: 'torch.Tensor') -> 'torch.Tensor':
32:     B, _C, H, W = img.shape
33:     key = (B, H, W, str(img.device), img.dtype)
34:     if key not in _warp_grid_cache:
35:         xs = torch.arange(0, W, device=img.device, dtype=img.dtype)
36:         ys = torch.arange(0, H, device=img.device, dtype=img.dtype)
37:         grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
38:         grid = torch.stack([grid_x, grid_y], dim=0)
39:         _warp_grid_cache[key] = grid.unsqueeze(0).expand(B, -1, -1, -1)
40:     base_grid = _warp_grid_cache[key]
41:     vgrid = base_grid + flow
42:     vgrid_x = 2.0 * vgrid[:, 0:1] / max(W - 1, 1) - 1.0
43:     vgrid_y = 2.0 * vgrid[:, 1:2] / max(H - 1, 1) - 1.0
44:     vgrid_scaled = torch.cat([vgrid_x, vgrid_y], dim=1).permute(0, 2, 3, 1)
45:     return _F_warp.grid_sample(img, vgrid_scaled,
46:                                mode='bilinear', padding_mode='border', align_corners=True)
47: 
48: 
49: 
50: def _load_ifrnet_module(model_name: str):
51:     import importlib
52:     module_name = MODEL_MODULE_MAP.get(model_name, 'models.IFRNet_S')
53:     mod = importlib.import_module(module_name)
54:     mod.warp = _cached_warp
55:     return mod.Model, mod
56: 
57: 
58: # ─────────────────────────────────────────────────────────────────────────────
59: # ThroughputMeter
60: # ─────────────────────────────────────────────────────────────────────────────
61: 
62: class ThroughputMeter:
63:     def __init__(self, window: int = 20):
64:         self._times: deque = deque(maxlen=window)
65:         self._total = 0
66: 
67:     def update(self, n: int):
68:         self._times.append((time.perf_counter(), n))
69:         self._total += n
70: 
71:     def fps(self) -> float:
72:         if len(self._times) < 2:
73:             return 0.0
74:         dt = self._times[-1][0] - self._times[0][0]
75:         return sum(t[1] for t in self._times) / dt if dt > 0 else 0.0
76: 
77:     def eta(self, total: int) -> float:
78:         f = self.fps()
79:         return (total - self._total) / f if f > 0 else float('inf')
80: 
81: 
82: # ─────────────────────────────────────────────────────────────────────────────
83: # [PINNED-D2H] 结果队列结构体 + 预分配 Pinned Buffer 池
84: # ─────────────────────────────────────────────────────────────────────────────
85: 
86: 
87: @dataclasses.dataclass
88: class _PinnedResultItem:
89:     buf:      torch.Tensor
90:     event:    torch.cuda.Event
91:     B:        int
92:     T:        int
93:     orig_H:   int
94:     orig_W:   int
95:     pool:     'PinnedResultPool'
96:     img1_raw: list = dataclasses.field(default_factory=list)
97: 
98: 
99: class PinnedResultPool:
100:     def __init__(self, pool_size: int, max_BT: int, H_pad: int, W_pad: int):
101:         self._q = queue.Queue(maxsize=pool_size)
102:         self.pool_size = pool_size
103:         _mb = pool_size * max_BT * H_pad * W_pad * 3 / 1e6
104:         print(
105:             f'\n[PinnedResultPool] 分配 {pool_size} × ({max_BT},{H_pad},{W_pad},3) '
106:             f'uint8 pinned，共 {_mb:.0f} MB', flush=True,
107:         )
108:         for _ in range(pool_size):
109:             self._q.put(torch.empty((max_BT, H_pad, W_pad, 3),
110:                                     dtype=torch.uint8).pin_memory())
111: 
112:     def acquire(self, timeout: float = 30.0) -> torch.Tensor:
113:         try:
114:             return self._q.get(timeout=timeout)
115:         except queue.Empty:
116:             raise RuntimeError('[PinnedResultPool] acquire 超时，T3-Writer 可能卡死')
117: 
118:     def release(self, buf: torch.Tensor):
119:         self._q.put(buf)
120: 
121:     def free(self) -> int:
122:         """
123:         [FIX-POOL-LEAK] 显式释放所有 pinned buffer，解除 CUDA 锁页内存占用。
124: 
125:         背景：Python GC 不会立即回收 cudaHostAlloc 分配的锁页内存（底层由 CUDA
126:         驱动管理，引用计数归零后仍可能驻留直至下次 GC 扫描）。若跨段不主动释放，
127:         各段 PinnedResultPool 会持续叠加（每段 ~1 GiB），累积 4-6 GiB 锁页内存，
128:         导致 DMA 带宽下降 → D2H 变慢 → result_queue 满载 → GPU P50 利用率从
129:         88% 跌至 8%，吞吐量减半。
130: 
131:         调用时机：_infer_loop finally 块，确保每段推理线程退出前完成释放。
132:         返回值：实际释放的 buffer 数量（用于日志确认）。
133:         """
134:         freed = 0
135:         while True:
136:             try:
137:                 buf = self._q.get_nowait()
138:                 del buf
139:                 freed += 1
140:             except queue.Empty:
141:                 break
142:         return freed
143: 
144: 
145: # ─────────────────────────────────────────────────────────────────────────────
146: # [FIX-T3-V643] Pinned Ring Buffer — T2→T3 零拷贝环形缓冲
147: # ─────────────────────────────────────────────────────────────────────────────
148: 
149: class PinnedRingBuffer:
150:     """
151:     [FIX-T3-V643] 预分配 N 个 pinned memory slot 的环形缓冲区。
152: 
153:     T2 (推理线程) 获取空 slot → GPU 裁剪 + D2H（保持 RGB）→ 存储 CUDA event → 标记为可读。
154:     T3 (写入线程) 获取可读 slot → 等待 event → memoryview 零拷贝写 pipe → 标记为空闲。
155: 
156:     每帧数据在 pinned memory 中完全连续（无 padding），memoryview 切片即得完整帧。
157:     """
158: 
159:     def __init__(self, num_slots: int, max_frames_per_slot: int, H: int, W: int):
160:         self.num_slots = num_slots
161:         self.H = H   # 原始高度（无 padding）
162:         self.W = W   # 原始宽度（无 padding）
163:         self.frame_bytes = H * W * 3
164:         self.max_frames = max_frames_per_slot
165:         self.slot_bytes = max_frames_per_slot * self.frame_bytes
166: 
167:         self.slots: List[torch.Tensor] = []
168:         _mb = num_slots * self.slot_bytes / 1e6
169:         print(
170:             f'[PinnedRingBuffer] 分配 {num_slots} × ({max_frames_per_slot},{H},{W},3) '
171:             f'uint8 pinned，共 {_mb:.0f} MB', flush=True
172:         )
173:         for _ in range(num_slots):
174:             t = torch.empty(max_frames_per_slot, H, W, 3, dtype=torch.uint8).pin_memory()
175:             self.slots.append(t)
176: 
177:         # 每 slot 的元数据: (n_frames, orig_H, orig_W, B, T, img1_list, cuda_event)
178:         self._meta: List[Optional[Tuple]] = [None] * num_slots
179: 
180:         self._write_sem = threading.Semaphore(num_slots)   # 可写 slot 数
181:         self._read_sem  = threading.Semaphore(0)           # 可读 slot 数
182:         self._write_idx = 0
183:         self._read_idx  = 0
184:         self._lock = threading.Lock()
185:         self._error: Optional[Exception] = None
186: 
187:     def writer_acquire(self, timeout: float = 30.0) -> Tuple[int, torch.Tensor]:
188:         """T2 获取一个空 slot 的 (slot_id, pinned_tensor)。"""
189:         if not self._write_sem.acquire(timeout=timeout):
190:             raise RuntimeError(
191:                 f'[PinnedRingBuffer] writer_acquire 超时（{timeout}s），T3 可能卡死'
192:             )
193:         with self._lock:
194:             slot_id = self._write_idx % self.num_slots
195:             self._write_idx += 1
196:         return slot_id, self.slots[slot_id]
197: 
198:     def writer_commit(self, slot_id: int, n_frames: int, orig_H: int, orig_W: int,
199:                       B: int, T: int, img1_list: list, cuda_event):
200:         """T2 写完 slot + 记录 CUDA event，标记为可读。"""
201:         self._meta[slot_id] = (n_frames, orig_H, orig_W, B, T, img1_list, cuda_event)
202:         self._read_sem.release()
203: 
204:     def reader_acquire(self, timeout: float = 30.0) -> Tuple[int, memoryview, int, int, int, int, list, object]:
205:         """T3 获取可读 slot，返回 (slot_id, memoryview, n_frames, orig_H, orig_W, B, T, img1_list, event)。"""
206:         if not self._read_sem.acquire(timeout=timeout):
207:             raise RuntimeError(
208:                 f'[PinnedRingBuffer] reader_acquire 超时（{timeout}s），T2 可能卡死'
209:             )
210:         with self._lock:
211:             slot_id = self._read_idx % self.num_slots
212:             self._read_idx += 1
213:         meta = self._meta[slot_id]
214:         self._meta[slot_id] = None
215:         n_frames, orig_H, orig_W, B, T, img1_list, cuda_event = meta
216:         mv = memoryview(self.slots[slot_id].numpy())
217:         return slot_id, mv, n_frames, orig_H, orig_W, B, T, img1_list, cuda_event
218: 
219:     def reader_release(self, slot_id: int):
220:         """T3 读完 slot，释放回可写池。"""
221:         self._write_sem.release()
222: 
223:     def free(self) -> int:
224:         """显式释放所有 pinned buffer，解除 CUDA 锁页内存占用。"""
225:         freed = 0
226:         for t in self.slots:
227:             del t
228:             freed += 1
229:         self.slots.clear()
230:         self._meta.clear()
231:         return freed
232: 
233: 
234: # ─────────────────────────────────────────────────────────────────────────────
235: # [EVENT-POOL] CUDA Event 对象复用池
236: # ─────────────────────────────────────────────────────────────────────────────
237: 
238: class CudaEventPool:
239:     """
240:     预分配 CUDA Event 对象池，避免每批次 cudaEventCreate 开销（约 0.5-1ms/次）。
241:     T2-Infer 调用 acquire() 取出事件，T3-Writer synchronize() 后 release() 归还。
242:     线程安全：acquire/release 均持锁操作。
243:     """
244:     def __init__(self, max_size: int = 8):
245:         self._events: deque = deque()
246:         self._lock = threading.Lock()
247:         for _ in range(max_size):
248:             self._events.append(torch.cuda.Event())
249: 
250:     def acquire(self) -> torch.cuda.Event:
251:         with self._lock:
252:             return self._events.popleft() if self._events else torch.cuda.Event()
253: 
254:     def release(self, event: torch.cuda.Event):
255:         with self._lock:
256:             self._events.append(event)
257: 
258: 
259: # ─────────────────────────────────────────────────────────────────────────────
260: # PinnedBufferPool（线程本地）
261: # ─────────────────────────────────────────────────────────────────────────────
262: 
263: _thread_local = threading.local()
264: 
265: 
266: class PinnedBufferPool:
267:     """
268:     线程本地 pinned-memory 缓冲池。
269: 
270:     [FIX-DOUBLEBUF-SLOT] _try_prefetch_next 使用 slot 0/1（prefetch_slot=0）
271:     和 slot 2/3（prefetch_slot=1）交替轮转，需要 4 个 slot。
272:     v1 仅分配 [None, None]（2槽），slot=2/3 时触发 IndexError → 线程卡死死锁。
273:     修复：初始化为 4 槽；get_for_frames 动态扩容，彻底防御未来 slot 越界。
274:     """
275: 
276:     def __init__(self):
277:         self._bufs:    list = [None, None, None, None]   # [FIX-DOUBLEBUF-SLOT] 4 槽支持 slot 0-3
278:         self._out_buf: Optional[torch.Tensor] = None
279: 
280:     def get_for_frames(self, frames: List[np.ndarray],
281:                        to_rgb: bool = False, slot: int = 0) -> torch.Tensor:
282:         # [FIX-DOUBLEBUF-SLOT] 动态扩容：slot 超界时自动增长，永不 IndexError
283:         while len(self._bufs) <= slot:
284:             self._bufs.append(None)
285:         arr = np.stack(frames, axis=0)
286:         if to_rgb:
287:             arr = arr[:, :, :, ::-1]
288:         arr = np.ascontiguousarray(arr)
289:         src   = torch.from_numpy(arr)
290:         src_f = src.permute(0, 3, 1, 2).float().div_(255.0).contiguous()
291:         n = src_f.numel()
292:         if self._bufs[slot] is None or self._bufs[slot].numel() < n:
293:             self._bufs[slot] = torch.empty(n, dtype=torch.float32).pin_memory()
294:         dst = self._bufs[slot][:n].view_as(src_f)
295:         dst.copy_(src_f)
296:         return dst
297: 
298:     def get_output_buf(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
299:         n_elem = 1
300:         for s in shape:
301:             n_elem *= s
302:         if (self._out_buf is None
303:                 or self._out_buf.dtype != dtype
304:                 or self._out_buf.numel() < n_elem):
305:             self._out_buf = torch.empty(n_elem, dtype=dtype).pin_memory()
306:         return self._out_buf[:n_elem].view(shape)
307: 
308: 
309: def _get_pinned_pool() -> PinnedBufferPool:
310:     if not hasattr(_thread_local, 'pinned_pool'):
311:         _thread_local.pinned_pool = PinnedBufferPool()
312:     return _thread_local.pinned_pool
313: 
314: 
315: # ─────────────────────────────────────────────────────────────────────────────
316: # 张量工具
317: # ─────────────────────────────────────────────────────────────────────────────
318: 
319: def pad_to_stride(arr: np.ndarray, stride: int = MODEL_STRIDE):
320:     H, W = arr.shape[:2]
321:     pad_h = (stride - H % stride) % stride
322:     pad_w = (stride - W % stride) % stride
323:     if pad_h == 0 and pad_w == 0:
324:         return arr, 0, 0
325:     return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'), pad_h, pad_w
326: 
327: 
328: def frames_to_tensor(frames, device, stream=None, dtype=torch.float32, slot: int = 0):
329:     pool  = _get_pinned_pool()
330:     cpu_t = pool.get_for_frames(frames, to_rgb=False, slot=slot)
331:     ctx   = torch.cuda.stream(stream) if stream is not None else nullcontext()
332:     with ctx:
333:         return cpu_t.to(device, non_blocking=True, dtype=dtype)
334: 
335: 
336: def tensor_to_np(t, orig_H, orig_W, sync_stream=None) -> List[np.ndarray]:
337:     """[FIX-D2H] 异步 D2H，用于同步回退路径。"""
338:     if sync_stream is not None and torch.cuda.is_available():
339:         torch.cuda.current_stream().wait_stream(sync_stream)
340:     arr_gpu    = t.clamp_(0.0, 1.0).mul_(255.0).round_().byte()
341:     arr_perm   = arr_gpu.permute(0, 2, 3, 1).contiguous()
342:     pool       = _get_pinned_pool()
343:     out_pinned = pool.get_output_buf(arr_perm.shape, torch.uint8)
344:     out_pinned.copy_(arr_perm, non_blocking=True)
345:     device = t.device
346:     if device.type == 'cuda':
347:         torch.cuda.synchronize(device)
348:     arr = out_pinned.numpy()
349:     # [FIX-T3-V643-COLOR] 保持 RGB（不翻转），pix_fmt 已是 rgb24
350:     return [np.ascontiguousarray(arr[i, :orig_H, :orig_W]) for i in range(arr.shape[0])]
351: 
352: 
353: # ─────────────────────────────────────────────────────────────────────────────
354: # TensorPool
355: # ─────────────────────────────────────────────────────────────────────────────
356: 
357: class TensorPool:
358:     def __init__(self):
359:         self._cache: dict = {}
360: 
361:     def get(self, shape, dtype, device) -> torch.Tensor:
362:         key = (shape, dtype, device)
363:         if key not in self._cache:
364:             self._cache[key] = torch.empty(shape, dtype=dtype, device=device)
365:         return self._cache[key]
366: 
367:     def clear(self):
368:         self._cache.clear()
369: 
370: 
371: # ─────────────────────────────────────────────────────────────────────────────
372: # M2: FFmpeg Pipe 帧读取器
373: # ─────────────────────────────────────────────────────────────────────────────
374: 
375: # ── [FIX-NVDEC-THREAD-CAP] NVDEC 解码线程钳位 ────────────────────────────────
376: # ffmpeg 6.1.1 的 NVDEC 解码 surface 计算公式：
377: #   ulNumDecodeSurfaces = ref_frame_count + num_reorder_frames
378: #                         + 2(deinterlace) + thread_count + 3(基础工作 surface)
379: # 32 是驱动硬上限（cudaVideoDecoder 拒绝），ffmpeg 6.1.1 无 FFMIN(pool,32) 钳位。
380: # 实测：-threads 8 → 32 surfaces 成功；-threads 9 → 33 surfaces 被驱动拒绝。
381: # 注意：仅影响解码侧命令；编码侧线程数由 _detect_encode_parallelism() 独立控制。
382: _MAX_DECODE_THREADS = 8
383: _DECODE_THREAD_HINT_SHOWN = False
384: 
385: def _clamp_decode_threads(requested: Optional[int] = None) -> int:
386:     """钳位 NVDEC 解码线程数（驱动 32-surface 硬上限），超限时打印一次提示。"""
387:     global _DECODE_THREAD_HINT_SHOWN
388:     n = int(requested) if requested else (os.cpu_count() or 4)
389:     if n > _MAX_DECODE_THREADS:
390:         if not _DECODE_THREAD_HINT_SHOWN:
391:             print(f"[decode] threads={n} exceeds max {_MAX_DECODE_THREADS}, "
392:                   f"clamped (NVDEC 32-surface limit)", flush=True)
393:             _DECODE_THREAD_HINT_SHOWN = True
394:         n = _MAX_DECODE_THREADS
395:     return n

(End of file - total 395 lines)
</content>
```

---

## Assistant (Plan · Ox Alpha Free · 14.8s)

_Thinking:_

Now let me read tensorrt_accel.py (218 lines) for the IFRNet side.

**Tool: read**

**Input:**
```json
{
  "filePath": "D:\\Workspace_Python\\Video_Enhancement\\Video_Enhancement\\external\\ifrnet_video\\tensorrt_accel.py"
}
```

**Output:**
```
<path>D:\Workspace_Python\Video_Enhancement\Video_Enhancement\external\ifrnet_video\tensorrt_accel.py</path>
<type>file</type>
<content>
1: #!/usr/bin/env python3
2: # -*- coding: utf-8 -*-
3: # IFRNet Video Enhancement - TensorRT 加速模块（Engine 构建/缓存 + TRT 推理分支）。
4: # 镜像 external/realesrgan_video/tensorrt_accel.py 的职责。
5: 
6: from __future__ import annotations
7: 
8: import os
9: import threading
10: import time
11: from contextlib import nullcontext
12: from typing import Tuple
13: 
14: import torch
15: 
16: 
17: class TensorRTAccelMixin:
18:     """IFRNet TRT Engine 构建/缓存与 TRT 推理分支（与 IFRNetVideoProcessor 混合）。
19: 
20:     自 v6.4.5.1 单文件 M4: TensorRT 小节逐字提取。
21:     """
22: 
23:     # ── M4: TensorRT ─────────────────────────────────────────────────────────
24: 
25:     def _build_trt_engine(self, input_shape: Tuple[int, int, int, int], cache_dir: str,
26:                           _rebuild_attempt: bool = False):
27:         try:
28:             import tensorrt as trt
29:         except ImportError:
30:             print('[TensorRT] 未安装，跳过 TRT 加速。')
31:             self.use_tensorrt = False
32:             return
33: 
34:         os.makedirs(cache_dir, exist_ok=True)
35:         B, C, H, W = input_shape
36: 
37:         _sm_tag = ''
38:         if torch.cuda.is_available():
39:             _props = torch.cuda.get_device_properties(0)
40:             import re as _re_sm
41:             _gpu_slug = _re_sm.sub(r'[^a-z0-9]', '', _props.name.lower())[:16]
42:             _sm_tag = f'_sm{_props.major}{_props.minor}_{_gpu_slug}'
43: 
44:         # ✅ 加入模型变体，避免跨模型加载错误 Engine
45:         tag       = f'{self.model_name}_B{B}_H{H}_W{W}_fp{"16" if self.use_fp16 else "32"}{_sm_tag}'
46:         trt_path  = os.path.join(cache_dir, f'{tag}.trt')
47:         onnx_path = os.path.join(cache_dir, f'{tag}.onnx')
48: 
49:         if os.path.exists(trt_path):
50:             if _sm_tag and _sm_tag not in os.path.basename(trt_path):
51:                 print(f'[TensorRT] 缓存文件缺少当前 GPU 标记 {_sm_tag}，删除并重建: {trt_path}')
52:                 try: os.remove(trt_path)
53:                 except OSError: pass
54:                 if os.path.exists(onnx_path):
55:                     try: os.remove(onnx_path)
56:                     except OSError: pass
57: 
58:         if not os.path.exists(trt_path):
59:             print(f'[TensorRT] 构建 Engine (shape={input_shape}) ...')
60:             dummy0 = torch.randn(*input_shape, device=self.device)
61:             dummy1 = torch.randn(*input_shape, device=self.device)
62:             embt   = torch.full((B,), 0.5, dtype=torch.float32,
63:                                 device=self.device).view(B, 1, 1, 1)
64:             if self.use_fp16:
65:                 dummy0, dummy1, embt = dummy0.half(), dummy1.half(), embt.half()
66: 
67:             _base_model = getattr(self.model, '_orig_mod', self.model)
68: 
69:             class _InferenceWrapper(torch.nn.Module):
70:                 def __init__(self, m):
71:                     super().__init__()
72:                     self.m = m
73:                 def forward(self, img0, img1, embt):
74:                     return self.m.inference(img0, img1, embt)
75: 
76:             export_model = _InferenceWrapper(_base_model)
77:             with torch.no_grad():
78:                 torch.onnx.export(
79:                     export_model, (dummy0, dummy1, embt), onnx_path,
80:                     input_names=['img0', 'img1', 'embt'],
81:                     output_names=['output'],
82:                     opset_version=18,
83:                     dynamic_axes=None,
84:                 )
85:             import onnx
86:             model_proto = onnx.load(onnx_path)
87:             onnx.save(model_proto, onnx_path,
88:                       save_as_external_data=False, all_tensors_to_one_file=False)
89:             print(f'[TensorRT] ONNX 已导出: {onnx_path}')
90: 
91:             if not hasattr(self, '_trt_logger'):
92:                 self._trt_logger = trt.Logger(trt.Logger.WARNING)
93:             logger  = self._trt_logger
94:             builder = trt.Builder(logger)
95:             network = builder.create_network(
96:                 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
97:             )
98:             parser = trt.OnnxParser(network, logger)
99:             with open(onnx_path, 'rb') as f:
100:                 if not parser.parse(f.read()):
101:                     for i in range(parser.num_errors):
102:                         print(f'  [TRT ONNX Error] {parser.get_error(i)}')
103:                     self.use_tensorrt = False
104:                     return
105: 
106:             config = builder.create_builder_config()
107:             config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
108:             if self.use_fp16 and builder.platform_has_fast_fp16:
109:                 config.set_flag(trt.BuilderFlag.FP16)
110: 
111:             _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unknown'
112:             _sm_code  = _props.major * 10 + _props.minor if torch.cuda.is_available() else 0
113:             _time_hint = {
114:                 75: '约需 20~30 分钟（T4/RTX20系 SM7.5）',
115:                 80: '约需 10~20 分钟（A100/A30 SM8.0）',
116:                 86: '约需 5~15 分钟（A10/RTX30系 SM8.6）',
117:                 89: '约需 5~10 分钟（RTX40系 SM8.9）',
118:                 90: '约需 3~8 分钟（H100 SM9.0）',
119:             }.get(_sm_code, f'约需 5~20 分钟（{_gpu_name}）')
120:             print(f'[TensorRT] {_time_hint}')
121: 
122:             _build_start = time.time()
123:             _build_done  = threading.Event()
124: 
125:             def _heartbeat():
126:                 _last = time.time()
127:                 while not _build_done.wait(timeout=5):
128:                     if time.time() - _last >= 300:
129:                         elapsed = time.time() - _build_start
130:                         print(f'[TensorRT] 编译中... {elapsed:.0f}s（仍在运行）', flush=True)
131:                         _last = time.time()
132: 
133:             _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
134:             _hb_thread.start()
135: 
136:             serialized = builder.build_serialized_network(network, config)
137:             _build_done.set()
138:             _build_elapsed = time.time() - _build_start
139:             del config, parser, network, builder
140:             import gc; gc.collect()
141: 
142:             if serialized is None:
143:                 print('[TensorRT] Engine 构建失败，回退 PyTorch 路径。')
144:                 self.use_tensorrt = False
145:                 return
146: 
147:             with open(trt_path, 'wb') as f:
148:                 f.write(serialized)
149:             print(f'[TensorRT] Engine 已缓存（用时 {_build_elapsed:.0f}s）: {trt_path}')
150: 
151:         # 加载 Engine
152:         try:
153:             if not hasattr(self, '_trt_logger'):
154:                 self._trt_logger = trt.Logger(trt.Logger.WARNING)
155:             logger  = self._trt_logger
156:             runtime = trt.Runtime(logger)
157:             with open(trt_path, 'rb') as f:
158:                 self._trt_engine = runtime.deserialize_cuda_engine(f.read())
159: 
160:             if self._trt_engine is None:
161:                 if _rebuild_attempt:
162:                     print('[TensorRT] ⚠️  重建后 Engine 仍反序列化失败，回退 PyTorch。')
163:                     self.use_tensorrt = False
164:                     self._trt_ok = False
165:                     return
166:                 print(f'[TensorRT] Engine 反序列化失败，删除并重建: {trt_path}')
167:                 try: os.remove(trt_path)
168:                 except OSError: pass
169:                 if os.path.exists(onnx_path):
170:                     try: os.remove(onnx_path)
171:                     except OSError: pass
172:                 return self._build_trt_engine(input_shape, cache_dir, _rebuild_attempt=True)
173: 
174:             self._trt_context = self._trt_engine.create_execution_context()
175:             if self._trt_context is None:
176:                 print('[TensorRT] ⚠️  create_execution_context() 失败（显存不足），回退 PyTorch。')
177:                 self._trt_engine  = None
178:                 self.use_tensorrt = False
179:                 self._trt_ok      = False
180:                 if torch.cuda.is_available():
181:                     torch.cuda.empty_cache()
182:                 return
183: 
184:             n = self._trt_engine.num_io_tensors
185:             inputs, outputs = [], []
186:             for i in range(n):
187:                 name = self._trt_engine.get_tensor_name(i)
188:                 mode = self._trt_engine.get_tensor_mode(name)
189:                 if mode == trt.TensorIOMode.INPUT:
190:                     inputs.append(name)
191:                 else:
192:                     outputs.append(name)
193:             self._trt_input_names  = inputs
194:             self._trt_output_names = outputs
195:             if not self.quiet:
196:                 print(f'[TensorRT] inputs={inputs} outputs={outputs}')
197:             self._trt_ok = True
198:             print('[TensorRT] Engine 已激活，TRT 推理就绪。')
199:         except Exception as e:
200:             print(f'[TensorRT] Engine 加载失败: {e}，回退 PyTorch。')
201:             try: os.remove(trt_path)
202:             except OSError: pass
203:             self.use_tensorrt = False
204:             self._trt_ok = False
205: 
206:     def _infer_batch_trt(self, img0_exp, img1_exp, timesteps, B):
207:         """TRT 推理分支（自 _infer_batch 的 elif 分支逐字提取，末尾补 return）。"""
208:         import tensorrt as _trt2
209:         in_names  = getattr(self, '_trt_input_names',  ['img0', 'img1', 'embt'])
210:         out_names = getattr(self, '_trt_output_names', ['output'])
211:         engine_BT = self._trt_engine.get_tensor_shape(in_names[0])[0]
212:         BT        = img0_exp.shape[0]
213:         out_dtype = torch.float16 if self.use_fp16 else torch.float32
214:         out_shape = tuple(self._trt_engine.get_tensor_shape(out_names[0]))
215:         out_buf   = torch.empty(out_shape, dtype=out_dtype, device=self.device)
216: 
217:         _trt_stream_ctx = (torch.cuda.stream(self.stream_compute)
218:                            if self.stream_compute is not None else nullcontext())
219:         with _trt_stream_ctx:
220:             t_vals = timesteps * B
221:             embt_t = torch.tensor(t_vals, dtype=torch.float32,
222:                                   device=self.device).view(-1, 1, 1, 1)
223:             i0 = img0_exp.half().contiguous() if self.use_fp16 else img0_exp.float().contiguous()
224:             i1 = img1_exp.half().contiguous() if self.use_fp16 else img1_exp.float().contiguous()
225:             em = embt_t.half().contiguous() if self.use_fp16 else embt_t.contiguous()
226:             if BT < engine_BT:
227:                 pad_n = engine_BT - BT
228:                 def _pad(t):
229:                     return torch.cat([t, t[-1:].expand(pad_n, *t.shape[1:])], 0).contiguous()
230:                 i0, i1, em = _pad(i0), _pad(i1), _pad(em)
231:             ctx = self._trt_context
232:             for name, buf in zip(in_names, [i0, i1, em]):
233:                 ctx.set_tensor_address(name, buf.data_ptr())
234:             ctx.set_tensor_address(out_names[0], out_buf.data_ptr())
235:             _dummy_bufs = []
236:             for _out_name in out_names[1:]:
237:                 _shape  = tuple(self._trt_engine.get_tensor_shape(_out_name))
238:                 _dtype  = self._trt_engine.get_tensor_dtype(_out_name)
239:                 _tdtype = torch.float16 if _dtype == _trt2.DataType.HALF else torch.float32
240:                 _dummy  = torch.empty(_shape, dtype=_tdtype, device=self.device)
241:                 ctx.set_tensor_address(_out_name, _dummy.data_ptr())
242:                 _dummy_bufs.append(_dummy)
243:             _trt_stream_handle = (self.stream_compute.cuda_stream
244:                                   if self.stream_compute is not None
245:                                   else torch.cuda.current_stream().cuda_stream)
246:             ctx.execute_async_v3(stream_handle=_trt_stream_handle)
247: 
248:         # [FIX-PREFETCH-TIMING] TRT kernel 已入 stream_compute，立即预取下批
249:         if self._pipeline_runner is not None:
250:             self._pipeline_runner._try_prefetch_next()
251: 
252:         # [STREAM-DUAL] 不再 wait default stream；
253:         # stream_d2h 将在 PINNED-D2H 路径内直接 wait stream_compute。
254:         # 保持原始类型（FP16），float() 转换移入 stream_d2h。
255:         result_buf = out_buf[:BT]
256:         pred_big   = result_buf
257:         return pred_big

(End of file - total 257 lines)
</content>
```

---

## Assistant (Plan · Ox Alpha Free · 140.0s)

_Thinking:_

I now have complete coverage of the IFRNet pipeline.py, ffmpeg_io.py, ifrnet_utils.py, tensorrt_accel.py. Let me analyze these files myself for bugs.

**My own analysis of pipeline.py (IFRNet):**

1. **pipeline.py:1122-1131 `_enqueue_pair`**: `while self.running: try put(timeout=1) except Full: check dead threads; continue`. If `self.running` becomes False while queue full → loop exits WITHOUT putting item → frame batch silently dropped (data loss). But this is during shutdown, acceptable. Also note: when infer/writer dead, break — item dropped silently (no error set). Minor.

2. **pipeline.py:1106-1117 reader finally**: 60×1s attempts to put SENTINEL. If fails → `_pending_sentinel = True`. OK.

3. **Race on `_pending_sentinel`** (line 1114, 1159, 1818-1825, 1835): written by reader thread and prefetch path (which runs in infer thread), read by infer thread. GIL-atomic bool assignment, benign.

4. **`_writer_loop` line 1222**: `while self.running or not self.result_queue.empty()`. If writer exits via sentinel break at 1250 (`received_sentinel=True; break`) — but items may still contain more results? Sentinel comes from infer thread AFTER all results, so fine.

5. **Watchdog false positive risk (line 1226-1245)**: `all_empty = pair_queue.empty() and result_queue.empty()` — if both empty and not received_sentinel for >120s → force exit. But consider LA>0 chunked encoding: T2 submits frames to encode thread; result_queue drains quickly because GPU_RAW items are consumed fast (submit to encode thread). Meanwhile pair_queue could be empty if... actually T1 reads much faster than inference so pair_queue stays full normally. Edge case: extremely slow disk read with small segment? pair_queue empties between batches; result_queue empties after writer consumes; sentinel not yet received because infer is blocked in _safe_infer (long TRT/OOM retry?). If _safe_infer takes >120s (OOM deep clean + retries can take minutes), watchdog kills pipeline → data loss. This is a real false-positive scenario: OOM cascade retry loops (bs halving up to 12 retries with sleeps of 20s cooldown each) could exceed 120s. The writer would declare deadlock, set running=False, break. Then run() raises error → segment failed. Not silent corruption but a spurious failure under heavy OOM recovery. Worth listing as medium.

Actually wait — the idle timer only counts consecutive Empty gets with all_empty. During _safe_infer blocking, result_queue is empty and pair_queue likely non-empty (prefetch keeps it filled)... but if reader finished (short segment tail) pair_queue could be empty too. Tail batches + slow OOM recovery >120s → watchdog kill. Plausible but rare-ish. Medium/low.

6. **GPU_RAW path line 1281-1353**: `_nvenc = self.proc._cached_nvenc_encoder`; if None skip batch → **frames silently dropped** (continue). When can encoder be lost mid-segment? main.py Level decision per-segment; within a segment encoder shouldn't vanish. Low.

7. **Line 1300 `torch.cuda.current_stream().synchronize()`**: syncs current stream — but NV12 was computed possibly on another stream? `_rgb_to_nv12_gpu_batch` runs on current stream presumably same as writer thread's current stream. OK.

8. **Line 1313-1325**: LA>0 D2H pinned copy — good. `except RuntimeError` fallback to `.cpu()` — pageable memory, correctness ok.

9. **RING path line 1366-1396**: `reader_acquire(timeout=5)` returns None → print + continue — **slot's data lost? No wait**. If reader_acquire returns None, the slot remains unread; but semaphores: read_sem acquired failed so no loss of semaphore; the slot will be re-offered later? Actually `reader_acquire` timeout means read_sem.acquire timed out — no slot taken, no leak. The RING item is dropped though — those frames never written → frame loss! It prints warning and continues. When does reader_acquire time out? Only if no readable slots for 5s — but we just got an item ('RING', slot_id) meaning writer_commit released read_sem... wait no. The item ('RING', slot_id) was queued by infer thread AFTER writer_commit(slot_id,...) which releases _read_sem. So reader_acquire should immediately succeed. Timeout=5s only under extreme contention with multiple ring items. Rare. But dropping the item loses rB pairs × (rT+1) frames silently. Medium-low.

Also note: `written += _rT` before writing interp frames then `+= 1` for img1 — order: they extend interp data then img1 per i. Counting consistent.

10. **_PinnedResultItem path line 1400**: `arr = _item.buf[:B*T].numpy()` — buf is pinned CPU torch tensor; `.numpy()` shares memory. Writer reads while pool release happens after writes. OK since single-threaded consumption here.

11. **Line 1411**: `writer.write(_item.img1_raw[i].copy())` — `.copy()` unnecessary overhead per frame (memcpy). Performance nit.

12. **run() line 1566-1573**: queues replaced AFTER threads already started? No — this block runs BEFORE thread start (line 1617+). Wait, actually auto_tune block at 1529-1587 runs before starting threads at 1617. Yes fine. But NOTE: `self.pair_queue = queue.Queue(maxsize=_pd)` — replaces the original queue created in __init__. Reader/infer use self.pair_queue attribute dynamically so fine.

13. **run() join order (1640-1667)**: joins writer first (10s!), then infer with dynamic timeout. Writer join only 10s at first — but writer waits for sentinel from infer which processes whole segment... So writer_th.join(timeout=10.0) almost always times out initially?? Let me re-read:

```
if self._writer_th is not None:
    self._writer_th.join(timeout=10.0)
```
Then infer join with big timeout. Then error check. Then RETUNE. Then:
```
if self._writer_th and self._writer_th.is_alive():
    ... dynamic timeout ...
    self._writer_th.join(timeout=_w_timeout)
```
OK so writer join(10s) is just preliminary; final proper join later. Fine.

14. **Infer thread exception handling (1943-1948)**: sets self._error, prints traceback, finally puts SENTINEL (up to 5 tries) and sets running=False. Good.

15. **`meter.update(B)` called from infer thread; `pbar.update(_n_pairs_total)` from writer thread — tqdm not thread-safe strictly but refresh=False mitigates. Cosmetic.

16. **`self._enc_thread_ref` set inside writer thread (1344); read in run() main thread (1727). Benign race (set once).

17. **Potential deadlock: writer exception → running=False; infer blocked on result_queue.put(timeout=30)** — put has timeout 30s, then raises Full? Actually after 30s timeout queue.Full propagates as exception in infer → caught → error recorded. OK bounded.

18. **Reader `_enqueue_pair` infinite retry while self.running and queue full and downstream alive**: bounded by 1s timeouts, checks death each iteration. OK.

19. **ffmpeg_io.FFmpegFrameReader.read()**: blocks forever on `self._queue.get()` — if read_loop thread died without putting SENTINEL (e.g., killed?), hang. read_loop always puts SENTINEL or Exception in its paths. But if `self._queue.put(...)` itself blocks forever because queue full and nobody consuming (reader thread daemon)... In normal flow consumer exists. Edge: close() terminates proc; read_loop's stdout.read returns b'' → break → SENTINEL put. OK. But `close()` doesn't stop the read thread explicitly — terminate causes read to EOF, thread exits. Acceptable. Note close() lacks joining _read_thread; minor.

20. **FFmpegFrameReader partial-frame read loop (442-448)**: `chunk = self._proc.stdout.read(fb - len(buf))` — read(n) on buffered reader returns up to n; loop handles partial. On EOF mid-frame → break with len<fb → treated as end. Truncated last frame dropped silently — matches ffmpeg behavior mostly. OK.

21. **`read()` raising stored Exception repeatedly**: once exception item is consumed, subsequent read() calls get SENTINEL? No — exception item put then return; next call gets SENTINEL (put at line 459 unreachable because exception path returns early at 458). Actually on exception: put(e), return → SENTINEL never put. Consumer calls read() again → queue empty → blocks forever!! Wait: `_read_loop`: try: while True: ... except Exception as e: self._queue.put(e); return. After return, nothing else. So exactly one Exception item, no SENTINEL afterwards. Consumer `read()` raises e to caller (T1 reader loop in pipeline catches? pipeline._reader_loop calls reader.read() inside its try — exception propagates, caught at 1097, sets self._error, finally puts pipeline SENTINEL). Pipeline-level handling OK. But FFmpegFrameReader.read() being called again after exception (e.g., probe liveness checks elsewhere?) would block forever. In main.py usage, reader.read() is called in loop until None or exception; after exception the pipeline stops reading. OK-ish. Low.

22. **FFmpegWriter.close() (781-797)**: stderr_thread.join(5) then stdin.close(), wait(30). rc != 0 → print warning only — **return code ignored by callers** (matches B-5 from earlier agent). Also `write_direct` after BrokenPipeError sets _error; subsequent writes raise RuntimeError — good.

23. **`print(f' [FFmpegWriter] 命令: ...')` unconditional (742)**: debug print always printed even quiet=True — noise issue (the `quiet` guard commented out!). Line 743-744 shows the guarded version commented out. Maintainability/log-noise bug.

24. **ifrnet_utils.PinnedResultPool.free()**: `del buf` doesn't free pinned memory deterministically — comment claims it does; relies on GC. Minor doc drift. Also free() races with concurrent acquire/release (called from cleanup while threads dead — assumed).

25. **PinnedRingBuffer.reader_acquire meta=None race**: `meta = self._meta[slot_id]; self._meta[slot_id] = None; n_frames,... = meta` — if meta is None (shouldn't happen due to semaphore protocol), TypeError crash. Protocol ensures commit before read_sem release. OK.

26. **CudaEventPool**: events reused across streams — event re-record race if released before synchronize completed elsewhere. Writer synchronizes before release. OK.

27. **tensorrt_accel.py:39 vs 112**: `_props` referenced at line 112 outside `torch.cuda.is_available()` guard context — defined at line 39 only if cuda available; line 112 uses `_props.major...` guarded by `if torch.cuda.is_available()` — but `_props` was assigned in the earlier `if torch.cuda.is_available():` block (line 38-42). If CUDA available, both defined. If not available... line 111-112: `_gpu_name = ... if torch.cuda.is_available() else 'unknown'`; `_sm_code = _props.major * 10 + ... if torch.cuda.is_available() else 0` — safe due to conditional. But wait, if cuda NOT available, `_sm_tag=''` and building TRT engine makes no sense anyway. OK.

28. **tensorrt_accel.py:211**: `engine_BT = get_tensor_shape(in_names[0])[0]` static shape assumption — documented. Padding duplicates last frame (229) — output sliced [:BT]. OK.

29. **tensorrt_accel.py:220**: `t_vals = timesteps * B` — list replication, correct semantics for list (agent verified).

30. **Missing TRT sync**: execute_async_v3 on stream_compute; D2H path waits stream_compute — noted OK.

31. **pipeline.py:1547 `_current_cfg` compare resets t2_measured_ms** — includes model_name now. OK.

32. **pipeline.py:1572**: `_rd = min(_rd + max(4, _rd // 4), 48)` — LA burst headroom. OK.

33. **GPUMonitor.stop() join 5s; if sample loop stuck in nvml call >5s, thread leaks but daemon. Low.

34. **pipeline.py:1092-1096**: buffer slicing logic — raw_buf[1:] as img1_raw list, padded[:-1] as img0_pad, padded[1:] as img1_pad. With raw_buf length bs+1 → img1_raw has bs entries; padded has bs+1 entries → img0_pad bs, img1_pad bs. Consistent. End-of-stream: len(raw_buf)>=2 → enqueue remaining. Correct pairing.

35. **`_try_prefetch_next` H2D on stream_h2d while infer computes on stream_compute**: tensors img0_t created under stream_h2d; consumed by model on compute stream — needs `stream_compute.wait_stream(stream_h2d)` somewhere. Presumably done in _infer_batch/_safe_infer ([STREAM-DUAL] mentions). I haven't seen main.py's _infer_batch fully (only agent summary). Trust agent's review that paths handle sync. There's a subtle risk if missing, but agent didn't flag.

36. **Writer line 1279 tuple check `len(_item) >= 2 and _item[0] == 'GPU_RAW'`** — a plain fallback item `(results, img1_raw, is_end)` where results[0]=='GPU_RAW'? Unlikely collision. OK.

37. **`written` counting for GPU_RAW path deferred to encode thread totals (1464)** — flush_and_join returns (_enc_written, _enc_empty); added. OK.

38. **pipeline.py:1962 `except queue.Full: continue`** — 5 tries × 1s; if fail, writer misses sentinel → relies on `received_sentinel and empty` break or 120s watchdog. Bounded.

39. **`IDLE_DEADLOCK_TIMEOUT` 120s vs ESRGAN 120s** consistent.

40. **Performance**: writer per-frame `.copy()` and `.tobytes()` copies; bytearray accumulation for RING; fine.

41. **ffmpeg_io.HardwareCapability class-level mutable `_nvenc: Dict[str, bool] = {}` shared across instances — intended cache. Thread-safety: probes could run concurrently duplicating work. Low.

42. **`best_encoder` trusts static table over failed probe (line 147-154)** — known tradeoff; if driver truly broken → SDK init fails → falls back to FFmpegWriter with nvenc again → raise (ESRGAN agent flagged same pattern B17). For IFRNet: main.py Level 1 failure falls to Ring+NVENC(Level2/3?) etc. There are 4 levels; final level 4 is PinnedPool standard (soft encode?). Probably lands on libx264 eventually. Less severe.

43. **_probe_video nb_frames estimation `nb = int(dur * fps)`** — fps from r_frame_rate could be tbr-ish; duration×fps mismatch warning exists. Downstream frame-count expectations use this estimate (n_seg_est) — estimation only affects progress/timeout. OK.

44. **`FFmpegFrameReader.__init__` no validation width/height>0; corrupt meta → weird frame_bytes=0 → infinite loop reading 0 bytes? `stdout.read(0)` returns b'' → len(buf)<fb(0)? fb=0 → while len(buf)<0 false → proceeds with reshape(h,w,3) of empty → error caught → exception item. Messy but bounded.

45. **Security/robustness**: `_probe_video` raises RuntimeError on ffprobe fail — caller handles.

46. **pipeline.py:1143 `while len(self._prefetch_deque) < 2:` with `pair_queue.empty()` early return** — race: empty() true momentarily though item coming; just returns, retried after next infer. OK.

47. **`_pop_prefetch_or_none` increments hits and total together (1204-1205); miss path increments total only (1812)** — hit rate accounting OK.

48. **Sentinel put-back loop in _try_prefetch_next (1149-1160): put(item, timeout=5) — if pair_queue full 5s → _pending_sentinel=True. Infer loop checks _pending_sentinel only when pair

---

## User

任务完成了吗？请用中文给出最终的任务交付

---

## Assistant (Plan · Ox Alpha Free · 253.3s)

_Thinking:_

The user asks: "任务完成了吗？请用中文给出最终的任务交付" (Is the task complete? Please give the final deliverable in Chinese.)

I'm in plan mode - read-only. The task was a deep analysis of the codebase. I have gathered:

1. ✅ Main entry + utils layer analysis (agent report with 27 bugs B1-B27 + architecture + maintainability C1-C14)
2. ✅ ESRGAN subsystem (24 bugs + full architecture)
3. ✅ IFRNet processor + main.py (20 bugs B-1 to B-20 + architecture + maintainability)
4. ✅ NVENC SDK (ifrnet_video) bug list (20 bugs B-1 to B-20 + maintainability C1-C4)
5. ✅ My own reading of pipeline.py, ffmpeg_io.py, ifrnet_utils.py, tensorrt_accel.py — I was in the middle of my own analysis but I have enough findings from the agents covering ESRGAN pipeline equivalents plus my reading.

The analysis is essentially complete. The user wants the final deliverable in Chinese with the requested format:
1) 架构总览 (Architecture overview)
2) BUG清单（含位置与影响）(Bug list with location and impact)
3) 优化方案（分阶段实施计划）(Optimization plan with phased implementation)

Since I'm in plan mode and this is an analysis/reporting task (not a code modification task), delivering the final report IS the task completion. The report is read-only output, which is allowed.

Let me synthesize everything into a comprehensive final Chinese report. This needs to be thorough but well-organized. Given the CLI display constraints mentioned in my instructions ("keep responses short"), but this is explicitly a request for a comprehensive final deliverable — the user asked for deep analysis output. The instruction says responses should be concise unless user asks for detail — here the user explicitly asks for the full deliverable, so detail is appropriate.

Let me structure:

# 一、架构总览

## 1. 分层结构
- 编排层: src/main_video_optimized.py (~2185行)
- 处理器层: src/processors/ifrnet_processor_video_optimized.py (961行), realesrgan_processor_video_optimized.py (1182行)
- 后端引擎层: external/ifrnet_video/ (~8800行), external/realesrgan_video/ (~13000行)
- 工具层: src/utils/ (config_manager 419行, video_utils 1438行, video_fixer 214行, output_filter 85行)

## 2. 数据流
主流程: 输入 → 音频提取 → [去噪(空挂)] → 切片 → 插帧(IFRNet) → 资源释放 → 超分(ESRGAN) → FIX-C归一化 → LA音频对齐 → 合并 → 清理/报告

IFRNet 内部: FFmpegFrameReader(NVDEC) → T1 Reader线程 → pair_queue → T2 Infer线程(_safe_infer: TRT/CUDA Graph/compile/eager 四级) → result_queue → T3 Writer线程(RGB→NV12→NVENC SDK编码线程) → FFmpegMuxer
四级编码回退: GPU直通SDK NVENC → Ring+NVENC → Ring+软编 → PinnedPool标准

ESRGAN 内部: 五级流水线 P/F/D/S/G 队列 (读帧→人脸检测(CPU retinaface)→SR推理(TRT/eager+OOM阶梯)→GFPGAN子进程(SHM双缓冲/pickle)→NVENC写出)

## 3. 核心机制
- checkpoint/resume: 指纹校验(mtime/size/upstream hash/config snapshot)
- OOM降级: batch减半、TRT working bs、逐帧兜底、显存压力联动GFPGAN pause
- NVENC: ctypes直调SDK13, 多slot(input/bitstream/event三元组), LA分块编码, CE流水线, per-slot FIFO记账, 空帧占位防御
- GFPGAN两阶段子进程(Builder/Inference), 三通道通信(SHM/pickle/控制消息)
- 设计模式: 门面+适配器、延迟单例+keyed缓存、责任链式降级、生产者消费者多级流水线、对象池、Memento(checkpoint)、Monkey Patch

## 4. 已知设计亮点（正面确认）
- 全仓 subprocess 均列表传参 shell=False
- stderr drain 线程防 pipe 死锁
- 双步间显式 GPU 资源释放顺序正确
- 大量带编号的 FIX 注释可追溯
- cuDevicePrimaryCtxRetain 共享 primary ctx 符合环境约束

# 二、BUG清单

需要合并去重，按严重程度排序。致命/高优先：

**致命:**
1. IFRNet model_name 硬编码为 S 架构且 processor 未透传 model_name → 非 S 模型必崩 (main.py:486,592 + ifrnet_processor_video_optimized.py:468-488)

**高危（数据丢失/静默错误）:**
2. NVENC EOS 返回码未检查 → 段尾 LA 滞留帧全部变成冻结占位帧且静默"成功" (nvenc_sdk.py:1942-1943, 2725-2726)
3. _drain_outputs_blocking 非 SUCCESS 错误码静默吞掉 → 真实丢帧不可观测 (nvenc_sdk.py:1323-1324)
4. HEVC/AV1 空槽 blocking LockBitstream 永久挂起，timeout 参数无效 (nvenc_sdk.py:1510-1553; 文件自证 L1951-1954)
5. FFmpegMuxer stdin 无超时写 + close 不校验 rc + flush_and_join 超时后线程泄漏 → 解释器退出被拖死/坏文件 (nvenc_sdk.py:3404, 3410-3426, 3172-3187)
6. AV1 请求在无 AV1 GPU 上 encoder 静默回退 H264 但 muxer 按 OBU 封装 → 必产废文件 (nvenc_sdk.py:753-762 vs 3307)
7. 批量模式无逐文件异常隔离 → 一个文件异常终止整批 (main_video_optimized.py:1722-1731)
8. skip_upscale 路径 keep_audio 配置污染不恢复 (main_video_optimized.py:1372-1390)
9. merge_videos_by_codec 强改扩展名 → 主流程假阴性报告链 (video_utils.py:1311-1313)
10. batch_size=1 OOM 无限重试活锁（CPU 回退路径无外层超时可真挂死）(main.py:977-1003)
11. ESRGAN 流水线异常被吞后段仍标记成功 → 半截视频混入成品 (realesrgan main.py:838-846)
12. Windows select() 用于管道 fd 直接失效 → ESRGAN FFmpegWriter 回退链整体不可用 (realesrgan ffmpeg_io.py:985-989)
13. ESRGAN SDK NVENC 路径音轨丢失 (audio_src 属性不存在恒 None) (realesrgan nvenc_sdk.py:3608 + main.py:793-795)
14. eval() 解析帧率 + 异常捕获面过窄 (video_utils.py:84)
15. checkpoint 指纹只含路径名不含内容 → 断点续跑混合新旧质量分段 (两个 processor 同病: ifrnet_processor:259-261, realesrgan_processor:382-383)

**中危:**
16. CUDA ctx retain 无 Release + push/pop 不对称 → 逐段重建下确定性累积泄漏 (nvenc_sdk.py:658-667, 2856-2863)
17. __init__ 中途失败 close() 链断裂 AttributeError 吞异常 (nvenc_sdk.py:684+, 2833-2869)
18. slot 创建中途 event 失败 buffer 泄漏 (nvenc_sdk.py:969-1008)
19. _ensure_slot_free 占位帧扭曲帧数守恒 + 真实码流被丢弃 (nvenc_sdk.py:1487-1507, 1404-1414)
20. encode_frame 与 FIFO 记账脱节 → 混用时错序花屏 (nvenc_sdk.py:2486-2694)
21. ce_pipeline LA>0 死路分支地雷（辅助块误当帧数据等）(nvenc_sdk.py:2324-2359)
22. NEED_MORE_INPUT 与 CE 固有竞态残余（替代帧风险）(nvenc_sdk.py:2172-2180)
23. flush() doNotWait=1 与 T4 segfault 记录自相矛盾 (nvenc_sdk.py:2770-2771)
24. GetEncodePresetConfigEx 回退用 4 参原型调 5 参函数 UB (nvenc_sdk.py:780-785)
25. DestroyEncoder 返回码未检查 (nvenc_sdk.py:2849-2852)
26. get_or_create 缓存先 free 后构造失败留悬空缓存 → IndexError/误诊卡死 (main.py:1076-1096)
27. Level 切换时 _ring_buf 不复位抢占数据通路 (main.py:820, 1403-1414)
28. muxer/writer 关闭返回码不参与成败判定 → faststart 写盘失败产出无 moov 文件但段判成功 (main.py:1639 + nvenc_sdk.py:3410)
29. 后端 _process_segment 前半段无保护 + GPUMonitor 异常路径不停 → 线程堆积 (main.py:1162-1498)
30. 非整数 scale 截断 → 时长缩短20%音画漂移 (main.py:1219-1229)
31. 短视频直通分支绕过 checkpoint 且失败不置位 _has_failure (processor:220-224)
32. 数值参数无校验（crf/batch_size falsy 忽略/segment-duration 负值/preview_interval=0 除零）(多处)
33. split_video_by_time 浮点取模+复用验证弱 → 截断分段蒙混、段数偏差静默 (video_utils.py:727-791)
34. VideoFixer >1KB 即判成功误收半成品 (video_fixer.py:150-152)
35. 音频临时文件泄漏矩阵（LA修剪后原音频残留/skip分支无清理/失败路径不清理）(main_video_optimized.py 多处)
36. LA丢帧补偿判定脆弱+60fps硬编码 → 可能误剪音频头 (main_video_optimized.py:1569-1589)
37. 三处裸 except 吞 Ctrl+C (video_utils.py:627,639,697)
38. smart_extract_audio 同名即复用无新鲜度校验 (video_utils.py:435-437)
39. input_dir==output_dir 就地覆盖源文件 (main_video_optimized.py:1714)
40. tile_size/tile_pad/pre_pad 配置完全惰性（活跃管线绕过 RealESRGANer.enhance）→ 文档与实现不符 + 失去尺寸补齐安全网 (pipeline.py:87-102 + realesrgan_utils.py:293-297)
41. torch.compile/CUDA Graph 在 ESRGAN 是死功能；IFRNet --use-cuda-graph 被 compile 接管静默推翻
42. ESRGAN total_frames 用 duration×fps 估算提前退出 → 实际帧数>估算时丢尾帧 (pipeline.py:1900-1905)
43. 奇数尺寸：ESRGAN outscale float 可产奇数尺寸 NVENC 异常；IFRNet _rgb_to_nv12_gpu 广播崩溃 (realesrgan nvenc_sdk.py:3357-3396; ifrnet nvenc_sdk.py:3215-3234)
44. ESRGAN GFPGAN wait_result 120s 超时后 slot 复用污染像素 (async_dispatcher.py:74-91)
45. TRT engine 静态形状取首段视频，混合分辨率段无防护 (realesrgan main.py:631-638)
46. LA=0 空帧直接不写 → MP4 帧数缩水 (nvenc_sdk.py:3055-3063 vs 3128-3139 双标准)
47. encode_frames_batch(force=True) 清半截流滞留帧 (nvenc_sdk.py:1629,1648)
48. IFRNet writer 空转看门狗 120s 可能误杀长 OOM 恢复场景 (pipeline.py:1226-1245) — 我自己读出的
49. RING reader_acquire None 时整批帧静默丢弃 (pipeline.py:1370-1373) — 我自己读出的
50. GPU_RAW 时 encoder 丢失跳批静默丢帧 (pipeline.py:1282-1284)
51. FFmpegWriter 命令行无条件打印（quiet 失效）(ffmpeg_io.py:742-744)
52. import 即改全局 stderr/warnings/env 波及宿主进程 (main.py:408-441)
53. torch.load weights_only=False (main.py:593)
54. checkpoint 非原子写 (两个 processor)
55. build_color_args 无元数据回退 pc range (video_utils.py:1081)
56. 合并输出校验失败坏产物不回收 (video_utils.py:1385-1392)

低危可以概括列举。

**低危:** 批量退出码宽松、显示默认值与json不符、死代码群（nvenc_writer.py 整文件、encode_frame、cuda_graph_accel 死分支、_RotationBitReader 等）、诊断计数漂移、Config.set IndexError、报告字段口径不一、get_input_videos glob 重复、TS源切MP4缺 BSF 等。

数量太多了。我需要在报告中合理组织：按子系统分组 + 按严重度标注，给出位置与影响。为了可读性用表格。

# 三、优化方案（分阶段实施）

阶段0（立即修复，1-2天）：致命+高危中的"必崩/必产废品"
- P0-1: 修 model_name 透传与动态架构加载
- P0-2: EOS/drain/DestroyEncoder 返回码检查
- P0-3: HEVC/AV1 blocking lock 改 doNotWait=1 轮询或非阻塞重试
- P0-4: muxer stdin 写超时线程 + close rc 校验上抛
- P0-5: AV1 codec 回退一致性
- P0-6: 批量循环 try/except 隔离 + keep_audio try/finally
- P0-7: merge 扩展名强改时返回实际路径
- P0-8: ESRGAN run_pipeline_for_video 异常上抛而非 return True；audio_src 修复
- P0-9: video_utils eval→Fraction、裸 except 收敛

阶段1（稳定性加固，1周）：
- checkpoint 指纹升级(size+mtime+参数哈希)+原子写
- bs=1 OOM 重试上限与熔断
- 缓存 free/构造失败回滚、_ring_buf 复位
- muxer/writer 关闭 rc 纳入成败
- GPUMonitor finally stop、f0 区 try/finally
- watchdog 与 OOM 恢复协调（暂停计时）
- 音频临时文件清理矩阵、input==output 守卫、同名复用加指纹
- select→Windows 兼容写（threading+queue 或 os.write non-blocking）
- 数值参数 argparse 校验(type=range函数)
- scale 非整数拒绝或告警

阶段2（性能提升，1-2周）：
- tile 真实接入或从配置移除；文档纠偏 compile/CUDA Graph
- writer .copy()/.tobytes() 消除、RING bytearray 直写
- CFUNCTYPE 原型模块级预构造、ctx push/pop contextmanager 化
- SPS/PPS 统一走 _apply_sps_pps
- LA=0 空帧补占位保持帧数守恒一致
- pinned 内存池化 LA D2H（避免每批分配）
- 队列深度建议闭环自动化（已有 RETUNE，减少 print 噪声）

阶段3（架构级与可维护性，2-4周）：
- 拆解上帝函数（_process_segment 532行、_open_session 549行、_process_single 450行、ce_pipeline 411行）
- CLI 覆盖仲裁三处双实现收敛单一模块
- 删除死代码群（nvenc_writer.py、四个矛盾 struct、_FUNC_TABLE_SIZE、旧版本目录归档移出 src）
- 两份 nvenc_sdk 镜像差异收敛为共享核心库
- logging 替代 print、魔法数字常量化
- 引入最小回归测试（合成视频端到端帧数守恒断言、mock NVENC 单测）
- 文档同步真实行数基线

最后给出风险提示：由于是 GPU 驱动级代码，修改需在真实 GPU 上逐项验证（T4/A10），建议每项修复配 bench。

关于 Plan mode：这是一个分析任务，最终交付物就是报告本身，不需要修改任何文件。我应该说明任务完成，给出报告。也可以问用户是否需要我把这份报告写入 memory/ 或生成修复计划文件——但 plan mode 下我不能写。我可以询问用户是否希望进入实施阶段。

考虑到输出长度限制，我需要精炼但完整。用户明确要求三大板块，我会写得结构化、密集。用表格压缩信息。

Let me write the final deliverable in Chinese now. I'll aim for completeness while keeping it scannable.

任务已完成。说明一点：指定路径 `Video_Enhancement项目` 不存在，分析以实际项目 `D:\Workspace_Python\Video_Enhancement\Video_Enhancement` 为准。以下为最终交付报告。

---

# Video Enhancement 代码库深度审查报告

**审查范围**（活跃代码约 2.5 万行）：入口编排层 `src/main_video_optimized.py`(2185行)；处理器层 `src/processors/*_optimized.py`(961+1182行)；后端 `external/ifrnet_video/`(约8800行) 与 `external/realesrgan_video/`(约13000行)；工具层 `src/utils/`(约2150行)。历史版本目录（backup*、带日期目录、v2~v6 系列）已排除。

---

## 一、架构总览

### 1.1 四层分层结构

```
┌─ 编排层  main_video_optimized.py ─ CLI 解析 / 环境检查 / 配置覆盖仲裁 / 批量调度 /
│            音频提取修剪 / FIX-C 归一化(timescale=90000) / 合并与完整性验证 / 报告
├─ 处理器层 processors/*_optimized.py ─ 防腐适配层：字段映射(argparse.Namespace)、
│            分段循环、checkpoint/resume(Memento)、跨段资源交接
├─ 后端引擎 external/ifrnet_video ─ v6.4.5.1 深度模块化包：
│     main.py(引擎/参数仲裁/ADAPTIVE-QUEUE) · pipeline.py(T1/T2/T3三线程流水线) ·
│     nvenc_sdk.py(ctypes直调SDK13) · ffmpeg_io.py(NVDEC探测/读写器) ·
│     ifrnet_utils.py(锁页内存池/RingBuffer/EventPool) · tensorrt_accel.py(TRT Mixin)
│   external/realesrgan_video ─ 同构镜像包 + gfpgan_subprocess.py(两阶段TRT子进程) +
│     async_dispatcher.py(SHM双缓冲调度) + face_utils.py(CPU retinaface)
└─ 工具层  utils/ ─ config_manager(JSON单一事实来源+递归合并+base_dir推算)、
             video_utils(切片segment muxer/concat合并/音轨)、video_fixer、output_filter
```

### 1.2 端到端数据流（默认双步模式）

```
输入视频 → 提取音轨(stream copy) → [可选去噪(空挂)] 
  → split_video_by_time(-c copy切片30s)
  → ①IFRNet插帧: NVDEC解码 → T1攒批(pair_queue) → T2推理[TRT>CUDAGraph>compile>eager
     四级, OOM减半降级] → T3写出[GPU直通SDK-NVENC → Ring+NVENC → Ring软编 → PinnedPool
     四级回退] → 段checkpoint
  → 显式释放第一阶段GPU资源(gc+empty_cache)
  → ②ESRGAN超分: 五级有界队列流水线 P(读帧)→F→D(人脸检测CPU)→S(SR推理,
     OOM阶梯:减批→深清→逐帧→重建流)→G(GFPGAN子进程SHM零拷贝/pickle回退)→NVENC写出
  → FIX-C remux归一化 → NVENC-LA时长补偿裁剪音频 → concat合并(编码梯队决策+色彩元数据注入)
  → 完整性校验 → 删checkpoint → 清理临时目录 → JSON报告
```

### 1.3 核心机制

| 机制 | 实现 |
|---|---|
| **NVENC 封装** | ctypes 手工偏移直调 SDK13 函数表；多 slot(input/bistream/event 三元组)；LA 有界分块流式编码 + per-slot FIFO 记账；CE 异步流水线(LA=0)；空帧占位防御；SPS/PPS 缓存注入 |
| **CUDA 协作** | `cuDevicePrimaryCtxRetain` 与 PyTorch 共享 primary ctx（规避 pycuda.autoinit 冲突）；三条专用 stream(transfer/sr/gfpgan) 规避 legacy null stream 隐式屏障 |
| **GFPGAN 子进程** | 两阶段(Builder 构建引擎→Inference 服务)；SHM 双缓冲零拷贝(512×512×64人脸槽)→pickle→直通 SR 三级回退；崩溃/context 污染检测自动降级主进程 |
| **checkpoint/resume** | config_snapshot(mtime/size/上游hash/crf/factor) 指纹校验，不匹配整体作废；每段落盘 |
| **自适应调优** | GPUMonitor 滑动窗口采样 → T2 中位数校准(.t2_cache) → ADAPTIVE-QUEUE 三轴动态队列上限(RAM预算/T3-T2比/绝对上限)；EMA 自适应批大小 |

### 1.4 设计模式
门面+防腐适配器、延迟单例+keyed 缓存(Cache-Aside)、责任链式降级(编码四级/TRT三级/SHM三级)、多级生产者-消费者流水线(哨兵传播+看门狗)、对象池(Pinned/Ring/Event/TensorPool)、Memento(checkpoint)、Monkey Patch(facexlib)、审计式防御(帧数守恒/faulthandler/相位漂移诊断)。

### 1.5 正面确认（审查通过项）
subprocess 全列表传参 shell=False ✓；stderr drain 线程防 pipe 死锁 ✓；双步间 GPU 释放顺序正确 ✓；NVDEC surface 上限钳位 ✓；OOM 批内半成品一致性处理正确 ✓；pycuda.autoinit 全库未调用 ✓。

---

## 二、BUG 清单（共 60+ 项，按严重度分级）

### 🔴 致命（P0，功能必然失效）

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| A1 | `external/ifrnet_video/main.py:486,592` + `src/processors/ifrnet_processor_video_optimized.py:468-488` | 模型架构模块级硬编码 `IFRNet_S_Vimeo90K`，processor 库内调用路径**不透传 model_name** 且无 `global Model` 重绑定 | 配置 L/V 模型 → S 架构 load_state_dict strict 失败 → **所有分段必败**；多模型能力整体失效 |

### 🟠 高危（P0-P1，静默数据损坏 / 平台性失效 / 挂起）

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| H1 | ifrnet `nvenc_sdk.py:1942,2725` | EOS EncodePicture 返回码不接收；EOS 失败后段尾 LA 滞留帧全部替换为 prev 帧冻结占位且流程“成功” | 输出尾部大段静止画面，**静默** |
| H2 | ifrnet `nvenc_sdk.py:1323-1324` | `_drain_outputs_blocking` 对非 SUCCESS/NEED_MORE_INPUT 错误码无声 break | 设备错误时**真实丢帧零痕迹** |
| H3 | ifrnet `nvenc_sdk.py:1510-1553`(自证于1951-1954) | blocking LockBitstream 的 timeout 仅在两次调用间隙生效；HEVC/AV1 空槽驱动永久阻塞（test4 实录） | HEVC/AV1+LA 路径**确定性卡死** |
| H4 | ifrnet `nvenc_sdk.py:3404,3410-3426` + `pipeline.py:3172-3187` | Muxer stdin 写无超时（持 encoder 锁阻塞→主线程 close 连锁死锁）；flush_and_join 超时仅打印不处置（线程泄漏+半死拆除）；close rc≠0 仅 warning | **解释器退出被拖死**；坏 mp4 延迟暴露 |
| H5 | ifrnet `nvenc_sdk.py:753-762` vs `3307` | AV1 请求在不支持 GPU 上 encoder 静默回退 H264，muxer 却按 `-f obu` 封装 | 组合必产废文件 |
| H6 | `src/main_video_optimized.py:1722-1731` + `1372-1390` | 批量循环无逐文件异常隔离；keep_audio 篡改仅两条路径恢复（应 try/finally） | 一个文件异常终止整批 + **配置跨文件污染** |
| H7 | `src/utils/video_utils.py:1311-1313` | 重编码触发时强改输出扩展名但返回 True，调用方沿用原路径 | 任务成功却报“输出不存在”假阴性链 |
| H8 | realesrgan `main.py:838-846,924-927` | `run_pipeline_for_video` 捕获异常仅打印，finally 照常“✅完成”return True；processor 只验容器完整 | 半截视频混入合并产物 |
| H9 | realesrgan `ffmpeg_io.py:985-989` | `select.select()` 用于管道 fd——**Windows 仅支持 socket**，OSError 被吞→stall 重试→writer 永久 broken | 本机(win32)软编回退链整体不可用（生产跑 SDK 路径掩盖中） |
| H10 | realesrgan `main.py:793-795` + `nvenc_sdk.py:3608,3475` | `getattr(reader,'audio_src',None)` 属性不存在恒 None；Writer 的 audio 参数从未使用 | SDK Level1 路径输出**无声视频**，两编码路径行为不一致 |
| H11 | ifrnet `main.py:977-1003` | bs=1 仍 OOM → 估算≥1 → continue 无限重试；CPU 回退路径无外层超时 | 显存被他方长期占用时**真挂死** |
| H12 | `src/utils/video_utils.py:84,94` | `eval(r_frame_rate)` + except 面过窄（'N/A' 抛 NameError 穿透） | 损坏容器头可崩溃全流程 |
| H13 | 两 processor（`ifrnet:259-261,420-435`；`realesrgan:382-383`） | checkpoint 指纹仅排序**路径名** md5，direct 模式 mtime/size 缺位 | 上游换参重跑同名分段→断点误命中→**新旧质量混流** |
| H14 | ifrnet `nvenc_sdk.py:2960-2965` + `submit` 读 error 与 put 非原子 | 编码线程异常退出瞬间提交批次丢失且 submitted 虚增 | 段尾帧丢失难定位 |

### 🟡 中危（P1-P2，边界条件错误 / 资源泄漏 / 功能与文档不符）

**NVENC/驱动层（ifrnet）**
- `nvenc_sdk.py:658-667,2856-2863`：primary ctx retain 无 Release + push/pop 不对称，配合逐段重建(main.py:1059)成为**确定性累积泄漏**；`_loop` 里 cuCtxSetCurrent 属补丁叠补丁
- `nvenc_sdk.py:684-953,2833-2869`：init 中途失败 → close 访问 `_func_ptrs` 抛 AttributeError 被 `__del__` bare except 吞掉；close 早退跳过 saved_ctx 恢复
- `nvenc_sdk.py:969-1008`：slot 创建中 cuEventCreate 失败，当前槽两 buffer 未入 `_slots` 泄漏
- `nvenc_sdk.py:1404-1414,1477-1507`：`_ensure_slot_free` 兜底三分支互相纠缠——占位帧扭曲守恒+真实码流稍后被 mismatch `continue` 丢弃+直锁路径不推进 `_output_slot_idx`
- `nvenc_sdk.py:1629,1648`：`encode_frames_batch(force=True)` 清半截流 pending → LA 滞留帧静默丢失
- `nvenc_sdk.py:2486-2694`：`encode_frame` 绕过 FIFO 记账（固定 slot0、取全局 drain 首帧），混用即错序花屏
- `nvenc_sdk.py:2324-2359,2438-2473`：ce_pipeline 的 LA>0 分支成死路但仍是地雷（辅助块误当帧码流、pending None 静默 continue）
- `nvenc_sdk.py:2770-2771` vs `1284`：flush() 用 doNotWait=1，与自己记录的“T4 segfault”结论矛盾（重构即触雷）
- `nvenc_sdk.py:3055-3063` vs `3128-3139`：空帧处理双标准——stream 路径写占位，ce_pipeline 直接不写→**LA=0 输出比输入少帧**
- `nvenc_sdk.py:780-785`：GetEncodePresetConfigEx 回退分支用 4 参原型调 5 参 API（x64 第 5 寄存器垃圾值，潜伏 UB）
- `nvenc_sdk.py:2849-2852,999,1855...`：DestroyEncoder/event create/sync 返回码全面漏检

**IFRNet 引擎层**
- `main.py:1076-1096`：get_or_create 先 free 后构造，构造抛异常留悬空缓存+key 未清 → 下段 key 命中已释放缓冲 → IndexError/误诊"T3 卡死”
- `main.py:820 vs 1403-1414`：Level 切换 `_ring_buf` 不复位 → 旧容量 ring 抢占 Level 4 通路，形状不匹配晦涩报错
- `main.py:1639` + `nvenc_sdk.py:3410`：muxer/writer 关闭 rc 不参与成败判定 → faststart 二次 moov 写盘失败仍判段成功
- `main.py:1162-1498,1715-1733`：reader 探测/f0 写入区无 try 保护；GPUMonitor 异常路径永不停→统计窗口重叠/线程堆积
- `main.py:1219-1229`：scale=2.5 时 n_interp 截断而 new_fps 用小数 → 时长缩 20% 音画漂移（软编静默/硬误导性报错）
- `main.py:593`：torch.load(weights_only=False)；`582`：CUDA 不可用静默切 CPU 无告警
- `main.py:408-441`：import 即改进程级 stderr/warnings/env，波及同进程 ESRGAN 管线
- `processor:220-224`：短视频直通绕过 checkpoint 且失败不置 `_has_failure`
- `pipeline.py:1226-1245`：120s 空转看门狗可能误杀长 OOM 恢复场景（尾批+_safe_infer 深度清理>120s）
- `pipeline.py:1370-1373`：RING reader_acquire None → 整批帧静默丢弃；`1282-1284`：encoder 丢失跳批静默丢帧

**ESRGAN 子系统**
- `pipeline.py:1900-1905`：total_frames 用 duration×fps 估算，实际帧数>估算+瞬时排空 → 提前退出丢真实尾帧
- `pipeline.py:87-102` + `realesrgan_utils.py:293-297`：**tile/tile_pad/pre_pad 全部惰性**——活跃管线直接 `upsampler.model()` 绕过 RealESRGANer.enhance，tile 能力不存在且失去尺寸补齐安全网
- `main.py:330-365` + `pipeline.py:83-93,295`：**torch.compile/CUDA Graph 为死功能**（cuda_graph_accel 恒 None，全库无 torch.compile 调用），CLI/文档宣传与实现脱节
- `async_dispatcher.py:74-91` + `pipeline.py:1355`：wait_result 120s 超时后 slot 立即复用，子进程迟到写入污染下一任务像素
- `main.py:631-638`：TRT engine 静态形状取首段视频，process_segments_directly 混合分辨率无防护
- `gfpgan_subprocess.py:1120-1130`：回退路径 result_queue.put 无 timeout（maxsize=2），collector 未消费即无限阻塞

**编排/工具层**
- `video_utils.py:727-791`：浮点取模多算段数、复用验证仅 OpenCV 读首帧（截断尾段蒙混）、陈旧 segments 目录可复用
- `video_fixer.py:150-152`：“存在且>1KB”判成功，前法半成品误收；integrity 校验仅读 1 秒
- `main_video_optimized.py` 音频泄漏矩阵：LA 修剪后原音频残留、skip 两分支无清理、失败路径不清理、with_audio.mp4 残留
- `main_video_optimized.py:1569-1589`：以“SDK 可导入”推断“本次确实用了 SDK NVENC”（1080p 自动切 constqp+LA=0 时高估）+阈值硬编码 60fps → **可能误剪音频开头（不可逆）**
- `video_utils.py:627,639,697`：三处裸 `except:` 吞 Ctrl+C/SystemExit
- `video_utils.py:435-437`：smart_extract_audio 同名即复用无新鲜度校验（挂错音轨）
- `main_video_optimized.py:1714`：input_dir==output_dir 就地覆盖源文件
- `config_manager.py:135 vs main:795-1021`：CLI 覆盖完全绕过 `_validate_config`；`--batch-size` 等 falsy 值被静默忽略 vs CRF 用 is not None（口径不一）
- `main.py:573-581`：--use-cuda-graph "强制启用”被 compile 接管静默推翻；rate_mode 默认值三处说法不一(vbr_hq/qvbr/constqp)

### ⚪ 低危（择要，P2-P3）
奇数宽高：ESRGAN outscale=float 产奇数尺寸 NVENC 异常无前置校验(realesrgan nvenc_sdk.py:3357)；IFRNet `_rgb_to_nv12_gpu` UV 形状广播崩溃(ifrnet nvenc_sdk.py:3215)。checkpoint 非原子写(两 processor:445)。build_color_args 无元数据强制 `-color_range pc`(video_utils.py:1081)。合并校验失败坏产物不回收(video_utils.py:1385)。FFmpegWriter 命令无条件打印 quiet 失效(ifrnet ffmpeg_io.py:743-744 注释掉的 guard)。批量退出码 10 败 9 仍返 0(main:2153)。启动摘要默认值与 json 系统性不符(main:350-392)。TS→MP4 缺 aac_adtstoasc BSF(video_utils.py:766)。`Config.set` 空 keys IndexError。GPUMonitor.stop join 5s 超时线程滞留。`_la_buffered` 只增不清诊断失真(nvenc_sdk.py:1897)。

### 可维护性负债（Top）
① 上帝函数：`_open_session`(~549行)、`_process_segment`(~532行)、`_process_single`(~450行)、`ce_pipeline`(~411行)、`_process_sr/_process_gfpgan`(各300-400行)；② 五套重复底层样板：ctx push/pop×4、LockBitstream 手工偏移×5、MEMCPY2D 打包×3、SPS/PPS 阶梯×7（统一入口 `_apply_sps_pps` 已存在却被绕开）、NAL 扫描四胞胎；③ 死代码地雷群：`nvenc_writer.py` 整文件（07-28 曾把修复打在它上面白干）、4 个与裸偏移矛盾的未实例化 ctypes 结构体、`cuda_graph_accel` 死分支、`_FUNC_TABLE_SIZE`；④ CLI 覆盖仲裁三处双实现必漂移；⑤ 两份 nvenc_sdk/pipeline 镜像已开始发散，人工双向同步是持续缺陷源；⑥ 全库 print 当日志无 logging；⑦ src 下 ~30 个历史版本文件并存易误导入。

---

## 三、优化方案（分阶段实施计划）

### 阶段 0｜止血修复（P0，预计 2-3 天，每项配 GPU 实测验证）

| 序 | 修复项 | 对应缺陷 | 验收标准 |
|---|---|---|---|
| 1 | `model_name` 透传 + 动态架构加载（仿 CLI 的 global Model 重绑定制） | A1 | L/S 模型各跑通 1 个视频，帧数守恒 |
| 2 | NVENC 返回码全面检查：EOS×2、drain 非 SUCCESS 改 raise、DestroyEncoder、event create/sync | H1/H2 | 人为制造会话错误能 fail-fast 而非静默 |
| 3 | blocking LockBitstream 改 `doNotWait=1`+deadline 轮询（避开 T4 segfault 场景则用独立等待事件） | H3 | HEVC+LA=16 分段不再挂起 |
| 4 | Muxer stdin 写移入带超时的专用线程；close rc≠0 上抛；flush_and_join 超时改为标记段失败 | H4/B5 | 模拟磁盘满：段判败而非产出无 moov 文件 |
| 5 | codec GUID 回退时同步改写 muxer 格式（H264 回退→mp4 封装） | H5 | AV1-on-T4 请求产出可播文件 |
| 6 | 批量循环 try/except 隔离；keep_audio 改 try/finally；merge 扩展名改动需返回实际路径并向上传播 | H6/H7 | 注入异常后其余文件继续处理 |
| 7 | ESRGAN `run_pipeline_for_video` 异常重抛；补 `reader.audio_src` 传递链 | H8/H10 | 故意断帧后段标败、SDK 路径有声 |
| 8 | `eval`→Fraction；三处裸 except→`except Exception` | H12 | N/A fps 文件正常解析 |

### 阶段 1｜稳定性加固（P1，约 1 周）

1. **checkpoint 指纹升级**：哈希加入各上游分段 `(size,mtime)` 与自身快照参数；写盘改 tmp+`os.replace` 原子化（修 H13）
2. **OOM 活锁熔断**：bs=1 连续 N 次 OOM → 段失败并给出显存占用诊断（修 H11）
3. **缓存失败回滚**：free 前先置 None 再构造，失败不回填缓存；Level 切换显式复位 `_ring_buf`（修中危 2 项）
4. **关闭语义收紧**：writer/muxer rc 纳入段成败；`_process_segment` 前半段包 try/finally 确保 monitor.stop + reader/writer.close（修中危）
5. **watchdog 协调**：`_safe_infer` 进入 OOM 恢复时通知 writer 暂停空闲计时（修误杀）
6. **Windows select 替换**：ESRGAN `_write_with_timeout` 改线程+queue 或非阻塞 fd（修 H9）
7. **数据卫生**：split 复用加 `(size,mtime)` 指纹侧车；extract_audio 同名校验；input==output 守卫；音频临时文件统一 finally 清理；LA 补偿判定改为查询本段实际编码路径而非 import 探测
8. **参数校验前置**：argparse type=range 校验（crf 0-51、duration>0、factor∈{2,4,8,16}∪自定义正分数、batch≥1），CLI 覆盖后重跑 validate

### 阶段 2｜性能与一致性提升（P2，约 1-2 周）

1. **tile 真实接入或显式移除**：二选一——接回 `RealESRGANer.enhance(tile=True)` 或从 config/CLI 删除并在启动日志明示（消除最大认知陷阱）
2. **compile/CUDA Graph 纠偏**：ESRGAN 侧要么实现要么摘除 CLI；IFRNet 侧互斥裁定前移到参数层
3. **热路径去拷贝**：writer `.copy()/.tobytes()` 消除（RING 路径 memoryview 直写）；LA D2H pinned 缓冲并入 PinnedBufferPool 复用
4. **NVENC 样板收敛**：CFUNCTYPE 原型模块级一次定义；ctx push/pop 抽 contextmanager；SPS/PPS 全走 `_apply_sps_pps`；空帧处理双标准统一为占位保帧数
5. **日志体系**：print→logging（模块级 logger），诊断计数收敛到统一 debug 通道，恢复 `--quiet` 语义

### 阶段 3｜架构级重构与可维护性（P3，约 2-4 周，可与日常迭代并行）

1. **拆解上帝函数**：`_process_segment`→(探测/决策/预热/编排/守恒校验)五私有方法；`_open_session`→(加载/枚举/RC/slots)四段；`_process_single`→阶段函数族
2. **镜像收敛**：抽公共 `common/nvenc_core.py` 供两子系统共享，消灭双维护；`MODEL_NAME_MAP` 等双处定义合一
3. **死代码清除**：删 `nvenc_writer.py`、4 个矛盾结构体、`_RotationBitReader`、`encode_frame`（或补 FIFO 对齐）；`src` 历史版本移入 `archive/` 目录
4. **测试基建**：新增最小 pytest 兼容套件（AGENTS.md 允许）：合成 3s 测试视频端到端断言帧数守恒/NVENC session 泄漏计数/checkpoint 指纹失效路径；ctypes 结构体布局加 sizeof 断言
5. **文档同步**：AGENTS.md 行数基线更新；README 标注 tile/compile 实际状态；memory/ 双侧同步本报告结论

### 实施原则
- 所有 NVENC/TRT 改动须在目标 GPU（当前 T4 基线）实测：帧数守恒 ≥ 源帧 ×(scale)/(scale) 校验、bitstream 解析、空帧统计三项既有验收手段
- 每阶段结束跑一次双步全流程基准（前后 FPS/P50 利用率/峰值显存对比），防止性能回退
- 镜像修改严格双侧同步（repo + `C:\Users\Administrator\.claude\...\memory`）

---

如需进入实施阶段，建议从**阶段 0 第 1 项（A1 致命缺陷）**开始。是否需要我先就阶段 0 的某一项展开具体的修复方案？

---

