借鉴 D:\Workspace_Python\VidUtils，对比分析 D:\Workspace_Python\Video_Enhancement\Video_Enhancement 项目，重点梳理以下环节中不同 codec 编码器（如 libx264、libx265、h264_nvenc、hevc_nvenc 等）之间 crf/cq 自动转换的差异：

1. 原视频分割编码环节
2. 超分/插帧输出编码环节
3. 合并编码环节

分析内容应涵盖：
- 各编码器 crf/cq 参数的取值范围与默认值差异
- 不同编码器之间 crf/cq 自动转换的现有逻辑与存在的问题
- 软编与硬编（GPU）之间 crf/cq 映射关系的不一致之处

制定优化统一方案，包括但不限于：
- 新增统一化参数 --crf-ifrnet-ref / --cq-ifrnet-crf / --crf-esrgan-ref / --cq-esrgan-ref
- 新增 GPU 硬编参数 --cq-ifrnet / --cq-esrgan
- 统一各环节中软编与硬编之间的 crf/cq 转换规则与默认值映射表
- 确保各环节参数传递一致，消除重复或冲突的转换逻辑

请以清晰的对比分析报告形式输出，包含现状问题总结与具体的统一方案设计。