import pathlib

NEW = """QUALITY_MAP = {
    # 标定（2026-10-02，纯 CPU，**第八版 · 当前生效**）：统一锚点 **18/21/24/27/30**
    # + **17 条素材** + **同key 取均值**合并（**顺序无关、已用随机种子验证**）。
    #
    # 素材池（17 条，两仓实测并集；1138 点，合并重复点 192 个）：
    #   · VU 侧 7 条（6s）：new5_raw / new4_raw / cc_anim_300s（动画平涂）
    #     / cc_subs_105s（动画+烧录字幕）/ earth_dark_80s（暗场）
    #     / ui_screen_10s（屏幕 UI）/ natgeo_grass_40s（高细节纹理）
    #   · VE 侧原有 4 条（10s）：new5_raw / new4_raw / new1 / word_world_2
    #   · BBC 实拍剧集 3 条（1080p，素材库 `BBC 真人情景剧/Molly and Mack`）：
    #     S01E01 / S03E01 / S05E01 —— 10s（第五版补标）+ **6s（庚新增）**
    #   · 己新增 5 条（10s，**重新采集**，见下）：anim_10s（动画平涂）
    #     / anim_subs_10s（动画+歌词字幕）/ dark_10s（暗场·夜拍 4K 下采样）
    #     / ui_10s（屏幕 UI）/ texture_10s（高细节纹理·实拍）
    #   覆盖立项 §3 全部 6 类内容 + 实拍影视 + 暗场（4K 下采样）。
    #
    # ✅ **第八版：素材池 12 → 17 条，LOO 全面改善**
    #   本轮补齐了此前**单侧口径的素材缺口**（实测「只用 10s」⇒ svtav1 LOO 7.33 超门禁；
    #   「只用 6s」⇒ vp9 6.85 超门禁—— 砍掉任一半都会丢掉某类内容）：
    #     · 庚：补 6s 侧的 new1 / word_world_2 / BBC×3（5 素材 × 65 点 = 325 点）
    #       ⇒ 6s 池也覆盖全部 6 类内容
    #     · 己：补 10s 侧的 5 类内容（5 素材 × 65 点 = 325 点）
    #       ⚠ 原 6s clip **物理上不够长**（cc_subs 6.00s / earth_dark 6.15s /
    #         natgeo 6.16s / cc_anim 8.08s），无法补10s 口径 ⇒ 必须**重新采集**。
    #         新 clip 与原 6s clip **内容不同** ⇒ 它们是**新增素材**而非「同一素材补
    #         另一时长」，这正是素材池能扩到 17 条的原因。
    #     采集脚本：`/tmp/collect_10s.py`（marker-walk定位项目根，两仓通用）
    #     测量脚本：`/tmp/measure_uni.py`（`--src/--out/--tiers/--duration`）
    #
    # ⚠ **仍存在 7 条跨时长素材**（并非「每条单一时长」）：
    #   `new5_raw` / `new4_raw` / `new1` / `word_world_2` / BBC×3 —— 它们的**原有数据
    #   本就横跨两个时长**，本轮补测只是**新增了另一时长的副本**，并未消除旧观测。
    #   正因如此**「同 key 取均值」仍是必需规则**（不是可选兜底）：
    #   同 (素材,档位,参数) 的多个时长观测取算术均值。
    #   重复观测的差异来源已定性：**VMAF 曲线本身随片段长度变化**（时长效应，
    #   非测量误差 —— 同一份 points.json 内重复值跨度为 0.000），实测最大跨度 2.05
    #   （BBC S05E01 rav1e qp210）。
    #
    # ✅ **顺序无关性已验证**：3 个随机种子打乱文件加载顺序复算，
    #   **6 档 × 3 seed 共 18 个数值全部逐位一致** ⇒ 表值可复现，不依赖合并顺序。
    #   （第六版曾用「后者覆盖前者」⇒依赖顺序，实测三规则 `a` 差 ~1.5%，已废弃。）
    #
    # ✅ **精度：门禁按编码器分档，全部达标**（2026-10-02 仓主裁定）
    #   · 软编 4 档门禁 **≤ 5.9**：x265 4.24 / vp9 3.82 / svtav1 4.88 / aom 5.35 —— 全 ✅
    #   · rav1e 两档门禁 **≤ 7.5**：@10 5.99 / native 5.76 —— 全 ✅
    #   较第七版（12 素材）：vp9 4.71→**3.82** / x265 4.73→**4.24** 明显改善；
    #   svtav1 4.68→4.88 / aom 5.49→5.35 / rav1e 5.34→5.76 / rav1e@10 5.88→5.99 微升。
    #   变化源于素材数 12→17（覆盖更全），**不是精度退化**。
    #
    # ⚠ **仍达不到立项 M2 的 |ΔVMAF| < 1.0**，经仓主裁定放宽（见上）。误差来源已定性
    #   （非标定执行错误），四条排除性证据：
    #     (a) 非过拟合 —— 部分素材连**训练内** ΔVMAF 都达 0.8~3.0；
    #     (b) 非素材不足 —— 本轮 12→17 条后 LOO 仍为 3.8~6.0（结构上限，非数量）；
    #     (c) 非表格式 —— 每素材**专属**表 LOO 4.02~13.77，比共享直线更差；分段/二次无增益；
    #     (d) 非锚点位置 —— 已在**同批素材**上完成两套锚点的可比对照并选定更优者。
    #   根因是**结构性的**：`(a,b,lo,hi)` 单行仿射 + VMAF 反查，跨素材存在精度上限。
    #   ⚠ 换素材集后 LOO 可能变化；**若素材异质性显著增加，需重跑标定并重新定门禁**。
    #
    # ⚠ **锚点统一（2026-10-02）**：此前两仓锚点不同（A `18/22/26/30/34` /
    #   B `18/21/24/27/30`），因各自缺测对方锚点而无法直接比较。补测缺口后，在**同批
    #   素材/同口径**下做可比 LOO 对照，**B 套 6 个档位全部更优**，故统一到 B 套。
    #   机制：**决定因素不是 crf 位置，而是锚点是否落在各素材 VMAF 的可分辨区间**
    #   （A 套强拉到 crf34，对跨度小的素材过头、进入陡峭段，反查条件数反而变差）。
    #
    # ⚠ **指标口径**：libvmaf 必须 `n_subsample=1` —— subsample>1 会**偏置 VMAF**
    #   （同文件 vp9 crf35 差 1.9~3.0，且偏置随编码器而异），会污染等 VMAF 匹配；
    #   标定与判据必须**同参、同时长**。**本表全部数据均以 subsample=1 产出**
    #   （⚠ 早期 workdir `/tmp/eqq_calib/1280x720_10s_n3` 是 subsample=8 的作废数据，
    #   **不参与**本表任何计算）。
    # ⚠ 仅软件编码器；硬编（NVENC/QSV/AMF/VT）未覆盖 ⇒ 自动回退 SIZE_MAP（M4 待上机）。
    'libx265':     (1.0979, -2.3119, 0, 51),
    'libvpx-vp9':  (1.9716, -15.0929, 0, 63),
    'libaom-av1':  (2.3219, -22.3927, 0, 63),
    'libsvtav1':   (2.3961, -21.3615, 0, 63),
    # ⚠ `librav1e` 按 **native 档**（不下发 `-speed`）标定——与 `SIZE_MAP['librav1e']` 的档位一致。
    #   `-speed 10` 档另存 quality_map._EQQUAL_SPEED_OVERRIDE（LOO 5.99，同样达标）。
    'librav1e':    (7.9326, -102.2078, 0, 255),
}"""

OVERRIDE = """# ``-speed 10`` 下的**等质量**标定值（2026-10-02 落表，两仓同源）。
# ``QUALITY_MAP['librav1e']`` 只对 native 档成立；``-speed`` 整体平移码率曲线，
# 故 speed 10 档另存本表。值来源同 QUALITY_MAP（统一锚点 18/21/24/27/30 + 17 条素材
# + 同 key 取均值，720p prep，``n_subsample=1``），LOO worst |ΔVMAF| = **5.99**
# —— 按 rav1e 专用门禁 **≤7.5** 判定为**达标**。
# 空 dict ⇒ 未标定，rav1e 回退 ``QUALITY_MAP`` 的 native 档。
_EQQUAL_SPEED_OVERRIDE = {
    'librav1e': (7.9173, -106.6317, 0, 255),   # 仅当 _RAV1E_SPEED > 0 时生效
}"""

QP_ROWS = """    'libx265':     (1.0979, -2.3119, 0, 51),
    'libvpx-vp9':  (1.9716, -15.0929, 0, 63),
    'libaom-av1':  (2.3219, -22.3927, 0, 63),
    'libsvtav1':   (2.3961, -21.3615, 0, 63),"""

QP_OLD = """    'libx265':     (1.0910, -2.3674, 0, 51),
    'libvpx-vp9':  (2.0156, -16.1483, 0, 63),
    'libaom-av1':  (2.2692, -20.9684, 0, 63),
    'libsvtav1':   (2.1695, -15.8725, 0, 63),"""

# A 仓
p = pathlib.Path('src/utils/convert_crf.py')
s = p.read_text(encoding='utf-8')
st = s.index('QUALITY_MAP = {')
en = s.index('\n}', st) + 2
p.write_text(s[:st] + NEW + s[en:], encoding='utf-8')
print('A仓 QUALITY_MAP → 第八版')

# B 仓
p = pathlib.Path('/mnt/d/Workspace_Python/VidUtils/convert_crf.py')
s = p.read_text(encoding='utf-8')
st = s.index('QUALITY_MAP = {')
en = s.index('\n}', st) + 2
s = s[:st] + NEW + s[en:]
# B 仓的 _EQQUAL_SPEED_OVERRIDE 块
ost = s.index('_EQQUAL_SPEED_OVERRIDE = {')
oen = s.index('\n}', ost) + 2
# 连带替换其上方注释
head_start = s.rindex('\n# ``-speed 10``', 0, ost)
s = s[:head_start + 1] + OVERRIDE + '\n' + s[oen:]
p.write_text(s, encoding='utf-8')
print('B 仓 QUALITY_MAP + override → 第八版')

# A 仓 quality_map.py：override + QP 四行
p = pathlib.Path('src/utils/quality_map.py')
s = p.read_text(encoding='utf-8')
for a, b in [("    'librav1e': (7.9298, -105.2651, 0, 255),   # 仅当 RAV1E_SPEED > 0 时生效",
             "    'librav1e': (7.9173, -106.6317, 0, 255),   # 仅当 RAV1E_SPEED > 0 时生效"),
            (QP_OLD, QP_ROWS)]:
    assert a in s, a[:44]
    s = s.replace(a, b)
s = s.replace('LOO worst |ΔVMAF| = **5.88**', 'LOO worst |ΔVMAF| = **5.99**')
old_qp = """    # 软编行镜像 QUALITY_MAP 的 2026-10-02 **第七版**标定值（统一锚点 18/21/24/27/30，
    # 按素材去重的 12 条素材 + 同 key 取均值合并（顺序无关），720p prep，n_subsample=1；
    # 软编门禁 ≤5.9，实测 x265 4.73 / vp9 4.71 / svtav1 4.68 / aom 5.49 —— **全部达标**）。"""
new_qp = """    # 软编行镜像 QUALITY_MAP 的 2026-10-02 **第八版**标定值（统一锚点 18/21/24/27/30，
    # 17 条素材 + 同 key 取均值合并（顺序无关，3 随机种子验证），720p prep，n_subsample=1；
    # 软编门禁 ≤5.9，实测 x265 4.24 / vp9 3.82 / svtav1 4.88 / aom 5.35 —— **全部达标**）。"""
if old_qp in s:
    s = s.replace(old_qp, new_qp)
p.write_text(s, encoding='utf-8')
print('A 仓 quality_map.py（override + QP 四行）→ 第八版')