#!/usr/bin/env python3
"""
T4 验证：SDK 13.0 是否接受 rc_ptr[1]=32 (VBR_HQ)

在 T4 机器上运行，确认 SDK ctypes 路径是否仍支持 VBR_HQ。
结果决定 P0 范围：方案 A（仅 CLI 映射）vs 方案 B（CLI + ctypes 双迁移）。

用法: python nvenc_vbr_hq_verify.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from external.ifrnet_video.nvenc_sdk import NVENCEncoder


def verify_vbr_hq_sdk_acceptance():
    """测试 SDK 是否接受 rc_ptr[1]=32 (VBR_HQ)。

    返回:
        True  → SDK 接受 VBR_HQ=32，方案 A（P0 只改 CLI 映射）
        False → SDK 拒绝 VBR_HQ=32，方案 B（ctypes 加 vbr 分支）
    """
    try:
        enc = NVENCEncoder(
            width=320, height=240, fps=30,
            rate_mode="vbr_hq", codec="h264", preset="p4", qp=23
        )
        print("✅ rc_ptr[1]=32 (VBR_HQ) 被 SDK 接受 — 方案 A：P0 只改 CLI 映射")
        enc.close()
        return True
    except Exception as e:
        print(f"❌ rc_ptr[1]=32 被 SDK 拒绝: {e}")
        print("→ 方案 B：P0 范围扩大到 ctypes 加 vbr 分支")
        return False


def verify_vbr_ctypes_fallback():
    """测试 SDK 是否接受 rc_ptr[1]=1 (VBR)。

    仅在方案 B 下有意义——验证新增 vbr 分支的可行性。

    返回:
        True  → SDK 接受 VBR=1，可行
        False → SDK 也不接受 VBR，需另寻方案
    """
    try:
        enc = NVENCEncoder(
            width=320, height=240, fps=30,
            rate_mode="vbr", codec="h264", preset="p4", qp=23
        )
        print("✅ rc_ptr[1]=1 (VBR) 被 SDK 接受 — vbr 分支可行")
        enc.close()
        return True
    except Exception as e:
        print(f"❌ rc_ptr[1]=1 (VBR) 被 SDK 拒绝: {e}")
        print("→ vbr 分支不可行，需评估其他方案")
        return False


def print_sdk_info():
    """打印 SDK 相关信息，辅助判读。"""
    import subprocess

    # FFmpeg 版本
    rc, out, _ = subprocess.run(
        ["ffmpeg", "-hide_banner", "-version"],
        capture_output=True, text=True, timeout=10
    ).values()
    if rc == 0 and out:
        print(f"FFmpeg: {out.split(chr(10))[0]}")

    # NVENC 编码器列表
    rc, out, _ = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True, timeout=10
    ).values()
    if rc == 0:
        nvenc = [l.strip() for l in out.split(chr(10)) if "nvenc" in l.lower()]
        print(f"NVENC encoders: {nvenc}")

    # nvenc_rc_mode_diagnose.py 输出
    diag_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Accessory", "probe", "nvenc_rc_mode_diagnose.py"
    )
    if os.path.exists(diag_script):
        print(f"\n--- nvenc_rc_mode_diagnose.py ---")
        rc, out, err = subprocess.run(
            ["python3", diag_script],
            capture_output=True, text=True, timeout=30
        ).values()
        if rc == 0:
            # 只打印关键行
            for line in out.split(chr(10)):
                if any(k in line for k in ["VBR_HQ", "RC mode", "SDK", "rateControl"]):
                    print(line)
        else:
            print(f"  (rc={rc}) {err[:200]}")


if __name__ == "__main__":
    print("=" * 60)
    print("T4 NVENC vbr_hq 移除 · SDK 接受度验证")
    print("=" * 60)

    print_sdk_info()

    print(f"\n--- SDK ctypes 实编测试 ---")
    sdk_accepts_vbr_hq = verify_vbr_hq_sdk_acceptance()

    if not sdk_accepts_vbr_hq:
        print(f"\n--- SDK ctypes vbr 分支验证 ---")
        verify_vbr_ctypes_fallback()

    print(f"\n--- 结论 ---")
    if sdk_accepts_vbr_hq:
        print("P0 范围：仅 CLI 映射（方案 A）")
        print("  - ffmpeg_io.py: _NVENC_RC_MAP['vbr_hq'] → 'vbr'")
        print("  - ffmpeg_io.py: vbr 路径追加 -tune hq -multipass fullres")
        print("  - nvenc_sdk.py: 不动")
    else:
        print("P0 范围：CLI + ctypes 双迁移（方案 B）")
        print("  - ffmpeg_io.py: 同上")
        print("  - nvenc_sdk.py: 新增 vbr 分支 (rc_ptr[1]=1)")
        print("  - config/default_config.json: rate_mode vbr_hq → vbr")
        print("  - main.py: 默认参数 vbr_hq → vbr")