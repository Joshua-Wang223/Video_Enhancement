#!/usr/bin/env python
"""故障注入包装器：定时 dump Python 栈，定位 IFRNet NVENC 挂死点。

用法:
  IFRNET_NVENC_MAX_BS_BYTES=1024 python Accessory/infra/fault_injection_wrapper.py 150 \
      --frames 100 --codec hevc --la 8 --rc vbr_hq --qp 21 --chunk 32 --segments 1 \
      --out temp/retest/A1/prof_small.mp4
"""
import faulthandler
import runpy
import sys

_TIMEOUT = float(sys.argv[1])
sys.argv = [sys.argv[0]] + sys.argv[2:]
faulthandler.enable()          # SIGSEGV 时也 dump Python 栈
faulthandler.dump_traceback_later(_TIMEOUT, exit=True)
runpy.run_path(str(Path(__file__).resolve().parent.parent / "probe" / "ifrnet_lookahead_repro.py"), run_name="__main__")
