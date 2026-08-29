import faulthandler
import os
import shutil
import subprocess
import sys
import tempfile
import threading


def main():
    faulthandler.dump_traceback_later(15, exit=True)
    writer = (
        "import sys\n"
        "sys.stderr.write('e' * 300000)\n"   # > 64KB stderr, never flushed/closed
        "sys.stdout.buffer.write(b'x')\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stderr.flush()\n"
    )
    fd, tmp_path = tempfile.mkstemp(prefix='deadlock_', suffix='.es')
    try:
        proc = subprocess.Popen([sys.executable, '-c', writer],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if '--fixed' in sys.argv:
            # [v7] 修复后的模式: 双线程并发排空 stdout/stderr + wait(timeout)
            out_ok = {}
            err_chunks = []

            def _drain_out():
                try:
                    with os.fdopen(fd, 'wb') as out_f:
                        shutil.copyfileobj(proc.stdout, out_f, 64 * 1024)
                    out_ok['ok'] = True
                except Exception as e:
                    out_ok['exc'] = e

            def _drain_err():
                try:
                    while True:
                        chunk = proc.stderr.read(65536)
                        if not chunk:
                            break
                        err_chunks.append(chunk)
                except Exception:
                    pass

            t_out = threading.Thread(target=_drain_out, daemon=True)
            t_err = threading.Thread(target=_drain_err, daemon=True)
            t_out.start()
            t_err.start()
            try:
                rc = proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                print('FIXED-PATTERN FAIL: 并发排水仍超时（不应发生）')
                return
            t_out.join(timeout=5)
            t_err.join(timeout=5)
            print('FIXED-PATTERN OK: rc=%d stderr_bytes=%d ok=%s'
                  % (rc, sum(len(x) for x in err_chunks), out_ok.get('ok')))
        else:
            # 旧模式（v4 死锁复现）: 先排 stdout 再读 stderr
            with os.fdopen(fd, 'wb') as out_f:
                shutil.copyfileobj(proc.stdout, out_f, 64 * 1024)
            proc.stderr.read()
            proc.wait()
            print('OLD-PATTERN OK (意外): 未复现死锁')
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == '__main__':
    main()
