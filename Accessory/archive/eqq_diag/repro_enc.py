import importlib.util, pathlib, sys
s=importlib.util.spec_from_file_location('ch','Accessory/probe/calibrate_equal_quality.py')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
w=pathlib.Path('temp/_rav1e_probe2'); w.mkdir(parents=True,exist_ok=True)
prep=w/'prep.mp4'
if not prep.is_file():
    m.run(['ffmpeg','-nostdin','-y','-hide_banner','-loglevel','error',
           '-i','/mnt/d/Workspace_Python/input_videos/new5_raw.mp4','-t','6','-an',
           '-vf','scale=1280:720:flags=lanczos','-c:v','libx264','-preset','veryfast',
           '-crf','10','-pix_fmt','yuv420p',str(prep)])
print('prep size', prep.stat().st_size, flush=True)
out=w/'librav1e_63.mp4'
try:
    print('encode OK', m.encode(prep,'librav1e',63,out), flush=True)
except Exception as e:
    print('encode FAILED:', type(e).__name__, flush=True)
    print(str(e)[-1200:], flush=True)
print('exists after:', out.is_file(), out.stat().st_size if out.is_file() else '-', flush=True)
