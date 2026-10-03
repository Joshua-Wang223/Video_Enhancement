import re
STEPS = [('QUALITY_MAP_QUALITY_QP', 'QUALITY_MAP_QP'), ('QUALITY_MAP_QUALITY', '__EQ_TMP__'),
         ('QUALITY_MAP', 'SIZE_MAP'), ('__EQ_TMP__', 'QUALITY_MAP')]
BASE = '/mnt/d/Workspace_Python/'
for f in ['Video_Enhancement/src/utils/convert_crf.py',
          'Video_Enhancement/src/utils/quality_map.py']:
    s = open(BASE + f, encoding='utf-8').read()
    for a, b in STEPS:
        s = s.replace(a, b)
    print(f'── {f} ──')
    for ln in s.splitlines():
        if re.match(r'^\s*(SIZE_MAP|QUALITY_MAP|QUALITY_MAP_QP)\s*=', ln):
            print('   ', ln.split('#')[0].rstrip())
        elif re.search(r'^\s*_active_table|^\s*def _eqvol_model|^\s*def _qp_model', ln):
            print('   ', ln.rstrip())
