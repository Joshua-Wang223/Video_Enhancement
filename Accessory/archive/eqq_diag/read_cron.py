import json
import datetime

p = '/mnt/d/Workspace_Python/Video_Enhancement/.codebuddy/scheduled_task_executions.json'
d = json.load(open(p))
print('顶层类型:', type(d).__name__)
if isinstance(d, dict):
    print('keys:', list(d)[:8])

execs = d.get('executions', d) if isinstance(d, dict) else d
if isinstance(execs, dict):
    execs = list(execs.values())
if not isinstance(execs, list):
    execs = [execs]
print('执行记录 %d 条\n' % len(execs))

for e in execs[-14:]:
    if not isinstance(e, dict):
        print(' ', str(e)[:140])
        continue
    ts = e.get('timestamp') or e.get('startedAt') or e.get('time') or e.get('createdAt')
    t = '-'
    if ts and isinstance(ts, (int, float)) and ts > 1e11:
        t = datetime.datetime.fromtimestamp(ts / 1000).strftime('%m-%d %H:%M:%S')
    tid = str(e.get('taskId') or e.get('id') or '?')[:10]
    st = e.get('status') or e.get('state') or '?'
    err = str(e.get('error') or e.get('message') or '')[:70]
    print('  %-18s task=%-10s status=%-10s %s' % (t, tid, st, err))

# 统计本任务
mine = [e for e in execs if isinstance(e, dict) and str(e.get('taskId', '')).startswith('97ce9693')]
print('\n本任务(97ce9693) 执行 %d 次' % len(mine))
