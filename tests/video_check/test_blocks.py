import cv2, numpy as np

for label, path in [("NORMAL", "/workspace/input_videos/word_world_2.mp4"),
                      ("BAD", "/workspace/Video_Enhancement/temp/rank02_sync-drain-loop_pipe9_LA8.mp4")]:
    cap = cv2.VideoCapture(path)
    all_ratios_by_vt = {vt: [] for vt in [800, 1200, 1600, 2000, 2500, 3000, 4000, 5000]}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape; bs = 16
        blocks = []
        for y in range(0, h, bs):
            for x in range(0, w, bs):
                b = gray[y:min(y+bs,h), x:min(x+bs,w)]
                if b.size > 0: blocks.append(np.var(b))
        blocks.sort(reverse=True)
        percentiles = {p: blocks[int(p*len(blocks)/100)] if int(p*len(blocks)/100)<len(blocks) else 0 for p in [50, 75, 90, 95, 99]}
        if idx < 5:
            print(f"[{label}] Frame {idx} block var: p50={percentiles[50]:.0f} p95={percentiles[95]:.0f} p99={percentiles[99]:.0f} max={blocks[0]:.0f}")
        for vt in all_ratios_by_vt:
            all_ratios_by_vt[vt].append(sum(1 for v in blocks if v > vt) / len(blocks))
        idx += 1
    cap.release()

    print(f"\n=== {label}: Frames with ratio > threshold at different var_thresh ===")
    for vt in sorted(all_ratios_by_vt.keys()):
        for rt in [0.05, 0.10, 0.15, 0.20, 0.25]:
            cnt = sum(1 for r in all_ratios_by_vt[vt] if r > rt)
            if cnt > 0:
                print(f"  var_thresh={vt}, ratio>{rt}: {cnt} frames")
