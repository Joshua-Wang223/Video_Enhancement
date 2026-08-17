import cv2, numpy as np

for label, path in [("NORMAL", "/workspace/input_videos/word_world_2.mp4"),
                      ("BAD", "/workspace/Video_Enhancement/temp/rank02_sync-drain-loop_pipe9_LA8.mp4")]:
    cap = cv2.VideoCapture(path)
    idx = 0
    max_ratios_by_vt = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        bs = 16
        blocks = []
        for y in range(0, h, bs):
            for x in range(0, w, bs):
                b = gray[y:min(y+bs,h), x:min(x+bs,w)]
                if b.size > 0:
                    blocks.append(np.var(b))
        for vt in [800, 1200, 1600, 2000, 2500, 3000, 4000, 5000]:
            r = sum(1 for v in blocks if v > vt) / len(blocks)
            if vt not in max_ratios_by_vt:
                max_ratios_by_vt[vt] = 0
            if r > max_ratios_by_vt[vt]:
                max_ratios_by_vt[vt] = r
        idx += 1
    cap.release()

    print(f"\n=== {label} (max block ratio at different var_thresh) ===")
    for vt in sorted(max_ratios_by_vt.keys()):
        print(f"  var_thresh={vt}: max_ratio={max_ratios_by_vt[vt]:.4f}")
