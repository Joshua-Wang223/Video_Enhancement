import cv2, numpy as np

for label, path in [("NORMAL", "/workspace/input_videos/word_world_2.mp4"),
                      ("BAD", "/workspace/Video_Enhancement/temp/rank02_sync-drain-loop_pipe9_LA8.mp4")]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ratios = []
    entropies = []
    lap_vars = []
    mse_vals = []
    prev_gray = None
    idx = 0
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
        r = sum(1 for v in blocks if v > 800) / len(blocks)
        ratios.append(r)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap_vars.append(lap.var())
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        hist = hist[hist > 0]
        e = -np.sum(hist * np.log2(hist))
        entropies.append(e)
        if prev_gray is not None:
            m = np.sum((prev_gray.astype('float') - gray.astype('float'))**2) / (h*w)
            mse_vals.append(m)
        prev_gray = gray.copy()
        idx += 1
    cap.release()

    print(f"\n=== {label} ({idx}/{total} frames) ===")
    print(f"Entropy:     min={min(entropies):.4f} max={max(entropies):.4f}")
    print(f"LapVar:      min={min(lap_vars):.2f} max={max(lap_vars):.2f}")
    print(f"BlockRatio:  min={min(ratios):.4f} max={max(ratios):.4f}")
    print(f"MSE:         min={min(mse_vals):.4f} max={max(mse_vals):.2f}")
    # key thresholds
    cnt_r28 = sum(1 for r in ratios if r > 0.28)
    cnt_r26 = sum(1 for r in ratios if r > 0.26)
    cnt_r24 = sum(1 for r in ratios if r > 0.24)
    cnt_e3  = sum(1 for e in entropies if e < 3.0)
    cnt_e4  = sum(1 for e in entropies if e < 4.0)
    cnt_e5  = sum(1 for e in entropies if e < 5.0)
    print(f"Frames ratio>0.28: {cnt_r28}, >0.26: {cnt_r26}, >0.24: {cnt_r24}")
    print(f"Frames ent<3.0: {cnt_e3}, ent<4.0: {cnt_e4}, ent<5.0: {cnt_e5}")
    # combo: high ratio + low entropy
    combo = sum(1 for i in range(min(len(ratios),len(entropies))) if ratios[i] > 0.24 and entropies[i] < 5.0)
    print(f"Frames ratio>0.24 AND ent<5.0: {combo}")
