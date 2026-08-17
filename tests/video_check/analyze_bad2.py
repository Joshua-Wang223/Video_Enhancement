import cv2, numpy as np

cap = cv2.VideoCapture('/workspace/Video_Enhancement/temp/rank02_sync-drain-loop_pipe9_LA8.mp4')
ratios = []
entropies = []
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
    ratios.append((idx, r))
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    ent = -np.sum(hist * np.log2(hist))
    entropies.append((idx, ent))
    idx += 1
cap.release()

ratios.sort(key=lambda x: -x[1])
print('Top 30 HighVarBlockRatio (bad video):')
for f, r in ratios[:30]:
    print(f'  Frame {f}: ratio={r:.4f}')

entropies.sort(key=lambda x: x[1])
print('\nLowest 10 entropies (bad video):')
for f, e in entropies[:10]:
    print(f'  Frame {f}: entropy={e:.4f}')
