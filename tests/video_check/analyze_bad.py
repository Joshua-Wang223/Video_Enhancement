import cv2, numpy as np

cap = cv2.VideoCapture('/workspace/Video_Enhancement/temp/rank02_sync-drain-loop_pipe9_LA8.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'总帧数声明: {total}, 帧率: {fps:.2f}')

max_lap = 0
min_lap = float('inf')
max_mse_val = 0
min_mse_val = float('inf')
black_count = 0
white_count = 0
corrupt_ratio_25 = 0
prev_gray = None
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print(f'cap.read() 在帧 {frame_idx} 处返回 False')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    var_val = np.var(gray)

    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_var = lap.var()
    if lap_var > max_lap: max_lap = lap_var
    if lap_var < min_lap: min_lap = lap_var

    if mean_val < 8 and var_val < 4:
        black_count += 1
    if mean_val > 250 and var_val < 4:
        white_count += 1

    # block-based corruption
    h, w = gray.shape
    bs = 16
    blocks = []
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            block = gray[y:min(y+bs,h), x:min(x+bs,w)]
            if block.size > 0:
                blocks.append(np.var(block))
    high_var_blocks = sum(1 for v in blocks if v > 800)
    ratio = high_var_blocks / len(blocks)
    if ratio > 0.25:
        corrupt_ratio_25 += 1

    if prev_gray is not None:
        mse = np.sum((prev_gray.astype('float') - gray.astype('float'))**2) / (gray.shape[0]*gray.shape[1])
        if mse > max_mse_val: max_mse_val = mse
        if mse < min_mse_val: min_mse_val = mse
    prev_gray = gray.copy()
    frame_idx += 1

cap.release()
print(f'实际读取帧数: {frame_idx}')
print(f'LaplacianVar 范围: {min_lap:.2f} ~ {max_lap:.2f}')
print(f'帧间MSE 范围: {min_mse_val:.5f} ~ {max_mse_val:.2f}')
print(f'黑屏帧数: {black_count}, 白屏帧数: {white_count}')
print(f'分块法ratio>0.25的帧数: {corrupt_ratio_25}')
