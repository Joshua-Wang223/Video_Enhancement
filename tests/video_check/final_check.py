import cv2, numpy as np
for label, path in [("NORMAL", "/workspace/input_videos/word_world_2.mp4"),
                      ("BAD", "/workspace/Video_Enhancement/temp/rank02_sync-drain-loop_pipe9_LA8.mp4")]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_mse, cnt_mse_gt_4000, cnt_mse_gt_3000 = 0, 0, 0
    prev, idx = None, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            mse = np.sum((prev.astype('float')-gray.astype('float'))**2)/(gray.shape[0]*gray.shape[1])
            if mse > max_mse: max_mse = mse
            if mse > 4000: cnt_mse_gt_4000 += 1
            if mse > 3000: cnt_mse_gt_3000 += 1
        prev = gray.copy(); idx += 1
    cap.release()
    print(f"{label}: read={idx}/{total}, max_MSE={max_mse:.1f}, MSE>4000={cnt_mse_gt_4000}, MSE>3000={cnt_mse_gt_3000}")
