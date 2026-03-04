import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import warnings
warnings.filterwarnings('ignore')

# 添加 IFRNet 到 Python 路径
base_dir = '/workspace/video_enhancement'
models_ifrnet = f'{base_dir}/models_ifrnet/checkpoints'

sys.path.insert(0, f'{base_dir}/IFRNet')
sys.path.insert(0, f'{models_ifrnet}')

from models.IFRNet_S import Model

class IFRNetVideoProcessor:
    def __init__(self, model_path=f'{models_ifrnet}/IFRNet_S_Vimeo90K.pth', device='cuda', batch_size=4):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.model = Model()

        print(f"加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

        print(f"IFRNet 模型加载完成, 使用设备: {self.device}, 批大小: {self.batch_size}")

    def process_frame_batch(self, img0_list, img1_list, timestep_list):
        """
        处理一批帧对，生成对应时间步的中间帧。
        参数：
            img0_list: list of PIL Images (起始帧)
            img1_list: list of PIL Images (结束帧)
            timestep_list: list of float (每个帧对的时间步)
        返回：
            list of PIL Images (预测帧)，顺序与输入帧对一致
        """
        # 转换为 tensor 并堆叠成 batch
        img0_tensors = []
        img1_tensors = []
        for img0, img1 in zip(img0_list, img1_list):
            img0_tensors.append(self.to_tensor(img0).unsqueeze(0))
            img1_tensors.append(self.to_tensor(img1).unsqueeze(0))
        img0_batch = torch.cat(img0_tensors, dim=0).to(self.device)  # (B, C, H, W)
        img1_batch = torch.cat(img1_tensors, dim=0).to(self.device)

        # 构造 embt batch (B, 1, 1, 1)
        embt_batch = torch.tensor(timestep_list, dtype=torch.float32).view(-1, 1, 1, 1).to(self.device)

        # 粗略线性插值作为辅助输入
        imgt_batch = img0_batch * (1 - embt_batch) + img1_batch * embt_batch

        with torch.no_grad():
            output = self.model(img0_batch, img1_batch, embt_batch, imgt_batch)
            if isinstance(output, tuple):
                pred_batch = output[0]  # 模型可能返回 (pred, flow, ...)
            else:
                pred_batch = output

        # 转回 PIL 列表
        pred_list = []
        for i in range(pred_batch.size(0)):
            pred_tensor = pred_batch[i].cpu().clamp(0, 1)
            pred_list.append(self.to_pil(pred_tensor))
        return pred_list

    def process_video(self, input_path, output_path, scale=2.0):
        """处理整个视频，使用批处理提高 GPU 利用率"""
        if not os.path.exists(input_path):
            print(f"错误: 输入文件不存在 - {input_path}")
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"无法打开视频: {input_path}")
            return False

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height}, {fps:.2f} FPS, {total_frames} 帧")
        new_fps = fps * scale
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))

        # 根据 scale 确定需要生成的时间步列表
        if scale == 2.0:
            timesteps = [0.5]
        elif scale == 4.0:
            timesteps = [0.25, 0.5, 0.75]
        else:
            # 通用：生成 scale-1 个均匀时间步
            timesteps = [i / scale for i in range(1, int(scale))]

        print(f"开始批处理插帧, 目标帧率: {new_fps:.2f} FPS, 时间步: {timesteps}")

        # 缓冲区存储原始帧（PIL 格式）
        frame_buffer = []
        frame_count = 0
        processed_frames = 0

        # 读取第一帧并写入
        ret, first_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            return False
        out.write(first_frame)
        processed_frames += 1
        frame_buffer.append(Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)))
        frame_count += 1

        # 进度报告点
        progress_thresholds = [int(total_frames * p) for p in [0.25, 0.5, 0.75]]
        next_progress = 0

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            frame_count += 1
            curr_pil = Image.fromarray(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))
            frame_buffer.append(curr_pil)

            # 当缓冲区达到 batch_size+1 时，处理一批
            if len(frame_buffer) == self.batch_size + 1:
                # 提取帧对：img0 = buffer[0:batch_size], img1 = buffer[1:batch_size+1]
                img0_list = frame_buffer[:-1]
                img1_list = frame_buffer[1:]

                # 对每个时间步进行批处理
                step_preds = []  # 每个元素是对应时间步的预测列表，长度为 batch_size
                for t in timesteps:
                    # 每个帧对使用相同的时间步
                    t_list = [t] * len(img0_list)
                    preds = self.process_frame_batch(img0_list, img1_list, t_list)
                    step_preds.append(preds)

                # 按顺序写入：中间帧 + 原始帧（第一个原始帧已写入，这里从第二个开始）
                for i in range(self.batch_size):
                    # 写入所有时间步的中间帧（第 i 个帧对的）
                    for step_pred in step_preds:
                        mid_frame = cv2.cvtColor(np.array(step_pred[i]), cv2.COLOR_RGB2BGR)
                        out.write(mid_frame)
                        processed_frames += 1
                    # 写入原始帧（第 i+1 帧）
                    orig_frame = cv2.cvtColor(np.array(img1_list[i]), cv2.COLOR_RGB2BGR)
                    out.write(orig_frame)
                    processed_frames += 1

                # 更新缓冲区：保留最后一个帧作为下一批的起始
                frame_buffer = [frame_buffer[-1]]

            # 进度报告
            if next_progress < len(progress_thresholds) and frame_count >= progress_thresholds[next_progress]:
                print(f"已处理: {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%)")
                next_progress += 1

        # 处理剩余帧对（缓冲区中可能还有不足 batch_size+1 的帧）
        if len(frame_buffer) > 1:
            img0_list = frame_buffer[:-1]
            img1_list = frame_buffer[1:]
            step_preds = []
            for t in timesteps:
                t_list = [t] * len(img0_list)
                preds = self.process_frame_batch(img0_list, img1_list, t_list)
                step_preds.append(preds)

            for i in range(len(img0_list)):
                for step_pred in step_preds:
                    mid_frame = cv2.cvtColor(np.array(step_pred[i]), cv2.COLOR_RGB2BGR)
                    out.write(mid_frame)
                    processed_frames += 1
                # 写入当前帧对的结束帧（即 img1_list[i]）
                orig_frame = cv2.cvtColor(np.array(img1_list[i]), cv2.COLOR_RGB2BGR)
                out.write(orig_frame)
                processed_frames += 1

        cap.release()
        out.release()

        print(f"插帧完成!")
        print(f"原始帧数: {frame_count}")
        print(f"处理后帧数: {processed_frames}")
        print(f"输出视频: {output_path}")
        return True

def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description='IFRNet 视频插帧处理（GPU 并行批处理版）')
    parser.add_argument('--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, required=True, help='输出视频路径')
    parser.add_argument('--scale', type=float, default=2.0, help='插帧倍数')
    parser.add_argument('--model', type=str, default='IFRNet_S_Vimeo90K', help='模型名称')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='使用的设备')
    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小（帧对数量），根据显存调整')

    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("步骤1: 使用 IFRNet 进行视频插帧（GPU 并行批处理）")
    print("=" * 60)
    print(f"使用模型: {args.model}")
    print(f"插帧倍数: {args.scale}x")
    print(f"使用设备: {args.device}")
    print(f"批处理大小: {args.batch_size}")
    print()

    # 确定模型路径
    model_name_map = {
        'IFRNet_Vimeo90K': 'IFRNet_Vimeo90K.pth',
        'IFRNet_S_Vimeo90K': 'IFRNet_S_Vimeo90K.pth',
        'IFRNet_L_Vimeo90K': 'IFRNet_L_Vimeo90K.pth'
    }
    if args.model in model_name_map:
        model_path = f'{models_ifrnet}/{model_name_map[args.model]}'
    else:
        model_path = args.model

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return

    processor = IFRNetVideoProcessor(model_path=model_path, device=args.device, batch_size=args.batch_size)
    success = processor.process_video(args.input, args.output, args.scale)

    end_time = time.time()
    elapsed_minutes = int((end_time - start_time) / 60)
    elapsed_seconds = int((end_time - start_time) % 60)

    print(f"⏱️  插帧耗时: {elapsed_minutes}分{elapsed_seconds}秒")
    if success:
        print("✅ 视频插帧处理完成")
        if os.path.exists(args.output):
            print(f"✅ 输出文件已创建: {args.output}")
    else:
        print("❌ 视频插帧处理失败")

if __name__ == '__main__':
    main()