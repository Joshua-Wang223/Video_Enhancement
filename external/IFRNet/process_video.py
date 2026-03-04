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
    def __init__(self, model_path=f'{models_ifrnet}/IFRNet_S_Vimeo90K.pth', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = Model()

        print(f"加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

        print(f"IFRNet 模型加载完成,使用设备: {self.device}")

    def process_frame_pair(self, img0, img1, timestep=0.5):
        """处理一对帧,生成中间帧"""
        # 转换图像为tensor
        img0_tensor = self.to_tensor(img0).unsqueeze(0).to(self.device)
        img1_tensor = self.to_tensor(img1).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            embt = torch.FloatTensor([timestep]).view(1, 1, 1, 1).to(self.device)
            imgt = img0_tensor * (1 - timestep) + img1_tensor * timestep
            output = self.model(img0_tensor, img1_tensor, embt, imgt)

            # 修复:处理模型返回的元组
            if isinstance(output, tuple):
                pred = output[0]  # 取第一个元素(预测的帧)
            else:
                pred = output

        # 转换为PIL图像
        pred_tensor = pred.squeeze(0) if pred.dim() == 4 else pred
        pred_img = self.to_pil(pred_tensor.cpu().clamp(0, 1))
        return pred_img

    def process_video(self, input_path, output_path, scale=2.0):
        """处理整个视频"""
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

        print(f"开始插帧处理,目标帧率: {new_fps:.2f} FPS")

        # 读取第一帧
        ret, prev_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            return False

        frame_count = 0
        processed_frames = 0

        # 写入第一帧
        out.write(prev_frame)
        processed_frames += 1
        prev_pil = Image.fromarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))

        # 计算每个百分比对应的帧数
        target_frames = [
            int(total_frames * 0.25),  # 25%
            int(total_frames * 0.50),  # 50%
            int(total_frames * 0.75),  # 75%
            total_frames              # 100%
        ]
        target_index = 0

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            frame_count += 1
            curr_pil = Image.fromarray(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))

            try:
                if scale == 2.0:
                    # 生成1个中间帧
                    mid_pil = self.process_frame_pair(prev_pil, curr_pil, timestep=0.5)
                    mid_frame = cv2.cvtColor(np.array(mid_pil), cv2.COLOR_RGB2BGR)
                    out.write(mid_frame)
                    processed_frames += 1

                elif scale == 4.0:
                    # 生成3个中间帧
                    for i in range(1, 4):
                        timestep = i / 4.0
                        mid_pil = self.process_frame_pair(prev_pil, curr_pil, timestep=timestep)
                        mid_frame = cv2.cvtColor(np.array(mid_pil), cv2.COLOR_RGB2BGR)
                        out.write(mid_frame)
                        processed_frames += 1

                else:
                    # 自定义倍数
                    num_inter_frames = int(scale) - 1
                    for i in range(1, num_inter_frames + 1):
                        timestep = i / scale
                        mid_pil = self.process_frame_pair(prev_pil, curr_pil, timestep=timestep)
                        mid_frame = cv2.cvtColor(np.array(mid_pil), cv2.COLOR_RGB2BGR)
                        out.write(mid_frame)
                        processed_frames += 1

            except Exception as e:
                print(f"处理帧 {frame_count} 时出错: {e}")

            # 写入当前帧
            out.write(curr_frame)
            processed_frames += 1
            prev_pil = curr_pil

            # if frame_count % 10 == 0:
            #     print(f"已处理: {frame_count}/{total_frames} 帧")
            # 进度检查：只在达到特定帧数时打印
            if target_index < len(target_frames) and frame_count >= target_frames[target_index]:
                progress_percent = (frame_count / total_frames) * 100
                print(f"已处理: {frame_count}/{total_frames} 帧 ({progress_percent:.1f}%)")
                target_index += 1

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

    parser = argparse.ArgumentParser(description='IFRNet 视频插帧处理')
    parser.add_argument('--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, required=True, help='输出视频路径')
    parser.add_argument('--scale', type=float, default=2.0, help='插帧倍数')
    parser.add_argument('--model', type=str, default='IFRNet_S_Vimeo90K', help='模型名称')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                        help='使用的设备 (cuda 或 cpu)')

    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("步骤1: 使用 IFRNet 进行视频插帧")
    print("=" * 60)
    print(f"使用模型: {args.model}")
    print(f"插帧倍数: {args.scale}x")
    print(f"使用设备: {args.device}")
    print()

    # 确定模型路径
    if args.model == 'IFRNet_Vimeo90K':
        model_path = f'{models_ifrnet}/IFRNet_Vimeo90K.pth'
    elif args.model == 'IFRNet_S_Vimeo90K':
        model_path = f'{models_ifrnet}/IFRNet_S_Vimeo90K.pth'
    elif args.model == 'IFRNet_L_Vimeo90K':
        model_path = f'{models_ifrnet}/IFRNet_L_Vimeo90K.pth'
    else:
        model_path = args.model

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return

    # 使用指定的设备
    processor = IFRNetVideoProcessor(model_path=model_path, device=args.device)
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
